"""
Temporal Trainer for SpatioTemporalGNN
=======================================
Trains the LSTM + HeteroGATv2 model on quarterly HeteroData snapshot lists
produced by graph_builder.build_temporal_graphs().

Key design:
- For each snapshot at index i, builds LSTM input by looking back T=12 quarters.
- Companies absent in a lookback snapshot receive a zero feature vector.
- Class weight is computed globally from the training set.
- Early stopping is on val Average Precision (primary metric for imbalanced data).
- Val/test snapshots can look back into training history for their LSTM sequences.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from training.metrics import (
    compute_metrics, compute_sector_metrics, compute_tier_recall,
    compute_tier_calibration, compute_tier_metrics,
    GICS_NAMES, print_metrics,
)

logger = logging.getLogger(__name__)

T_LOOKBACK = 12   # LSTM sequence length (quarters)


# ---------------------------------------------------------------------------
# Sequence builder
# ---------------------------------------------------------------------------

def build_sequence_tensor(
    graphs: List[HeteroData],
    i: int,
    T: int = T_LOOKBACK,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Build the LSTM input tensor for the snapshot at index i.

    For each company in graphs[i], collects its feature vector from the
    T most recent snapshots in [i-T+1 ... i], zero-padding when absent.

    Parameters
    ----------
    graphs : list of HeteroData
        Ordered list of quarterly snapshot graphs (train or train+val+test).
    i : int
        Index of the current snapshot in `graphs`.
    T : int
        Lookback length (default 12 quarters = 3 years).
    device : torch.device

    Returns
    -------
    seq : Tensor [N, T, in_dim]
        N = number of companies in graphs[i]
        T = lookback length
        in_dim = feature dimension (graphs[i]["company"].x.size(1))
    """
    g_current = graphs[i]
    gvkeys: List[str] = g_current["company"].gvkey
    N = len(gvkeys)
    in_dim = g_current["company"].x.size(1)

    # Build per-snapshot lookup: {gvkey -> np.array(in_dim)}
    lookback_range = list(range(max(0, i - T + 1), i + 1))
    pad_len = T - len(lookback_range)  # left-padding if history is short

    snapshot_lookups: List[Dict[str, np.ndarray]] = []
    for j in lookback_range:
        g = graphs[j]
        lookup = {
            gvk: g["company"].x[k].cpu().numpy()
            for k, gvk in enumerate(g["company"].gvkey)
        }
        snapshot_lookups.append(lookup)

    seq = np.zeros((N, T, in_dim), dtype=np.float32)
    for t_offset, lookup in enumerate(snapshot_lookups):
        t_pos = pad_len + t_offset
        for n_i, gvk in enumerate(gvkeys):
            if gvk in lookup:
                seq[n_i, t_pos, :] = lookup[gvk]

    return torch.tensor(seq, dtype=torch.float, device=device)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class TemporalTrainer:
    """
    Train SpatioTemporalGNN on quarterly HeteroData snapshot lists.

    Parameters
    ----------
    model : nn.Module
        SpatioTemporalGNN instance.
    train_graphs : List[HeteroData]
        Training snapshots (2001-Q1 to 2016-Q4).
    val_graphs : List[HeteroData]
        Validation snapshots (2017-Q1 to 2019-Q4).
    test_graphs : List[HeteroData]
        Test snapshots (2020-Q1 to 2024-Q4).
    device : torch.device
    lr : float
    weight_decay : float
    patience : int
        Early stopping patience in epochs.
    pos_weight_cap : float
        Maximum class weight for positive class (prevents training instability).
    survival_alpha : float
        DeepHit mixing: alpha * NLL + (1-alpha) * ranking_loss. Default 0.8.
    t_lookback : int
        LSTM sequence length. 12 for quarterly (3-year history); 52 for weekly
        (1-year history). Default T_LOOKBACK=12.
    force_binary_loss : bool
        If True, always use the 3-bin BCE/survival fallback loss regardless of
        whether y_bin labels are present. Use for weekly cadence where quarterly
        features create identical feature vectors across 52 different y_bin targets,
        making the 52-bin DeepHit loss untrainable.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_graphs: List[HeteroData],
        val_graphs: List[HeteroData],
        test_graphs: List[HeteroData],
        device: torch.device,
        lr: float = 5e-4,
        weight_decay: float = 1e-3,
        patience: int = 15,
        pos_weight_cap: float = 10.0,
        survival_alpha: float = 0.5,
        t_lookback: int = T_LOOKBACK,
        train_stride: int = 1,
        force_binary_loss: bool = False,
    ):
        self.model = model.to(device)
        self.train_graphs = train_graphs
        self.val_graphs = val_graphs
        self.test_graphs = test_graphs
        self.device = device
        self.patience = patience
        self.pos_weight_cap = pos_weight_cap
        self.survival_alpha = survival_alpha
        self.t_lookback = t_lookback
        self.train_stride = train_stride
        self.force_binary_loss = force_binary_loss

        # Pre-build shuffled-cycle offset schedule for stochastic monthly stride.
        # Each epoch draws the next offset from a repeating shuffled cycle of
        # [0, 1, ..., stride-1], guaranteeing uniform coverage across all epochs
        # while varying the snapshot set seen each epoch (temporal dropout effect).
        # Val always uses all snapshots regardless of stride.
        if train_stride > 1:
            cycle = list(range(train_stride))
            np.random.shuffle(cycle)
            self._stride_cycle = cycle
        else:
            self._stride_cycle = [0]

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Halve LR when val AP stops improving (patience=10 epochs, min_lr=1e-6)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6
        )

        # Per-horizon class weights from training set
        all_y_4q = torch.cat([g["company"].y.float() for g in train_graphs])
        n_neg = float((all_y_4q == 0).sum())
        n_pos = float((all_y_4q == 1).sum())

        def _pw(label_attr):
            all_y = torch.cat([getattr(g["company"], label_attr).float()
                               for g in train_graphs])
            _neg = float((all_y == 0).sum())
            _pos = float((all_y == 1).sum())
            return min(_neg / max(_pos, 1.0), pos_weight_cap)

        self.pw_1q = _pw("y_1q") if hasattr(train_graphs[0]["company"], "y_1q") else _pw("y")
        self.pw_2q = _pw("y_2q") if hasattr(train_graphs[0]["company"], "y_2q") else _pw("y")
        self.pw_4q = _pw("y")
        self.pos_weight = self.pw_4q  # keep for logging

        self.history: Dict[str, List[float]] = {
            "train_loss": [], "train_auc": [], "train_ap": [],
            "val_loss":   [], "val_auc":   [], "val_ap":   [],
        }
        self.best_val_ap = 0.0
        self.patience_counter = 0
        self.best_state: Optional[Dict] = None

        # State buffer for Temporal Contagion Memory (B1 + B3).
        # Keyed by integer gvkey -> {'h': Tensor[lstm_hidden], 'hazard': Tensor[1]}.
        # Reset at the start of each training epoch; populated after every forward pass.
        # During evaluate(), a separate _eval_buffer is used so eval state never
        # pollutes the training buffer.
        self._state_buffer: Dict[int, Dict] = {}

        # Precompute combined list (train then val then test) for lookback
        self._all_graphs = train_graphs + val_graphs + test_graphs
        self._val_start  = len(train_graphs)
        self._test_start = len(train_graphs) + len(val_graphs)

        logger.info(
            "TemporalTrainer: %d train | %d val | %d test snapshots | "
            "pos_weight=%.2f (pos=%d, neg=%d)",
            len(train_graphs), len(val_graphs), len(test_graphs),
            self.pos_weight, int(n_pos), int(n_neg),
        )
        stride_note = (f"  stride={train_stride} -> ~{len(train_graphs)//train_stride} snaps/epoch "
                       f"(shuffled-cycle, full coverage per {train_stride} epochs)"
                       if train_stride > 1 else "")
        print(
            f"TemporalTrainer ready: {len(train_graphs)} train / "
            f"{len(val_graphs)} val / {len(test_graphs)} test snapshots\n"
            f"  pos_weight = {self.pos_weight:.2f}  "
            f"(n_pos={int(n_pos)}, n_neg={int(n_neg)})"
            + (f"\n{stride_note}" if stride_note else "")
        )

        # Pre-build all LSTM sequence tensors on CPU once.
        # Avoids rebuilding T-step lookback dicts every forward pass every epoch.
        # For weekly T=52 this is the critical optimisation (~43k lookups -> 0).
        total = len(self._all_graphs)
        print(f"  Pre-computing {total} sequence tensors (T={t_lookback}) ...", end=" ", flush=True)
        self._seq_cache: List[torch.Tensor] = [
            build_sequence_tensor(self._all_graphs, i, T=t_lookback, device=torch.device("cpu"))
            for i in range(total)
        ]
        print("done.")

    # ------------------------------------------------------------------
    # State-buffer helpers (Temporal Contagion Memory B1 + B3)
    # ------------------------------------------------------------------

    def _lookup_prev_state(
        self, g: HeteroData, buffer: Dict
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Build [N, lstm_hidden] h_prev and [N, 1] hazard_prev from buffer.

        Firms not in the buffer get zero vectors (cold-start handling).
        Returns (None, None) if the buffer is empty or the model has no
        use_momentum / use_hazard_attn flags set.
        """
        use_m = getattr(self.model, 'use_momentum',    False)
        use_h = getattr(self.model, 'use_hazard_attn', False)
        if not (use_m or use_h) or not buffer:
            return None, None

        cids     = g['company'].company_ids.cpu().numpy()   # [N]
        N        = len(cids)
        dim      = getattr(self.model, 'lstm_hidden', 128)
        h_prev   = torch.zeros(N, dim,  dtype=torch.float)
        haz_prev = torch.zeros(N, 1,    dtype=torch.float)
        for k, cid in enumerate(cids):
            entry = buffer.get(int(cid))
            if entry is not None:
                h_prev[k]   = entry['h']
                haz_prev[k] = entry['hazard']
        return h_prev.to(self.device), haz_prev.to(self.device)

    def _update_buffer(
        self, g: HeteroData, h_dict: Dict, f_terminal: torch.Tensor,
        buffer: Dict
    ) -> None:
        """Populate buffer with this snapshot's LSTM embeddings and hazard scores."""
        company_emb = h_dict.get('_company_emb')
        if company_emb is None:
            return
        cids = g['company'].company_ids.cpu().numpy()       # [N]
        emb  = company_emb.detach().cpu()                   # [N, 128]
        haz  = f_terminal.detach().cpu().unsqueeze(1)       # [N, 1]
        for k, cid in enumerate(cids):
            buffer[int(cid)] = {'h': emb[k], 'hazard': haz[k]}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _forward(
        self, i_global: int, buffer: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               Optional[torch.Tensor], Optional[torch.Tensor],
               Dict, torch.Tensor]:
        """
        Forward pass for snapshot at global index i_global.

        Parameters
        ----------
        buffer : optional state buffer for TCM (B1+B3). When provided, h_prev and
                 hazard_prev are looked up and passed to the model.

        Returns
        -------
        phi     : Tensor [N, 52]   52-bin survival PMF logits (primary)
        logits  : Tensor [N, 3]    legacy 3-bin logits (1q, 2q, 4q)
        labels  : Tensor [N, 3]    columns = (y_1q, y_2q, y_4q)
        y_bin   : Tensor [N] or None  survival bin index (0-51)
        y_event : Tensor [N] or None  event indicator (0/1)
        h_dict  : Dict             model output h_dict (contains '_company_emb' for buffer)
        f_term  : Tensor [N]       terminal CIF risk score (for buffer update)
        """
        g   = self._all_graphs[i_global].to(self.device)
        seq = self._seq_cache[i_global].to(self.device)
        edge_attr_dict = {
            et: g[et].edge_attr
            for et in g.edge_types
            if hasattr(g[et], 'edge_attr') and g[et].edge_attr is not None
        }
        sector_idx = getattr(g['company'], 'sector_idx', None)

        # Look up previous-snapshot state for TCM (None if buffer empty or flags off)
        h_prev, hazard_prev = self._lookup_prev_state(g, buffer or {})

        # GATv2+NDR needs the snapshot's company_ids to rebuild its NDR feature
        # column; all other models ignore it (absorbed via **kw).
        extra = ({'company_ids': g['company'].company_ids}
                 if hasattr(self.model, 'ndr_lookup') else {})

        phi, logits, h_dict = self.model(
            g.x_dict, g.edge_index_dict,
            edge_attr_dict=edge_attr_dict if edge_attr_dict else None,
            temporal_sequences=seq,
            sector_idx=sector_idx,
            h_prev=h_prev,
            hazard_prev=hazard_prev,
            **extra,
        )

        # Compute risk score (used for buffer and metrics)
        with torch.no_grad():
            if self.force_binary_loss:
                f_term = torch.sigmoid(logits[:, 2])
            else:
                h_t    = torch.sigmoid(phi)
                log_st = torch.log((1.0 - h_t).clamp(1e-7)).cumsum(dim=1)
                f_term = 1.0 - torch.exp(log_st[:, -1])

        y_4q = g["company"].y.float().to(self.device)
        y_1q = getattr(g["company"], "y_1q", g["company"].y).float().to(self.device)
        y_2q = getattr(g["company"], "y_2q", g["company"].y).float().to(self.device)
        labels = torch.stack([y_1q, y_2q, y_4q], dim=1)  # [N, 3]

        y_bin_raw   = getattr(g["company"], "y_bin",   None)
        y_event_raw = getattr(g["company"], "y_event", None)
        y_bin   = y_bin_raw.to(self.device)   if y_bin_raw   is not None else None
        y_event = y_event_raw.to(self.device) if y_event_raw is not None else None

        return phi, logits, labels, y_bin, y_event, h_dict, f_term

    @staticmethod
    def _survival_loss(
        logits: torch.Tensor,   # [N, 3]  cols = (1q, 2q, 4q)
        labels: torch.Tensor,   # [N, 3]
        sigma: float = 0.1,
        alpha: float = 0.8,
    ) -> torch.Tensor:
        """
        DeepHit discrete-time survival loss (Lee et al. 2018).

        Interprets the 3 output columns as hazard logits at discretised
        intervals T1, T2, T3 (e.g. 1q/2q/4q for quarterly; 4w/8w/52w weekly):
            h_k = sigmoid(logits[:, k])   P(event in interval k | survived k-1)

        Survival:
            S1 = 1 - h1
            S2 = S1 * (1 - h2)
            S3 = S2 * (1 - h3)

        Event timing from multi-horizon labels:
            d1: y_1=1                -> defaulted in interval 1
            d2: y_1=0, y_2=1        -> defaulted in interval 2
            d3: y_2=0, y_4=1        -> defaulted in interval 3
            c:  y_4=0               -> censored (no default in window)

        Loss = alpha * L_nll + (1 - alpha) * L_rank
            L_nll  : negative log-likelihood of discrete hazard model
            L_rank : pairwise concordance penalty (earlier defaulters ranked higher)

        References: Lee et al. (2018) DeepHit; Kvamme et al. (2019) pycox.
        """
        eps = 1e-7
        h1 = torch.sigmoid(logits[:, 0])
        h2 = torch.sigmoid(logits[:, 1])
        h3 = torch.sigmoid(logits[:, 2])

        S1 = 1.0 - h1
        S2 = S1 * (1.0 - h2)
        S3 = S2 * (1.0 - h3)

        y1, y2, y4 = labels[:, 0], labels[:, 1], labels[:, 2]
        d1   = y1
        d2   = (1.0 - y1) * y2
        d3   = (1.0 - y2) * y4
        cens = 1.0 - y4

        # Negative log-likelihood of the discrete survival model
        nll = -(
            d1   * torch.log(h1.clamp(eps)) +
            d2   * (torch.log(S1.clamp(eps)) + torch.log(h2.clamp(eps))) +
            d3   * (torch.log(S2.clamp(eps)) + torch.log(h3.clamp(eps))) +
            cens * torch.log(S3.clamp(eps))
        ).mean()

        # Cumulative incidence F(t) used for concordance ranking
        F1 = h1            # P(default by T1)
        F2 = 1.0 - S2      # P(default by T2)
        F4 = 1.0 - S3      # P(default by T3)

        def _rank(Fi: torch.Tensor, Fj: torch.Tensor,
                  mask_i: torch.Tensor, mask_j: torch.Tensor) -> torch.Tensor:
            """Pairwise ranking: penalise when earlier defaulter i is ranked <= j."""
            if mask_i.sum() == 0 or mask_j.sum() == 0:
                return torch.tensor(0.0, device=logits.device)
            fi = Fi[mask_i].unsqueeze(1)   # [n_i, 1]
            fj = Fj[mask_j].unsqueeze(0)   # [1, n_j]
            return torch.exp(-sigma * (fi - fj)).mean()

        r1 = _rank(F1, F1, d1.bool(), (~y1.bool()))
        r2 = _rank(F2, F2, d2.bool(), (~y2.bool()))
        r3 = _rank(F4, F4, d3.bool(), (cens.bool()))
        rank_loss = (r1 + r2 + r3) / 3.0

        return alpha * nll + (1.0 - alpha) * rank_loss

    @staticmethod
    def _deephit_loss(
        phi:     torch.Tensor,   # [N, K] per-bin hazard logits
        y_bin:   torch.Tensor,   # [N] int, event/censor bin index
        y_event: torch.Tensor,   # [N] float, 1=event 0=censored
        sigma:   float = 0.1,
        alpha:   float = 0.8,
    ) -> torch.Tensor:
        """
        52-bin DeepHit discrete-time survival loss (Lee et al. 2018).

        Uses per-bin sigmoid hazard (discrete-time Cox) formulation:

            h_k = sigmoid(phi_k)            P(event in bin k | survived k-1)
            S(t) = prod_{k<=t} (1 - h_k)   survival function
            F(t) = 1 - S(t)                cumulative incidence (CIF)

        This avoids the softmax PMF trap where S(K-1)=0 by construction,
        which would give -log(0) = inf for every censored firm.

        NLL: event firms use log(h_at) + log(S(t-1)); censored use log(S_at).
        Rank: pairwise concordance penalty on F_terminal = F[:, -1].
        Loss = alpha * NLL + (1-alpha) * Rank
        """
        eps = 1e-7
        N, K = phi.shape
        h = torch.sigmoid(phi)                                    # [N, K] hazard

        # log S(t) = sum_{k=0..t} log(1 - h_k)
        log_s = torch.log((1.0 - h).clamp(eps, 1.0 - eps)).cumsum(dim=1)  # [N, K]
        S = torch.exp(log_s)                                      # [N, K]

        idx  = y_bin.clamp(0, K - 1).unsqueeze(1)                # [N, 1]
        h_at = h.gather(1, idx).squeeze(1).clamp(eps)            # [N] hazard at event bin

        # S(t-1): survival up to the bin *before* the event bin
        S_prev = torch.cat([torch.ones(N, 1, device=phi.device), S[:, :-1]], dim=1)
        S_at_prev = S_prev.gather(1, idx).squeeze(1).clamp(eps)  # [N]
        S_at = S.gather(1, idx).squeeze(1).clamp(eps)            # [N] for censored

        ev = y_event.float()
        # Event firms: NLL = -[log h_k + log S(k-1)]
        # Censored firms: NLL = -log S(k)  (survived at least to censor time)
        nll = -(ev * (torch.log(h_at) + torch.log(S_at_prev))
                + (1.0 - ev) * torch.log(S_at)).mean()

        # Time-aware ranking (Lee et al. 2018 eq. 4):
        # For each event firm i defaulting at t_i, compare F_i(t_i) vs F_j(t_i)
        # for all j with t_j > t_i (survived longer than i's event time).
        # Both firms are evaluated at i's event time -- not terminal CIF.
        # This ensures a week-10 defaulter ranks above a week-40 defaulter.
        F = 1.0 - S                                               # [N, K] CIF
        ev_mask = ev.bool()
        rank_terms: List[torch.Tensor] = []
        if ev_mask.sum() >= 2:
            for t_i in y_bin[ev_mask].unique():
                ev_at_ti     = ev_mask & (y_bin == t_i)           # firms that defaulted at t_i
                surv_past_ti = y_bin > t_i                        # firms that survived beyond t_i
                if surv_past_ti.sum() == 0:
                    continue
                F_ev   = F[ev_at_ti,     t_i]                     # [n_ev]  CIF at t_i
                F_surv = F[surv_past_ti, t_i]                     # [n_surv] CIF at t_i
                rank_terms.append(
                    torch.exp(-sigma * (F_ev.unsqueeze(1) - F_surv.unsqueeze(0))).mean()
                )
        rank_loss = (torch.stack(rank_terms).mean()
                     if rank_terms else torch.tensor(0.0, device=phi.device))

        # Calibration: mean F_terminal should track empirical default rate.
        # Prevents hazard collapse (all h_k -> 1 -> F_terminal = 1 for everyone).
        F_terminal = 1.0 - S[:, -1]
        calib_loss = (F_terminal.mean() - ev.mean()).pow(2)

        return alpha * nll + (1.0 - alpha) * rank_loss + 0.1 * calib_loss

    def _loss(
        self,
        phi:     torch.Tensor,
        logits:  torch.Tensor,
        labels:  torch.Tensor,
        y_bin:   Optional[torch.Tensor] = None,
        y_event: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Primary: 52-bin DeepHit when y_bin is available; fallback to 3-bin.
        If force_binary_loss=True, always use the 3-bin fallback regardless."""
        if not self.force_binary_loss and y_bin is not None and y_event is not None:
            return self._deephit_loss(phi, y_bin, y_event.float(),
                                      alpha=self.survival_alpha)
        return self._survival_loss(logits, labels, alpha=self.survival_alpha)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, num_epochs: int = 100, print_every: int = 5) -> Dict:
        """
        Train for up to num_epochs epochs with early stopping on val AP.

        Returns
        -------
        history : Dict with train/val loss, AUROC, AP per epoch.
        """
        print(f"\n{'='*70}")
        print(f"Training SpatioTemporalGNN for up to {num_epochs} epochs")
        print(f"{'='*70}")

        n_train = len(self.train_graphs)
        val_indices = list(range(self._val_start,
                                 self._val_start + len(self.val_graphs)))

        # Extend stride cycle to cover all epochs (repeat as needed)
        cycle_len = len(self._stride_cycle)

        for epoch in range(num_epochs):
            t0 = datetime.now()

            # Stochastic monthly stride: pick offset from shuffled cycle.
            # Reshuffles each full cycle so the pattern never exactly repeats.
            cycle_pos = epoch % cycle_len
            if self.train_stride > 1 and cycle_pos == 0 and epoch > 0:
                np.random.shuffle(self._stride_cycle)
            offset = self._stride_cycle[cycle_pos] if self.train_stride > 1 else 0
            train_indices = list(range(offset, n_train, self.train_stride))

            # Reset state buffer at epoch start: embeddings from prior epoch are stale
            # because model weights have changed. The buffer fills up during the epoch.
            self._state_buffer = {}

            # ---- TRAIN ----
            self.model.train()
            t_losses, t_probs_list, t_labels_list = [], [], []

            for i in train_indices:
                self.optimizer.zero_grad()
                phi, logits, y, y_bin, y_event, h_dict, f_term = self._forward(
                    i, buffer=self._state_buffer
                )
                loss = self._loss(phi, logits, y, y_bin, y_event)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                # Update buffer AFTER gradient step (detached, no memory overhead)
                g = self._all_graphs[i]
                self._update_buffer(g, h_dict, f_term, self._state_buffer)

                t_losses.append(loss.item())
                t_probs_list.append(f_term.detach().cpu().numpy())
                t_labels_list.append(y[:, 2].cpu().numpy())

            # ---- VALIDATE ----
            # Val snapshots continue the buffer populated during training,
            # giving the first val snapshot access to the last training snapshot's state.
            self.model.eval()
            v_losses, v_probs_list, v_labels_list = [], [], []

            with torch.no_grad():
                for i_g in val_indices:
                    phi, logits, y, y_bin, y_event, h_dict_v, f_term = self._forward(
                        i_g, buffer=self._state_buffer
                    )
                    v_losses.append(self._loss(phi, logits, y, y_bin, y_event).item())
                    g_v = self._all_graphs[i_g]
                    self._update_buffer(g_v, h_dict_v, f_term, self._state_buffer)
                    v_probs_list.append(f_term.cpu().numpy())
                    v_labels_list.append(y[:, 2].cpu().numpy())

            # Aggregate metrics (4q horizon for early stopping)
            t_m = compute_metrics(
                np.concatenate(t_labels_list), np.concatenate(t_probs_list)
            )
            v_m = compute_metrics(
                np.concatenate(v_labels_list), np.concatenate(v_probs_list)
            )

            tl = float(np.mean(t_losses))
            vl = float(np.mean(v_losses))
            self.history["train_loss"].append(tl)
            self.history["train_auc"].append(t_m.get("auc", 0.0))
            self.history["train_ap"].append(t_m.get("ap", 0.0))
            self.history["val_loss"].append(vl)
            self.history["val_auc"].append(v_m.get("auc", 0.0))
            self.history["val_ap"].append(v_m.get("ap", 0.0))

            val_ap = v_m.get("ap", 0.0)
            elapsed = (datetime.now() - t0).total_seconds()
            is_best = val_ap > self.best_val_ap
            # Only flag [BEST] in output when improvement is >= 0.5pp to avoid
            # printing every epoch for tiny incremental improvements.
            is_notable = val_ap > self.best_val_ap + 0.005

            # Step scheduler on val AP (halves LR if no improvement for 5 epochs)
            self.scheduler.step(val_ap)
            current_lr = self.optimizer.param_groups[0]['lr']

            if (epoch + 1) % print_every == 0 or epoch == 0 or is_notable:
                print(
                    f"Ep {epoch+1:3d} | "
                    f"Train loss {tl:.4f} AUC {t_m.get('auc',0):.3f} AP {t_m.get('ap',0):.3f} | "
                    f"Val loss {vl:.4f} AUC {v_m.get('auc',0):.3f} AP {v_m.get('ap',0):.3f}"
                    + (f"  [BEST]  {elapsed:.0f}s" if is_notable else f"  {elapsed:.0f}s")
                    + (f"  lr={current_lr:.2e}" if current_lr < 5e-4 else "")
                )

            if is_best:
                self.best_val_ap = val_ap
                self.patience_counter = 0
                self.best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch+1} "
                          f"(best val AP = {self.best_val_ap:.4f})")
                    break

        # Restore best checkpoint
        if self.best_state is not None:
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in self.best_state.items()}
            )
            print(f"\nRestored best model (val AP = {self.best_val_ap:.4f})")

        return self.history

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, split: str = "test", verbose: bool = True,
                 threshold: float = 0.5) -> Dict:
        """
        Evaluate the model on a held-out split.

        Parameters
        ----------
        split : 'test' or 'val'
        verbose : print per-snapshot results
        threshold : float, default 0.5
            Decision threshold for binary metrics (tier recall, sector recall,
            F1, precision, recall). Use the val-calibrated threshold (e.g. from
            precision_recall_curve F1-max) rather than 0.5 for DeepHit models,
            which output low hazard probabilities relative to the base rate.

        Returns
        -------
        Dict with 'per_snapshot' (list of per-snapshot metrics)
              and 'aggregate' (metrics over all nodes in the split).
        """
        if split == "test":
            start = self._test_start
            graphs = self.test_graphs
        elif split == "val":
            start = self._val_start
            graphs = self.val_graphs
        else:
            raise ValueError(f"Unknown split '{split}': choose 'val' or 'test'")

        self.model.eval()
        per_snapshot = []
        all_probs, all_labels = [], []
        all_phi, all_y_bin, all_y_event = [], [], []

        if verbose:
            print(f"\n{'='*70}")
            print(f"Evaluation: {split.upper()} set ({len(graphs)} snapshots)")
            print(f"{'='*70}")

        HORIZONS = [(0, "1q"), (1, "2q"), (2, "4q")]

        # Separate eval buffer: does not pollute the training state buffer.
        # Populated sequentially so each snapshot sees the previous one's state.
        _eval_buffer: Dict = {}

        with torch.no_grad():
            for i_s, i_g in enumerate(range(start, start + len(graphs))):
                phi, logits, y, y_bin, y_event, h_dict_e, f_term = self._forward(
                    i_g, buffer=_eval_buffer
                )
                g_e = self._all_graphs[i_g]
                self._update_buffer(g_e, h_dict_e, f_term, _eval_buffer)
                probs_all  = f_term.cpu().numpy()          # [N] terminal CIF
                probs_3bin = torch.sigmoid(logits).cpu().numpy()  # [N, 3] legacy
                labels_all = y.cpu().numpy()
                snap_t = getattr(self._all_graphs[i_g]["company"], "t", f"snap_{i_s}")

                # Primary metrics using F(t=52) risk score
                m4 = compute_metrics(labels_all[:, 2], probs_all)
                snap_entry = {
                    "snapshot":     str(snap_t),
                    "n_nodes":      labels_all.shape[0],
                    "default_rate": float(labels_all[:, 2].mean()),
                    **m4,
                }
                # Legacy per-horizon AP / AUC from 3-bin heads
                for col, tag in HORIZONS:
                    m = compute_metrics(labels_all[:, col], probs_3bin[:, col])
                    snap_entry[f"auc_{tag}"] = m.get("auc", 0.0)
                    snap_entry[f"ap_{tag}"]  = m.get("ap",  0.0)

                # Tier recall (IG / SG / HS) using raw rating_numeric
                rating_raw = getattr(self._all_graphs[i_g]["company"],
                                     "rating_numeric_raw", None)
                if rating_raw is not None:
                    y_pred_bin = (probs_all >= threshold).astype(int)
                    tier = compute_tier_recall(
                        labels_all[:, 2].astype(int), y_pred_bin,
                        rating_raw.cpu().numpy()
                    )
                    snap_entry.update(tier)

                per_snapshot.append(snap_entry)
                all_probs.append(probs_all)
                all_labels.append(labels_all[:, 2])

                # Collect survival arrays for aggregate C-index / Delta-L
                all_phi.append(phi.cpu().numpy())
                if y_bin is not None:
                    all_y_bin.append(y_bin.cpu().numpy())
                    all_y_event.append(y_event.cpu().numpy())

                if verbose:
                    m1 = compute_metrics(labels_all[:, 0], probs_3bin[:, 0])
                    m2 = compute_metrics(labels_all[:, 1], probs_3bin[:, 1])
                    print(
                        f"  {str(snap_t)[:10]}  n={labels_all.shape[0]:4d}  "
                        f"DR={labels_all[:,2].mean():.2%}  "
                        f"1q AP={m1.get('ap',0):.3f}  "
                        f"2q AP={m2.get('ap',0):.3f}  "
                        f"surv AUC={m4.get('auc',0):.3f}  surv AP={m4.get('ap',0):.3f}"
                    )

        agg = compute_metrics(
            np.concatenate(all_labels), np.concatenate(all_probs)
        )

        # Aggregate tier recall and sector metrics across all snapshots
        all_ratings = []
        all_sectors = []
        for i_g in range(start, start + len(graphs)):
            g_company = self._all_graphs[i_g]["company"]
            rating_raw = getattr(g_company, "rating_numeric_raw", None)
            if rating_raw is not None:
                all_ratings.append(rating_raw.cpu().numpy())
            sector_raw = getattr(g_company, "sector_idx", None)
            if sector_raw is not None:
                all_sectors.append(sector_raw.cpu().numpy())

        all_probs_np  = np.concatenate(all_probs)
        all_labels_np = np.concatenate(all_labels)
        y_pred_thresh = (all_probs_np >= threshold).astype(int)

        # Aggregate survival metrics (C-index, Delta-L) if y_bin labels present
        if all_y_bin:
            from training.metrics import compute_survival_metrics, compute_lead_time_gain
            phi_all    = np.concatenate(all_phi,     axis=0)
            y_bin_all  = np.concatenate(all_y_bin,   axis=0)
            y_ev_all   = np.concatenate(all_y_event, axis=0)
            surv_m = compute_survival_metrics(phi_all, y_bin_all, y_ev_all)
            agg.update({k: v for k, v in surv_m.items() if k not in ('S', 'F_terminal')})
            # Delta-L uses a fixed CIF threshold (0.5) independent of the val-calibrated
            # binary threshold -- the binary threshold reflects class imbalance (~1.5% DR)
            # and is not meaningful as a survival early-warning trigger level.
            lead_m = compute_lead_time_gain(surv_m['S'], surv_m['F_terminal'],
                                             y_bin_all, y_ev_all, threshold=0.5)
            agg.update(lead_m)

        if all_ratings:
            all_ratings_np = np.concatenate(all_ratings)
            agg_tier = compute_tier_recall(
                all_labels_np.astype(int), y_pred_thresh, all_ratings_np
            )
            agg.update(agg_tier)

        if all_sectors:
            all_sectors_np = np.concatenate(all_sectors)
            agg_sector = compute_sector_metrics(
                all_labels_np.astype(int), all_probs_np, y_pred_thresh, all_sectors_np
            )
            agg.update(agg_sector)

        if verbose:
            print_metrics(agg, title=f"\nAggregate {split.upper()} metrics (survival F(t=52) primary)")
            if "surv_cindex" in agg:
                print(f"\n  Survival Metrics:")
                print(f"    C-index       : {agg.get('surv_cindex', float('nan')):.4f}")
                print(f"    AP [F(t=52)]  : {agg.get('surv_ap',     float('nan')):.4f}")
                print(f"    Delta-L (weeks): {agg.get('delta_l_weeks', float('nan')):.1f}")
                print(f"    Early signal  : {agg.get('pct_early_signal', float('nan')):.1%}  (n_event={agg.get('surv_n_event',0)})")
            if "rec_ig" in agg:
                print(f"\n  Tier Recall (threshold={threshold:.3f}):")
                print(f"    IG  (AAA-BBB-) n={agg.get('n_ig',0):4d}  recall={agg.get('rec_ig', float('nan')):.3f}")
                print(f"    SG  (BB+ -B-)  n={agg.get('n_sg',0):4d}  recall={agg.get('rec_sg', float('nan')):.3f}")
                print(f"    HS  (CCC+-C)   n={agg.get('n_hs',0):4d}  recall={agg.get('rec_hs', float('nan')):.3f}")
            if any(k.startswith("sec_") for k in agg):
                print(f"\n  Sector Metrics (AP / AUC / Recall @ {threshold:.3f} / DR):")
                print(f"    {'Sector':<14}  {'n':>6}  {'DR':>6}  {'AP':>6}  {'AUC':>6}  {'Recall':>6}")
                print(f"    {'-'*56}")
                for idx in range(12):
                    name = GICS_NAMES[idx]
                    prefix = f"sec_{name}_"
                    n = agg.get(f"{prefix}n")
                    if n is None:
                        continue
                    ap  = agg.get(f"{prefix}ap",  float('nan'))
                    auc = agg.get(f"{prefix}auc", float('nan'))
                    rec = agg.get(f"{prefix}rec", float('nan'))
                    dr  = agg.get(f"{prefix}dr",  float('nan'))
                    ap_s  = f"{ap:.3f}"  if not np.isnan(ap)  else "  nan"
                    auc_s = f"{auc:.3f}" if not np.isnan(auc) else "  nan"
                    rec_s = f"{rec:.3f}" if not np.isnan(rec) else "  nan"
                    print(f"    {name:<14}  {n:>6}  {dr:>6.2%}  {ap_s:>6}  {auc_s:>6}  {rec_s:>6}")

        return {"per_snapshot": per_snapshot, "aggregate": agg}

    # ------------------------------------------------------------------
    # Permuted-edge negative control (RQ3)
    # ------------------------------------------------------------------

    def evaluate_permuted_edges(
        self,
        n_permutations: int = 5,
        seed: int = 42,
        verbose: bool = True,
    ) -> Dict:
        """
        Permuted-edge negative control for RQ3.

        Randomly permutes destination indices within each edge type on every
        test snapshot, preserving node count and in-degree distribution but
        destroying all meaningful topology. If graph structure carries genuine
        predictive value, AP/AUC should drop relative to the real-edge baseline.

        Parameters
        ----------
        n_permutations : int
            Number of independent random permutations to average over.
        seed : int
            Base RNG seed; permutation k uses seed+k.
        verbose : bool

        Returns
        -------
        Dict with keys:
            'mean_ap', 'std_ap', 'mean_auc', 'std_auc' -- aggregate stats
            'runs'      -- list of per-permutation dicts {'probs', 'labels', 'agg'}
            'snap_meta' -- per-snapshot {'t', 'sector_idx', 'n'} for stratification
        """
        self.model.eval()

        # Collect snapshot metadata once (timestamps, sector indices, node counts)
        snap_meta = []
        for i_g in range(self._test_start, self._test_start + len(self.test_graphs)):
            g = self._all_graphs[i_g]
            snap_meta.append({
                't':          getattr(g['company'], 't', None),
                'sector_idx': getattr(g['company'], 'sector_idx', None),
                'n':          g['company'].x.shape[0],
            })

        runs = []

        for k in range(n_permutations):
            rng = np.random.default_rng(seed + k)
            all_probs, all_labels = [], []
            _eval_buffer: Dict = {}

            with torch.no_grad():
                for i_g in range(self._test_start,
                                 self._test_start + len(self.test_graphs)):
                    g   = self._all_graphs[i_g].to(self.device)
                    seq = self._seq_cache[i_g].to(self.device)

                    # Permute destination indices within each edge type.
                    # Source order (and edge_attr alignment) is preserved.
                    perm_edge_index = {}
                    for et, ei in g.edge_index_dict.items():
                        new_dst = torch.tensor(
                            rng.permutation(ei[1].cpu().numpy()),
                            dtype=ei.dtype, device=self.device,
                        )
                        perm_edge_index[et] = torch.stack([ei[0], new_dst], dim=0)

                    edge_attr_dict = {
                        et: g[et].edge_attr
                        for et in g.edge_types
                        if hasattr(g[et], 'edge_attr') and g[et].edge_attr is not None
                    }
                    sector_idx = getattr(g['company'], 'sector_idx', None)
                    h_prev, hazard_prev = self._lookup_prev_state(g, _eval_buffer)

                    extra = ({'company_ids': g['company'].company_ids}
                             if hasattr(self.model, 'ndr_lookup') else {})

                    phi, logits, h_dict = self.model(
                        g.x_dict, perm_edge_index,
                        edge_attr_dict=edge_attr_dict if edge_attr_dict else None,
                        temporal_sequences=seq,
                        sector_idx=sector_idx,
                        h_prev=h_prev,
                        hazard_prev=hazard_prev,
                        **extra,
                    )

                    if self.force_binary_loss:
                        f_term = torch.sigmoid(logits[:, 2])
                    else:
                        h_t    = torch.sigmoid(phi)
                        log_st = torch.log((1.0 - h_t).clamp(1e-7)).cumsum(dim=1)
                        f_term = 1.0 - torch.exp(log_st[:, -1])

                    self._update_buffer(g, h_dict, f_term, _eval_buffer)

                    y_4q = g['company'].y.float().to(self.device)
                    labels = (y_4q[:, 2] if y_4q.dim() == 2 else y_4q).cpu().numpy()

                    all_probs.append(f_term.cpu().numpy())
                    all_labels.append(labels)

            probs_arr  = np.concatenate(all_probs)
            labels_arr = np.concatenate(all_labels)
            agg = compute_metrics(labels_arr, probs_arr)
            runs.append({'probs': probs_arr, 'labels': labels_arr, 'agg': agg})

            if verbose:
                print(f"  Permutation {k+1}/{n_permutations}  "
                      f"AUC={agg.get('auc', float('nan')):.4f}  "
                      f"AP={agg.get('ap', float('nan')):.4f}")

        aps  = [r['agg'].get('ap',  float('nan')) for r in runs]
        aucs = [r['agg'].get('auc', float('nan')) for r in runs]
        result = {
            'mean_ap':  float(np.nanmean(aps)),
            'std_ap':   float(np.nanstd(aps)),
            'mean_auc': float(np.nanmean(aucs)),
            'std_auc':  float(np.nanstd(aucs)),
            'runs':     runs,
            'snap_meta': snap_meta,
        }
        if verbose:
            print(f"\nPermuted-edge control ({n_permutations} runs):")
            print(f"  AP  = {result['mean_ap']:.4f} +/- {result['std_ap']:.4f}")
            print(f"  AUC = {result['mean_auc']:.4f} +/- {result['std_auc']:.4f}")
        return result
