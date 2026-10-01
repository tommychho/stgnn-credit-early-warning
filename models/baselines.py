"""Baseline and ablation models used in the paper.

LSTMOnly          -- Sequential baseline (no graph)
GATv2Only         -- Static attention snapshot proxy (no LSTM)
GATv2NDR          -- GATv2 + Node Default Rate feature (LANR-type, Han et al. 2026)
HomogeneousRGCN   -- Multi-relational proxy (homogeneous, no LSTM)
NoGraphWrapper    -- Ablation: ST-GNN+GRS with all edges zeroed at inference

All learned models return a 3-tuple (phi, logits, h_dict) matching HeteroEdgeModel:
  phi     [N, 52]  52-bin survival PMF logits (primary, trained by DeepHit loss)
  logits  [N, 3]   3-bin hazard logits (1q, 2q, 4q; fallback / legacy)
  h_dict           node-embedding dict (empty for non-stateful baselines)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .temporal_encoder import TemporalEncoder
from .stgnn import SpatioTemporalGNN, HeteroEdgeModel


class LSTMOnly(nn.Module):
    """Variant A: Firm LSTM -> 3-bin + 52-bin survival heads.  No macro gate, no GNN."""
    def __init__(self, firm_dim=11, lstm_hidden=128, dropout=0.3):
        super().__init__()
        self.firm_dim = firm_dim
        self.encoder  = TemporalEncoder(input_dim=firm_dim, hidden_dim=lstm_hidden, num_layers=1)
        self.trunk    = nn.Sequential(nn.Linear(lstm_hidden, lstm_hidden), nn.ReLU(), nn.Dropout(dropout))
        self.head_1q  = nn.Linear(lstm_hidden, 1)
        self.head_2q  = nn.Linear(lstm_hidden, 1)
        self.head_4q  = nn.Linear(lstm_hidden, 1)
        self.head_surv = nn.Linear(lstm_hidden, 52)   # 52-bin survival PMF logits (phi)
        nn.init.constant_(self.head_surv.bias, -8.0)

    def forward(self, x_dict, edge_index_dict, temporal_sequences=None, **kw):
        emb    = self.encoder(temporal_sequences[:, :, :self.firm_dim])
        trunk  = self.trunk(emb)
        logits = torch.stack([self.head_1q(trunk).squeeze(-1),
                              self.head_2q(trunk).squeeze(-1),
                              self.head_4q(trunk).squeeze(-1)], dim=1)   # [N, 3]
        phi    = self.head_surv(trunk)                                   # [N, 52]
        return phi, logits, {}


class GATv2Only(nn.Module):
    """Variant B: Raw features -> Hetero GATv2 -> 3 heads.  No LSTM."""
    def __init__(self, metadata, in_dim=16, hidden=64, num_layers=2, heads=4, dropout=0.3):
        super().__init__()
        self.gnn = HeteroEdgeModel(metadata=metadata, in_channels_dict={'company': in_dim},
                                   hidden_channels=hidden, num_layers=num_layers,
                                   heads=heads, dropout=dropout)

    def forward(self, x_dict, edge_index_dict, temporal_sequences=None, **kw):
        return self.gnn(x_dict, edge_index_dict)


class GATv2NDR(nn.Module):
    """GATv2 + Node Default Rate (LANR-type approximation, Han et al. 2026).

    Static heterogeneous GATv2 (no LSTM) with the Node Default Rate appended as
    one extra input feature on each company node:
        NDR_i = fraction of firm i's training-graph neighbours that defaulted.
    The per-firm NDR values are precomputed on the training split only (see
    compute_ndr_lookup in train.py) and supplied as ``ndr_lookup`` (a dict
    mapping integer company_id -> NDR scalar). At forward time the column is
    rebuilt from the snapshot's ``company_ids``; firms unseen in training (or
    when company_ids is None) receive NDR = 0.

    This is the only difference from GATv2Only, so in_dim must be one larger
    (e.g. 17 vs 16) to accommodate the appended NDR feature.
    """
    def __init__(self, metadata, in_dim=17, hidden=64, num_layers=2, heads=4,
                 dropout=0.3, ndr_lookup=None):
        super().__init__()
        self.gnn = HeteroEdgeModel(metadata=metadata, in_channels_dict={'company': in_dim},
                                   hidden_channels=hidden, num_layers=num_layers,
                                   heads=heads, dropout=dropout)
        # Stored in state_dict-adjacent buffer via plain attribute; also saved to
        # the checkpoint by train.py so evaluation reuses the exact same lookup.
        self.ndr_lookup = dict(ndr_lookup) if ndr_lookup else {}

    def _ndr_column(self, company_ids, n, device):
        if company_ids is None:
            return torch.zeros(n, 1, device=device)
        cids = (company_ids.detach().cpu().numpy()
                if torch.is_tensor(company_ids) else company_ids)
        vals = [self.ndr_lookup.get(int(c), 0.0) for c in cids]
        return torch.tensor(vals, dtype=torch.float32, device=device).unsqueeze(1)

    def forward(self, x_dict, edge_index_dict, company_ids=None,
                temporal_sequences=None, **kw):
        comp    = x_dict['company']
        ndr_col = self._ndr_column(company_ids, comp.shape[0], comp.device)
        x_aug   = dict(x_dict)
        x_aug['company'] = torch.cat([comp, ndr_col], dim=1)
        return self.gnn(x_aug, edge_index_dict)


class HomogeneousRGCN(nn.Module):
    """Variant C: Homogeneous R-GCN on company nodes only, no LSTM.
    Only company->company edges are used; relations renumbered 0..K-1.
    """
    def __init__(self, in_dim=16, hidden=64, num_relations=4, dropout=0.3):
        super().__init__()
        from torch_geometric.nn import RGCNConv
        self.n_rels  = num_relations
        self.proj    = nn.Linear(in_dim, hidden)
        self.conv1   = RGCNConv(hidden, hidden, num_relations=num_relations)
        self.conv2   = RGCNConv(hidden, hidden, num_relations=num_relations)
        self.trunk   = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.head_1q = nn.Linear(hidden, 1)
        self.head_2q = nn.Linear(hidden, 1)
        self.head_4q = nn.Linear(hidden, 1)
        self.head_surv = nn.Linear(hidden, 52)   # 52-bin survival PMF logits (phi)
        nn.init.constant_(self.head_surv.bias, -8.0)
        self.drop    = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict, temporal_sequences=None, **kw):
        x = self.proj(x_dict['company'])
        N = x.shape[0]
        ei_list, et_list = [], []
        ri = 0
        for (src_t, _, dst_t), ei in edge_index_dict.items():
            if src_t != 'company' or dst_t != 'company':
                continue
            if ei.numel() > 0 and ri < self.n_rels:
                mask  = (ei[0] < N) & (ei[1] < N)
                ei_ok = ei[:, mask]
                if ei_ok.numel() > 0:
                    ei_list.append(ei_ok)
                    et_list.append(torch.full((ei_ok.shape[1],), ri,
                                              dtype=torch.long, device=ei.device))
            ri += 1
        if ei_list:
            edge_index = torch.cat(ei_list, dim=1)
            edge_type  = torch.cat(et_list)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=x.device)
            edge_type  = torch.zeros(0, dtype=torch.long, device=x.device)
        h     = self.drop(F.relu(self.conv1(x, edge_index, edge_type)))
        h     = F.relu(self.conv2(h, edge_index, edge_type)) + x
        trunk = self.trunk(h)
        logits = torch.stack([self.head_1q(trunk).squeeze(-1),
                              self.head_2q(trunk).squeeze(-1),
                              self.head_4q(trunk).squeeze(-1)], dim=1)   # [N, 3]
        phi    = self.head_surv(trunk)                                   # [N, 52]
        return phi, logits, {}


class NoGraphWrapper(nn.Module):
    """Ablation: ST-GNN+GRS with all graph edges zeroed at inference.

    Wraps any SpatioTemporalGNN instance and replaces every edge tensor with an
    empty index before the forward pass.  The architecture (weights, gate, LSTM)
    is identical to the wrapped model; only neighbourhood aggregation is disabled.

    NOTE ON INTERPRETATION. An earlier version of this docstring read the gap
    between this ablation and the full model as evidence that the graph was
    necessary. It is not: on the checkpoints reported in the paper, removing every
    graph-derived path from a trained model changes average precision by
    -0.0014 +- 0.0202, and this wrapper, trained with edges empty throughout,
    reaches the highest AP of the four variants. Masking edges at inference is
    also the weaker of the two available tests, because a branch whose
    contribution is redundant is nearly invariant to what is fed into it, and so
    cannot distinguish "the model internalised the graph" from "the model does not
    need it". Removing each path from the trained checkpoint separates them; this
    wrapper does not.

    Note that this wrapper is also the control for a DIFFERENT question. Because
    it is built before training, a model trained through it never sees an edge in
    any forward pass, which is what tests whether the graph shaped the gates
    during training, as distinct from propagating at inference.
    """
    def __init__(self, model: SpatioTemporalGNN):
        super().__init__()
        self.model = model

    def forward(self, x_dict, edge_index_dict,
                edge_attr_dict=None, temporal_sequences=None, **kw):
        empty = {
            k: torch.zeros(2, 0, dtype=torch.long, device=v.device)
            for k, v in edge_index_dict.items()
        }
        empty_attr = (
            {k: torch.zeros(0, v.shape[1] if v.dim() > 1 else 1,
                            device=v.device)
             for k, v in edge_attr_dict.items()}
            if edge_attr_dict is not None else None
        )
        return self.model(
            x_dict, empty,
            edge_attr_dict=empty_attr,
            temporal_sequences=temporal_sequences,
            **kw,
        )
