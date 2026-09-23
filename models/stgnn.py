"""ST-GNN model variants used in the paper (ICDM 2026).

SectorAttentionFusion    -- Mod 2b sector gate (SAF variant)
HeteroEdgeModel          -- Mod 4 GATv2 + Mod 6 Classifier
NetworkSystematicModule  -- Mod 3 network systemic risk pool
SpatioTemporalGNN        -- Full proposed architecture (base / SAF / GRS variants)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv
from typing import Dict, List, Tuple, Optional


class SectorAttentionFusion(nn.Module):
    """
    Adaptive Cross-Modal Gated Fusion for sector conditioning.

    Replaces ResidualFiLMBlock. The gate is conditioned on BOTH the firm
    embedding and the sector embedding, so each firm receives a personalised
    sector adjustment rather than a uniform affine penalty.

    Formula:
        z_s'    = sector_proj(sector_embed(sector_idx))    # [N, embed_dim]
        z_s_b   = tanh(z_s') * scale                       # [N, embed_dim], bounded ±scale
        g_i     = sigmoid(W_g * [h_i || z_s'] + b_g)      # [N, embed_dim], starts ~0.018
        h_fused = h_i + g_i * z_s_b                        # residual; h_i when g_i -> 0

    Two stabilisers for quarterly compatibility:
    1. tanh * scale (default 0.1) bounds sector influence to ±0.1 per element,
       matching FiLM's gamma bound and preventing z_s_proj norm growth from
       overwhelming firm embeddings under DeepHit loss.
    2. bias=-4.0 -> gate~0.018 at init, giving the survival head time to
       stabilise before sector conditioning activates.

    Ref: Gated Multimodal Units (Arevalo et al., 2017).

    Parameters
    ----------
    embed_dim   : int  Firm embedding dimension (= lstm_hidden, default 128).
    num_sectors : int  GICS sector classes incl. 0=unknown/padding (default 12).
    sector_dim  : int  Sector embedding size before projection (default 32).
    """

    GSECTOR_MAP: Dict[str, int] = {
        '10': 1, '15': 2, '20': 3, '25': 4, '30': 5,
        '35': 6, '40': 7, '45': 8, '50': 9, '55': 10, '60': 11,
    }  # 0 reserved for unknown/NR

    def __init__(self, embed_dim: int, num_sectors: int = 12, sector_dim: int = 32,
                 scale: float = 0.1):
        super().__init__()
        self.sector_embed   = nn.Embedding(num_sectors, sector_dim, padding_idx=0)
        self.sector_proj    = nn.Linear(sector_dim, embed_dim)
        self.attention_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )
        self.scale = scale
        # Zero weights + negative bias -> gate starts at ~0.018 (near identity)
        nn.init.zeros_(self.attention_gate[0].weight)
        nn.init.constant_(self.attention_gate[0].bias, -4.0)

    def forward(self, firm_h: torch.Tensor,
                sector_idx: Optional[torch.Tensor]) -> torch.Tensor:
        if sector_idx is None:
            return firm_h
        z_s      = self.sector_embed(sector_idx)                              # [N, sector_dim]
        z_s_proj = self.sector_proj(z_s)                                      # [N, embed_dim]
        # tanh bounds sector influence to ±scale (mirrors FiLM's gamma bound of 0.1)
        z_s_bounded = torch.tanh(z_s_proj) * self.scale                       # [N, embed_dim]
        gate     = self.attention_gate(torch.cat([firm_h, z_s_proj], dim=-1)) # [N, embed_dim]
        return firm_h + gate * z_s_bounded


class HeteroEdgeModel(nn.Module):
    """
    Heterogeneous Graph Attention Network with Edge Attributes.

    This model uses GATv2Conv for attention-based message passing across
    different node and edge types in a credit risk graph.

    Architecture:
    1. Node-type specific input projections
    2. Multiple HeteroConv layers with GATv2Conv
    3. Layer normalization per node type
    4. Dropout for regularization
    5. Borrower-specific classification head

    Parameters
    ----------
    metadata : Tuple[List[str], List[Tuple[str, str, str]]]
        Graph metadata (node_types, edge_types)
    in_channels_dict : Dict[str, int]
        Input feature dimensions per node type
    hidden_channels : int, default=32
        Hidden dimension size
    num_layers : int, default=3
        Number of GNN layers
    heads : int, default=4
        Number of attention heads
    dropout : float, default=0.5
        Dropout probability

    Attributes
    ----------
    node_types : List[str]
        List of node type names
    edge_types : List[Tuple[str, str, str]]
        List of edge type tuples (src, rel, dst)
    proj : nn.ModuleDict
        Input projection layers per node type
    convs : nn.ModuleList
        HeteroConv layers
    norms : nn.ModuleList
        LayerNorm layers per node type per layer
    classifier : nn.Sequential
        Classification head for borrower nodes

    Example
    -------
    >>> metadata = (
    ...     ['borrower', 'facility'],
    ...     [('borrower', 'has_facility', 'facility')]
    ... )
    >>> in_channels = {'borrower': 5, 'facility': 5}
    >>> model = HeteroEdgeModel(metadata, in_channels, hidden_channels=32)
    >>> logits, embeddings = model(x_dict, edge_index_dict, edge_attr_dict)
    """

    def __init__(
        self,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        in_channels_dict: Dict[str, int],
        hidden_channels: int = 32,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.5,
        net_sys_dim: int = 0,
        use_hazard_attn: bool = False,
        use_jk: bool = False,
        use_no_skip: bool = False,
        use_gated_residual: bool = False,
    ):
        super().__init__()

        node_types, edge_types = metadata
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.net_sys_dim = net_sys_dim
        self.use_hazard_attn = use_hazard_attn
        self.use_jk = use_jk
        self.use_no_skip = use_no_skip
        self.use_gated_residual = use_gated_residual and not use_no_skip

        if self.use_gated_residual:
            self.residual_gates = nn.ModuleDict({
                ntype: nn.Sequential(
                    nn.Linear(hidden_channels * 2, hidden_channels),
                    nn.Sigmoid(),
                )
                for ntype in self.node_types
            })

        # =====================================================================
        # INPUT PROJECTIONS (node-type specific)
        # =====================================================================
        self.proj = nn.ModuleDict({
            ntype: nn.Linear(in_channels_dict[ntype], hidden_channels)
            for ntype in self.node_types
        })

        # =====================================================================
        # HETEROCONV LAYERS with GATv2Conv
        # =====================================================================
        def make_hetero_layer():
            """Create one HeteroConv layer with relation-specific GATv2Conv."""
            rel_convs = {}

            for (src, rel, dst) in self.edge_types:
                # Determine edge attribute dimension based on relation type
                edge_dim = self._get_edge_dim(rel)

                rel_convs[(src, rel, dst)] = GATv2Conv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    heads=heads,
                    edge_dim=edge_dim,
                    add_self_loops=False,
                    concat=False  # Average multi-head outputs
                )

            return HeteroConv(rel_convs, aggr="sum")

        self.convs = nn.ModuleList([make_hetero_layer() for _ in range(num_layers)])

        # =====================================================================
        # LAYER NORMALIZATION (per node type, per layer)
        # =====================================================================
        self.norms = nn.ModuleList([
            nn.ModuleDict({
                ntype: nn.LayerNorm(hidden_channels)
                for ntype in self.node_types
            })
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # =====================================================================
        # CLASSIFICATION HEADS - one per prediction horizon (1q, 2q, 4q)
        # Shared trunk -> 3 separate output layers for multi-task learning.
        # =====================================================================
        # When use_jk=True, Layer-1 and Layer-2 company outputs are concatenated
        # before the trunk, giving hidden_channels * num_layers input channels.
        trunk_in = (hidden_channels * num_layers if use_jk else hidden_channels) + net_sys_dim
        self.classifier_trunk = nn.Sequential(
            nn.Linear(trunk_in, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head_1q = nn.Linear(hidden_channels, 1)  # 1-quarter horizon (legacy)
        self.head_2q = nn.Linear(hidden_channels, 1)  # 2-quarter horizon (legacy)
        self.head_4q = nn.Linear(hidden_channels, 1)  # 4-quarter horizon (legacy)
        # 52-bin survival head: outputs per-bin hazard logits h_k = sigmoid(phi_k)
        # Bias init: sigmoid(-8) ~ 0.00034/bin, so F_terminal ~ 1.7% ~ base rate
        self.n_survival_bins = 52
        self.head_surv = nn.Linear(hidden_channels, self.n_survival_bins)
        nn.init.constant_(self.head_surv.bias, -8.0)

    # Supply-chain edge types carry a binary edge_weight attribute (dim=1).
    # When use_hazard_attn=True, all edge types additionally carry the
    # destination node's previous-snapshot hazard score (+1 dim, B3).
    _SUPPLY_CHAIN_RELS = frozenset({
        'customer_of', 'supplier_of', 'supplier_ext_of', 'customer_ext_of',
    })

    def _get_edge_dim(self, rel: str) -> Optional[int]:
        """Edge attribute dimension: supply-chain weight (1) + hazard flag (+1 if B3)."""
        base = 1 if rel in self._SUPPLY_CHAIN_RELS else 0
        if self.use_hazard_attn:
            base += 1
        return base if base > 0 else None

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
        m_local: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the model.

        Parameters
        ----------
        x_dict : Dict[str, torch.Tensor]
            Node features per node type
        edge_index_dict : Dict[Tuple[str, str, str], torch.Tensor]
            Edge indices per edge type
        edge_attr_dict : Optional[Dict[Tuple[str, str, str], torch.Tensor]]
            Edge attributes per edge type

        Returns
        -------
        logits : torch.Tensor
            Classification logits for borrower nodes, shape (num_borrowers,)
        h_dict : Dict[str, torch.Tensor]
            Node embeddings after final GNN layer, per node type

        Raises
        ------
        ValueError
            If borrower node type not found in output
        """
        # =====================================================================
        # 1. PROJECT INPUT FEATURES TO HIDDEN SPACE
        # =====================================================================
        h_dict = {}
        for ntype, x in x_dict.items():
            if x is not None and x.numel() > 0:
                h_dict[ntype] = self.proj[ntype](x)
            else:
                # Handle empty node types
                num_nodes = x.shape[0] if x is not None else 1
                h_dict[ntype] = torch.zeros(
                    (num_nodes, self.hidden_channels),
                    device=x.device if x is not None else 'cpu',
                    dtype=torch.float
                )

        # Store projected embeddings for residual skip (ensures isolated nodes
        # retain a unique, input-derived representation after GATv2 layers)
        h_init = {ntype: h.clone() for ntype, h in h_dict.items()}

        # =====================================================================
        # 2. APPLY HETEROCONV LAYERS
        # =====================================================================
        jk_outputs = [] if self.use_jk else None  # B2: accumulate per-layer company emb

        for layer_idx, conv in enumerate(self.convs):
            # Message passing (pass edge_attr_dict so GATv2 uses supply-chain weights)
            # Only pass edge_attr_dict as kwarg when non-None - PyG HeteroConv
            # raises TypeError if it receives None and tries to iterate over it.
            if edge_attr_dict:
                h_dict_new = conv(h_dict, edge_index_dict,
                                  edge_attr_dict=edge_attr_dict)
            else:
                h_dict_new = conv(h_dict, edge_index_dict)

            # Handle nodes that didn't receive messages
            for ntype in self.node_types:
                if ntype not in h_dict_new or h_dict_new[ntype] is None:
                    h_dict_new[ntype] = h_dict.get(ntype, None)

                    if h_dict_new[ntype] is None:
                        # Create zero tensor for missing node types
                        device = list(h_dict.values())[0].device
                        h_dict_new[ntype] = torch.zeros(
                            (1, self.hidden_channels),
                            device=device,
                            dtype=torch.float
                        )

            # Layer normalization + activation + dropout
            layer_norms = self.norms[layer_idx]
            for ntype in self.node_types:
                if h_dict_new[ntype] is not None and h_dict_new[ntype].numel() > 0:
                    h = layer_norms[ntype](h_dict_new[ntype])
                    h = F.relu(h)
                    h_dict_new[ntype] = self.dropout(h)

            h_dict = h_dict_new

            # B2: collect this layer's company representation for JK concatenation
            if self.use_jk and 'company' in h_dict and h_dict['company'] is not None:
                jk_outputs.append(h_dict['company'])

        # Residual skip: three-way branch controlled by use_no_skip / use_gated_residual.
        # use_no_skip=True  -> skip omitted entirely; GATv2 output used directly.
        # use_gated_residual=True -> learned per-node sigmoid gate g:
        #   h = g * GATv2_out + (1-g) * LSTM_init
        #   Gate learns g~1 for corp-structure firms (real signal), g~0 for isolated.
        # Default -> unconditional additive skip (original behaviour).
        # Note: applied before JK concat, which overrides h_dict['company'] below.
        if not self.use_no_skip:
            if self.use_gated_residual:
                for ntype in self.node_types:
                    if ntype in h_dict and h_dict[ntype] is not None and ntype in h_init:
                        if h_dict[ntype].shape == h_init[ntype].shape:
                            g = self.residual_gates[ntype](
                                torch.cat([h_dict[ntype], h_init[ntype]], dim=-1)
                            )
                            h_dict[ntype] = g * h_dict[ntype] + (1.0 - g) * h_init[ntype]
            else:
                for ntype in self.node_types:
                    if ntype in h_dict and h_dict[ntype] is not None and ntype in h_init:
                        if h_dict[ntype].shape == h_init[ntype].shape:
                            h_dict[ntype] = h_dict[ntype] + h_init[ntype]

        # B2: Jumping Knowledge -- concatenate all-layer company representations.
        # Distinguishes immediate 1-hop shock (Layer 1) from systemic decay (Layer 2+).
        # classifier_trunk was constructed with trunk_in = hidden * num_layers + net_sys_dim.
        if self.use_jk and jk_outputs and len(jk_outputs) == self.num_layers:
            h_dict['company'] = torch.cat(jk_outputs, dim=1)  # [N, hidden * num_layers]

        # =====================================================================
        # 3. MULTI-HORIZON CLASSIFICATION
        # =====================================================================
        if "company" not in h_dict or h_dict["company"] is None:
            raise ValueError("'company' node type not found in model output")

        # Option B: concatenate GATv2 output with Vasicek channel-systematic vector
        company_h = h_dict["company"]
        if m_local is not None:
            company_h = torch.cat([company_h, m_local], dim=1)  # [N, hidden + net_sys_dim]
        trunk = self.classifier_trunk(company_h)            # [N, hidden]
        # Legacy 3-bin logits (retained for ablation / backward compat)
        logits = torch.stack([
            self.head_1q(trunk).squeeze(-1),   # [N] - 1-quarter horizon
            self.head_2q(trunk).squeeze(-1),   # [N] - 2-quarter horizon
            self.head_4q(trunk).squeeze(-1),   # [N] - 4-quarter horizon (primary)
        ], dim=1)                               # [N, 3]
        # 52-bin survival PMF logits (primary survival output)
        phi = self.head_surv(trunk)             # [N, 52]

        return phi, logits, h_dict

    def get_num_parameters(self) -> int:
        """
        Get total number of model parameters.

        Returns
        -------
        int
            Total parameter count
        """
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_parameters(self) -> int:
        """
        Get number of trainable parameters.

        Returns
        -------
        int
            Trainable parameter count
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class NetworkSystematicModule(nn.Module):
    """
    Module 3: Network Systematic Risk  (M_local in Two-Factor Vasicek).

    Two-Factor Vasicek interpretation
    ----------------------------------
    The standard Vasicek model has one systematic factor M_global (captured
    by the Macro Gate).  This module adds a local systematic factor M_local(i)
    that aggregates *peer distress* from the firm's direct network neighbours,
    grouped into four economically meaningful contagion channels:

        ch[0] supply_chain   : customer_of, supplier_of, supplier_ext_of, customer_ext_of
        ch[1] corp_structure : subsidiary_of, parent_of
        ch[2] common_owner   : co_owned_by, co_owner_of
        ch[3] competitor     : competitor_of, competitor_of_rev

    For each channel the gated-LSTM embeddings of neighbours are mean-pooled
    then projected to a scalar signal via a learned weight vector.
    Isolated nodes (zero neighbours in a channel) receive signal = 0.

    The [N, 4] output is concatenated with the gated embedding [N, lstm_hidden]
    before GATv2, giving the attention mechanism an explicit pre-aggregated
    network-risk signal alongside the firm-level embedding.

    Parameters
    ----------
    in_channels : int   dimension of the gated-LSTM embedding (lstm_hidden).

    Output
    ------
    Tensor [N, 4]   one scalar per contagion channel per firm.
    """

    CHANNELS: List[Tuple[str, List[str]]] = [
        ('supply_chain',   ['customer_of', 'supplier_of',
                            'supplier_ext_of', 'customer_ext_of']),
        ('corp_structure', ['subsidiary_of', 'parent_of']),
        ('common_owner',   ['co_owned_by',   'co_owner_of']),
        ('competitor',     ['competitor_of', 'competitor_of_rev']),
    ]
    OUT_DIM: int = 4

    def __init__(self, in_channels: int):
        super().__init__()
        # One bias-free linear projection per channel: [D] -> scalar
        self.proj = nn.ModuleList([
            nn.Linear(in_channels, 1, bias=False)
            for _ in self.CHANNELS
        ])

    def forward(
        self,
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        h: torch.Tensor,                                    # [N, D]  gated firm embeddings
        h_prev: Optional[torch.Tensor] = None,              # [N, D]  previous-snapshot embeddings (B1)
        edge_attr_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
    ) -> torch.Tensor:                                      # [N, 4]
        """
        Pool peer signals per contagion channel.

        When h_prev is provided (B1 Contagion Velocity), pools the embedding
        delta (h - h_prev) instead of raw embeddings, so Module 3 signals the
        *acceleration* of peer distress rather than its absolute level.
        Firms absent from the previous snapshot receive h_prev=0, making their
        delta equal to the raw embedding (cold-start degrades gracefully).

        When edge_attr_dict is provided (use_weighted_module3=True), supply-chain
        pooling uses revenue_percent weights (attr[:,0]) instead of uniform counts.
        Only the 9% of edges with disclosed revenue_percent benefit; the 91% proxy
        edges fall through to the unweighted path via the same weight value.
        """
        pool_input = (h - h_prev) if h_prev is not None else h
        N, D = pool_input.shape
        device = pool_input.device
        signals: List[torch.Tensor] = []

        for ch_idx, (_, rel_types) in enumerate(self.CHANNELS):
            agg   = torch.zeros(N, D, device=device)
            count = torch.zeros(N, 1, device=device)

            for rel in rel_types:
                key = ('company', rel, 'company')
                if key not in edge_index_dict:
                    continue
                ei        = edge_index_dict[key]          # [2, E]
                src, dst  = ei[0], ei[1]
                if (edge_attr_dict is not None and key in edge_attr_dict
                        and edge_attr_dict[key] is not None):
                    w = edge_attr_dict[key][:, 0:1].to(device)   # [E, 1] revenue weight
                    agg.scatter_add_(
                        0, dst.unsqueeze(1).expand(-1, D), pool_input[src] * w
                    )
                    count.scatter_add_(0, dst.unsqueeze(1), w)
                else:
                    agg.scatter_add_(
                        0, dst.unsqueeze(1).expand(-1, D), pool_input[src]
                    )
                    count.scatter_add_(
                        0, dst.unsqueeze(1),
                        torch.ones(src.shape[0], 1, device=device)
                    )

            has_nbr = (count > 0).float()                 # [N, 1]
            agg     = agg / (count + 1e-8) * has_nbr      # mean, zero isolated
            sig     = self.proj[ch_idx](agg) * has_nbr    # [N, 1]
            signals.append(sig)

        return torch.cat(signals, dim=1)                  # [N, 4]


class SpatioTemporalGNN(nn.Module):
    """
    Two-Factor Spatiotemporal GNN (Vasicek-Grounded Architecture).

    Architecture
    ------------
    1. Firm LSTM    : [N, T, firm_dim]  -> [N, lstm_hidden]       (epsilon_i)
       Encodes each company's idiosyncratic financial trajectory.

    2. Macro LSTM   : [1, T, macro_dim] -> [1, macro_hidden]
       Encodes the shared macroeconomic context (VIX, HY spread,
       Fed Funds, unemployment, GDP growth).

    3. Macro Gate   : macro_emb -> gate [lstm_hidden]              (M_global)
       Additive gate: firm_emb * (1 + gate).  Implements the global
       systematic factor from the Vasicek two-factor extension.

    4. Network Sys. : gated_emb [N, lstm_hidden] -> [N, 4]         (M_local)
       Per-channel mean-pool of neighbour gated embeddings, projected
       to 4 scalar contagion signals (supply-chain, corporate structure,
       common ownership, competitor). Implements the local systematic
       factor (peer distress). Carried forward as a separate residual
       path, bypassing GATv2 entirely.

    5. GATv2        : [N, lstm_hidden] -> [N, hidden_channels]
       Heterogeneous attention over the firm embedding only.
       m_local bypasses this module entirely.

    6. Classifier   : [N, hidden_channels + 4] -> logits [N, 3]   (1q/2q/4q)
       Trunk input = cat([GATv2 output, m_local]). The Vasicek
       channel-systematic signal enters the classifier independently
       of GATv2 neighbourhood aggregation.

    Parameters
    ----------
    metadata : graph metadata (node_types, edge_types)
    temporal_dim : int
        Total features per time step (firm_dim + macro_dim).
    macro_dim : int
        Number of macro features (last columns of temporal_sequences).
        Default 5: vix, hy_spread, fed_funds, unemployment, gdp_growth.
    lstm_hidden : int
        Firm LSTM hidden dimension.
    macro_hidden : int
        Macro LSTM hidden dimension.
    lstm_layers : int
        Number of LSTM layers (firm encoder).
    hidden_channels : int
        GATv2 hidden dimension.
    num_layers : int
        Number of GATv2 layers.
    heads : int
        Number of attention heads.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        temporal_dim: int = 16,
        macro_dim: int = 5,
        lstm_hidden: int = 128,
        macro_hidden: int = 32,
        lstm_layers: int = 1,
        hidden_channels: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
        sector_dim: int = 32,
        use_momentum: bool = False,
        use_hazard_attn: bool = False,
        use_jk: bool = False,
        use_saf: bool = True,
        use_no_skip: bool = False,
        use_gated_residual: bool = False,
        use_weighted_module3: bool = False,
    ):
        super().__init__()

        from .temporal_encoder import TemporalEncoder

        self.firm_dim       = temporal_dim - macro_dim   # 11
        self.macro_dim      = macro_dim                  # 5
        self.lstm_hidden    = lstm_hidden
        self.use_momentum   = use_momentum
        self.use_hazard_attn = use_hazard_attn
        self.use_jk         = use_jk
        self.use_saf        = use_saf
        self.use_weighted_module3 = use_weighted_module3

        # Module 1 - Firm LSTM: [N, T, firm_dim] -> [N, lstm_hidden]
        self.firm_encoder = TemporalEncoder(
            input_dim=self.firm_dim,
            hidden_dim=lstm_hidden,
            num_layers=lstm_layers,
        )

        # Module 2 - Macro LSTM: [1, T, macro_dim] -> [1, macro_hidden]
        self.macro_encoder = TemporalEncoder(
            input_dim=macro_dim,
            hidden_dim=macro_hidden,
            num_layers=1,
        )

        # Module 2 cont - Macro Gate: additive scale in (1, 2)
        self.macro_gate = nn.Sequential(
            nn.Linear(macro_hidden, lstm_hidden),
            nn.Sigmoid(),
        )

        # Module 2b - Residual FiLM: bounded, identity-init sector modulation
        self.film = SectorAttentionFusion(embed_dim=lstm_hidden, num_sectors=12, sector_dim=sector_dim)

        # Module 3 - Network Systematic Risk: [N, lstm_hidden] -> [N, 4]
        self.net_sys = NetworkSystematicModule(in_channels=lstm_hidden)

        # Module 4 - GATv2: input is firm embedding only (M_local bypasses as residual path)
        # Module 5 - GRS bypass gate [*Proposed]: learnable per-node gate after GATv2
        # Module 6 - Classifier trunk: receives [GATv2 output || M_local] = [hidden + 4]
        self.gnn = HeteroEdgeModel(
            metadata=metadata,
            in_channels_dict={'company': lstm_hidden},  # 128, not 132
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            net_sys_dim=NetworkSystematicModule.OUT_DIM,  # 4
            use_hazard_attn=use_hazard_attn,
            use_jk=use_jk,
            use_no_skip=use_no_skip,
            use_gated_residual=use_gated_residual,
        )

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
        temporal_sequences: Optional[torch.Tensor] = None,
        temporal_lengths: Optional[torch.Tensor] = None,
        sector_idx: Optional[torch.Tensor] = None,
        h_prev: Optional[torch.Tensor] = None,
        hazard_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass: Firm LSTM -> Macro Gate -> FiLM -> Network Systematic -> GATv2.

        Parameters
        ----------
        x_dict : node features (used for device/shape reference only).
        edge_index_dict : edge indices per edge type.
        edge_attr_dict : edge attributes per edge type (binary weight=1 for supply-chain).
        temporal_sequences : Tensor [N, T, temporal_dim]
            Combined firm + macro quarterly sequences.
            First firm_dim columns are firm features; last macro_dim are macro.
        temporal_lengths : optional packed-LSTM sequence lengths.
        sector_idx : Tensor [N] GICS sector index (0=unknown, 1-11). If None, FiLM skipped.
        h_prev : Tensor [N, lstm_hidden] or None.
            Previous-snapshot LSTM embeddings from state buffer (B1 Contagion Velocity).
            When provided and use_momentum=True, Module 3 pools the embedding delta
            (company_emb - h_prev) instead of raw company_emb.
        hazard_prev : Tensor [N, 1] or None.
            Previous-snapshot terminal hazard F(t=52) from state buffer (B3 Directional
            Distress Attention). Injected as an additional edge attribute so GATv2 attends
            more to high-hazard neighbours. Only used when use_hazard_attn=True.

        Returns
        -------
        phi    : Tensor [N, 52]  raw PMF logits for 52-bin survival head (primary)
        logits : Tensor [N, 3]   legacy 3-bin hazard logits (1q, 2q, 4q)
        h_dict : Dict[str, Tensor]  node embeddings after GATv2;
                 also stores h_dict['_company_emb'] = company_emb [N, lstm_hidden]
                 so the trainer can populate the state buffer without a second forward pass.
        """
        N      = x_dict['company'].shape[0]
        device = x_dict['company'].device

        if temporal_sequences is not None:
            # Module 1: firm trajectory -> idiosyncratic embedding
            firm_seq  = temporal_sequences[:, :, :self.firm_dim]      # [N, T, 11]
            macro_seq = temporal_sequences[0:1, :, self.firm_dim:]    # [1, T,  5]
            firm_emb  = self.firm_encoder(firm_seq, temporal_lengths)  # [N, 128]

            # Module 2: macro context -> additive gate (M_global)
            macro_emb   = self.macro_encoder(macro_seq)                # [1,  32]
            gate        = self.macro_gate(macro_emb)                   # [1, 128]
            company_emb = firm_emb * (1.0 + gate)                     # [N, 128]
        else:
            # Graceful degradation: neutral gate (no macro conditioning)
            company_emb = torch.zeros(N, self.lstm_hidden, device=device)

        # Module 2b: SAF sector gate - skipped if use_saf=False (quarterly model)
        # or if sector_idx not provided (ablation runs).
        # SAF disabled for quarterly: 64 snapshots + DeepHit loss insufficient
        # for gate alignment; additive perturbations add noise at quarterly cadence.
        if self.use_saf and sector_idx is not None:
            company_emb = self.film(company_emb, sector_idx)          # [N, 128]

        # Module 3: network peer distress -> local systematic signal (M_local)
        # B1 (use_momentum): pass h_prev so Module 3 pools velocity (delta) signals
        # rather than raw embedding levels. Cold-start firms have h_prev=0 -> delta=emb.
        m_local = self.net_sys(
            edge_index_dict,
            company_emb,
            h_prev=h_prev if self.use_momentum else None,
            edge_attr_dict=edge_attr_dict if self.use_weighted_module3 else None,
        )                                                              # [N, 4]

        # Module 4: GATv2 takes firm embedding only; m_local bypasses GATv2
        # and is concatenated at the classifier head (Option B residual path)
        gnn_x_dict = dict(x_dict)
        gnn_x_dict['company'] = company_emb                           # [N, 128]

        # B3 (use_hazard_attn): inject hazard_{j,t-1} as edge attribute so GATv2
        # computes asymmetric, distress-aware attention weights.
        # Supply-chain edges: [weight, hazard] (dim=2); all others: [hazard] (dim=1).
        gnn_edge_attr = edge_attr_dict
        if self.use_hazard_attn and hazard_prev is not None:
            gnn_edge_attr = {} if edge_attr_dict is None else dict(edge_attr_dict)
            for rel_key, ei in edge_index_dict.items():
                _, rel, _ = rel_key
                dst_idx   = ei[1]                                     # [E]
                h_j       = hazard_prev[dst_idx]                      # [E, 1]
                if rel in HeteroEdgeModel._SUPPLY_CHAIN_RELS and rel_key in gnn_edge_attr:
                    gnn_edge_attr[rel_key] = torch.cat(
                        [gnn_edge_attr[rel_key], h_j], dim=1
                    )                                                  # [E, 2]
                else:
                    gnn_edge_attr[rel_key] = h_j                      # [E, 1]

        phi, logits, h_dict = self.gnn(
            gnn_x_dict, edge_index_dict,
            edge_attr_dict=gnn_edge_attr,
            m_local=m_local,
        )
        # Store LSTM embedding in h_dict so trainer can extract it for the state buffer
        # without a second forward pass (no gradient needed here).
        h_dict['_company_emb'] = company_emb
        return phi, logits, h_dict

    def get_num_parameters(self) -> int:
        """Get total number of model parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


