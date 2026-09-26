"""train.py -- Train one model variant with one random seed.

Usage:
    python train.py --model stgnn --seed 42 --freq W
    python train.py --model grs   --seed 123 --output outputs/
    python train.py --model xgb   --seed 42  --freq W
    python train.py --model coxph --freq W          # deterministic, no seed

Models (deep learning -- saved as .pt checkpoints):
    lstm      LSTMOnly (sequential baseline, no graph)
    stgnn     ST-GNN base (Mod1-5, SAF off, GRS off)
    saf       ST-GNN + Sector Attention Fusion (Mod2b)
    grs       ST-GNN + Gated Residual Skip (isolate-aware bypass)
    nograph   ST-GNN+GRS with all edges zeroed (no-graph ablation)
    rgcn      HomogeneousRGCN (multi-relational proxy)
    gatv2     GATv2-Static (attention snapshot proxy)
    ndr       GATv2+NDR (label-aware neighbour proxy)

Models (tabular/survival -- saved as .pkl checkpoints):
    xgb       XGBoost (financial ratios, 3-seed)
    lgbm      LightGBM (financial ratios, 3-seed)
    coxph     Cox Proportional Hazards (discrete-time hazard, deterministic)
"""

import argparse
import os
import pickle
import random
import sys

import numpy as np
import torch

# Allow running from code_release/ directory directly
sys.path.insert(0, os.path.dirname(__file__))

from data.graph_builder import build_temporal_graphs
from eval.splits import shifted_split
from models.baselines import LSTMOnly, GATv2Only, GATv2NDR, HomogeneousRGCN, NoGraphWrapper
from models.stgnn import SpatioTemporalGNN
from training.trainer import TemporalTrainer

TABULAR_MODELS = {"xgb", "lgbm", "coxph"}


# -----------------------------------------------------------------------
# Constants matching the paper
# -----------------------------------------------------------------------
FIRM_DIM   = 11      # financial ratios (Section 3 Data)
MACRO_DIM  = 5       # FRED series
LSTM_HID   = 128     # Mod1 LSTM hidden
GNN_HID    = 64      # Mod4 GATv2 hidden
GNN_HEADS  = 4
GNN_LAYERS = 2
DROPOUT    = 0.3
LR         = 5e-4
WD         = 1e-3
PATIENCE   = 15
EPOCHS     = 100
T_Q        = 12      # quarterly lookback (3 years)
T_W        = 52      # weekly lookback (1 year)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def compute_ndr_lookup(train_graphs) -> dict:
    """Per-firm Node Default Rate from the training split only.

    NDR_i = (# of firm i's company->company neighbours that defaulted) /
            (# of firm i's company->company neighbour edges), summed over all
            training snapshots. Returns {company_id: NDR}. Firms never seen as a
            source default to 0 at lookup time.
    """
    from collections import defaultdict
    counts = defaultdict(lambda: [0, 0])   # gvkey -> [defaulted_neighbours, total_neighbours]
    for g in train_graphs:
        cids = g['company'].company_ids.cpu().numpy()
        ys   = g['company'].y.cpu().numpy()
        for (src_t, _, dst_t), ei in g.edge_index_dict.items():
            if src_t != 'company' or dst_t != 'company' or ei.numel() == 0:
                continue
            for s_idx, d_idx in zip(ei[0].cpu().numpy(), ei[1].cpu().numpy()):
                if s_idx < len(cids) and d_idx < len(cids):
                    gvk = int(cids[s_idx])
                    counts[gvk][0] += int(ys[d_idx])
                    counts[gvk][1] += 1
    return {gvk: c[0] / max(c[1], 1) for gvk, c in counts.items()}


def build_model(name: str, metadata, device, ndr_lookup=None) -> torch.nn.Module:
    """Construct model from name string."""
    temporal_dim = FIRM_DIM + MACRO_DIM

    if name == "lstm":
        return LSTMOnly(firm_dim=FIRM_DIM, lstm_hidden=LSTM_HID, dropout=DROPOUT)

    if name == "stgnn":
        return SpatioTemporalGNN(
            metadata=metadata,
            temporal_dim=temporal_dim,
            macro_dim=MACRO_DIM,
            lstm_hidden=LSTM_HID,
            hidden_channels=GNN_HID,
            num_layers=GNN_LAYERS,
            heads=GNN_HEADS,
            dropout=DROPOUT,
            use_saf=False,
            use_gated_residual=False,
        )

    if name == "saf":
        return SpatioTemporalGNN(
            metadata=metadata,
            temporal_dim=temporal_dim,
            macro_dim=MACRO_DIM,
            lstm_hidden=LSTM_HID,
            hidden_channels=GNN_HID,
            num_layers=GNN_LAYERS,
            heads=GNN_HEADS,
            dropout=DROPOUT,
            use_saf=True,
            use_gated_residual=False,
        )

    if name == "grs":
        return SpatioTemporalGNN(
            metadata=metadata,
            temporal_dim=temporal_dim,
            macro_dim=MACRO_DIM,
            lstm_hidden=LSTM_HID,
            hidden_channels=GNN_HID,
            num_layers=GNN_LAYERS,
            heads=GNN_HEADS,
            dropout=DROPOUT,
            use_saf=False,
            use_gated_residual=True,
        )

    if name == "rgcn":
        # Number of company->company relation types (supply-chain, corp, owner, competitor)
        return HomogeneousRGCN(in_dim=temporal_dim, hidden=GNN_HID,
                               num_relations=4, dropout=DROPOUT)

    if name == "gatv2":
        return GATv2Only(
            metadata=metadata,
            in_dim=temporal_dim,
            hidden=GNN_HID,
            num_layers=GNN_LAYERS,
            heads=GNN_HEADS,
            dropout=DROPOUT,
        )

    if name == "ndr":
        # GATv2 + Node Default Rate: one extra input feature (in_dim + 1).
        return GATv2NDR(
            metadata=metadata,
            in_dim=temporal_dim + 1,
            hidden=GNN_HID,
            num_layers=GNN_LAYERS,
            heads=GNN_HEADS,
            dropout=DROPOUT,
            ndr_lookup=ndr_lookup,
        )

    if name == "nograph":
        grs = SpatioTemporalGNN(
            metadata=metadata,
            temporal_dim=temporal_dim,
            macro_dim=MACRO_DIM,
            lstm_hidden=LSTM_HID,
            hidden_channels=GNN_HID,
            num_layers=GNN_LAYERS,
            heads=GNN_HEADS,
            dropout=DROPOUT,
            use_saf=False,
            use_gated_residual=True,
        )
        return NoGraphWrapper(grs)

    raise ValueError(
        f"Unknown model '{name}'. "
        "Choose from: lstm stgnn saf grs nograph rgcn gatv2 ndr xgb lgbm coxph"
    )


def extract_flat_features(graphs):
    """Extract (X, y) flat arrays from graph snapshots for tabular models.

    Uses each node's current-snapshot feature vector (x_dict['company']) and
    the 4q horizon label.  One row per firm-snapshot.
    """
    X_list, y_list = [], []
    for g in graphs:
        x = g['company'].x.cpu().numpy()
        y = g['company'].y.cpu().numpy()  # 4q horizon label (stored 1-D [N])
        X_list.append(x)
        y_list.append(y)
    return np.concatenate(X_list), np.concatenate(y_list)


def train_tabular(model_name: str, seed: int, train_graphs, val_graphs, output_dir: str):
    """Train XGBoost, LightGBM, or Cox PH and save as .pkl."""
    X_train, y_train = extract_flat_features(train_graphs)
    X_val,   y_val   = extract_flat_features(val_graphs)
    scale_pos = float((y_train == 0).sum()) / max((y_train == 1).sum(), 1)

    if model_name == "xgb":
        import xgboost as xgb
        clf = xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            eval_metric="aucpr", early_stopping_rounds=30,
            random_state=seed, verbosity=0,
        )
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        model = clf

    elif model_name == "lgbm":
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            metric="average_precision",
            random_state=seed, verbose=-1,
        )
        clf.fit(X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(30, verbose=False),
                           lgb.log_evaluation(period=-1)])
        model = clf

    elif model_name == "coxph":
        import pandas as pd
        from lifelines import CoxPHFitter
        feat_cols = [f"x{i}" for i in range(X_train.shape[1])]
        df = pd.DataFrame(X_train, columns=feat_cols)
        df["T"] = 1          # discrete-time: duration = 1 per snapshot
        df["E"] = y_train.astype(int)
        model = CoxPHFitter(penalizer=0.1)
        model.fit(df, duration_col="T", event_col="E")

    os.makedirs(output_dir, exist_ok=True)
    suffix = "" if model_name == "coxph" else f"_seed{seed}"
    ckpt_path = os.path.join(output_dir, f"{model_name}{suffix}.pkl")
    with open(ckpt_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nCheckpoint saved: {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description="Train ST-GNN credit risk model")
    parser.add_argument("--model",  required=True,
                        choices=["lstm", "stgnn", "saf", "grs", "nograph",
                                 "rgcn", "gatv2", "ndr",
                                 "xgb", "lgbm", "coxph"],
                        help="Model variant")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--freq",   choices=["Q", "W"], default="W",
                        help="Snapshot cadence: Q=quarterly, W=weekly (paper default)")
    parser.add_argument("--data",   default="data/processed",
                        help="Root directory containing processed parquet files")
    parser.add_argument("--cache",  default=None,
                        help="Path to graph cache pickle (built on first run, reused after)")
    parser.add_argument("--output", default="outputs/models",
                        help="Directory to save checkpoint")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Model: {args.model}  |  Seed: {args.seed}  |  Freq: {args.freq}")

    # ----------------------------------------------------------------
    # Load data (with optional pickle cache for faster re-runs)
    # ----------------------------------------------------------------
    if args.cache and os.path.exists(args.cache):
        print(f"\nLoading graphs from cache {args.cache} ...")
        with open(args.cache, "rb") as f:
            _c = pickle.load(f)
        train_graphs, val_graphs, test_graphs = _c["train"], _c["val"], _c["test"]
    else:
        print(f"\nLoading graphs from {args.data} ...")
        train_graphs, val_graphs, test_graphs = build_temporal_graphs(
            base_dir=args.data, freq=args.freq,
        )
        if args.cache:
            os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
            with open(args.cache, "wb") as f:
                pickle.dump({"train": train_graphs, "val": val_graphs,
                             "test": test_graphs}, f)
            print(f"  Graph cache saved: {args.cache}")

    # ----------------------------------------------------------------
    # APPLY THE EMBARGO. build_temporal_graphs (and the cache it writes) returns the
    # PUBLISHED split: train to 2016-12-31, validation 2017-2019, test from 2020. That
    # split has no embargo, and because company.y looks 52 weeks forward a validation
    # window ending where testing begins carries positive labels for firms defaulting
    # after the boundary: 1,030 of 3,059 validation positives are contaminated that way,
    # spanning 35 firms. The paper reports the embargoed split, so it is derived here
    # rather than left to the caller. shifted_split slides validation back one full label
    # horizon and pays for it out of training, giving 783 / 156 / 261 with 2019 in
    # neither set. Applied exactly once, immediately after the published split is obtained.
    _spec = shifted_split(train_graphs + val_graphs + test_graphs,
                          len(train_graphs), len(val_graphs))
    train_graphs, val_graphs, test_graphs = _spec.train, _spec.val, _spec.test
    print(f"  Train: {len(train_graphs)}  Val: {len(val_graphs)}  Test: {len(test_graphs)}")

    # ----------------------------------------------------------------
    # Tabular models (XGBoost / LightGBM / Cox PH) -- separate flow
    # ----------------------------------------------------------------
    if args.model in TABULAR_MODELS:
        train_tabular(args.model, args.seed, train_graphs, val_graphs, args.output)
        return

    metadata = train_graphs[0].metadata()

    # ----------------------------------------------------------------
    # Build model
    # ----------------------------------------------------------------
    # GATv2+NDR needs the per-firm Node Default Rate computed on the training
    # split only; saved to the checkpoint so evaluation reuses the same lookup.
    ndr_lookup = compute_ndr_lookup(train_graphs) if args.model == "ndr" else None
    if ndr_lookup is not None:
        nz = sum(1 for v in ndr_lookup.values() if v > 0)
        print(f"  NDR computed for {len(ndr_lookup)} firms ({nz} with non-zero NDR)")

    model = build_model(args.model, metadata, device, ndr_lookup=ndr_lookup)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # ----------------------------------------------------------------
    # Train
    # ----------------------------------------------------------------
    t_lookback = T_W if args.freq == "W" else T_Q
    trainer = TemporalTrainer(
        model=model,
        train_graphs=train_graphs,
        val_graphs=val_graphs,
        test_graphs=test_graphs,
        device=device,
        lr=LR,
        weight_decay=WD,
        patience=PATIENCE,
        t_lookback=t_lookback,
        force_binary_loss=(args.freq == "Q"),
    )
    trainer.train(num_epochs=args.epochs)

    # ----------------------------------------------------------------
    # Evaluate on test split and save checkpoint
    # ----------------------------------------------------------------
    results = trainer.evaluate(split="test", verbose=True)

    os.makedirs(args.output, exist_ok=True)
    ckpt_path = os.path.join(args.output, f"{args.model}_seed{args.seed}.pt")
    torch.save({
        "model_name":   args.model,
        "seed":         args.seed,
        "freq":         args.freq,
        "state_dict":   model.state_dict(),
        "val_ap":       trainer.best_val_ap,
        "test_metrics": results["aggregate"],
        "metadata":     metadata,
        "ndr_lookup":   ndr_lookup,   # None unless model == "ndr"
        "hyperparams": {
            "temporal_dim":   FIRM_DIM + MACRO_DIM,
            "macro_dim":      MACRO_DIM,
            "lstm_hidden":    LSTM_HID,
            "hidden_channels": GNN_HID,
            "num_layers":     GNN_LAYERS,
            "heads":          GNN_HEADS,
            "dropout":        DROPOUT,
        },
    }, ckpt_path)
    print(f"\nCheckpoint saved: {ckpt_path}")


if __name__ == "__main__":
    main()
