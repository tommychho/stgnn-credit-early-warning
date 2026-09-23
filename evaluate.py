"""evaluate.py -- Reproduce all paper tables, figures, and statistics.

Loads all checkpoints from --checkpoints directory and the test graphs,
then outputs:
    outputs/results/detection_at_capacity.csv   AP, DR and lead time at a budget
    outputs/results/capacity_curve.csv          detection across review budgets
    outputs/results/regime_lift.csv             ranking quality by period, as lift
    outputs/results/stats.txt                   seed-level Welch tests under Holm
    outputs/figures/lt_survival.pdf             lead-time survival curve

Two protocol choices are explicit flags, because both decide what a detection rate
means: --protocol {capacity,recall} and --anchor {default,label_onset}. The defaults
reproduce the paper; the alternatives reproduce the superseded protocol.

Usage:
    python evaluate.py --checkpoints outputs/models --output outputs
"""

import argparse
import os
import pickle
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu, ttest_ind
from statsmodels.stats.contingency_tables import mcnemar

sys.path.insert(0, os.path.dirname(__file__))

from data.graph_builder import build_temporal_graphs
from eval import detection as _det
from models.baselines import LSTMOnly, GATv2Only, GATv2NDR, HomogeneousRGCN, NoGraphWrapper
from models.stgnn import SpatioTemporalGNN
from training.metrics import compute_metrics
from training.trainer import TemporalTrainer, build_sequence_tensor, T_LOOKBACK


# -----------------------------------------------------------------------
# Evaluation protocol (see the paper, "Evaluation Protocol")
# -----------------------------------------------------------------------
# Two choices decide what a detection rate means, and both are set here rather
# than left implicit.
#
#   1. The THRESHOLD is indexed to review capacity, not to a recall target.
#      For a budget c, the operating threshold is the (1-c) quantile of the
#      pooled score distribution over the evaluation window, so alerts consume
#      c of firm-weeks on average and the budget is identical across models.
#      Thresholding to a fixed recall target instead leaves the alert rate free:
#      on this panel a uniform random scorer reaches 47.6% detection under that
#      rule. Pass --protocol recall to reproduce that result.
#
#   2. The EVENT is anchored on the firm's default date, not on the first
#      snapshot where the horizon label turns on. Label-onset anchoring is
#      roughly default minus 52 weeks and leaves 44 of the 88 defaulters with no
#      lookback to score, capping achievable detection at 50%. Pass
#      --anchor label_onset to reproduce that.
CAPACITY_DEFAULT = 0.015      # 1.5% of live firms per week
CAPACITY_CURVE = (0.0025, 0.015, 0.03, 0.05, 0.10)
WINDOW_WEEKS = 52             # confirmation window, matches the label horizon


def load_default_dates(data_root: str) -> Dict[str, list]:
    """{gvkey: [default_date, ...]} from the Compustat/S&P default record."""
    path = os.path.join(data_root, "raw", "defaults",
                        "compustat_sp_defaults_fallback.parquet")
    df = pd.read_parquet(path)
    df["default_date"] = pd.to_datetime(df["default_date"])
    return df.groupby("gvkey")["default_date"].apply(lambda s: sorted(s)[0]).to_dict()


def build_scores(firm_data: Dict) -> Dict[Tuple[int, int], float]:
    """{(company_id, snapshot_idx): score} in the shape eval.detection expects."""
    return {(cid, si): p
            for cid, d in firm_data.items()
            for si, p in d["probs"]}


def build_anchors(all_graphs, test_snap_idx, data_root: str, mode: str):
    """Event anchors and the defaulter list, on the corrected or published anchor."""
    if mode == "label_onset":
        return _det.build_event_anchors(all_graphs, test_snap_idx, mode="label_onset")
    cid_to_gvkey = {}
    for si in test_snap_idx:
        g = all_graphs[si]
        for gvk in g["company"].gvkey:
            cid_to_gvkey[int(gvk)] = str(gvk)
    return _det.build_event_anchors(
        all_graphs, test_snap_idx, mode="default",
        gvkey_to_defdate=load_default_dates(data_root),
        cid_to_gvkey=cid_to_gvkey,
    )


# -----------------------------------------------------------------------
# Paper constants
# -----------------------------------------------------------------------
FIRM_DIM   = 11
MACRO_DIM  = 5
LSTM_HID   = 128
GNN_HID    = 64
GNN_HEADS  = 4
GNN_LAYERS = 2
DROPOUT    = 0.3
T_W        = 52   # weekly lookback
T_Q        = 12   # quarterly lookback

# 88 test defaulters; periods defined in Table IV
PERIOD_COVID     = (pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31"))
PERIOD_POSTCOVID = (pd.Timestamp("2022-01-01"), pd.Timestamp("2024-12-31"))

# All trained models use 3 seeds (42, 123, 456); only Cox PH is deterministic (1 fit).
MULTI_SEED_MODELS  = {"lstm", "stgnn", "saf", "grs", "nograph",
                      "rgcn", "gatv2", "ndr", "xgb", "lgbm"}
TABULAR_MODELS     = {"xgb", "lgbm", "coxph"}
DISPLAY_NAMES = {
    "xgb":     "XGBoost (financial ratios)",
    "lgbm":    "LightGBM (financial ratios)",
    "coxph":   "Cox PH (hazard baseline)",
    "rgcn":    "R-GCN (adapted implementation)",
    "gatv2":   "GATv2-Static (static relational adaptation)",
    "ndr":     "GATv2+NDR (snapshot attention baseline)",
    "lstm":    "LSTM-only",
    "stgnn":   "ST-GNN base",
    "saf":     "ST-GNN+SAF",
    "nograph": "ST-GNN+GRS (no graph, ablation)",
    "grs":     "ST-GNN+GRS",
}


# -----------------------------------------------------------------------
# Model reconstruction
# -----------------------------------------------------------------------

def rebuild_model(ckpt: dict) -> torch.nn.Module:
    """Reconstruct model from saved checkpoint."""
    name   = ckpt["model_name"]
    hp     = ckpt["hyperparams"]
    meta   = ckpt["metadata"]
    tdim   = hp["temporal_dim"]

    if name == "lstm":
        m = LSTMOnly(firm_dim=FIRM_DIM, lstm_hidden=LSTM_HID, dropout=DROPOUT)
    elif name == "stgnn":
        m = SpatioTemporalGNN(metadata=meta, temporal_dim=tdim, macro_dim=MACRO_DIM,
                              lstm_hidden=LSTM_HID, hidden_channels=GNN_HID,
                              num_layers=GNN_LAYERS, heads=GNN_HEADS, dropout=DROPOUT,
                              use_saf=False, use_gated_residual=False)
    elif name == "saf":
        m = SpatioTemporalGNN(metadata=meta, temporal_dim=tdim, macro_dim=MACRO_DIM,
                              lstm_hidden=LSTM_HID, hidden_channels=GNN_HID,
                              num_layers=GNN_LAYERS, heads=GNN_HEADS, dropout=DROPOUT,
                              use_saf=True, use_gated_residual=False)
    elif name == "grs":
        m = SpatioTemporalGNN(metadata=meta, temporal_dim=tdim, macro_dim=MACRO_DIM,
                              lstm_hidden=LSTM_HID, hidden_channels=GNN_HID,
                              num_layers=GNN_LAYERS, heads=GNN_HEADS, dropout=DROPOUT,
                              use_saf=False, use_gated_residual=True)
    elif name == "nograph":
        grs = SpatioTemporalGNN(metadata=meta, temporal_dim=tdim, macro_dim=MACRO_DIM,
                                lstm_hidden=LSTM_HID, hidden_channels=GNN_HID,
                                num_layers=GNN_LAYERS, heads=GNN_HEADS, dropout=DROPOUT,
                                use_saf=False, use_gated_residual=True)
        m = NoGraphWrapper(grs)
    elif name == "rgcn":
        m = HomogeneousRGCN(in_dim=tdim, hidden=GNN_HID, num_relations=4, dropout=DROPOUT)
    elif name == "gatv2":
        m = GATv2Only(metadata=meta, in_dim=tdim, hidden=GNN_HID,
                      num_layers=GNN_LAYERS, heads=GNN_HEADS, dropout=DROPOUT)
    elif name == "ndr":
        # in_dim + 1 for the appended NDR feature; reuse the training-split lookup.
        m = GATv2NDR(metadata=meta, in_dim=tdim + 1, hidden=GNN_HID,
                     num_layers=GNN_LAYERS, heads=GNN_HEADS, dropout=DROPOUT,
                     ndr_lookup=ckpt.get("ndr_lookup"))
    else:
        raise ValueError(f"Unknown model: {name}")

    m.load_state_dict(ckpt["state_dict"])
    return m


# -----------------------------------------------------------------------
# Persistence filter (double-lock) for DR and LT
# -----------------------------------------------------------------------

def apply_persistence_filter(
    firm_probs: Dict[int, List[Tuple[int, float]]],
    threshold: float = 0.5,
    window_snaps: int = 26,
    n_consecutive: int = 2,
) -> Dict[int, int]:
    """SUPERSEDED. The published rule: absolute threshold, 26-week window.

    Retained so the earlier protocol stays inspectable, and because a reader may want
    to see what a fixed 0.5 cut-off does. It is not used by the reporting path, which
    goes through ``eval.detection`` with a capacity-indexed threshold and a 52-week
    window. Detection computed this way is confounded by alert volume: the threshold
    is not tied to any review budget, so the alert rate floats free across models.

    Double-lock persistence filter.

    For each firm, find the first snapshot where >= n_consecutive exceedances
    of `threshold` occur within a `window_snaps`-snapshot window.

    Parameters
    ----------
    firm_probs : {firm_id: [(snap_i, prob), ...]} ordered by snap_i
    threshold : risk score threshold (0.5 CIF threshold used in paper)
    window_snaps : look-back window (26 weeks = ~6 months)
    n_consecutive : consecutive exceedances required (2 in paper)

    Returns
    -------
    {firm_id: first_confirmed_snap_i}  -- only firms with a confirmed signal
    """
    confirmed = {}
    for fid, history in firm_probs.items():
        history = sorted(history, key=lambda x: x[0])
        for j in range(len(history)):
            si, pi = history[j]
            if pi < threshold:
                continue
            # count consecutive exceedances within window
            count = 1
            for k in range(j + 1, len(history)):
                sj, pj = history[k]
                if sj - si > window_snaps:
                    break
                if pj >= threshold:
                    count += 1
                    if count >= n_consecutive:
                        confirmed[fid] = si
                        break
                else:
                    count = 1
                    si = sj
            if fid in confirmed:
                break
    return confirmed


# -----------------------------------------------------------------------
# Per-firm inference
# -----------------------------------------------------------------------

def run_tabular_inference(tabular_model, model_name: str, all_graphs, test_start: int):
    """Score test snapshots with a tabular or survival model.

    Uses each node's current-snapshot feature vector (x_dict['company']).
    Returns the same (firm_data, results) structure as run_inference so the
    downstream persistence filter and metric computation are unchanged.
    """
    from train import extract_flat_features  # reuse feature extractor
    test_graphs = all_graphs[test_start:]
    X_test, y_test = extract_flat_features(test_graphs)

    if model_name == "coxph":
        import pandas as pd
        feat_cols = [f"x{i}" for i in range(X_test.shape[1])]
        df = pd.DataFrame(X_test, columns=feat_cols)
        df["T"] = 1
        df["E"] = y_test.astype(int)
        # Partial hazard as risk score (higher = more risk)
        scores = model_name and tabular_model.predict_partial_hazard(df).values
    else:
        scores = tabular_model.predict_proba(X_test)[:, 1]

    # Package into per-snapshot list matching run_inference output format
    firm_data: Dict = {}
    offset = 0
    results = []
    for g in test_graphs:
        n = g["company"].x.shape[0]
        snap_scores = scores[offset: offset + n]
        snap_labels = y_test[offset: offset + n]
        results.append({"scores": snap_scores, "labels": snap_labels})
        offset += n
    return firm_data, results


def run_inference(
    model: torch.nn.Module,
    all_graphs, seq_cache, test_start: int,
    device: torch.device,
    freq: str = "W",
) -> Tuple[Dict, List]:
    """
    Run model on all test snapshots, returning per-firm predictions.

    Returns
    -------
    firm_data : {firm_id: {'probs': [(snap_i, prob)], 'label': int,
                           'event_snap': int or None, 'snap_ts': timestamp or None,
                           'degree': int, 'corp_edges': int}}
    all_results : [(probs_np, labels_np, phi_np, snap_t, cids_np), ...]
    """
    model.eval()
    firm_data: Dict = defaultdict(lambda: {'probs': [], 'label': 0,
                                            'event_snap': None, 'snap_ts': None,
                                            'degree': 0, 'corp_edges': 0})
    all_results = []
    _eval_buffer: Dict = {}
    force_binary = (freq == "Q")

    # The 52-bin survival head (head_surv -> phi) is only meaningful if the DeepHit
    # 52-bin objective actually trained it. TemporalTrainer._loss uses _deephit_loss(phi)
    # only when per-bin labels (y_bin, y_event) are available; the baseline checkpoints
    # reported in the paper are trained through _survival_loss(logits [N,3]) instead, which
    # never touches head_surv. Those checkpoints keep head_surv at initialisation
    # (bias == -8.0 constant, see nn.init.constant_ in the model definition), so scoring
    # phi there reads a fixed random projection of the classifier trunk rather than a
    # calibrated hazard. Detect that case and fall back to the trained 3-bin head.
    def _head_surv_is_trained(m) -> bool:
        hs = getattr(getattr(m, "gnn", m), "head_surv", None)
        if hs is None or not hasattr(hs, "bias") or hs.bias is None:
            return False
        b = hs.bias.detach()
        return not bool(torch.allclose(b, torch.full_like(b, -8.0)))

    phi_trained = _head_surv_is_trained(model)
    use_binary_head = force_binary or not phi_trained
    if not force_binary and not phi_trained:
        print("  [scoring] head_surv is at initialisation (untrained under the 3-bin "
              "objective); scoring the trained 3-bin hazard head instead of phi.")

    def _f_terminal(phi, logits):
        if use_binary_head:
            return torch.sigmoid(logits[:, 2])
        h_t    = torch.sigmoid(phi)
        log_st = torch.log((1.0 - h_t).clamp(1e-7)).cumsum(dim=1)
        return 1.0 - torch.exp(log_st[:, -1])

    with torch.no_grad():
        n_test = len(all_graphs) - test_start
        for k in range(n_test):
            i_g = test_start + k
            g   = all_graphs[i_g].to(device)
            seq = seq_cache[i_g].to(device)

            ea = {et: g[et].edge_attr for et in g.edge_types
                  if hasattr(g[et], 'edge_attr') and g[et].edge_attr is not None}
            sid = getattr(g['company'], 'sector_idx', None)

            # GATv2+NDR rebuilds its NDR feature from company_ids; others ignore it.
            extra = ({'company_ids': g['company'].company_ids}
                     if hasattr(model, 'ndr_lookup') else {})
            out = model(g.x_dict, g.edge_index_dict,
                        edge_attr_dict=ea if ea else None,
                        temporal_sequences=seq,
                        sector_idx=sid,
                        **extra)
            if len(out) == 3:
                phi, logits, h_dict = out
            else:
                logits, h_dict = out
                phi = logits   # fallback

            f_term = _f_terminal(phi, logits)

            # Update eval buffer if model has state
            if hasattr(model, 'use_momentum') and '_company_emb' in h_dict:
                cids = g['company'].company_ids.cpu().numpy()
                emb  = h_dict['_company_emb'].detach().cpu()
                haz  = f_term.detach().cpu().unsqueeze(1)
                for j, cid in enumerate(cids):
                    _eval_buffer[int(cid)] = {'h': emb[j], 'hazard': haz[j]}

            probs_np  = f_term.cpu().numpy()
            labels_np = g['company'].y.float().cpu().numpy()
            cids_np   = g['company'].company_ids.cpu().numpy()
            snap_t    = getattr(g['company'], 't', None)

            # Connectivity info
            degrees = torch.zeros(g['company'].x.shape[0], dtype=torch.long)
            corp_degrees = torch.zeros(g['company'].x.shape[0], dtype=torch.long)
            for (src_t, rel, dst_t), ei in g.edge_index_dict.items():
                if dst_t == 'company':
                    degrees.scatter_add_(0, ei[1].cpu(),
                                         torch.ones(ei.shape[1], dtype=torch.long))
                    if rel in ('subsidiary_of', 'parent_of'):
                        corp_degrees.scatter_add_(0, ei[1].cpu(),
                                                   torch.ones(ei.shape[1], dtype=torch.long))

            for j, cid in enumerate(cids_np.tolist()):
                p  = float(probs_np[j])
                lb = int(labels_np[j])
                firm_data[cid]['probs'].append((i_g, p))
                firm_data[cid]['label'] = max(firm_data[cid]['label'], lb)
                firm_data[cid]['degree'] = max(firm_data[cid]['degree'],
                                                int(degrees[j]))
                firm_data[cid]['corp_edges'] = max(firm_data[cid]['corp_edges'],
                                                    int(corp_degrees[j]))
                if lb == 1 and firm_data[cid]['event_snap'] is None:
                    firm_data[cid]['event_snap'] = i_g
                    firm_data[cid]['snap_ts']    = snap_t

            phi_np = phi.cpu().numpy() if not force_binary else None
            all_results.append((probs_np, labels_np, phi_np, snap_t, cids_np))

    return dict(firm_data), all_results


# -----------------------------------------------------------------------
# Table builders
# -----------------------------------------------------------------------

def compute_table1_row(model_name, predictions_runs, anchors, defaulters,
                       capacity=CAPACITY_DEFAULT, protocol="capacity"):
    """One detection-table row: ranking quality plus detection at a stated budget.

    Seed handling differs by quantity on purpose. AP is averaged over seeds and its
    spread reported, because it is a property of the run. Detection is computed on
    each seed and averaged, so the reported DR is a mean over runs rather than a
    single run's value, which is what the paper reports.
    """
    ap_per_run, dr_per_run, ltm_per_run, ltmed_per_run = [], [], [], []
    all_probs, all_labels = [], []

    for fd, results in predictions_runs:
        probs_ = np.concatenate([r[0] for r in results])
        labels_ = np.concatenate([r[1] for r in results])
        all_probs.append(probs_)
        all_labels.append(labels_)
        ap_per_run.append(compute_metrics(labels_, probs_).get("ap", float("nan")))

        scores = build_scores(fd)
        vals = np.asarray(list(scores.values()), dtype=float)
        if protocol == "capacity":
            thr = _det.capacity_threshold(vals, capacity)
        else:
            thr = _det.recall_threshold(labels_, probs_, target=0.50)
        r = _det.detection_rate(scores, anchors, defaulters, thr, window=WINDOW_WEEKS)
        dr_per_run.append(r["dr_pct"])
        ltm_per_run.append(r["lt_mean"])
        ltmed_per_run.append(r["lt_median"])

    m = compute_metrics(np.concatenate(all_labels), np.concatenate(all_probs))
    return {
        "model":    DISPLAY_NAMES.get(model_name, model_name),
        "auroc":    round(m.get("auc", float("nan")), 3),
        "ap_4q":    round(float(np.mean(ap_per_run)), 4),
        "ap_sigma": round(float(np.std(ap_per_run)), 3) if len(ap_per_run) > 1 else float("nan"),
        "dr_pct":   round(float(np.mean(dr_per_run)), 1),
        "dr_seed_sd": round(float(np.std(dr_per_run)), 1) if len(dr_per_run) > 1 else float("nan"),
        "lt_mean":  round(float(np.mean(ltm_per_run)), 1),
        "lt_median": round(float(np.mean(ltmed_per_run)), 1),
    }


def compute_capacity_curve(predictions_runs_by_model, anchors, defaulters):
    """Detection and lead time across review budgets, not at a single point.

    Reports one row per model per capacity. The paper groups these into sequential
    and tree families; the raw rows are left here so the grouping stays checkable.
    """
    rows = []
    for model_name, runs in sorted(predictions_runs_by_model.items()):
        for cap in CAPACITY_CURVE:
            dr, ltm = [], []
            for fd, _ in runs:
                scores = build_scores(fd)
                vals = np.asarray(list(scores.values()), dtype=float)
                thr = _det.capacity_threshold(vals, cap)
                r = _det.detection_rate(scores, anchors, defaulters, thr,
                                        window=WINDOW_WEEKS)
                dr.append(r["dr_pct"])
                ltm.append(r["lt_mean"])
            rows.append({
                "model":      DISPLAY_NAMES.get(model_name, model_name),
                "capacity_pct": round(100 * cap, 2),
                "dr_pct":     round(float(np.mean(dr)), 1),
                "lt_mean":    round(float(np.mean(ltm)), 1),
                "n_seeds":    len(runs),
            })
    return rows


def report_metric_ceiling(predictions_runs_by_model, anchors, defaulters):
    """How many defaulters ANY rule could detect under this anchor and window.

    Printed next to the detection rates because DR is capped by the persistence rule,
    not by 100%. Under label-onset anchoring this is 44 of 88; under the default-date
    anchor it is materially higher, and that difference is larger than any
    architectural effect in the paper.
    """
    fd, _ = next(iter(predictions_runs_by_model.values()))[0]
    reach = _det.reachable_defaulters(build_scores(fd), anchors, defaulters,
                                      window=WINDOW_WEEKS)
    return len(reach), len(defaulters)



def compute_regime_table(predictions_runs_by_model, all_graphs, test_snap_idx):
    """Ranking quality by period, reported as lift over each period's own base rate.

    Lift rather than raw AP, because the base rate differs between periods and a raw
    comparison reports the window rather than the model. Detection is deliberately not
    split by period here: the per-period cohorts are far too small for the design to
    resolve differences between models (see the paper's limitations).
    """
    periods = [("COVID 2020-21", PERIOD_COVID),
               ("Post-COVID 2022-24", PERIOD_POSTCOVID)]
    rows = []
    for model_name, runs in sorted(predictions_runs_by_model.items()):
        for period_name, (p_start, p_end) in periods:
            aps, bases = [], []
            for _fd, results in runs:
                pr, lb = [], []
                for probs_np, labels_np, _phi, snap_t, _cids in results:
                    if snap_t is None:
                        continue
                    ts = pd.Timestamp(snap_t)
                    if not (p_start <= ts <= p_end):
                        continue
                    pr.append(probs_np)
                    lb.append(labels_np)
                if not pr:
                    continue
                pr = np.concatenate(pr)
                lb = np.concatenate(lb)
                if lb.sum() == 0:
                    continue
                aps.append(compute_metrics(lb, pr).get("ap", float("nan")))
                bases.append(float(lb.mean()))
            if not aps:
                continue
            ap = float(np.mean(aps))
            base = float(np.mean(bases))
            rows.append({"model": DISPLAY_NAMES.get(model_name, model_name),
                         "period": period_name,
                         "base_rate_pct": round(100 * base, 2),
                         "ap": round(ap, 4),
                         "lift": round(ap / base, 1) if base > 0 else float("nan")})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------
# Statistical tests
# -----------------------------------------------------------------------

def bootstrap_ap_ci(labels, probs, n_boot=1000, ci=0.95, seed=42):
    """Bootstrap confidence interval for AP."""
    rng = np.random.default_rng(seed)
    aps = []
    n = len(labels)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if labels[idx].sum() == 0:
            continue
        m = compute_metrics(labels[idx], probs[idx])
        aps.append(m.get("ap", float("nan")))
    aps = np.array([a for a in aps if not np.isnan(a)])
    lo = float(np.percentile(aps, (1 - ci) / 2 * 100))
    hi = float(np.percentile(aps, (1 + ci) / 2 * 100))
    return lo, hi


def mcnemar_test(detected_a: set, detected_b: set, all_defaulters: set):
    """SUPERSEDED. Firm-level paired test, retained for reference only.

    The reporting path compares models on seed-level means (Welch, Holm-corrected),
    because the unit of replication is the training run rather than the firm-week.
    """
    """McNemar test on per-firm detection (paired binary outcomes)."""
    n01, n10 = 0, 0
    for fid in all_defaulters:
        in_a = fid in detected_a
        in_b = fid in detected_b
        if in_a and not in_b:
            n10 += 1
        elif in_b and not in_a:
            n01 += 1
    table = np.array([[0, n01], [n10, 0]])
    if n01 + n10 < 2:
        return float("nan"), float("nan")
    result = mcnemar(table, exact=False, correction=True)
    return result.statistic, result.pvalue


# -----------------------------------------------------------------------
# Figure generation
# -----------------------------------------------------------------------

def plot_lt_survival(predictions_runs_by_model, anchors, output_path):
    """Figure 2: LT survival curve (fraction detectable >= w weeks)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib not available -- skipping lt_survival.pdf")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {"lstm": "#4878CF", "stgnn": "#6ACC65",
              "saf": "#E8590C", "grs": "#D65F5F"}
    weeks = np.arange(0, 27)

    for model_name in ("lstm", "stgnn", "saf", "grs"):
        if model_name not in predictions_runs_by_model:
            continue
        fd, _ = predictions_runs_by_model[model_name][0]
        scores = build_scores(fd)
        vals = np.asarray(list(scores.values()), dtype=float)
        thr = _det.capacity_threshold(vals, CAPACITY_DEFAULT)
        confirmed = {}
        for fid in fd:
            if fid in anchors:
                lt = _det.first_detection_lt(dict(fd[fid]['probs']),
                                             anchors[fid], thr,
                                             window=WINDOW_WEEKS)
                if lt > 0:
                    confirmed[fid] = lt

        defaulters = {fid for fid, d in fd.items() if d['label'] == 1}
        lead_times_all = []
        for fid in defaulters:
            if fid in confirmed and fd[fid]['event_snap'] is not None:
                lt = max(0, fd[fid]['event_snap'] - confirmed[fid])
                lead_times_all.append(lt)
            else:
                lead_times_all.append(-1)   # not detected

        lt_arr = np.array(lead_times_all)
        n_all  = len(defaulters)
        fracs  = [np.sum(lt_arr >= w) / n_all for w in weeks]
        ax.plot(weeks, fracs, label=DISPLAY_NAMES.get(model_name, model_name),
                color=colors.get(model_name, "gray"),
                linestyle="--" if model_name == "grs" else "-")

    ax.set_xlabel("Lead time (weeks)")
    ax.set_ylabel("Fraction of defaulters detectable")
    ax.set_xlim(0, 26)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Reproduce all paper tables and figures")
    parser.add_argument("--checkpoints", default="outputs/models",
                        help="Directory containing .pt checkpoint files")
    parser.add_argument("--data",   default="data/processed",
                        help="Root directory containing processed parquet files")
    parser.add_argument("--cache",  default=None,
                        help="Path to graph cache pickle (built on first run, reused after)")
    parser.add_argument("--output", default="outputs",
                        help="Output root directory")
    parser.add_argument("--freq",   choices=["Q", "W"], default="W")
    parser.add_argument("--capacity", type=float, default=CAPACITY_DEFAULT,
                        help="Weekly review budget as a fraction of live firms "
                             "(default 0.015 = 1.5%%)")
    parser.add_argument("--protocol", choices=["capacity", "recall"],
                        default="capacity",
                        help="capacity: threshold at the (1-c) quantile of pooled "
                             "scores (paper). recall: fixed 50%% recall target, which "
                             "leaves the alert rate free and is the superseded rule.")
    parser.add_argument("--anchor", choices=["default", "label_onset"],
                        default="default",
                        help="default: anchor detection on the firm's default date "
                             "(paper). label_onset: first snapshot with y=1, the "
                             "superseded anchor that caps detection at 50%%.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.join(args.output, "results"),  exist_ok=True)
    os.makedirs(os.path.join(args.output, "figures"),  exist_ok=True)

    # ----------------------------------------------------------------
    # Load graphs (with optional pickle cache for faster re-runs)
    # ----------------------------------------------------------------
    if args.cache and os.path.exists(args.cache):
        print(f"Loading graphs from cache {args.cache} ...")
        with open(args.cache, "rb") as f:
            _c = pickle.load(f)
        train_graphs, val_graphs, test_graphs = _c["train"], _c["val"], _c["test"]
    else:
        print(f"Loading graphs from {args.data} ...")
        train_graphs, val_graphs, test_graphs = build_temporal_graphs(
            base_dir=args.data, freq=args.freq,
        )
        if args.cache:
            os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
            with open(args.cache, "wb") as f:
                pickle.dump({"train": train_graphs, "val": val_graphs,
                             "test": test_graphs}, f)
            print(f"  Graph cache saved: {args.cache}")
    all_graphs = train_graphs + val_graphs + test_graphs
    test_start = len(train_graphs) + len(val_graphs)
    t_lookback = T_W if args.freq == "W" else T_Q

    print(f"Building sequence cache for {len(all_graphs)} snapshots ...")
    seq_cache = [
        build_sequence_tensor(all_graphs, i, T=t_lookback,
                              device=torch.device("cpu"))
        for i in range(len(all_graphs))
    ]
    print("  Done.")

    # ----------------------------------------------------------------
    # Load all checkpoints (.pt for DL models, .pkl for tabular/survival)
    # ----------------------------------------------------------------
    ckpt_dir = args.checkpoints
    all_files = sorted(os.listdir(ckpt_dir))
    pt_files  = [f for f in all_files if f.endswith(".pt")]
    pkl_files = [f for f in all_files if f.endswith(".pkl")]
    print(f"\nFound {len(pt_files)} DL checkpoints + {len(pkl_files)} tabular checkpoints")

    # Group by model name: {model_name: [(firm_data, results), ...]}
    predictions_runs_by_model: Dict[str, list] = defaultdict(list)

    for fname in pt_files:
        path = os.path.join(ckpt_dir, fname)
        print(f"  Loading {fname} ...")
        ckpt  = torch.load(path, map_location="cpu", weights_only=False)
        model = rebuild_model(ckpt).to(device)
        model.eval()

        fd, results = run_inference(
            model, all_graphs, seq_cache, test_start, device, args.freq
        )
        model_name = ckpt["model_name"]
        predictions_runs_by_model[model_name].append((fd, results))
        print(f"    {model_name}  seed={ckpt['seed']}"
              f"  val_ap={ckpt.get('val_ap', float('nan')):.4f}")

    for fname in pkl_files:
        path = os.path.join(ckpt_dir, fname)
        print(f"  Loading {fname} ...")
        with open(path, "rb") as f:
            tabular_model = pickle.load(f)
        # Derive model name from filename (e.g. xgb_seed42.pkl -> xgb)
        model_name = fname.split("_seed")[0].replace(".pkl", "")
        fd, results = run_tabular_inference(
            tabular_model, model_name, all_graphs, test_start
        )
        predictions_runs_by_model[model_name].append((fd, results))
        print(f"    {model_name}  (tabular)")

    # ----------------------------------------------------------------
    # Event anchors: the choice that decides what a detection rate means
    # ----------------------------------------------------------------
    test_snap_idx = list(range(test_start, len(all_graphs)))
    anchors, defaulters = build_anchors(all_graphs, test_snap_idx,
                                        args.data, args.anchor)
    print(f"\nAnchor: {args.anchor}   defaulters in test window: {len(defaulters)}")

    n_reach, n_def = report_metric_ceiling(predictions_runs_by_model,
                                           anchors, defaulters)
    print(f"Metric ceiling: {n_reach} of {n_def} defaulters are reachable by ANY rule "
          f"under this anchor and a {WINDOW_WEEKS}-week window.")
    if args.anchor == "label_onset":
        print("  NOTE: label-onset anchoring caps achievable detection well below 100%. "
              "This reproduces the superseded protocol; use --anchor default otherwise.")

    # ----------------------------------------------------------------
    # Detection at a stated review capacity
    # ----------------------------------------------------------------
    print(f"\n--- Detection at {100*args.capacity:.2f}% weekly capacity "
          f"(protocol={args.protocol}) ---")
    t1_rows = []
    for model_name in ("lstm", "stgnn", "saf", "grs", "xgb", "lgbm"):
        if model_name not in predictions_runs_by_model:
            continue
        row = compute_table1_row(model_name, predictions_runs_by_model[model_name],
                                 anchors, defaulters,
                                 capacity=args.capacity, protocol=args.protocol)
        t1_rows.append(row)
        n_seeds = len(predictions_runs_by_model[model_name])
        print(f"  {row['model']:<28}  AP={row['ap_4q']:.4f} (sd {row['ap_sigma']})"
              f"  DR={row['dr_pct']}% (sd {row['dr_seed_sd']})"
              f"  LT mean={row['lt_mean']}w  median={row['lt_median']}w"
              f"  ({n_seeds} seeds)")

    t1_path = os.path.join(args.output, "results", "detection_at_capacity.csv")
    pd.DataFrame(t1_rows).to_csv(t1_path, index=False)
    print(f"  Saved: {t1_path}")
    print("  NOTE: the spread across variants is smaller than their own seed standard "
          "deviations. No architecture claim is supported by this table.")

    # ----------------------------------------------------------------
    # The same result as a curve across review budgets
    # ----------------------------------------------------------------
    print("\n--- Capacity curve ---")
    curve = pd.DataFrame(compute_capacity_curve(predictions_runs_by_model,
                                                anchors, defaulters))
    print(curve.to_string(index=False))
    curve_path = os.path.join(args.output, "results", "capacity_curve.csv")
    curve.to_csv(curve_path, index=False)
    print(f"  Saved: {curve_path}")

    # ----------------------------------------------------------------
    # Ranking quality by period, as lift over each period's base rate
    # ----------------------------------------------------------------
    print("\n--- Ranking quality by period ---")
    reg = compute_regime_table(predictions_runs_by_model, all_graphs, test_snap_idx)
    print(reg.to_string(index=False))
    reg_path = os.path.join(args.output, "results", "regime_lift.csv")
    reg.to_csv(reg_path, index=False)
    print(f"  Saved: {reg_path}")

    # ----------------------------------------------------------------
    # Figures
    # ----------------------------------------------------------------
    print("\n--- Figures ---")
    plot_lt_survival(
        predictions_runs_by_model, anchors,
        os.path.join(args.output, "figures", "lt_survival.pdf"),
    )

    # ----------------------------------------------------------------
    # Statistical tests
    # ----------------------------------------------------------------
    print("\n--- Statistical Tests ---")
    stats_lines = []

    # Bootstrap CI for AP@4q (GRS vs LSTM)
    for model_name in ("lstm", "grs"):
        if model_name not in predictions_runs_by_model:
            continue
        fd, results = predictions_runs_by_model[model_name][0]
        probs_  = np.concatenate([r[0] for r in results])
        labels_ = np.concatenate([r[1] for r in results])
        lo, hi  = bootstrap_ap_ci(labels_, probs_, n_boot=1000)
        line    = f"Bootstrap CI AP@4q  {DISPLAY_NAMES[model_name]}: [{lo:.4f}, {hi:.4f}]"
        print(f"  {line}")
        stats_lines.append(line)

    # McNemar: GRS vs LSTM, GRS vs SAF, SAF vs LSTM
    # ------------------------------------------------------------------
    # Seed-level comparisons, corrected within each family by Holm-Bonferroni.
    # The unit of replication is the TRAINING RUN, not the firm-week: a firm-level
    # paired test (McNemar) treats firms as independent replicates and answers a
    # narrower question than "does this architecture help". With three seeds these
    # are tests on two to four degrees of freedom, so a retained null bounds very
    # little; the paper reports the smallest detectable effect alongside.
    # ------------------------------------------------------------------
    def _seed_level(model_name, capacity):
        aps, drs = [], []
        for fd, results in predictions_runs_by_model[model_name]:
            probs_ = np.concatenate([r[0] for r in results])
            labels_ = np.concatenate([r[1] for r in results])
            aps.append(compute_metrics(labels_, probs_).get("ap", float("nan")))
            scores = build_scores(fd)
            vals = np.asarray(list(scores.values()), dtype=float)
            thr = _det.capacity_threshold(vals, capacity)
            drs.append(_det.detection_rate(scores, anchors, defaulters, thr,
                                           window=WINDOW_WEEKS)["dr_pct"])
        return np.array(aps), np.array(drs)

    graph_family = [m for m in ("stgnn", "saf", "grs")
                    if m in predictions_runs_by_model]
    if "lstm" in predictions_runs_by_model and graph_family:
        base_ap, base_dr = _seed_level("lstm", args.capacity)
        comparisons = []
        for m in graph_family:
            ap, dr = _seed_level(m, args.capacity)
            for label, a, b in (("AP@4q", ap, base_ap),
                                (f"DR@{100*args.capacity:.1f}%", dr, base_dr)):
                if len(a) < 2 or len(b) < 2:
                    continue
                t, pv = ttest_ind(a, b, equal_var=False)
                comparisons.append({"family": label,
                                    "comparison": f"{m} vs lstm",
                                    "effect": float(np.mean(a) - np.mean(b)),
                                    "p": float(pv)})
        for fam in sorted({c["family"] for c in comparisons}):
            fam_rows = sorted((c for c in comparisons if c["family"] == fam),
                              key=lambda c: c["p"])
            m_tests = len(fam_rows)
            rejected = True
            for i, c in enumerate(fam_rows):
                holm = 0.05 / (m_tests - i)
                outcome = "rejected" if (rejected and c["p"] <= holm) else "retained"
                if outcome == "retained":
                    rejected = False
                line = (f"{fam:<12} {c['comparison']:<14} effect={c['effect']:+.4f}"
                        f"  p={c['p']:.4f}  Holm={holm:.4f}  -> {outcome.upper()}")
                print(f"  {line}")
                stats_lines.append(line)
        stats_lines.append(
            "Unit of replication is the training run. Seed counts here are small, so a "
            "retained null bounds little on its own; see the paper's minimum detectable "
            "effect discussion."
        )

    stats_path = os.path.join(args.output, "results", "stats.txt")
    with open(stats_path, "w") as f:
        f.write("\n".join(stats_lines) + "\n")
    print(f"\n  Saved: {stats_path}")
    print("\nDone. All outputs written to:", args.output)


if __name__ == "__main__":
    main()
