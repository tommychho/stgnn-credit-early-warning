"""component_audit.py -- which components of a trained model contribute to a score?

Reproduces the paper's component audit and path ablation. For each component, the model is
loaded, that component is REMOVED, and the test window is re-scored. The comparison is paired
inside a single checkpoint: same weights, same data, only the component changes, so the
comparator for these deltas is their own spread rather than the run-to-run spread across
independently trained models.

    python component_audit.py --checkpoints outputs/models --data ./data/processed/raw/

WHY THIS EXISTS, AND WHY IT IS NOT audit_checkpoint.py. Parameter norms diagnose the optimiser,
not the contribution. A checkpoint whose convolutions are 0.0% denormal still moves average
precision by +0.0404 when those convolutions are zeroed, and a block can sit far below its
initialisation scale while still carrying signal. `audit_checkpoint.py` reports the parameter
statistics and is useful for understanding training dynamics; only removal and re-scoring
settles what a component contributes. This script does the latter.

CHOOSING THE REMOVAL. Each removal must be *identity for that block's algebra*, or it measures
something other than the component's contribution:

  convs, net_sys   zero the weights. Both feed the trunk additively through concatenation, so
                   zeroing removes their content without rescaling anything else.
  film (SAF)       zero the weights. The block is `h + gate * tanh(z_s) * 0.1`, a residual term,
                   so zeroing returns `h` exactly.
  macro_gate       FREEZE at its own time-mean, do not zero. The block is `h * (1 + g)` with
                   `g = sigmoid(Linear(.))`, so zeroing the linear gives `sigmoid(0) = 0.5` and
                   `h * 1.5`: a uniform rescale of every firm, not a removal. The classifier
                   trunk is not scale-invariant, so a zeroed macro gate would move scores for
                   reasons unrelated to macro information. Freezing removes the week-to-week
                   variation while preserving the scale the model was trained at.

The macro gate is also computed from one broadcast row, so it carries no firm-specific signal
and cannot separate firms within a week whatever its value; its across-week variation is the
only thing it could contribute, and that is what freezing removes.

Requires the same data and graph cache as `evaluate.py`. No training, and GPU only for the
forward passes.
"""

import argparse
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr, ttest_1samp, wilcoxon
from sklearn.metrics import average_precision_score

from data.graph_builder import build_temporal_graphs
from evaluate import (CAPACITY_DEFAULT, T_Q, T_W, WINDOW_WEEKS, build_anchors,
                      build_scores, build_sequence_tensor, rebuild_model, run_inference,
                      _threshold)
import eval.detection as _det

# Components, the pattern that identifies their parameters, and how they are removed.
# "zero" is valid only where the block enters additively; see the module docstring.
COMPONENTS = [
    ("convs",       ("gnn.convs", "convs."),      "zero"),
    ("net_sys",     ("net_sys.",),                "zero"),
    ("film (SAF)",  ("film.",),                   "zero"),
    ("macro_gate",  ("macro_gate.",),             "freeze"),
]
# The 2x2 of Table IV: the two graph-derived paths, separately and together.
PATHS = [("convs", ("convs.",)), ("net_sys", ("net_sys.",)),
         ("both", ("convs.", "net_sys."))]


class _FrozenGate(nn.Module):
    """Stands in for macro_gate, returning a fixed gate vector for every input."""

    def __init__(self, g):
        super().__init__()
        self.register_buffer("g", g)

    def forward(self, x):
        return self.g.to(x.device).expand(x.shape[0], -1)


def _zero(model, patterns):
    n = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if any(pat in name for pat in patterns):
                p.zero_()
                n += p.numel()
    return n


def _inner(model):
    """NoGraphWrapper keeps the real model under .model; everything else is itself."""
    return getattr(model, "model", model)


@torch.no_grad()
def _gate_time_mean(model, seq_cache, snap_idx, device):
    """Per-dimension mean of gate(t) over the evaluation window."""
    m = _inner(model)
    out = []
    for si in snap_idx:
        seq = seq_cache[si].to(device)
        macro = seq[0:1, :, m.firm_dim:]
        out.append(m.macro_gate(m.macro_encoder(macro)).squeeze(0).cpu())
    return torch.stack(out).mean(0)


def _score(model, all_graphs, seq_cache, test_start, device, freq, snap_idx):
    fd, results = run_inference(model, all_graphs, seq_cache, test_start, device,
                                freq=freq, snap_idx=snap_idx)
    probs = np.concatenate([r[0] for r in results]) if results else np.array([])
    labels = np.concatenate([r[1] for r in results]) if results else np.array([])
    return build_scores(fd), probs, labels


def _summarise(rows, key):
    """Mean, s.d. and a two-sided p over runs, plus the sign count."""
    v = np.asarray([r[key] for r in rows], dtype=float)
    v = v[~np.isnan(v)]
    if v.size < 2:
        return dict(mean=float(v.mean()) if v.size else float("nan"),
                    sd=float("nan"), p=float("nan"), n_neg=int((v < 0).sum()), n=v.size)
    p_t = float(ttest_1samp(v, 0).pvalue)
    try:
        p_w = float(wilcoxon(v).pvalue)
    except ValueError:
        p_w = float("nan")
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)), p=p_t, p_wilcoxon=p_w,
                n_neg=int((v < 0).sum()), n=int(v.size))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoints", default="outputs/models")
    ap.add_argument("--data", default="./data/processed/raw/")
    ap.add_argument("--cache", default="outputs/graph_cache.pkl")
    ap.add_argument("--output", default="outputs")
    ap.add_argument("--freq", default="W")
    ap.add_argument("--anchor", choices=["default", "label_onset"], default="default")
    ap.add_argument("--capacity", type=float, default=CAPACITY_DEFAULT)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.join(args.output, "results"), exist_ok=True)

    if args.cache and os.path.exists(args.cache):
        with open(args.cache, "rb") as f:
            c = pickle.load(f)
        train_g, val_g, test_g = c["train"], c["val"], c["test"]
    else:
        train_g, val_g, test_g = build_temporal_graphs(base_dir=args.data, freq=args.freq)
    all_graphs = train_g + val_g + test_g
    test_start = len(train_g) + len(val_g)
    t_lookback = T_W if args.freq == "W" else T_Q
    print(f"Building sequence cache for {len(all_graphs)} snapshots ...")
    seq_cache = [build_sequence_tensor(all_graphs, i, T=t_lookback,
                                       device=torch.device("cpu"))
                 for i in range(len(all_graphs))]
    test_snap_idx = list(range(test_start, len(all_graphs)))
    anchors, defaulters = build_anchors(all_graphs, test_snap_idx, args.data, args.anchor)
    print(f"{len(test_snap_idx)} test snapshots, {len(defaulters)} defaulters, "
          f"anchor={args.anchor}\n")

    comp_rows, path_rows = [], []
    ckpts = sorted(f for f in os.listdir(args.checkpoints) if f.endswith(".pt"))
    for fn in ckpts:
        ck = torch.load(os.path.join(args.checkpoints, fn), map_location=device,
                        weights_only=False)
        name = ck.get("model_name", "?")
        if name not in ("stgnn", "saf", "grs"):
            continue   # the graph variants are the ones with components to audit
        base = rebuild_model(ck).to(device)
        sc0, p0, y0 = _score(base, all_graphs, seq_cache, test_start, device,
                             args.freq, test_snap_idx)
        ap0 = float(average_precision_score(y0, p0)) if y0.size else float("nan")
        keys = sorted(sc0)
        a0 = np.array([sc0[k] for k in keys])
        print(f"{fn}: AP {ap0:.4f}")

        for label, pats, how in COMPONENTS:
            m = rebuild_model(ck).to(device)
            if how == "zero":
                n = _zero(m, pats)
                if n == 0:
                    continue        # component absent from this variant
            else:
                g = _gate_time_mean(m, seq_cache, test_snap_idx, device)
                _inner(m).macro_gate = _FrozenGate(g.clone())
                n = int(g.numel())
            sc, p, y = _score(m, all_graphs, seq_cache, test_start, device,
                              args.freq, test_snap_idx)
            apx = float(average_precision_score(y, p)) if y.size else float("nan")
            rho = float(spearmanr(a0, np.array([sc[k] for k in keys])).correlation)
            comp_rows.append(dict(checkpoint=fn, model=name, component=label, how=how,
                                  params=n, ap_full=ap0, ap_removed=apx,
                                  dAP=apx - ap0, rho=rho))
            print(f"    {label:<12} {how:<6} {n:>8,} params  AP {apx:.4f}  "
                  f"dAP {apx - ap0:+.4f}  rho {rho:+.4f}")
            del m

        for label, pats in PATHS:
            m = rebuild_model(ck).to(device)
            if _zero(m, pats) == 0:
                continue
            sc, p, y = _score(m, all_graphs, seq_cache, test_start, device,
                              args.freq, test_snap_idx)
            apx = float(average_precision_score(y, p)) if y.size else float("nan")
            thr = _threshold(sc, np.asarray(list(sc.values())), args.capacity, "per_week")
            dr = _det.detection_rate(sc, anchors, defaulters, thr,
                                     window=WINDOW_WEEKS)["dr_pct"]
            path_rows.append(dict(checkpoint=fn, model=name, zeroed=label,
                                  ap_full=ap0, ap_zeroed=apx, dAP=apx - ap0, dr_pct=dr,
                                  rho=float(spearmanr(
                                      a0, np.array([sc[k] for k in keys])).correlation)))
            del m
        del base

    if not comp_rows:
        raise SystemExit("No graph-variant checkpoints found under --checkpoints.")

    comp = pd.DataFrame(comp_rows)
    path = pd.DataFrame(path_rows)
    comp.to_csv(os.path.join(args.output, "results", "component_audit.csv"), index=False)
    path.to_csv(os.path.join(args.output, "results", "path_ablation.csv"), index=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 92)
    print("COMPONENT AUDIT   dAP when the component is removed, pooled over checkpoints")
    print("=" * 92)
    print(f"  {'component':<14}{'mean dAP':>10}{'s.d.':>9}{'neg/n':>8}{'p':>9}"
          f"{'Wilcoxon':>10}{'mean rho':>10}")
    for label, _, _ in COMPONENTS:
        sub = comp[comp["component"] == label]
        if sub.empty:
            continue
        s = _summarise(sub.to_dict("records"), "dAP")
        print(f"  {label:<14}{s['mean']:>+10.4f}{s['sd']:>9.4f}"
              f"{str(s['n_neg']) + '/' + str(s['n']):>8}{s['p']:>9.4f}"
              f"{s.get('p_wilcoxon', float('nan')):>10.4f}{sub['rho'].mean():>10.4f}")

    print("\n" + "=" * 92)
    print("PATH ABLATION   the two graph-derived paths, separately and together")
    print("=" * 92)
    for label, _ in PATHS:
        sub = path[path["zeroed"] == label]
        if sub.empty:
            continue
        s = _summarise(sub.to_dict("records"), "dAP")
        print(f"  {label:<10} dAP {s['mean']:+.4f} +- {s['sd']:.4f}  "
              f"({s['n_neg']}/{s['n']} negative, p={s['p']:.4f})  "
              f"rho {sub['rho'].mean():+.4f}")

    print("\nHOW TO READ THIS")
    print("  A component whose removal leaves AP unchanged and rho at 1.0000 contributes")
    print("  nothing to any score. A stable SIGN across checkpoints matters more than the")
    print("  magnitude: the paper's one live component is the network pool, negative in")
    print("  every run. Do not read these deltas against the run-to-run spread of")
    print("  independently trained models; the comparison here is paired inside a checkpoint,")
    print("  so the comparator is the spread of the deltas themselves.")
    print("\nwrote outputs/results/component_audit.csv and path_ablation.csv")


if __name__ == "__main__":
    main()
