"""audit_checkpoint.py -- is the message-passing branch of a trained model alive?

Reads a checkpoint and reports the branch-health statistics behind the convergence
audit in the paper. It needs the checkpoint and nothing else: no data, no GPU, no
retraining. Every quantity is read from saved parameters.

    python audit_checkpoint.py outputs/models/stgnn_grs_seed42_w.pt

Why this exists. Under coupled L2 decay through an adaptive optimiser, one branch of a
gated architecture can be driven to numerical zero while the loss, AUROC and average
precision all stay ordinary. A performance table cannot distinguish a working graph
component from an absent one, because the other branch absorbs the difference. The
check below costs one file read and would have caught it.

What "dead" means here, and the two forms it takes:

  1. Decay to denormal values. The convolution weights underflow toward 1e-38 and the
     branch contributes nothing.
  2. Decay without underflow. The weights sit at a small fraction of their
     initialisation scale with the LayerNorm scale intact. They never underflow, so a
     check that looks only for denormals calls them healthy. Four of the fifteen
     checkpoints reported in the paper fail this way.

A third signature is independent of both: the LayerNorm scale immediately downstream of
the branch. It initialises to exactly 1.0, and if it trains to 0.0000 it annihilates the
block output regardless of what the convolution produces.

The path ablation quoted in the paper, zeroing the convolutions and re-scoring, needs
the evaluation data and therefore runs through `evaluate.py` rather than here.
"""

import argparse
import math
import sys
from collections import defaultdict

import torch

DENORMAL = 1e-30          # 0 < |w| < DENORMAL counts as underflowed
ALIVE_DENORMAL_PCT = 5.0  # alive: under this share denormal ...
ALIVE_STD_INIT = 0.10     # ... and still this fraction of initialisation scale

GROUPS = (
    ("convs",          ("gnn.convs", ".conv.")),
    ("norms",          ("gnn.norms", "layer_norm", "norm")),
    ("residual_gates", ("residual_gates",)),
    ("proj (h_init)",  (".proj.",)),
)


def group_of(name: str) -> str:
    for label, needles in GROUPS:
        if any(n in name for n in needles):
            return label
    return "other"


def init_std(t: torch.Tensor) -> float:
    """Expected std of a freshly initialised nn.Linear weight of this shape.

    torch's default is U(-1/sqrt(fan_in), +1/sqrt(fan_in)), whose std is
    1/sqrt(3 * fan_in). Returns nan for tensors this does not apply to.
    """
    if t.ndim != 2:
        return float("nan")
    fan_in = t.shape[1]
    return 1.0 / math.sqrt(3.0 * fan_in) if fan_in else float("nan")


def load_state_dict(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "state_dict", "model"):
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key], ckpt
    if isinstance(ckpt, dict) and all(torch.is_tensor(v) for v in ckpt.values()):
        return ckpt, {}
    raise SystemExit(f"Could not find a state_dict inside {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("checkpoint")
    args = ap.parse_args()

    sd, meta = load_state_dict(args.checkpoint)
    print(f"checkpoint : {args.checkpoint}")
    for k in ("model_name", "seed", "val_ap"):
        if k in meta:
            print(f"{k:<11}: {meta[k]}")
    print()

    stats = defaultdict(lambda: {"n": 0, "n_denorm": 0, "nz_min": float("inf"),
                                 "ratios": []})
    gamma_rows, gate_rows = [], []

    for name, t in sd.items():
        if not torch.is_tensor(t) or not t.is_floating_point():
            continue
        g = group_of(name)
        v = t.detach().float()
        a = v.abs()
        s = stats[g]
        s["n"] += v.numel()
        s["n_denorm"] += int(((a > 0) & (a < DENORMAL)).sum())
        nz = a[a > 0]
        if nz.numel():
            s["nz_min"] = min(s["nz_min"], float(nz.min()))
        isd = init_std(v)
        if not math.isnan(isd) and isd > 0:
            s["ratios"].append(float(v.std()) / isd)

        # LayerNorm scale: initialises to exactly 1.0
        if name.endswith(".weight") and ("norm" in name.lower()) and v.ndim == 1:
            gamma_rows.append((name, float(v.mean()), float(v.std()), float(v.min())))

        # Gate bias: a gate whose weights died rests at sigmoid(bias), a constant
        if "residual_gates" in name and name.endswith(".bias"):
            b = float(v.mean())
            gate_rows.append((name, b, 1.0 / (1.0 + math.exp(-b))))

    print("Parameter groups against initialisation scale")
    print(f"  {'group':<16}{'n':>10}{'% denormal':>12}{'min |w|>0':>13}{'std/init':>10}")
    for g in ("convs", "norms", "residual_gates", "proj (h_init)", "other"):
        if g not in stats:
            continue
        s = stats[g]
        pct = 100.0 * s["n_denorm"] / max(s["n"], 1)
        mn = "-" if not math.isfinite(s["nz_min"]) else f"{s['nz_min']:.2e}"
        ri = "-" if not s["ratios"] else f"{sum(s['ratios']) / len(s['ratios']):.4f}"
        print(f"  {g:<16}{s['n']:>10}{pct:>11.1f}%{mn:>13}{ri:>10}")

    if gamma_rows:
        print("\nLayerNorm scales (initialise to exactly 1.0; 0.0000 annihilates the block)")
        for n, mean, sd_, mn in gamma_rows:
            print(f"  {n:<52} mean={mean:.4f}  std={sd_:.4f}  min={mn:.4f}")

    if gate_rows:
        print("\nGate resting value (a gate with dead weights is the constant sigmoid(bias))")
        for n, b, sig in gate_rows:
            print(f"  {n:<52} bias={b:+.4f}  sigmoid(bias)={sig:.4f}")

    print("\nVerdict")
    if "convs" not in stats or stats["convs"]["n"] == 0:
        print("  No message-passing parameters found. This looks like a non-graph variant.")
        return 0

    c = stats["convs"]
    pct = 100.0 * c["n_denorm"] / max(c["n"], 1)
    ratio = (sum(c["ratios"]) / len(c["ratios"])) if c["ratios"] else float("nan")
    gamma_min = min((g[3] for g in gamma_rows), default=float("nan"))

    alive = pct < ALIVE_DENORMAL_PCT and (math.isnan(ratio) or ratio > ALIVE_STD_INIT)
    if pct >= ALIVE_DENORMAL_PCT:
        print(f"  DEAD by underflow: {pct:.1f}% of convolution weights are denormal.")
    elif not math.isnan(ratio) and ratio <= ALIVE_STD_INIT:
        print(f"  DEAD without underflow: convolutions sit at {ratio:.4f} of "
              f"initialisation scale and never underflowed.")
        print("  A check looking only for denormals would call this healthy.")
    else:
        print(f"  Convolutions look ALIVE: {pct:.1f}% denormal, "
              f"std/init {ratio:.4f}.")

    if not math.isnan(gamma_min) and abs(gamma_min) < 1e-6:
        print("  A LayerNorm scale has trained to 0.0000: the branch output is "
              "annihilated downstream regardless of the convolutions above.")
        alive = False

    if not alive:
        print("\n  Any comparison between this checkpoint and a no-graph ablation measures")
        print("  the optimiser, not the graph. `training/optim.py` provides the decoupled")
        print("  decay that revives the branch; the paper reports that the branch still")
        print("  does not improve average precision once alive.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
