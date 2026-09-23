"""Detection-rate and lead-time metrics: ONE definition, shared by every notebook.

Why this module exists
----------------------
The event anchor, the persistence rule and the DR/LT computation were implemented
separately in four places: ``notebooks/03_baseline_comparison.ipynb`` [Cell 24],
``notebooks/04_early_warning_evaluation.ipynb`` [Cell 2],
``notebooks/06_explainability.ipynb`` [Cell 2] and ``code_release/evaluate.py``.
All four carried the same defect, and a fifth copy would have carried it too. See
``markdowns/EVENT_ANCHOR_DEFECT.md``.

The anchor defect
-----------------
``company.y`` is ``has_default_in_window(t, horizon, freq='W')``, so ``y == 1`` marks
*every* snapshot within roughly 52 weeks before a default, not the default itself. The
notebooks anchored the event at the FIRST ``y == 1`` snapshot, which is approximately
``default - 52 weeks``. Measured gap between that anchor and the last ``y == 1``
snapshot: median 37 weeks, mean 46.6.

Two consequences:

1. Lead time measured weeks to *label onset*, not to the default, while both papers
   state it is "the weeks from first detection to the default date".
2. 43 of 88 test-window defaulters had their anchor pinned to the first test snapshot
   (2020-01-03) with an empty lookback, making them undetectable by any model and
   capping DR at 50.0%.

Correcting the anchor moves DR from about 14% to about 76% and the ceiling from 50.0%
to 95.5%.

Modes
-----
``mode='label_onset'`` reproduces every published number exactly. Keep it: a disclosed
correction is only verifiable if the original can still be regenerated.
``mode='default'`` is correct and is what the papers claim to measure.

Both are provided so old-versus-new is a parameter change rather than a
re-implementation, which makes the comparison trustworthy by construction.
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "build_event_anchors", "first_detection_lt", "detection_rate",
    "capacity_threshold", "reachable_defaulters", "recall_threshold",
]

DEFAULT_WINDOW = 26


# ---------------------------------------------------------------------------
# Event anchors
# ---------------------------------------------------------------------------

def build_event_anchors(
    all_graphs: Sequence,
    test_snap_idx: Sequence[int],
    mode: str = "default",
    gvkey_to_defdate: Optional[Dict[str, "object"]] = None,
    cid_to_gvkey: Optional[Dict[int, str]] = None,
) -> Tuple[Dict[int, int], List[int]]:
    """Return ``(anchors, defaulters)``.

    ``anchors`` maps company id to the snapshot index the detection window ends at.

    mode='label_onset'
        First snapshot in ``test_snap_idx`` where ``y == 1``. Reproduces the published
        numbers. This is label onset, roughly ``default - 52 weeks``.

    mode='default'
        Last snapshot strictly BEFORE the firm's true default date, taken from
        ``gvkey_to_defdate`` (Compustat/S&P). Firms without a precise date fall back to
        the label-onset anchor, and the caller can detect this by comparing modes.
    """
    if mode not in ("label_onset", "default"):
        raise ValueError(f"mode must be 'label_onset' or 'default', got {mode!r}")

    onset: Dict[int, int] = {}
    for si in test_snap_idx:
        g = all_graphs[si]
        cids = g["company"].company_ids.cpu().numpy()
        ys = g["company"].y.cpu().numpy()
        for k in range(len(cids)):
            if int(ys[k]) == 1:
                gvk = int(cids[k])
                if gvk not in onset or si < onset[gvk]:
                    onset[gvk] = si
    defaulters = list(onset.keys())

    if mode == "label_onset":
        return onset, defaulters

    if gvkey_to_defdate is None or cid_to_gvkey is None:
        raise ValueError(
            "mode='default' needs gvkey_to_defdate and cid_to_gvkey; without the true "
            "default dates the anchor cannot be corrected."
        )

    t_of = {i: all_graphs[i]["company"].t for i in range(len(all_graphs))}
    anchors: Dict[int, int] = {}
    for gvk in defaulters:
        dd = gvkey_to_defdate.get(cid_to_gvkey.get(gvk, ""))
        if dd is None:
            anchors[gvk] = onset[gvk]          # fall back; caller should count these
            continue
        prior = [i for i in range(len(all_graphs)) if t_of[i] < dd]
        anchors[gvk] = max(prior) if prior else onset[gvk]
    return anchors, defaulters


def score_range(anchors: Dict[int, int], test_snap_idx: Sequence[int],
                window: int = DEFAULT_WINDOW) -> List[int]:
    """Snapshot indices that must be scored so every anchor has a full lookback.

    With the corrected anchor some windows reach before the test boundary. Those graphs
    exist and the model can score them, and scoring a firm before its event is precisely
    what early warning means. NOTE: snapshots inside the 2017-2019 validation window were
    seen during early stopping and threshold calibration, so their use must be disclosed.
    """
    start = min(min(anchors.values()), min(test_snap_idx)) - window - 1
    return list(range(max(0, start), max(test_snap_idx) + 1))


# ---------------------------------------------------------------------------
# Persistence rule
# ---------------------------------------------------------------------------

def first_detection_lt(firm_scores: Dict[int, float], anchor: int, thr: float,
                       window: int = DEFAULT_WINDOW) -> int:
    """Weeks from first confirmed detection to ``anchor``; 0 if never detected.

    Detection requires TWO CONSECUTIVE snapshots above ``thr`` inside
    ``[anchor - window, anchor)``. This is the paper's "double-lock" trigger.
    """
    lo = anchor - window
    cands = sorted(si for si, p in firm_scores.items() if lo <= si < anchor and p >= thr)
    for i in range(len(cands) - 1):
        if cands[i + 1] == cands[i] + 1:
            return max(0, anchor - cands[i])
    return 0


def _by_firm(scores: Dict[Tuple[int, int], float]) -> Dict[int, Dict[int, float]]:
    out: Dict[int, Dict[int, float]] = {}
    for (gvk, si), p in scores.items():
        out.setdefault(gvk, {})[si] = p
    return out


def detection_rate(scores: Dict[Tuple[int, int], float], anchors: Dict[int, int],
                   defaulters: Iterable[int], thr: float,
                   window: int = DEFAULT_WINDOW) -> Dict[str, float]:
    """DR and lead time.

    Returns ``dr_pct`` (over all defaulters), ``dr_reachable_pct`` (over those any rule
    could detect), ``lt_mean`` / ``lt_median`` over detected firms, and the counts.

    Always report ``dr_pct`` next to ``n_reachable``: DR is capped by the persistence
    rule, not by 100%.
    """
    fs = _by_firm(scores)
    defaulters = list(defaulters)
    lts = [first_detection_lt(fs.get(g, {}), anchors[g], thr, window) for g in defaulters]
    det = [v for v in lts if v > 0]
    n_reach = len(reachable_defaulters(scores, anchors, defaulters, window))
    return {
        "dr_pct": 100.0 * len(det) / max(len(defaulters), 1),
        "dr_reachable_pct": 100.0 * len(det) / max(n_reach, 1),
        "lt_mean": float(np.mean(det)) if det else 0.0,
        "lt_median": float(np.median(det)) if det else 0.0,
        "n_detected": len(det),
        "n_defaulters": len(defaulters),
        "n_reachable": n_reach,
    }


def reachable_defaulters(scores: Dict[Tuple[int, int], float], anchors: Dict[int, int],
                         defaulters: Iterable[int],
                         window: int = DEFAULT_WINDOW) -> List[int]:
    """Defaulters a constant "flag everything" score would detect.

    This is the metric's ceiling. Under the old anchor it was 44 of 88 (50.0%); under
    the corrected anchor with an extended scoring range it is 84 of 88 (95.5%).
    """
    fs = _by_firm(scores)
    out = []
    for g in defaulters:
        snaps = {si: 1.0 for si in fs.get(g, {})}
        if first_detection_lt(snaps, anchors[g], 0.5, window) > 0:
            out.append(g)
    return out


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

def capacity_threshold(values: np.ndarray, capacity: float) -> float:
    """Threshold alerting on exactly ``capacity`` of observations (e.g. 0.015 = 1.5%).

    Preferred over a recall target: it fixes the REVIEW BUDGET, which is the constraint a
    credit function actually faces, and it is well defined whatever shape the score
    distribution has. A recall target fixes recall and lets the alert rate float, so DR
    comparisons across models silently become comparisons of how indiscriminately each
    one alerts.
    """
    return float(np.quantile(np.asarray(values), 1.0 - capacity))


def recall_threshold(labels: np.ndarray, probs: np.ndarray,
                     target: float = 0.50) -> float:
    """First threshold reaching ``target`` recall. Reproduces the published protocol.

    Retained for reproducibility only; prefer ``capacity_threshold`` for new work.
    """
    from sklearn.metrics import precision_recall_curve
    _, rc, th = precision_recall_curve(labels, probs)
    idx = np.where(rc[:-1] >= target)[0]
    return float(th[idx[-1]]) if len(idx) else float(th[0])
