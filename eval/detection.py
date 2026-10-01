"""Detection rate and lead time: one definition, used everywhere in this release.

Two choices decide what a detection rate means in operational early warning, and both
are parameters here rather than assumptions baked into a call site.

The event anchor
----------------
``company.y`` is ``has_default_in_window(t, horizon, freq='W')``, so ``y == 1`` marks
*every* snapshot in roughly the 52 weeks before a default, not the default itself.
Anchoring the event at the first ``y == 1`` snapshot therefore measures lead time to
label onset, about a year early, and it leaves 44 of the 88 test-window defaulters with
their anchor pinned to the first scoreable snapshot and no lookback behind it. Those
firms are undetectable by any model, so the metric is capped at 50% before any
comparison begins. Anchoring on the default date removes the cap.

``mode='label_onset'`` reproduces the earlier protocol; ``mode='default'`` is what the
paper reports. Both are kept so that comparing them is a parameter change rather than a
re-implementation, which is what makes the comparison trustworthy.

The threshold
-------------
``capacity_threshold`` fixes the review budget: the operating threshold is the
``(1 - capacity)`` quantile of the pooled score distribution, so alerts consume a fixed
share of firm-weeks and the budget is identical across models. ``recall_threshold``
fixes recall instead and lets the alert rate float, which is the confounded rule the
paper reports on: under it a uniform random scorer reaches 47.6% detection on this
panel.

Always read ``dr_pct`` next to ``n_reachable``. Detection is capped by the persistence
rule and the anchor, not by 100%, and ``reachable_defaulters`` reports that ceiling.
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

def first_detection_lt(firm_scores: Dict[int, float], anchor: int, thr,
                       window: int = DEFAULT_WINDOW) -> int:
    """Weeks from first confirmed detection to ``anchor``; 0 if never detected.

    Detection requires TWO CONSECUTIVE snapshots above ``thr`` inside
    ``[anchor - window, anchor)``. This is the paper's "double-lock" trigger.

    ``thr`` is either a float (one threshold for the window, the pooled arm) or a dict mapping
    snapshot index to threshold (the per-week rule). Both are accepted so that switching rules
    is a parameter change rather than a second implementation.
    """
    lo = anchor - window
    at = (lambda si: thr.get(si, float("inf"))) if isinstance(thr, dict) else (lambda si: thr)
    cands = sorted(si for si, p in firm_scores.items() if lo <= si < anchor and p >= at(si))
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
                   defaulters: Iterable[int], thr,
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

def per_week_threshold(scores: Dict[Tuple[int, int], float],
                       capacity: float) -> Dict[int, float]:
    """PRIMARY RULE. A threshold per snapshot, alerting ceil(capacity * n_t) names in week t.

    This is the standard top-k cross-section applied per week. It fixes the number of files
    opened in every week, which is what a review budget actually constrains.

    Why not the pooled quantile (``capacity_threshold`` below). A quantile taken over the whole
    evaluation window fixes the share of firm-weeks alerted ON AVERAGE and lets the weekly load
    float. On the paper's panel that rule averages the intended 14.4 names a week with a standard
    deviation of 12.1, a range of 0 to 59, and a lag-1 autocorrelation of +0.993: the load moves
    in year-long regimes, one 52-week stretch averaging 33 names a week against 4.6 for another.
    That is the same defect as a recall target, one level down, so the pooled form is reported
    only as a sensitivity arm.

    Exact ties at the cut are broken arbitrarily.
    """
    by_snap: Dict[int, list] = {}
    for (_, si), p in scores.items():
        by_snap.setdefault(si, []).append(p)
    out: Dict[int, float] = {}
    for si, vals in by_snap.items():
        arr = np.asarray(vals)
        k = int(np.ceil(capacity * arr.size))
        out[si] = float(np.partition(arr, -k)[-k]) if 0 < k <= arr.size else float("inf")
    return out


def capacity_threshold(values: np.ndarray, capacity: float) -> float:
    """SENSITIVITY ARM. One threshold for the whole window, at the ``(1 - capacity)`` quantile.

    Fixes the share of firm-weeks alerted on average, not the weekly count; see
    ``per_week_threshold`` for why that distinction is material. Still preferable to a recall
    target, which fixes recall and lets the alert rate float, so that DR comparisons across
    models silently become comparisons of how indiscriminately each one alerts.
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
