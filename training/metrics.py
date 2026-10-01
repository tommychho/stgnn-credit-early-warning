"""
Evaluation Metrics for Credit Risk Models
"""

import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from typing import Dict, Optional


def compute_tier_recall(
    y_true: np.ndarray,
    y_pred_binary: np.ndarray,
    rating_numeric: np.ndarray,
) -> Dict[str, float]:
    """
    Recall by S&P credit tier (threshold = 0.5 applied upstream).

    Tiers (firms with rating >= 21 = SD/NR already in default are excluded):
        IG  Investment Grade   : 0-9   (AAA to BBB-)
        SG  Speculative Grade  : 10-15 (BB+ to B-)
        HS  Highly Speculative : 16-20 (CCC+ to C/CC)

    Returns NaN for a tier if no positive labels exist in that slice.
    """
    active = rating_numeric < 21
    ig_mask = (rating_numeric <= 9)  & active
    sg_mask = (rating_numeric >= 10) & (rating_numeric <= 15) & active
    hs_mask = (rating_numeric >= 16) & (rating_numeric <= 20) & active

    def _recall(mask: np.ndarray) -> float:
        if mask.sum() == 0 or y_true[mask].sum() == 0:
            return float('nan')
        return float(recall_score(y_true[mask], y_pred_binary[mask], zero_division=0))

    return {
        'rec_ig': _recall(ig_mask),  'n_ig': int(ig_mask.sum()),
        'rec_sg': _recall(sg_mask),  'n_sg': int(sg_mask.sum()),
        'rec_hs': _recall(hs_mask),  'n_hs': int(hs_mask.sum()),
    }


def compute_tier_calibration(
    y_true: np.ndarray,
    y_score: np.ndarray,
    rating_numeric: np.ndarray,
    min_positives: int = 3,
    global_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Calibrate a separate F1-max threshold per credit tier on a val set.

    Returns tier-specific thresholds. Falls back to global_threshold for
    any tier with fewer than min_positives positive labels.

    Tiers:  IG 0-9 | SG 10-15 | HS 16-20
    """
    active = rating_numeric < 21
    tier_masks = {
        'IG': (rating_numeric <= 9)  & active,
        'SG': (rating_numeric >= 10) & (rating_numeric <= 15) & active,
        'HS': (rating_numeric >= 16) & (rating_numeric <= 20) & active,
    }
    thresholds: Dict[str, float] = {}
    for tier, mask in tier_masks.items():
        pos = int(y_true[mask].sum())
        if pos < min_positives:
            thresholds[tier] = global_threshold
            continue
        scores_t = y_score[mask]
        labels_t = y_true[mask]
        candidates = np.unique(scores_t)
        best_f1, best_thr = 0.0, global_threshold
        for thr in candidates:
            preds = (scores_t >= thr).astype(int)
            f1 = float(f1_score(labels_t, preds, zero_division=0))
            if f1 > best_f1:
                best_f1, best_thr = f1, float(thr)
        thresholds[tier] = best_thr
    return thresholds


def compute_tier_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    rating_numeric: np.ndarray,
    thresholds: Dict[str, float],
) -> Dict[str, float]:
    """
    Precision, recall, F1 per tier using tier-specific thresholds.

    thresholds: dict from compute_tier_calibration() with keys 'IG','SG','HS'.
    """
    if rating_numeric is None:
        return {}
    active = rating_numeric < 21
    tier_masks = {
        'IG': (rating_numeric <= 9)  & active,
        'SG': (rating_numeric >= 10) & (rating_numeric <= 15) & active,
        'HS': (rating_numeric >= 16) & (rating_numeric <= 20) & active,
    }
    out: Dict[str, float] = {}
    for tier, mask in tier_masks.items():
        thr = thresholds.get(tier, 0.5)
        n = int(mask.sum())
        n_pos = int(y_true[mask].sum())
        if n == 0:
            continue
        preds = (y_score[mask] >= thr).astype(int)
        out[f'tier_{tier.lower()}_n']         = n
        out[f'tier_{tier.lower()}_pos']       = n_pos
        out[f'tier_{tier.lower()}_thr']       = thr
        out[f'tier_{tier.lower()}_recall']    = float(recall_score(y_true[mask], preds, zero_division=0))
        out[f'tier_{tier.lower()}_precision'] = float(precision_score(y_true[mask], preds, zero_division=0))
        out[f'tier_{tier.lower()}_f1']        = float(f1_score(y_true[mask], preds, zero_division=0))
    return out


GICS_NAMES: Dict[int, str] = {
    0:  "Unknown/NR",
    1:  "Energy",
    2:  "Materials",
    3:  "Industrials",
    4:  "ConsDisc",
    5:  "ConsStaples",
    6:  "Healthcare",
    7:  "Financials",
    8:  "IT",
    9:  "CommSvcs",
    10: "Utilities",
    11: "RealEstate",
}


def compute_sector_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred_binary: np.ndarray,
    sector_idx: np.ndarray,
) -> Dict[str, float]:
    """
    AP, AUC, recall, default rate, and node count per GICS sector.

    sector_idx values (0 = unknown/NR, 1-11 = GICS sectors) match
    graph_builder.GSECTOR_MAP: 1=Energy, 2=Materials, 3=Industrials,
    4=ConsDisc, 5=ConsStaples, 6=Healthcare, 7=Financials, 8=IT,
    9=CommSvcs, 10=Utilities, 11=RealEstate.

    Returns a flat dict keyed by   sec_{name}_{metric}
    so it can be merged directly into the aggregate metrics dict.
    NaN is returned for AP/AUC when a sector has fewer than 2 positive
    labels (not enough to compute rank metrics).
    """
    out: Dict[str, float] = {}
    for idx, name in GICS_NAMES.items():
        mask = sector_idx == idx
        n = int(mask.sum())
        if n == 0:
            continue
        prefix = f"sec_{name}_"
        n_pos = int(y_true[mask].sum())
        out[f"{prefix}n"]   = n
        out[f"{prefix}dr"]  = float(y_true[mask].mean())
        out[f"{prefix}rec"] = (
            float(recall_score(y_true[mask], y_pred_binary[mask], zero_division=0))
            if n_pos > 0 else float('nan')
        )
        try:
            out[f"{prefix}ap"] = (
                float(average_precision_score(y_true[mask], y_score[mask]))
                if n_pos >= 2 else float('nan')
            )
            out[f"{prefix}auc"] = (
                float(roc_auc_score(y_true[mask], y_score[mask]))
                if n_pos >= 2 and (1 - y_true[mask]).sum() >= 2 else float('nan')
            )
        except ValueError:
            out[f"{prefix}ap"]  = float('nan')
            out[f"{prefix}auc"] = float('nan')
    return out


def compute_survival_metrics(
    phi:      np.ndarray,
    y_bin:    np.ndarray,
    y_event:  np.ndarray,
) -> Dict:
    """
    Survival metrics from 52-bin DeepHit output.

    Parameters
    ----------
    phi     : [N, K] raw PMF logits from head_surv
    y_bin   : [N] int, event/censor bin index (0 to K-1)
    y_event : [N] int, 1=event 0=censored

    Returns
    -------
    Dict with surv_cindex, surv_ap, surv_auc, surv_n_event,
    S  ([N, K] survival curves), F_terminal ([N] terminal CIF).
    """
    # Per-bin sigmoid hazard (discrete-time Cox) - avoids S(K-1)=0 trap
    h = 1.0 / (1.0 + np.exp(-phi))                 # [N, K] hazard per bin
    log_s = np.log(np.clip(1.0 - h, 1e-7, 1.0)).cumsum(axis=1)  # [N, K]
    S = np.exp(log_s)                               # [N, K] survival
    F = 1.0 - S                                     # [N, K] CIF
    F_terminal = F[:, -1]                           # [N] 52-week default prob

    n_event = int(y_event.sum())
    out: Dict = {'surv_n_event': n_event, 'S': S, 'F_terminal': F_terminal}

    # Harrell's C-index on survival times
    if n_event >= 2:
        try:
            from lifelines.utils import concordance_index
            ci = float(concordance_index(y_bin, -F_terminal, y_event))
        except ImportError:
            ci = float(roc_auc_score(y_event, F_terminal)) if n_event > 1 else float('nan')
        out['surv_cindex'] = ci
    else:
        out['surv_cindex'] = float('nan')

    # AP and AUC from terminal CIF (backward-comparable with binary baselines)
    try:
        out['surv_ap']  = float(average_precision_score(y_event, F_terminal)) \
                          if n_event >= 2 else float('nan')
        out['surv_auc'] = float(roc_auc_score(y_event, F_terminal)) \
                          if n_event >= 2 else float('nan')
    except ValueError:
        out['surv_ap']  = float('nan')
        out['surv_auc'] = float('nan')

    return out


def compute_lead_time_gain(
    S:         np.ndarray,
    F_terminal: np.ndarray,
    y_bin:     np.ndarray,
    y_event:   np.ndarray,
    threshold: float = 0.5,
) -> Dict:
    """
    Delta-L: mean weeks of early warning for event=1 firms.

    For each defaulting firm, find the first weekly bin k where
    F(t_k) = 1 - S(t_k) >= threshold. Delta-L = mean(K - k) over
    event firms that crossed the threshold at all.

    Parameters
    ----------
    S          : [N, K] survival curves from compute_survival_metrics
    F_terminal : [N] terminal CIF (unused, kept for API symmetry)
    y_bin      : [N] actual event bin
    y_event    : [N] event indicator
    threshold  : CIF threshold defining a high-hazard signal

    Returns
    -------
    Dict with delta_l_weeks and pct_early_signal.
    """
    F = 1.0 - S                                          # [N, K] CIF
    K = S.shape[1]
    ev_mask = y_event.astype(bool)
    crossing_weeks = []
    for fi in F[ev_mask]:
        cross = np.where(fi >= threshold)[0]
        crossing_weeks.append(int(cross[0]) if len(cross) > 0 else K)
    if not crossing_weeks:
        return {'delta_l_weeks': float('nan'), 'pct_early_signal': float('nan')}
    cw = np.array(crossing_weeks)
    return {
        'delta_l_weeks':    float(np.mean(K - cw)),
        'pct_early_signal': float(np.mean(cw < K)),
    }


def compute_cindex(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Harrell's C-index (concordance statistic) for binary default prediction.

    For binary outcomes C-index is mathematically equivalent to ROC-AUC:
    it counts concordant pairs (i, j) where firm i defaulted and the model
    scored i higher than j. This implementation uses sklearn's optimised
    roc_auc_score. When survival/TTE labels are added, upgrade to
    lifelines.utils.concordance_index.

    Parameters
    ----------
    y_true  : np.ndarray  binary labels (0/1)
    y_score : np.ndarray  model risk scores (higher = more risk)

    Returns
    -------
    float in [0, 1]; 0.5 = random, 1.0 = perfect. NaN if single class.
    """
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float('nan')
    return float(roc_auc_score(y_true, y_score))


def compute_metrics(
    y_true: torch.Tensor,
    y_pred_proba: torch.Tensor,
    threshold: float = 0.5,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics for binary classification.

    Parameters
    ----------
    y_true : torch.Tensor
        Ground truth labels (0 or 1)
    y_pred_proba : torch.Tensor
        Predicted probabilities (0 to 1)
    threshold : float, default=0.5
        Decision threshold for converting probabilities to binary predictions
    prefix : str, default=""
        Prefix for metric names (e.g., "train_", "val_")

    Returns
    -------
    Dict[str, float]
        Dictionary of metrics:
        - auc: Area under ROC curve
        - ap: Average precision (PR-AUC)
        - f1: F1 score
        - precision: Precision
        - recall: Recall
        - tn, fp, fn, tp: Confusion matrix values
        - accuracy: Classification accuracy
        - default_rate: Observed default rate

    Example
    -------
    >>> y_true = torch.tensor([0, 1, 0, 1])
    >>> y_pred = torch.tensor([0.1, 0.9, 0.2, 0.8])
    >>> metrics = compute_metrics(y_true, y_pred, prefix="test_")
    >>> print(f"Test AUC: {metrics['test_auc']:.3f}")
    """
    # Accept both torch.Tensor and numpy arrays
    import numpy as _np
    y_true_np = y_true.cpu().numpy() if hasattr(y_true, 'cpu') else _np.asarray(y_true)
    y_pred_proba_np = y_pred_proba.cpu().numpy() if hasattr(y_pred_proba, 'cpu') else _np.asarray(y_pred_proba)

    # Binary predictions
    y_pred_binary = (y_pred_proba_np >= threshold).astype(int)

    metrics = {}

    # AUC-ROC (threshold-independent).
    # Guarded on the class count rather than on an exception. Older sklearn RAISED
    # ValueError when y_true held one class; current versions emit UndefinedMetricWarning
    # and return nan instead, so the old `except ValueError` never fired and every
    # single-class snapshot printed two warnings. The condition itself is expected: at a
    # ~1% positive rate a given week often has no positives at the 1q or 2q horizon, so
    # the metric is undefined for that slice. nan says that; 0.0 would assert a real AUC.
    _n_pos = int((y_true_np == 1).sum())
    _n_neg = int((y_true_np == 0).sum())
    metrics[f"{prefix}auc"] = (float(roc_auc_score(y_true_np, y_pred_proba_np))
                               if _n_pos >= 1 and _n_neg >= 1 else float("nan"))

    # C-index (Harrell's concordance) - equals AUC for binary labels
    metrics[f"{prefix}cindex"] = compute_cindex(y_true_np, y_pred_proba_np)

    # Average Precision (PR-AUC). Same guard as the AUC above: with no positives the
    # precision-recall curve is undefined and current sklearn warns rather than raising.
    metrics[f"{prefix}ap"] = (float(average_precision_score(y_true_np, y_pred_proba_np))
                              if _n_pos >= 1 else float("nan"))

    # F1 Score
    metrics[f"{prefix}f1"] = f1_score(y_true_np, y_pred_binary, zero_division=0)

    # Precision & Recall
    metrics[f"{prefix}precision"] = precision_score(y_true_np, y_pred_binary, zero_division=0)
    metrics[f"{prefix}recall"] = recall_score(y_true_np, y_pred_binary, zero_division=0)

    # Confusion Matrix (safe ravel for single-class edge cases)
    try:
        cm = confusion_matrix(y_true_np, y_pred_binary, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
    except ValueError:
        tn = fp = fn = tp = 0
    metrics[f"{prefix}tn"] = int(tn)
    metrics[f"{prefix}fp"] = int(fp)
    metrics[f"{prefix}fn"] = int(fn)
    metrics[f"{prefix}tp"] = int(tp)

    # Accuracy
    metrics[f"{prefix}accuracy"] = (tp + tn) / (tp + tn + fp + fn)

    # Default rate
    metrics[f"{prefix}default_rate"] = y_true_np.mean()

    return metrics


def print_metrics(
    metrics: Dict[str, float],
    title: str = "Metrics",
    show_confusion: bool = True
) -> None:
    """
    Print formatted metrics.

    Parameters
    ----------
    metrics : Dict[str, float]
        Metrics dictionary from compute_metrics()
    title : str, default="Metrics"
        Title for the metrics section
    show_confusion : bool, default=True
        Whether to show confusion matrix
    """
    print(f"\n{title}")
    print("=" * 60)

    # Extract prefix (if any)
    prefix = ""
    for key in metrics.keys():
        if key.endswith("auc"):
            prefix = key[:-3]
            break

    # Core metrics
    print(f"  AUC:            {metrics.get(f'{prefix}auc', 0):.4f}")
    print(f"  C-index:        {metrics.get(f'{prefix}cindex', float('nan')):.4f}")
    print(f"  AP (PR-AUC):    {metrics.get(f'{prefix}ap', 0):.4f}")
    print(f"  F1 Score:       {metrics.get(f'{prefix}f1', 0):.4f}")
    print(f"  Precision:      {metrics.get(f'{prefix}precision', 0):.4f}")
    print(f"  Recall:         {metrics.get(f'{prefix}recall', 0):.4f}")
    print(f"  Accuracy:       {metrics.get(f'{prefix}accuracy', 0):.4f}")
    print(f"  Default Rate:   {metrics.get(f'{prefix}default_rate', 0):.4f}")

    # Confusion matrix
    if show_confusion:
        tp = metrics.get(f'{prefix}tp', 0)
        fp = metrics.get(f'{prefix}fp', 0)
        fn = metrics.get(f'{prefix}fn', 0)
        tn = metrics.get(f'{prefix}tn', 0)

        print(f"\n  Confusion Matrix:")
        print(f"                  Predicted")
        print(f"                  0       1")
        print(f"    Actual  0   {tn:5d}   {fp:5d}")
        print(f"            1   {fn:5d}   {tp:5d}")

    print("=" * 60)


def compute_all_splits_metrics(
    model: torch.nn.Module,
    data,
    edge_attr_dict: Dict,
    device: torch.device
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for train/val/test splits.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model
    data : HeteroData
        Graph data with masks
    edge_attr_dict : Dict
        Edge attributes dictionary
    device : torch.device
        Device for computation

    Returns
    -------
    Dict[str, Dict[str, float]]
        Nested dictionary with metrics for each split

    Example
    -------
    >>> all_metrics = compute_all_splits_metrics(model, data, edge_attr_dict, device)
    >>> print(f"Test AUC: {all_metrics['test']['test_auc']:.3f}")
    """
    model.eval()

    with torch.no_grad():
        # Forward pass
        logits, _ = model(data.x_dict, data.edge_index_dict, edge_attr_dict)
        probs = torch.sigmoid(logits)

        y = data["borrower"].y
        train_mask = data["borrower"].train_mask
        val_mask = data["borrower"].val_mask
        test_mask = data["borrower"].test_mask

        # Compute metrics for each split
        results = {}

        results["train"] = compute_metrics(
            y[train_mask],
            probs[train_mask],
            prefix="train_"
        )

        results["val"] = compute_metrics(
            y[val_mask],
            probs[val_mask],
            prefix="val_"
        )

        results["test"] = compute_metrics(
            y[test_mask],
            probs[test_mask],
            prefix="test_"
        )

    return results


if __name__ == "__main__":
    # Example usage
    y_true = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred_proba = torch.tensor([0.1, 0.3, 0.7, 0.9, 0.2, 0.8, 0.4, 0.6])

    metrics = compute_metrics(y_true, y_pred_proba, prefix="test_")
    print_metrics(metrics, title="Test Metrics")
