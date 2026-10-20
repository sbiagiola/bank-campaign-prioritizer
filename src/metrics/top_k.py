"""Ranking, Top-K and calibration metrics engine for campaign prioritization."""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve


def get_top_k_indices(y_prob: np.ndarray, k: float = 0.20) -> np.ndarray:
    """Returns the indices of the top-k highest scoring candidates."""
    n = len(y_prob)
    k_count = max(1, int(round(k * n)))
    # Negative for descending order
    sorted_indices = np.argsort(-y_prob, kind="stable")
    return sorted_indices[:k_count]


def recall_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: float = 0.20) -> float:
    """Computes the fraction of total positive instances captured in the top-k contacts."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    total_positives = np.sum(y_true == 1)
    if total_positives == 0:
        return 0.0
    top_indices = get_top_k_indices(y_prob, k)
    captured_positives = np.sum(y_true[top_indices] == 1)
    return float(captured_positives / total_positives)


def precision_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: float = 0.20) -> float:
    """Computes the precision (conversion rate) within the top-k contacts."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    top_indices = get_top_k_indices(y_prob, k)
    k_count = len(top_indices)
    if k_count == 0:
        return 0.0
    captured_positives = np.sum(y_true[top_indices] == 1)
    return float(captured_positives / k_count)


def lift_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: float = 0.20) -> float:
    """Computes the Lift at top-k: Precision@k / Base Rate."""
    y_true = np.asarray(y_true)
    base_rate = float(np.mean(y_true == 1))
    if base_rate == 0.0:
        return 1.0
    prec = precision_at_k(y_true, y_prob, k)
    return float(prec / base_rate)


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k: float = 0.20,
) -> Dict[str, float]:
    """Computes primary and secondary metrics for a model's probabilistic predictions."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    rec_k = recall_at_k(y_true, y_prob, k)
    prec_k = precision_at_k(y_true, y_prob, k)
    lift_k = lift_at_k(y_true, y_prob, k)
    brier = float(brier_score_loss(y_true, y_prob))
    pr_auc_score = float(average_precision_score(y_true, y_prob))
    roc_auc_val = float(roc_auc_score(y_true, y_prob))
    base_rate = float(np.mean(y_true == 1))

    return {
        f"recall_at_{int(k*100)}pct": round(rec_k, 4),
        f"precision_at_{int(k*100)}pct": round(prec_k, 4),
        f"lift_at_{int(k*100)}pct": round(lift_k, 4),
        "pr_auc": round(pr_auc_score, 4),
        "roc_auc": round(roc_auc_val, 4),
        "brier_score": round(brier, 4),
        "base_rate": round(base_rate, 4),
    }


def compute_cumulative_gain_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_points: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes percentiles from 0 to 1 and corresponding cumulative gains (recall)."""
    percentages = np.linspace(0.01, 1.0, n_points)
    gains = [recall_at_k(y_true, y_prob, p) for p in percentages]
    # prepend 0, 0
    return np.concatenate(([0.0], percentages)), np.concatenate(([0.0], gains))


def compute_decile_lifts(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, Any]:
    """Computes decile lifts (deciles 1 to 10)."""
    deciles = [0.1 * i for i in range(1, 11)]
    lifts = [lift_at_k(y_true, y_prob, d) for d in deciles]
    precisions = [precision_at_k(y_true, y_prob, d) for d in deciles]
    recalls = [recall_at_k(y_true, y_prob, d) for d in deciles]
    return {
        "deciles": deciles,
        "lifts": [round(l, 3) for l in lifts],
        "precisions": [round(p, 4) for p in precisions],
        "recalls": [round(r, 4) for r in recalls],
    }
