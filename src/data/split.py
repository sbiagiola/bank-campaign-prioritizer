"""Sequential temporal splitting module for out-of-time evaluation."""

from typing import Tuple, Dict, Any
import pandas as pd


def sequential_temporal_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_pct: float = 0.60,
    selection_pct: float = 0.15,
    calibration_pct: float = 0.10,
    test_pct: float = 0.15,
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """
    Partitions dataset strictly in chronological order without shuffling:
      - Train: first train_pct (e.g. 60%)
      - Selection / Validation: next selection_pct (e.g. 15%)
      - Calibration: next calibration_pct (e.g. 10%)
      - Final Evaluation / Test: remaining test_pct (e.g. 15%)

    Ensures contiguous, non-overlapping partitions covering 100% of data.
    """
    total_pct = train_pct + selection_pct + calibration_pct + test_pct
    if abs(total_pct - 1.0) > 1e-6:
        raise ValueError(f"Split percentages must sum to 1.0, got {total_pct}")

    n = len(X)
    n_train = int(n * train_pct)
    n_sel = int(n * selection_pct)
    n_cal = int(n * calibration_pct)
    # The remainder guarantees no lost rows due to integer rounding
    n_test = n - (n_train + n_sel + n_cal)

    idx_train_end = n_train
    idx_sel_end = n_train + n_sel
    idx_cal_end = idx_sel_end + n_cal

    X_train, y_train = X.iloc[:idx_train_end], y.iloc[:idx_train_end]
    X_sel, y_sel = X.iloc[idx_train_end:idx_sel_end], y.iloc[idx_train_end:idx_sel_end]
    X_cal, y_cal = X.iloc[idx_sel_end:idx_cal_end], y.iloc[idx_sel_end:idx_cal_end]
    X_test, y_test = X.iloc[idx_cal_end:], y.iloc[idx_cal_end:]

    assert len(X_train) + len(X_sel) + len(X_cal) + len(X_test) == n, "Data loss during split!"

    return {
        "train": (X_train, y_train),
        "selection": (X_sel, y_sel),
        "calibration": (X_cal, y_cal),
        "test": (X_test, y_test),
    }


def get_split_summary(splits: Dict[str, Tuple[pd.DataFrame, pd.Series]]) -> Dict[str, Any]:
    """Generates descriptive metadata for the splits."""
    summary = {}
    total_n = sum(len(X) for X, _ in splits.values())
    for name, (X_part, y_part) in splits.items():
        summary[name] = {
            "n_samples": len(X_part),
            "pct_of_total": round(len(X_part) / total_n * 100, 2),
            "positive_count": int(y_part.sum()),
            "positive_rate": round(float(y_part.mean()), 4),
        }
    return summary
