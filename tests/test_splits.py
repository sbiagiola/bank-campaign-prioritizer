"""Unit test for sequential temporal partitions."""

import pytest
import pandas as pd
import numpy as np
from src.data.split import sequential_temporal_split


def test_sequential_temporal_split_properties():
    # Synthetic dataset of 100 rows
    n = 100
    df = pd.DataFrame({"feat": np.arange(n)})
    y = pd.Series(np.zeros(n))

    splits = sequential_temporal_split(
        df, y,
        train_pct=0.60,
        selection_pct=0.15,
        calibration_pct=0.10,
        test_pct=0.15,
    )

    X_train, y_train = splits["train"]
    X_sel, y_sel = splits["selection"]
    X_cal, y_cal = splits["calibration"]
    X_test, y_test = splits["test"]

    # 1. Total rows match
    total_len = len(X_train) + len(X_sel) + len(X_cal) + len(X_test)
    assert total_len == n

    # 2. Exact partition sizes
    assert len(X_train) == 60
    assert len(X_sel) == 15
    assert len(X_cal) == 10
    assert len(X_test) == 15

    # 3. Non-overlapping and strictly contiguous in order
    assert list(X_train["feat"]) == list(range(0, 60))
    assert list(X_sel["feat"]) == list(range(60, 75))
    assert list(X_cal["feat"]) == list(range(75, 85))
    assert list(X_test["feat"]) == list(range(85, 100))


def test_split_percentage_validation():
    df = pd.DataFrame({"feat": [1, 2, 3]})
    y = pd.Series([0, 1, 0])
    with pytest.raises(ValueError):
        sequential_temporal_split(df, y, train_pct=0.5, selection_pct=0.1, calibration_pct=0.1, test_pct=0.1)
