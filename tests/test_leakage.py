"""Unit test to ensure strict pre-contact availability and absence of data leakage."""

import os
import pytest
import pandas as pd
from src.data.loader import load_raw_dataset


def test_duration_leakage_excluded():
    raw_path = "data/raw/bank-additional/bank-additional-full.csv"
    if not os.path.exists(raw_path):
        pytest.skip("Dataset file not found for local test")

    X, y = load_raw_dataset(raw_path, exclude_cols=["duration"])

    # 1. Assert duration is completely purged
    assert "duration" not in X.columns, "Leakage detected: 'duration' is present in feature matrix!"

    # 2. Assert target is excluded from X
    assert "y" not in X.columns, "Target 'y' found in feature matrix X!"

    # 3. Assert y only contains 0 and 1
    assert set(y.unique()).issubset({0, 1}), f"Target y has unexpected values: {y.unique()}"


def test_loader_custom_exclusion():
    # Synthetic test dataframe
    df = pd.DataFrame({
        "age": [30, 40],
        "duration": [120, 240],
        "y": ["yes", "no"]
    })
    temp_path = "/tmp/test_bank_sample.csv"
    df.to_csv(temp_path, sep=";", index=False)

    try:
        X, y = load_raw_dataset(temp_path, delimiter=";", exclude_cols=["duration"])
        assert "duration" not in X.columns
        assert list(y) == [1, 0]
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
