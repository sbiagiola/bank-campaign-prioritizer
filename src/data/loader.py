"""Data loader and validation module for the Bank Marketing campaign dataset."""

import os
from typing import Tuple, List, Optional
import pandas as pd


def load_raw_dataset(
    filepath: str,
    delimiter: str = ";",
    target_col: str = "y",
    positive_val: str = "yes",
    exclude_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads raw dataset preserving original chronological sequence.
    Excludes specified leakage columns (e.g. 'duration') and converts target to binary 0/1.

    Args:
        filepath: Path to CSV file.
        delimiter: CSV separator (default ';').
        target_col: Name of the binary target column.
        positive_val: Value in target_col indicating positive event (subscription).
        exclude_cols: List of column names to strictly purge before any processing.

    Returns:
        (X, y): Feature DataFrame and Target Series (0 or 1).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    exclude_cols = exclude_cols or ["duration"]
    df = pd.read_csv(filepath, sep=delimiter)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    # Explicit leakage audit check
    for col in exclude_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Convert target to integer 1/0
    y = (df[target_col] == positive_val).astype(int)
    X = df.drop(columns=[target_col])

    return X, y
