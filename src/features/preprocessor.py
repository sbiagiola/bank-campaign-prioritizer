"""Feature preprocessing utilities for tabular banking data."""

from typing import List, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def build_linear_preprocessor(
    categorical_cols: List[str],
    numerical_cols: List[str],
) -> ColumnTransformer:
    """
    Builds a scikit-learn ColumnTransformer suitable for linear models:
    - Numerical features: Standard scaled (mean 0, variance 1)
    - Categorical features: One-Hot Encoded with unknown handling
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_cols,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )
    return preprocessor


def prepare_catboost_data(
    df: pd.DataFrame,
    categorical_cols: List[str],
    numerical_cols: List[str],
) -> pd.DataFrame:
    """
    Prepares DataFrame for CatBoost:
    Ensures categorical columns are string types without missing values.
    """
    df_out = df[categorical_cols + numerical_cols].copy()
    for col in categorical_cols:
        df_out[col] = df_out[col].fillna("unknown").astype(str)
    for col in numerical_cols:
        df_out[col] = df_out[col].astype(float)
    return df_out
