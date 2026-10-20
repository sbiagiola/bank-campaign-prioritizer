"""Logistic Regression benchmark model with preprocessor and calibration."""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from src.features.preprocessor import build_linear_preprocessor


class LogisticRegressionModel:
    """Logistic Regression pipeline with categorical one-hot encoding, scaling, and calibration."""

    def __init__(
        self,
        categorical_cols: List[str],
        numerical_cols: List[str],
        params: Optional[Dict[str, Any]] = None,
    ):
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.params = params or {
            "C": 0.1,
            "max_iter": 1000,
            "solver": "lbfgs",
            "class_weight": "balanced",
            "random_state": 42,
        }
        self.preprocessor = build_linear_preprocessor(categorical_cols, numerical_cols)
        self.base_classifier = LogisticRegression(**self.params)
        self.pipeline = Pipeline(
            [
                ("preprocessor", self.preprocessor),
                ("classifier", self.base_classifier),
            ]
        )
        self.calibrated_model: Optional[CalibratedClassifierCV] = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "LogisticRegressionModel":
        """Fits the base pipeline on training data."""
        self.pipeline.fit(X_train, y_train)
        return self

    def calibrate(
        self,
        X_cal: pd.DataFrame,
        y_cal: pd.Series,
        method: str = "sigmoid",
    ) -> "LogisticRegressionModel":
        """
        Calibrates the pre-fitted pipeline probabilities on the dedicated calibration set.
        Uses FrozenEstimator for scikit-learn 1.6+ compatibility.
        """
        try:
            from sklearn.calibration import FrozenEstimator
            estimator = FrozenEstimator(self.pipeline)
            self.calibrated_model = CalibratedClassifierCV(
                estimator=estimator,
                method=method,
            )
        except ImportError:
            # Fallback for older scikit-learn
            self.calibrated_model = CalibratedClassifierCV(
                estimator=self.pipeline,
                cv="prefit",
                method=method,
            )
        self.calibrated_model.fit(X_cal, y_cal)
        return self

    def predict_proba(self, X: pd.DataFrame, calibrated: bool = False) -> np.ndarray:
        """Returns predicted probability for positive class (1)."""
        if calibrated and self.calibrated_model is not None:
            return self.calibrated_model.predict_proba(X)[:, 1]
        return self.pipeline.predict_proba(X)[:, 1]
