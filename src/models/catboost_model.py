"""CatBoost gradient boosting model with categorical handling and probability calibration."""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from src.features.preprocessor import prepare_catboost_data


class CatBoostModel:
    """CatBoost classifier with native categorical handling and probability calibration."""

    def __init__(
        self,
        categorical_cols: List[str],
        numerical_cols: List[str],
        params: Optional[Dict[str, Any]] = None,
    ):
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.params = params or {
            "iterations": 350,
            "learning_rate": 0.05,
            "depth": 5,
            "auto_class_weights": "Balanced",
            "random_seed": 42,
            "verbose": 0,
        }
        self.model = CatBoostClassifier(
            cat_features=self.categorical_cols,
            **self.params,
        )
        self.calibrated_model: Optional[CalibratedClassifierCV] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "CatBoostModel":
        """Fits CatBoost classifier on train data with optional early stopping on validation."""
        X_train_prep = prepare_catboost_data(X_train, self.categorical_cols, self.numerical_cols)
        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_prep = prepare_catboost_data(X_val, self.categorical_cols, self.numerical_cols)
            eval_set = (X_val_prep, y_val)

        self.model.fit(
            X_train_prep,
            y_train,
            eval_set=eval_set,
            early_stopping_rounds=40 if eval_set is not None else None,
            verbose=False,
        )
        return self

    def calibrate(
        self,
        X_cal: pd.DataFrame,
        y_cal: pd.Series,
        method: str = "sigmoid",
    ) -> "CatBoostModel":
        """
        Calibrates the pre-fitted CatBoost probabilities on the calibration set.
        Uses FrozenEstimator for scikit-learn 1.6+ compatibility.
        """
        X_cal_prep = prepare_catboost_data(X_cal, self.categorical_cols, self.numerical_cols)
        try:
            from sklearn.calibration import FrozenEstimator
            estimator = FrozenEstimator(self.model)
            self.calibrated_model = CalibratedClassifierCV(
                estimator=estimator,
                method=method,
            )
        except ImportError:
            self.calibrated_model = CalibratedClassifierCV(
                estimator=self.model,
                cv="prefit",
                method=method,
            )
        self.calibrated_model.fit(X_cal_prep, y_cal)
        return self

    def predict_proba(self, X: pd.DataFrame, calibrated: bool = False) -> np.ndarray:
        """Returns predicted probability for positive class (1)."""
        X_prep = prepare_catboost_data(X, self.categorical_cols, self.numerical_cols)
        if calibrated and self.calibrated_model is not None:
            return self.calibrated_model.predict_proba(X_prep)[:, 1]
        return self.model.predict_proba(X_prep)[:, 1]
