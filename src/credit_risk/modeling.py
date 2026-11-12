"""Leakage-aware preprocessing, baseline/model training, and calibration evaluation."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    log_loss, precision_score, recall_score, f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_cal: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_cal: pd.Series
    y_test: pd.Series


def stratified_split(
    X: pd.DataFrame, y: pd.Series, random_state: int = 42
) -> SplitData:
    """Create train/calibration/test splits; calibration data never fits base models."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=random_state, stratify=y
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )
    return SplitData(X_train, X_cal, X_test, y_train, y_cal, y_test)


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ], remainder="drop")


def make_models(X: pd.DataFrame, random_state: int = 42) -> dict[str, Pipeline]:
    """Compare linear, bagged-tree and boosted-tree models with fixed settings."""
    return {
        "Logistic regression": Pipeline([
            ("preprocess", make_preprocessor(X)),
            ("model", LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=random_state
            )),
        ]),
        "Random forest": Pipeline([
            ("preprocess", make_preprocessor(X)),
            ("model", RandomForestClassifier(
                n_estimators=400, min_samples_leaf=10, max_features="sqrt",
                class_weight="balanced_subsample", n_jobs=-1,
                random_state=random_state,
            )),
        ]),
        "Gradient boosting": Pipeline([
            ("preprocess", make_preprocessor(X)),
            ("model", HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.08, max_leaf_nodes=15,
                min_samples_leaf=30, l2_regularization=1.0,
                class_weight="balanced", early_stopping=False,
                random_state=random_state,
            )),
        ]),
    }


def fit_models(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42):
    models = make_models(X_train, random_state)
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def calibrate_probabilities(
    fitted_model: Pipeline,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    method: str = "sigmoid",
) -> CalibratedClassifierCV:
    """Calibrate a pretrained estimator on a held-out calibration set.

    FrozenEstimator avoids refitting the base estimator on calibration data (sklearn >=1.6).
    For older versions, this function uses a compatible prefit calibration interface.
    """
    try:
        from sklearn.frozen import FrozenEstimator
        calibrated = CalibratedClassifierCV(FrozenEstimator(fitted_model), method=method)
        calibrated.fit(X_cal, y_cal)
    except ImportError:
        # Compatibility for sklearn versions prior to FrozenEstimator.
        calibrated = CalibratedClassifierCV(fitted_model, method=method, cv="prefit")
        calibrated.fit(X_cal, y_cal)
    return calibrated


def expected_calibration_error(y_true, probabilities, n_bins: int = 10) -> float:
    """Weighted absolute gap between mean predicted probability and observed event rate."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError("y_true and probabilities must be non-empty and equal length.")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite and within [0, 1].")
    edges = np.linspace(0, 1, n_bins + 1)
    total = len(y)
    error = 0.0
    for i in range(n_bins):
        # Last bin includes probability 1.
        mask = (p >= edges[i]) & ((p < edges[i + 1]) if i < n_bins - 1 else (p <= edges[i + 1]))
        if mask.any():
            error += mask.sum() / total * abs(p[mask].mean() - y[mask].mean())
    return float(error)


def evaluate_probabilities(y_true, probabilities, threshold: float = 0.5) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    pred = (p >= threshold).astype(int)
    return {
        "ROC_AUC": float(roc_auc_score(y, p)),
        "PR_AUC_average_precision": float(average_precision_score(y, p)),
        "Brier_score": float(brier_score_loss(y, p)),
        "Log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ECE_10_bins": expected_calibration_error(y, p, n_bins=10),
        "Precision_at_threshold": float(precision_score(y, pred, zero_division=0)),
        "Recall_at_threshold": float(recall_score(y, pred, zero_division=0)),
        "F1_at_threshold": float(f1_score(y, pred, zero_division=0)),
    }


def calibration_points(y_true, probabilities, n_bins: int = 10):
    observed, predicted = calibration_curve(
        y_true, probabilities, n_bins=n_bins, strategy="quantile"
    )
    return predicted, observed


def confusion_matrix_values(y_true, probabilities, threshold: float = 0.5):
    pred = (np.asarray(probabilities) >= threshold).astype(int)
    return confusion_matrix(y_true, pred, labels=[0, 1])
