"""
Model training module for Business Entity Resolution.

Supports:
- Strict feature matrix validation (no NaN, no inf, numeric only, identical column alignment, no IDs or labels)
- Model 1: Logistic Regression with StandardScaler and class weighting
- Model 2: Gradient-Boosted Decision Trees (HistGradientBoostingClassifier) with histogram binning and class weighting
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

FORBIDDEN_COLUMNS = {
    "entity_id",
    "source1_entity_id",
    "candidate_entity_id",
    "candidate_source",
    "label",
    "matched_entity_ids",
    "blocking_rules",
    "business_name",
    "normalized_name",
    "address",
    "normalized_address",
    "country",
}


def validate_feature_matrix(
    X_train: pd.DataFrame,
    X_val: Optional[pd.DataFrame] = None,
) -> List[str]:
    """
    Validates feature matrix before model training or prediction.

    Enforces strict rules:
    - No NaN values
    - No infinite values
    - All columns must be numeric (float or integer)
    - No duplicate column names
    - No target/label or entity ID columns
    - If X_val is provided, column names and order must be 100% identical.

    Raises:
        ValueError if any validation rule is violated.

    Returns:
        List of verified feature column names.
    """
    # 1. Check duplicate columns
    cols = list(X_train.columns)
    if len(cols) != len(set(cols)):
        duplicates = [c for c in cols if cols.count(c) > 1]
        raise ValueError(f"Duplicate feature columns found in X: {duplicates}")

    # 2. Check forbidden columns (labels or raw IDs or raw text)
    forbidden_present = set(cols) & FORBIDDEN_COLUMNS
    if forbidden_present:
        raise ValueError(f"Forbidden ID/label/text column found in feature matrix X: {forbidden_present}")

    # 3. Check dtypes (must all be numeric)
    for col in cols:
        if not np.issubdtype(X_train[col].dtype, np.number):
            raise ValueError(f"Feature '{col}' has non-numeric dtype: {X_train[col].dtype}")

    # 4. Check NaN and Inf in X_train
    null_count = int(X_train.isnull().sum().sum())
    if null_count > 0:
        bad_cols = X_train.columns[X_train.isnull().any()].tolist()
        raise ValueError(f"Feature matrix X_train contains {null_count} NaN values in columns: {bad_cols}")

    arr_train = X_train.values
    if np.isinf(arr_train).any():
        raise ValueError("Feature matrix X_train contains infinite values.")

    # 5. Check X_val if provided
    if X_val is not None:
        val_cols = list(X_val.columns)
        if val_cols != cols:
            raise ValueError(
                f"Feature columns mismatch between train and val! "
                f"Train cols ({len(cols)}), Val cols ({len(val_cols)}). Difference: {set(cols) ^ set(val_cols)}"
            )

        val_null_count = int(X_val.isnull().sum().sum())
        if val_null_count > 0:
            bad_cols = X_val.columns[X_val.isnull().any()].tolist()
            raise ValueError(f"Feature matrix X_val contains {val_null_count} NaN values in columns: {bad_cols}")

        arr_val = X_val.values
        if np.isinf(arr_val).any():
            raise ValueError("Feature matrix X_val contains infinite values.")

    return cols


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    class_weight: Union[str, Dict[int, float]] = "balanced",
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
) -> Tuple[Pipeline, float]:
    """
    Trains a Logistic Regression baseline model with standard scaling.

    Args:
        X_train: DataFrame of numerical features
        y_train: Binary labels (0 or 1)
        class_weight: Weighting scheme ('balanced' or custom dict)
        C: Inverse regularization strength
        max_iter: Maximum solver iterations
        random_state: Random state seed

    Returns:
        (pipeline, training_time_seconds)
    """
    validate_feature_matrix(X_train)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            class_weight=class_weight,
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            solver="lbfgs",
        )),
    ])

    t0 = time.time()
    pipeline.fit(X_train, y_train)
    t_train = time.time() - t0

    logger.info("Logistic Regression trained in %.3f seconds (samples=%d, features=%d).",
                t_train, len(X_train), X_train.shape[1])
    return pipeline, t_train


def train_tree_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    class_weight: Union[str, Dict[int, float]] = "balanced",
    max_iter: int = 150,
    max_leaf_nodes: int = 31,
    learning_rate: float = 0.1,
    min_samples_leaf: int = 20,
    random_state: int = 42,
) -> Tuple[HistGradientBoostingClassifier, float]:
    """
    Trains a Gradient-Boosted Decision Tree model (HistGradientBoostingClassifier).

    Args:
        X_train: DataFrame of numerical features
        y_train: Binary labels (0 or 1)
        class_weight: Weighting scheme ('balanced' or None)
        max_iter: Number of boosting iterations (trees)
        max_leaf_nodes: Max leaves per tree
        learning_rate: Shrinkage parameter
        min_samples_leaf: Minimum samples required per leaf
        random_state: Random state seed

    Returns:
        (model, training_time_seconds)
    """
    validate_feature_matrix(X_train)

    model = HistGradientBoostingClassifier(
        class_weight=class_weight,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        learning_rate=learning_rate,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )

    t0 = time.time()
    model.fit(X_train, y_train)
    t_train = time.time() - t0

    logger.info("Tree model (HistGradientBoosting) trained in %.3f seconds (samples=%d, features=%d).",
                t_train, len(X_train), X_train.shape[1])
    return model, t_train
