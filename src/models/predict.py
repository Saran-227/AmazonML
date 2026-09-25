"""
Model prediction and probability inference module for Business Entity Resolution.

Supports:
- Safe probability generation with bounds checking [0.0, 1.0] and finite guarantees
- Conversion of probabilities to entity-level match sets supporting 0, 1, or multiple matches
- Structured prediction DataFrame generation keeping identifiers strictly aligned
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.models.train import validate_feature_matrix

logger = logging.getLogger(__name__)


def predict_probabilities(
    model: Any,
    X: pd.DataFrame,
) -> Tuple[np.ndarray, float]:
    """
    Computes match probabilities P(match | x) for all candidate pairs in X.

    Verifications:
    - Feature matrix validity
    - Number of predictions == len(X)
    - All probabilities are finite
    - All probabilities are strictly within [0.0, 1.0]

    Returns:
        (probabilities_array, prediction_time_seconds)
    """
    validate_feature_matrix(X)

    t0 = time.time()
    raw_proba = model.predict_proba(X)
    pred_time = time.time() - t0

    # Extract probability of positive class (label = 1)
    if raw_proba.ndim == 2 and raw_proba.shape[1] >= 2:
        proba = raw_proba[:, 1]
    else:
        proba = raw_proba.ravel()

    # Integrity assertions
    if len(proba) != len(X):
        raise ValueError(f"Prediction count mismatch: expected {len(X)}, got {len(proba)}")

    if not np.isfinite(proba).all():
        raise ValueError("Non-finite values encountered in predicted probabilities.")

    if (proba < 0.0).any() or (proba > 1.0).any():
        raise ValueError(
            f"Probabilities out of bounds [0, 1]: min={proba.min()}, max={proba.max()}"
        )

    return proba, pred_time


def predict_matches_at_threshold(
    probabilities: np.ndarray,
    metadata_df: pd.DataFrame,
    threshold: float,
    all_s1_ids: Optional[Iterable[str]] = None,
) -> Dict[str, Set[str]]:
    """
    Translates candidate-level probabilities into entity-level match sets.

    Supports:
    - 0 matches (empty set)
    - 1 match (singleton set)
    - Multiple matches (one-to-many set)

    Args:
        probabilities: Array of P(match) for each candidate pair
        metadata_df: DataFrame containing 'source1_entity_id' and 'candidate_entity_id'
        threshold: Decision threshold in [0.0, 1.0]
        all_s1_ids: Optional collection of all target S1 IDs to guarantee presence in output dict

    Returns:
        Dict mapping source1_entity_id -> set of predicted candidate entity IDs
    """
    if len(probabilities) != len(metadata_df):
        raise ValueError("Length mismatch between probabilities and metadata DataFrame.")

    # Initialize all target S1 IDs with empty sets
    pred_map: Dict[str, Set[str]] = {}
    if all_s1_ids is not None:
        for s1_id in all_s1_ids:
            pred_map[s1_id] = set()

    s1_col = metadata_df["source1_entity_id"].values
    cand_col = metadata_df["candidate_entity_id"].values

    above_thresh_indices = np.where(probabilities >= threshold)[0]
    for idx in above_thresh_indices:
        s1 = s1_col[idx]
        c_id = cand_col[idx]
        if s1 not in pred_map:
            pred_map[s1] = set()
        pred_map[s1].add(c_id)

    return pred_map


def build_prediction_dataframe(
    metadata_df: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    y_true: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Constructs an aligned DataFrame with metadata, probabilities, and binary predictions.
    """
    df = metadata_df.copy()
    df["predicted_probability"] = np.round(probabilities, 4)
    df["predicted_match"] = (probabilities >= threshold).astype(int)

    if y_true is not None:
        df["true_match"] = y_true.astype(int)

    return df
