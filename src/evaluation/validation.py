"""
Validation splitting and leakage-safe grouping module for Business Entity Resolution.

Ensures that candidate pairs are strictly partitioned by Source-1 entity groups:
- Zero group leakage between train and validation splits
- Stratified or random grouped splitting
- GroupKFold cross-validation iterator
- Programmatic leakage verification assertion
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

logger = logging.getLogger(__name__)


def verify_group_leakage(
    train_s1_ids: Set[str],
    val_s1_ids: Set[str],
) -> None:
    """
    Verifies programmatically that set(train_s1_ids) & set(val_s1_ids) is EMPTY.
    Raises ValueError immediately if any overlap is discovered.
    """
    overlap = set(train_s1_ids) & set(val_s1_ids)
    if overlap:
        raise ValueError(
            f"DATA LEAKAGE DETECTED! Found {len(overlap)} Source-1 entities present in both "
            f"train and validation splits: {list(overlap)[:5]}"
        )


def split_grouped_dataset(
    metadata_df: pd.DataFrame,
    val_ratio: float = 0.25,
    group_col: str = "source1_entity_id",
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Set[str], Set[str]]:
    """
    Partitions rows into train and validation splits strictly by group_col.

    Args:
        metadata_df: DataFrame containing at minimum group_col
        val_ratio: Fraction of groups assigned to validation split (default: 0.25)
        group_col: Column name identifying group entity (default: 'source1_entity_id')
        random_state: Random seed for deterministic reproducibility

    Returns:
        train_indices: np.ndarray of row indices for training
        val_indices: np.ndarray of row indices for validation
        train_groups: Set of unique group IDs in training
        val_groups: Set of unique group IDs in validation
    """
    groups = metadata_df[group_col].values
    unique_groups = np.unique(groups)

    rng = np.random.RandomState(random_state)
    shuffled_groups = rng.permutation(unique_groups)

    n_val_groups = int(np.round(len(unique_groups) * val_ratio))
    val_group_set = set(shuffled_groups[:n_val_groups])
    train_group_set = set(shuffled_groups[n_val_groups:])

    # Programmatic leakage check
    verify_group_leakage(train_group_set, val_group_set)

    is_val = metadata_df[group_col].isin(val_group_set).values
    val_indices = np.where(is_val)[0]
    train_indices = np.where(~is_val)[0]

    return train_indices, val_indices, train_group_set, val_group_set


def iter_grouped_folds(
    metadata_df: pd.DataFrame,
    n_splits: int = 3,
    group_col: str = "source1_entity_id",
) -> Iterator[Tuple[np.ndarray, np.ndarray, Set[str], Set[str]]]:
    """
    Generates GroupKFold cross-validation splits.

    Yields for each fold:
        (train_indices, val_indices, train_group_set, val_group_set)
    """
    gkf = GroupKFold(n_splits=n_splits)
    groups = metadata_df[group_col].values
    X_dummy = np.zeros(len(metadata_df))

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_dummy, groups=groups)):
        train_groups = set(metadata_df.iloc[train_idx][group_col].unique())
        val_groups = set(metadata_df.iloc[val_idx][group_col].unique())
        verify_group_leakage(train_groups, val_groups)
        yield train_idx, val_idx, train_groups, val_groups
