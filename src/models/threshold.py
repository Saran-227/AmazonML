"""
Threshold grid search and optimization module for Business Entity Resolution.

Evaluates thresholds over a fine-grained grid (0.10 to 0.95, step 0.05) on the
official validation Entity-Level Macro F0.5 metric, selecting the optimal threshold
and breaking ties toward the higher, more precision-conservative threshold.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_entity_level
from src.models.predict import predict_matches_at_threshold

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = [
    0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
]


def evaluate_threshold_grid(
    probabilities: np.ndarray,
    metadata_df: pd.DataFrame,
    gt_map: Dict[str, Set[str]],
    all_s1_ids: Optional[Iterable[str]] = None,
    thresholds: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Evaluates candidate predictions across a grid of thresholds against Ground Truth.

    Calculates for each threshold:
    - entity-level Macro F0.5
    - Macro Precision
    - Macro Recall
    - Micro Precision
    - Micro Recall
    - Micro F0.5
    - Total predicted match count
    - Empty prediction count
    - Singleton prediction count
    - Multi-match prediction count

    Returns:
        pd.DataFrame containing all evaluated metrics per threshold row.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    if all_s1_ids is None:
        target_s1_ids = sorted(list(set(metadata_df["source1_entity_id"].unique()) | set(gt_map.keys())))
    else:
        target_s1_ids = sorted(list(set(all_s1_ids)))

    records: List[Dict[str, Any]] = []

    for thresh in thresholds:
        pred_map = predict_matches_at_threshold(
            probabilities=probabilities,
            metadata_df=metadata_df,
            threshold=thresh,
            all_s1_ids=target_s1_ids,
        )

        res = evaluate_entity_level(
            gt_map=gt_map,
            pred_map=pred_map,
            all_s1_ids=target_s1_ids,
        )

        row = {
            "threshold": round(thresh, 2),
            "macro_f05": res["macro_f05"],
            "macro_precision": res["macro_precision"],
            "macro_recall": res["macro_recall"],
            "micro_precision": res["micro_precision"],
            "micro_recall": res["micro_recall"],
            "micro_f05": res["micro_f05"],
            "predicted_matches": res["total_predicted_matches"],
            "empty_predictions": res["empty_prediction_count"],
            "singleton_predictions": res["singleton_prediction_count"],
            "multimatch_predictions": res["multimatch_prediction_count"],
        }
        records.append(row)

    df_grid = pd.DataFrame(records)
    return df_grid


def select_best_threshold(
    grid_df: pd.DataFrame,
) -> Tuple[float, Dict[str, Any], bool, List[float]]:
    """
    Selects the optimal threshold maximizing Macro F0.5.

    Tie-breaking rule:
    If multiple thresholds achieve the exact maximum Macro F0.5, selects the
    higher (more precision-conservative) threshold, and reports the tie.

    Returns:
        (best_threshold, best_metrics_dict, is_tied, tied_thresholds_list)
    """
    max_f05 = grid_df["macro_f05"].max()
    candidates = grid_df[grid_df["macro_f05"] == max_f05]

    is_tied = len(candidates) > 1
    tied_thresholds = candidates["threshold"].tolist()

    # Prefer highest threshold (more conservative) on ties
    best_row = candidates.sort_values(by="threshold", ascending=False).iloc[0]
    best_thresh = float(best_row["threshold"])
    best_metrics = best_row.to_dict()

    if is_tied:
        logger.info(
            "Threshold tie detected at max Macro F0.5=%.4f among %s. "
            "Selected higher threshold %.2f (precision-conservative rule).",
            max_f05, tied_thresholds, best_thresh
        )

    return best_thresh, best_metrics, is_tied, tied_thresholds
