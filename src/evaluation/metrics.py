"""
Evaluation metrics module for Business Entity Resolution.

Provides standardized calculations for:
- Blocking recall (overall, per-source, macro-averaged)
- Candidate set distribution statistics (percentiles, mean, min, max)
- Official competition metric: Entity-level Macro F0.5 (with strict handling of empty/singleton/multi-match sets)
- Entity-level Macro Precision, Macro Recall, Micro Precision, Micro Recall
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
import numpy as np


def calculate_blocking_recall(captured_matches: int, total_matches: int) -> float:
    """Calculates recall percentage: captured / total * 100."""
    if total_matches <= 0:
        return 100.0 if captured_matches == 0 else 0.0
    return (captured_matches / total_matches) * 100.0


def calculate_candidate_distribution(
    counts: Sequence[int],
) -> Dict[str, float]:
    """
    Computes distribution metrics for candidate set sizes per entity:
    mean, median, std, p90, p95, p99, min, max.
    """
    if not counts:
        return {
            "total_entities": 0,
            "total_candidates": 0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "min": 0,
            "max": 0,
        }

    arr = np.array(counts, dtype=np.int64)
    return {
        "total_entities": int(len(arr)),
        "total_candidates": int(np.sum(arr)),
        "mean": round(float(np.mean(arr)), 2),
        "median": float(np.median(arr)),
        "std": round(float(np.std(arr)), 2),
        "p90": round(float(np.percentile(arr, 90)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "p99": round(float(np.percentile(arr, 99)), 2),
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
    }


def calculate_entity_f05(
    true_set: Set[str],
    pred_set: Set[str],
) -> Tuple[float, float, float]:
    """
    Computes (Precision, Recall, F0.5) for a single Source-1 entity.

    Formula:
        F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
             = ((1 + beta^2) * P * R) / (beta^2 * P + R) with beta = 0.5

    Official Special Cases:
    - true == empty and predicted == empty  -> F0.5 = 1.0 (P = 1.0, R = 1.0)
    - true == empty and predicted != empty  -> F0.5 = 0.0 (P = 0.0, R = 1.0)
    - true != empty and predicted == empty  -> F0.5 = 0.0 (P = 1.0, R = 0.0)
    """
    num_true = len(true_set)
    num_pred = len(pred_set)

    if num_true == 0 and num_pred == 0:
        return 1.0, 1.0, 1.0
    if num_true == 0 and num_pred > 0:
        return 0.0, 1.0, 0.0
    if num_true > 0 and num_pred == 0:
        return 1.0, 0.0, 0.0

    tp = len(true_set & pred_set)
    precision = tp / num_pred
    recall = tp / num_true

    denom = 0.25 * precision + recall
    if denom <= 0.0 or (precision == 0.0 and recall == 0.0):
        f05 = 0.0
    else:
        f05 = (1.25 * precision * recall) / denom

    return precision, recall, f05


def evaluate_entity_level(
    gt_map: Dict[str, Set[str]],
    pred_map: Dict[str, Set[str]],
    all_s1_ids: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """
    Calculates Macro and Micro metrics across all evaluated Source-1 entities.

    Macro F0.5 = mean(F0.5 over ALL Source-1 entities)

    Args:
        gt_map: Dict mapping source1_entity_id -> set of true matched candidate IDs
        pred_map: Dict mapping source1_entity_id -> set of predicted candidate IDs
        all_s1_ids: Optional collection of all relevant S1 entity IDs. If None,
                    the union of keys in gt_map and pred_map is evaluated.

    Returns:
        Dictionary containing Macro F0.5, Macro Precision, Macro Recall,
        Micro Precision, Micro Recall, and prediction breakdown statistics.
    """
    if all_s1_ids is None:
        target_s1_ids = sorted(list(set(gt_map.keys()) | set(pred_map.keys())))
    else:
        target_s1_ids = sorted(list(set(all_s1_ids)))

    if not target_s1_ids:
        return {
            "entity_count": 0,
            "macro_f05": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f05": 0.0,
            "total_predicted_matches": 0,
            "empty_prediction_count": 0,
            "singleton_prediction_count": 0,
            "multimatch_prediction_count": 0,
        }

    f05_list: List[float] = []
    p_list: List[float] = []
    r_list: List[float] = []

    total_tp = 0
    total_pred = 0
    total_true = 0

    empty_pred_cnt = 0
    singleton_pred_cnt = 0
    multi_pred_cnt = 0

    for s1_id in target_s1_ids:
        t_set = gt_map.get(s1_id, set())
        p_set = pred_map.get(s1_id, set())

        p_val, r_val, f05_val = calculate_entity_f05(t_set, p_set)
        f05_list.append(f05_val)
        p_list.append(p_val)
        r_list.append(r_val)

        tp = len(t_set & p_set)
        total_tp += tp
        total_pred += len(p_set)
        total_true += len(t_set)

        if len(p_set) == 0:
            empty_pred_cnt += 1
        elif len(p_set) == 1:
            singleton_pred_cnt += 1
        else:
            multi_pred_cnt += 1

    macro_f05 = float(np.mean(f05_list))
    macro_p = float(np.mean(p_list))
    macro_r = float(np.mean(r_list))

    micro_p = (total_tp / total_pred) if total_pred > 0 else (1.0 if total_true == 0 else 0.0)
    micro_r = (total_tp / total_true) if total_true > 0 else (1.0 if total_pred == 0 else 0.0)

    micro_denom = 0.25 * micro_p + micro_r
    if micro_denom <= 0.0 or (micro_p == 0.0 and micro_r == 0.0):
        micro_f05 = 0.0
    else:
        micro_f05 = (1.25 * micro_p * micro_r) / micro_denom

    return {
        "entity_count": len(target_s1_ids),
        "macro_f05": round(macro_f05, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        "micro_f05": round(micro_f05, 4),
        "total_true_matches": total_true,
        "total_predicted_matches": total_pred,
        "total_tp": total_tp,
        "empty_prediction_count": empty_pred_cnt,
        "singleton_prediction_count": singleton_pred_cnt,
        "multimatch_prediction_count": multi_pred_cnt,
    }
