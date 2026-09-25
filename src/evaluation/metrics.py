"""
Evaluation metrics module for Business Entity Resolution.

Provides standardized calculations for:
- Blocking recall (overall, per-source, macro-averaged)
- Candidate set distribution statistics (percentiles, mean, min, max)
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Union
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
