"""
Blocking rule and candidate source feature module (Feature Groups E & F).

Extracts binary flags for which Phase-2 blocking rules generated the candidate,
the total number of triggering rules, and candidate source indicators.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Set


ALL_BLOCKING_RULES = [
    "exact_name",
    "name_country",
    "name_token",
    "compressed_name",
    "exact_address",
    "address_anchor",
]


def compute_blocking_and_source_features(
    blocking_rules: Optional[Iterable[str]] = None,
    candidate_source: str = "",
    candidate_id: str = "",
) -> Dict[str, float]:
    """
    Computes blocking rule indicators, rule count, and candidate source flags.
    """
    rules_set: Set[str] = set(blocking_rules) if blocking_rules else set()

    features: Dict[str, float] = {}
    for r in ALL_BLOCKING_RULES:
        features[f"block_{r}"] = 1.0 if r in rules_set else 0.0

    features["blocking_rule_count"] = float(len(rules_set))

    # Candidate source features (S2 vs S3)
    src = candidate_source.upper()
    if not src and candidate_id:
        if candidate_id.startswith("S2-"):
            src = "S2"
        elif candidate_id.startswith("S3-"):
            src = "S3"

    features["is_source2"] = 1.0 if src == "S2" or src.startswith("S2") else 0.0
    features["is_source3"] = 1.0 if src == "S3" or src.startswith("S3") else 0.0

    return features
