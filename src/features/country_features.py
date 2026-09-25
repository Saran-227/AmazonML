"""
Country feature extraction module (Feature Group C).

Computes open-set country agreement and missingness features without hard-coding
any country labels (e.g. US, India, France).
"""

from __future__ import annotations

from typing import Any, Dict


def compute_country_features(
    s1_rec: Dict[str, Any], cand_rec: Dict[str, Any]
) -> Dict[str, float]:
    """
    Computes pairwise country agreement features.
    Treats country strictly as an open-set string.
    """
    c1 = (s1_rec.get("normalized_country", "") or "").strip().lower()
    c2 = (cand_rec.get("normalized_country", "") or "").strip().lower()

    missing_s1 = 1.0 if not c1 else 0.0
    missing_c = 1.0 if not c2 else 0.0

    if c1 and c2:
        match = 1.0 if c1 == c2 else 0.0
        mismatch = 1.0 if c1 != c2 else 0.0
    else:
        match = 0.0
        mismatch = 0.0

    return {
        "country_exact_match": match,
        "country_mismatch": mismatch,
        "country_missing_s1": missing_s1,
        "country_missing_candidate": missing_c,
    }
