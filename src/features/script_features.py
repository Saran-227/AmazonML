"""
Script and multilingual feature extraction module (Feature Group G).

Computes deterministic, Unicode-safe features capturing script compatibility
and Latin vs non-Latin character distributions (e.g. Devanagari, Tamil, Kannada).
"""

from __future__ import annotations

from typing import Any, Dict

from src.features.feature_utils import compute_script_ratios


def compute_script_features(
    s1_rec: Dict[str, Any], cand_rec: Dict[str, Any]
) -> Dict[str, float]:
    """
    Computes pairwise script features capturing cross-script variation.
    Uses raw and normalized business names to measure Unicode script presence.
    """
    name1 = s1_rec.get("business_name", "") or ""
    name2 = cand_rec.get("business_name", "") or ""

    s1_latin, s1_non_latin = compute_script_ratios(name1)
    cand_latin, cand_non_latin = compute_script_ratios(name2)

    # Classification: predominantly Latin (> 0.5) vs predominantly non-Latin
    s1_is_latin = s1_latin >= 0.5
    cand_is_latin = cand_latin >= 0.5

    script_match = 1.0 if (s1_is_latin == cand_is_latin) else 0.0
    is_cross_script = 1.0 if (s1_is_latin != cand_is_latin) else 0.0

    # Flag if any Indic / non-ASCII character is present in either entity
    has_non_ascii = 1.0 if not (name1.isascii() and name2.isascii()) else 0.0

    return {
        "s1_latin_ratio": s1_latin,
        "cand_latin_ratio": cand_latin,
        "s1_non_latin_ratio": s1_non_latin,
        "cand_non_latin_ratio": cand_non_latin,
        "script_match": script_match,
        "is_cross_script": is_cross_script,
        "has_indic_or_non_ascii": has_non_ascii,
    }
