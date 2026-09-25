"""
Country normalizer module.
Provides generic country normalization without hard-coding specific country rules.
"""

from typing import Any
from src.preprocessing.text_utils import clean_multilingual_text, is_null_or_empty


def normalize_country(country: Any) -> str:
    """
    Normalizes a country name or code generically.
    - Handles missing/null values safely (returns "")
    - Normalizes case (lowercase) and collapses whitespace
    - Preserves actual country information without country-specific branching
    """
    if is_null_or_empty(country):
        return ""

    return clean_multilingual_text(country)
