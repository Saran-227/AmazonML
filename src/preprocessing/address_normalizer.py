"""
Business address normalizer module.
Provides deterministic, multilingual-safe address normalization and tokenization.
Preserves numbers, unit/apartment/floor/PO box details, and postal codes across countries.
"""

from typing import Any, List
from src.preprocessing.text_utils import clean_multilingual_text, tokenize_text


def normalize_business_address(address: Any) -> str:
    """
    Normalizes a business address:
    - Handles missing/null values safely (returns "")
    - Normalizes Unicode (NFKC, folds Latin accents while preserving Indic scripts)
    - Normalizes case to lowercase
    - Normalizes whitespace
    - Normalizes punctuation (replaces commas, hyphens, slashes, colons with spaces)
    - Preserves important numbers (street numbers, building numbers, postal codes)
    - Preserves unit, apartment, suite, floor, and PO Box details
    - Strips noisy standalone 'null' tokens resulting from raw data concatenation
    - Completely country-agnostic (does not assume any single country format)

    Example:
        "1795 Westchester Drive, High Point, NC" -> "1795 westchester drive high point nc"
        "2100 Cameron Drive, Unit APARTMENT G, Dundalk, MD" -> "2100 cameron drive unit apartment g dundalk md"
        "1056-1060 BELDEN AVE, PO BOX 8807, AKRON, OH" -> "1056 1060 belden ave po box 8807 akron oh"
        "45ND TERRACE, null, KANSAS CITY, MO" -> "45nd terrace kansas city mo"
    """
    return clean_multilingual_text(address, remove_null_words=True)


def tokenize_address(address: Any) -> List[str]:
    """
    Deterministically tokenizes a business address into a list of word tokens.
    Can accept either a raw or already normalized address.

    Example:
        "1795 Westchester Drive, High Point, NC" ->
        ["1795", "westchester", "drive", "high", "point", "nc"]
    """
    return tokenize_text(address, remove_null_words=True)
