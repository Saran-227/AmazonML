"""
Business name normalizer module.
Provides deterministic, multilingual-safe business name normalization and tokenization.
"""

from typing import Any, List
from src.preprocessing.text_utils import clean_multilingual_text, tokenize_text


def normalize_business_name(name: Any) -> str:
    """
    Normalizes a business name:
    - Handles missing/null values safely (returns "")
    - Converts input safely to string
    - Normalizes Unicode (NFKC, folds Latin accents while preserving Indic scripts)
    - Normalizes case to lowercase
    - Normalizes whitespace
    - Handles punctuation (removes noise, replaces separators with spaces, ampersand -> 'and')
    - Preserves meaningful alphanumeric content
    - Produces deterministic output without destructive transformations

    Example:
        " Prime Money, Inc. " -> "prime money inc"
        "Orelee's Barbershop" -> "orelees barbershop"
        "PAYNE-ENRTPRMISES" -> "payne enrtprmises"
        "राम मार्केटिंग प्राइवेट लिमिटेड" -> "राम मार्केटिंग प्राइवेट लिमिटेड"
    """
    return clean_multilingual_text(name, remove_null_words=False)


def tokenize_name(name: Any) -> List[str]:
    """
    Deterministically tokenizes a business name into a list of word tokens.
    Can accept either a raw or already normalized business name.

    Example:
        "Prime Money Inc" -> ["prime", "money", "inc"]
        "Orelee's Barbershop" -> ["orelees", "barbershop"]
    """
    return tokenize_text(name, remove_null_words=False)
