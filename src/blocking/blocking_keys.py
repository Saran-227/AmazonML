"""
Blocking key extraction module for Business Entity Resolution.

Extracts deterministic, multilingual-safe, and country-agnostic blocking keys
from normalized business records produced by the preprocessing pipeline.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

# Common generic legal suffixes and ubiquitous business stopwords across jurisdictions
DEFAULT_LEGAL_STOPWORDS: Set[str] = {
    # English / US / International
    "inc", "incorporated", "llc", "corp", "corporation", "ltd", "limited",
    "co", "company", "and", "the", "of", "in", "for", "llp", "pc", "pllc",
    "services", "enterprises", "solutions", "holdings", "group", "associates",
    "center", "international", "global", "trading", "technologies", "com",
    # India
    "pvt", "private", "trust", "society",
    # France / Europe
    "sa", "sarl", "sas", "sasu", "sci", "gmbh", "spa", "bv", "amicale",
}

# Common address stopwords that are too generic to form distinctive address blocks
DEFAULT_ADDRESS_STOPWORDS: Set[str] = {
    # Road / building types
    "st", "street", "rd", "road", "ave", "avenue", "dr", "drive", "blvd",
    "boulevard", "ln", "lane", "ct", "court", "pl", "place", "way", "terrace",
    "hwy", "highway", "pkwy", "parkway", "cswy", "causeway", "circle", "cir",
    # Units / components
    "apt", "apartment", "unit", "ste", "suite", "fl", "floor", "bldg", "building",
    "no", "box", "po", "dept", "room", "rm", "lot", "door", "flat", "plot",
    # Landmarks & directions
    "near", "opp", "opposite", "behind", "beside", "above", "below", "adj",
    "adjacent", "cross", "main", "stage", "phase", "block", "sector", "layout",
    "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th",
    "north", "south", "east", "west", "central", "upper", "lower", "new", "old",
}


def normalize_number_token(tok: str) -> str:
    """Normalize a numeric token by stripping leading zeros (e.g., '0684' -> '684')."""
    stripped = tok.lstrip("0")
    return stripped if stripped else "0"


def extract_exact_name_key(record: Dict[str, Any]) -> Optional[str]:
    """
    Extracts the exact normalized business name key.
    Returns None if normalized_name is empty.
    """
    name = record.get("normalized_name", "").strip()
    return name if name else None


def extract_name_country_key(record: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """
    Extracts (normalized_name, normalized_country) key.
    Returns None if normalized_name is empty.
    """
    name = record.get("normalized_name", "").strip()
    if not name:
        return None
    country = record.get("normalized_country", "").strip()
    return (name, country)


def extract_name_tokens_keys(
    record: Dict[str, Any],
    min_len: int = 3,
    stopwords: Optional[Set[str]] = None,
    with_country: bool = True,
) -> List[Tuple[str, ...]]:
    """
    Extracts distinctive name token keys.
    Each token must be >= min_len characters and not in the stopwords set.
    Returns a list of tuples: (token, country) if with_country=True, else (token,).
    """
    if stopwords is None:
        stopwords = DEFAULT_LEGAL_STOPWORDS

    tokens = record.get("name_tokens", [])
    if not tokens and record.get("normalized_name"):
        tokens = record["normalized_name"].split()

    country = record.get("normalized_country", "").strip()
    keys: List[Tuple[str, ...]] = []

    for tok in tokens:
        tok_clean = tok.strip()
        if len(tok_clean) >= min_len and tok_clean not in stopwords:
            if with_country:
                keys.append((tok_clean, country))
            else:
                keys.append((tok_clean,))

    return keys


def extract_compressed_name_keys(
    record: Dict[str, Any],
    min_len: int = 5,
    stopwords: Optional[Set[str]] = None,
    with_country: bool = True,
) -> List[Tuple[str, ...]]:
    """
    Extracts compressed/concatenated name keys (bridging domain names like 'celestialmemorialtrust com'
    and concatenated trade names).
    """
    if stopwords is None:
        stopwords = DEFAULT_LEGAL_STOPWORDS

    tokens = record.get("name_tokens", [])
    if not tokens and record.get("normalized_name"):
        tokens = record["normalized_name"].split()

    country = record.get("normalized_country", "").strip()
    non_stop = [t for t in tokens if t not in stopwords and len(t) >= 2]
    keys: List[Tuple[str, ...]] = []

    if non_stop:
        # Full concatenated non-stop tokens
        comp_full = "".join(non_stop)
        if len(comp_full) >= min_len:
            if with_country:
                keys.append((comp_full, country))
            else:
                keys.append((comp_full,))

        # First 2 non-stop tokens concatenated (e.g. 'first seven' -> 'firstseven')
        if len(non_stop) >= 2:
            comp_first2 = "".join(non_stop[:2])
            if len(comp_first2) >= min_len and comp_first2 != comp_full:
                if with_country:
                    keys.append((comp_first2, country))
                else:
                    keys.append((comp_first2,))

    return keys


def extract_exact_address_key(record: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """
    Extracts exact (normalized_address, normalized_country) key.
    Returns None if normalized_address is empty.
    """
    addr = record.get("normalized_address", "").strip()
    if not addr:
        return None
    country = record.get("normalized_country", "").strip()
    return (addr, country)


def extract_address_anchor_keys(
    record: Dict[str, Any],
    addr_stopwords: Optional[Set[str]] = None,
    min_word_len: int = 4,
    max_words_per_number: int = 3,
) -> List[Tuple[str, str, str]]:
    """
    Extracts robust address anchor keys: (country, normalized_number, distinctive_word).
    Normalizes street/building numbers by stripping leading zeros (e.g. '0684' -> '684').
    Pairs numbers with distinctive locality/city tokens (skipping generic street words like 'road').
    
    This bridges cases where business names are synthetic aliases, web domains,
    or cross-script transliterations with identical physical locations.
    """
    if addr_stopwords is None:
        addr_stopwords = DEFAULT_ADDRESS_STOPWORDS

    tokens = record.get("address_tokens", [])
    if not tokens and record.get("normalized_address"):
        tokens = record["normalized_address"].split()

    if not tokens:
        return []

    country = record.get("normalized_country", "").strip()

    # Separate numeric tokens and distinctive textual words
    numbers: List[str] = []
    distinct_words: List[str] = []

    for t in tokens:
        if t.isdigit():
            norm_num = normalize_number_token(t)
            if norm_num not in numbers:
                numbers.append(norm_num)
        elif len(t) >= min_word_len and t not in addr_stopwords:
            if t not in distinct_words:
                distinct_words.append(t)

    anchors: List[Tuple[str, str, str]] = []
    if numbers and distinct_words:
        # Use primary numbers (up to 2) and distinctive locality words
        for num in numbers[:2]:
            for w in distinct_words[:max_words_per_number]:
                anchors.append((country, num, w))

    return anchors
