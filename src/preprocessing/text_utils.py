"""
Core text utilities for multilingual normalization and tokenization.
Handles Unicode safely, folding Latin diacritics while preserving Indic scripts
(Devanagari, Tamil, Kannada, etc.) and their combining vowel marks (matras).
"""

import math
import string
import unicodedata
from typing import Any, List, Optional

# Precomputed fast-path translation table for ASCII text
_ASCII_PUNCT_MAP = str.maketrans({
    c: " " for c in string.punctuation if c not in ("'", "`")
})
_ASCII_PUNCT_MAP[ord("'")] = None
_ASCII_PUNCT_MAP[ord("`")] = None


def is_null_or_empty(val: Any) -> bool:
    """Check if value is None, NaN, or an empty string representation."""
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return True
    return False


def remove_latin_accents(text: str) -> str:
    """
    Folds accented Latin characters (e.g. é -> e, ó -> o, É -> E)
    while preserving non-Latin combining marks such as Indic vowel signs (matras).
    """
    decomposed = unicodedata.normalize("NFKD", text)
    result = []
    for c in decomposed:
        if unicodedata.combining(c):
            # Only strip combining marks if preceding character is a Latin letter
            if result and ("a" <= result[-1].lower() <= "z"):
                continue
        result.append(c)
    return unicodedata.normalize("NFC", "".join(result))


def clean_multilingual_text(text: Any, remove_null_words: bool = False) -> str:
    """
    Cleans and normalizes text in a deterministic, multilingual-safe manner:
    1. Handles missing/null values safely -> returns ""
    2. Fast path for ASCII strings using maketrans and lowercasing
    3. Multilingual Unicode path for non-ASCII text:
       - Unicode NFKC normalization
       - Latin accent folding (preserving Indic matras)
       - Lowercasing
       - Ampersand '&' converted to ' and '
       - Apostrophes removed cleanly (e.g. Orelee's -> orelees)
       - Punctuation and symbols replaced with spaces
       - Whitespace collapsed and trimmed
    4. Optional removal of standalone 'null' tokens (e.g. for concatenated addresses)
    """
    if is_null_or_empty(text):
        return ""

    if not isinstance(text, str):
        text = str(text)

    # Strip surrounding whitespace
    text = text.strip()
    if not text:
        return ""

    # Fast path: Pure ASCII string
    if text.isascii():
        s = text.lower().replace("&", " and ").translate(_ASCII_PUNCT_MAP)
        res = " ".join(s.split())
        if remove_null_words and "null" in res:
            res = " ".join([w for w in res.split() if w != "null"])
        return res

    # Multilingual Unicode path
    # NFKC normalizes compatibility characters (ligatures, full-width, etc.)
    s = unicodedata.normalize("NFKC", text)
    # Fold Latin accents safely
    s = remove_latin_accents(s).lower()
    s = s.replace("&", " and ")

    out = []
    for c in s:
        if c in ("'", "’", "‘", "`"):
            continue  # remove apostrophe cleanly
        cat = unicodedata.category(c)
        # L = Letter, N = Number, M = Mark (combining vowel marks/viramas in Indic scripts)
        if cat[0] in ("L", "N", "M"):
            out.append(c)
        else:
            out.append(" ")

    res = " ".join("".join(out).split())
    if remove_null_words and "null" in res:
        res = " ".join([w for w in res.split() if w != "null"])

    return res


def tokenize_text(text: Any, remove_null_words: bool = False) -> List[str]:
    """
    Deterministically normalizes and tokenizes text into a list of string tokens.
    Returns an empty list for empty/null inputs.
    """
    cleaned = clean_multilingual_text(text, remove_null_words=remove_null_words)
    if not cleaned:
        return []
    return cleaned.split()
