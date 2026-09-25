"""
Feature engineering utility functions for Business Entity Resolution.

Provides fast, deterministic, pure-Python implementations of:
- Levenshtein distance and normalized Levenshtein similarity
- Jaro and Jaro-Winkler string similarity
- Token set similarities (Jaccard, Overlap Coefficient / Simpson)
- Character n-gram Jaccard similarity
- Numeric address token extraction and leading-zero normalization
- Unicode script character analysis (Latin vs non-Latin ratios)
"""

from __future__ import annotations

import unicodedata
from typing import Any, Iterable, List, Optional, Set, Tuple


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Computes Levenshtein edit distance between two strings using two-row DP buffer.
    Time: O(len(s1) * len(s2)), Space: O(min(len(s1), len(s2))).
    """
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)

    # Ensure s2 is the shorter string to minimize allocated memory
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1] * (len(s2) + 1)
        for j, c2 in enumerate(s2):
            insertions = prev[j + 1] + 1
            deletions = curr[j] + 1
            substitutions = prev[j] + (c1 != c2)
            curr[j + 1] = min(insertions, deletions, substitutions)
        prev = curr

    return prev[-1]


def normalized_levenshtein_similarity(s1: str, s2: str) -> float:
    """
    Computes normalized Levenshtein similarity in [0.0, 1.0].
    sim = 1.0 - (lev_dist / max(len(s1), len(s2))).
    Returns 1.0 if both strings are empty.
    """
    if s1 == s2:
        return 1.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(s1, s2)
    return round(max(0.0, min(1.0, 1.0 - (dist / max_len))), 4)


def jaro_similarity(s1: str, s2: str) -> float:
    """
    Computes Jaro string similarity in [0.0, 1.0].
    """
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    match_distance = max(len1, len2) // 2 - 1
    if match_distance < 0:
        match_distance = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j]:
                continue
            if s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    transpositions //= 2
    jaro = (matches / len1 + matches / len2 + (matches - transpositions) / matches) / 3.0
    return round(max(0.0, min(1.0, jaro)), 4)


def jaro_winkler_similarity(s1: str, s2: str, p: float = 0.1, max_l: int = 4) -> float:
    """
    Computes Jaro-Winkler string similarity with prefix scaling bonus.
    sim in [0.0, 1.0].
    """
    jaro = jaro_similarity(s1, s2)
    if jaro < 0.7 or jaro >= 1.0:
        return jaro

    # Calculate common prefix length up to max_l
    l = 0
    for c1, c2 in zip(s1[:max_l], s2[:max_l]):
        if c1 == c2:
            l += 1
        else:
            break

    jw = jaro + l * p * (1.0 - jaro)
    return round(max(0.0, min(1.0, jw)), 4)


def token_jaccard(tokens1: Iterable[str], tokens2: Iterable[str]) -> float:
    """
    Computes Jaccard similarity between two token sets: |A ∩ B| / |A ∪ B|.
    Returns 1.0 if both sets are empty, 0.0 if one is empty.
    """
    s1 = set(tokens1)
    s2 = set(tokens2)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    intersection = len(s1 & s2)
    union = len(s1 | s2)
    return round(intersection / union, 4) if union > 0 else 0.0


def token_overlap_coefficient(tokens1: Iterable[str], tokens2: Iterable[str]) -> float:
    """
    Computes Overlap Coefficient (Simpson coefficient): |A ∩ B| / min(|A|, |B|).
    Measures degree of containment (e.g. if one entity name is an abbreviated subset of the other).
    """
    s1 = set(tokens1)
    s2 = set(tokens2)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    intersection = len(s1 & s2)
    denom = min(len(s1), len(s2))
    return round(intersection / denom, 4) if denom > 0 else 0.0


def character_ngram_jaccard(s1: str, s2: str, n: int = 3) -> float:
    """
    Computes Jaccard similarity over character n-grams.
    Robust to slight transpositions, concats, and minor misspellings.
    """
    if s1 == s2:
        return 1.0
    c1 = s1.replace(" ", "")
    c2 = s2.replace(" ", "")
    if not c1 and not c2:
        return 1.0
    if not c1 or not c2:
        return 0.0

    if len(c1) < n:
        grams1 = {c1}
    else:
        grams1 = {c1[i : i + n] for i in range(len(c1) - n + 1)}

    if len(c2) < n:
        grams2 = {c2}
    else:
        grams2 = {c2[i : i + n] for i in range(len(c2) - n + 1)}

    return token_jaccard(grams1, grams2)


def extract_numeric_tokens(tokens: Iterable[str]) -> List[str]:
    """
    Extracts numeric tokens normalized by stripping leading zeros (e.g. '0684' -> '684').
    """
    res = []
    for t in tokens:
        if t.isdigit():
            stripped = t.lstrip("0")
            res.append(stripped if stripped else "0")
    return res


def compute_script_ratios(text: str) -> Tuple[float, float]:
    """
    Computes (latin_char_ratio, non_latin_char_ratio) for alphabetic characters in text.
    Returns (1.0, 0.0) if text has no letters.
    """
    latin_count = 0
    non_latin_count = 0
    for ch in text:
        if ch.isalpha():
            name = unicodedata.name(ch, "").lower()
            if "latin" in name:
                latin_count += 1
            else:
                non_latin_count += 1

    total = latin_count + non_latin_count
    if total == 0:
        return 1.0, 0.0
    return round(latin_count / total, 4), round(non_latin_count / total, 4)
