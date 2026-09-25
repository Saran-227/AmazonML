"""
Business name feature extraction module (Feature Group A).

Computes deterministic pairwise similarity and length features for business names
using normalized names and tokens from preprocessing.
"""

from __future__ import annotations

from typing import Any, Dict

from src.features.feature_utils import (
    character_ngram_jaccard,
    jaro_winkler_similarity,
    normalized_levenshtein_similarity,
    token_jaccard,
    token_overlap_coefficient,
)


def compute_name_features(
    s1_rec: Dict[str, Any], cand_rec: Dict[str, Any]
) -> Dict[str, float]:
    """
    Computes pairwise name features between Source 1 record and Candidate record.
    Returns a dictionary of float values.
    """
    n1 = s1_rec.get("normalized_name", "") or ""
    n2 = cand_rec.get("normalized_name", "") or ""

    t1 = s1_rec.get("name_tokens", [])
    if not t1 and n1:
        t1 = n1.split()
    t2 = cand_rec.get("name_tokens", [])
    if not t2 and n2:
        t2 = n2.split()

    set1 = set(t1)
    set2 = set(t2)
    shared_tokens = set1 & set2

    len_s1 = len(n1)
    len_c = len(n2)
    max_len = max(len_s1, len_c)
    len_ratio = round(min(len_s1, len_c) / max_len, 4) if max_len > 0 else 1.0

    tok_len_s1 = len(t1)
    tok_len_c = len(t2)

    exact_match = 1.0 if (n1 and n1 == n2) else 0.0

    return {
        "name_exact_match": exact_match,
        "name_token_jaccard": token_jaccard(t1, t2),
        "name_token_overlap_coefficient": token_overlap_coefficient(t1, t2),
        "name_shared_token_count": float(len(shared_tokens)),
        "name_token_count_s1": float(tok_len_s1),
        "name_token_count_candidate": float(tok_len_c),
        "name_token_count_difference": float(abs(tok_len_s1 - tok_len_c)),
        "name_length_s1": float(len_s1),
        "name_length_candidate": float(len_c),
        "name_length_difference": float(abs(len_s1 - len_c)),
        "name_length_ratio": len_ratio,
        "name_levenshtein_sim": normalized_levenshtein_similarity(n1, n2),
        "name_jaro_winkler": jaro_winkler_similarity(n1, n2),
        "name_char_ngram_jaccard": character_ngram_jaccard(n1, n2, n=3),
    }
