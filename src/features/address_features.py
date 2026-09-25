"""
Business address feature extraction module (Feature Group B).

Computes deterministic pairwise similarity, length, and numeric anchor features
for business addresses. Handles missing addresses safely without false mismatches.
"""

from __future__ import annotations

from typing import Any, Dict, Set

from src.blocking.blocking_keys import DEFAULT_ADDRESS_STOPWORDS
from src.features.feature_utils import (
    extract_numeric_tokens,
    normalized_levenshtein_similarity,
    token_jaccard,
    token_overlap_coefficient,
)


def compute_address_features(
    s1_rec: Dict[str, Any], cand_rec: Dict[str, Any]
) -> Dict[str, float]:
    """
    Computes pairwise address features between Source 1 record and Candidate record.
    Returns a dictionary of float values. Missing addresses default cleanly to 0.0.
    """
    a1 = s1_rec.get("normalized_address", "") or ""
    a2 = cand_rec.get("normalized_address", "") or ""

    t1 = s1_rec.get("address_tokens", [])
    if not t1 and a1:
        t1 = a1.split()
    t2 = cand_rec.get("address_tokens", [])
    if not t2 and a2:
        t2 = a2.split()

    len_s1 = len(a1)
    len_c = len(a2)
    tok_len_s1 = len(t1)
    tok_len_c = len(t2)

    has_both_addrs = bool(a1 and a2)
    exact_match = 1.0 if (has_both_addrs and a1 == a2) else 0.0

    if has_both_addrs:
        set1 = set(t1)
        set2 = set(t2)
        shared_tokens = set1 & set2
        max_len = max(len_s1, len_c)
        len_ratio = round(min(len_s1, len_c) / max_len, 4) if max_len > 0 else 1.0

        jaccard = token_jaccard(t1, t2)
        overlap = token_overlap_coefficient(t1, t2)
        lev_sim = normalized_levenshtein_similarity(a1, a2)

        # Numeric tokens (street / building numbers normalized of leading zeros)
        nums1 = extract_numeric_tokens(t1)
        nums2 = extract_numeric_tokens(t2)
        shared_nums = set(nums1) & set(nums2)
        shared_numeric_count = float(len(shared_nums))
        num_overlap = token_overlap_coefficient(nums1, nums2) if (nums1 and nums2) else 0.0

        # Primary street number agreement (first numeric token in address)
        primary_match = 1.0 if (nums1 and nums2 and nums1[0] == nums2[0]) else 0.0

        # Address anchor agreement: shared number + at least one shared distinctive locality token
        distinct_words1 = {w for w in t1 if len(w) >= 4 and not w.isdigit() and w not in DEFAULT_ADDRESS_STOPWORDS}
        distinct_words2 = {w for w in t2 if len(w) >= 4 and not w.isdigit() and w not in DEFAULT_ADDRESS_STOPWORDS}
        anchor_match = 1.0 if (shared_numeric_count > 0 and (distinct_words1 & distinct_words2)) else 0.0

    else:
        shared_tokens = set()
        len_ratio = 0.0
        jaccard = 0.0
        overlap = 0.0
        lev_sim = 0.0
        shared_numeric_count = 0.0
        num_overlap = 0.0
        primary_match = 0.0
        anchor_match = 0.0

    return {
        "address_exact_match": exact_match,
        "address_token_jaccard": jaccard,
        "address_token_overlap_coefficient": overlap,
        "address_shared_token_count": float(len(shared_tokens)),
        "address_token_count_s1": float(tok_len_s1),
        "address_token_count_candidate": float(tok_len_c),
        "address_token_count_difference": float(abs(tok_len_s1 - tok_len_c)),
        "address_length_s1": float(len_s1),
        "address_length_candidate": float(len_c),
        "address_length_difference": float(abs(len_s1 - len_c)),
        "address_length_ratio": len_ratio,
        "address_levenshtein_sim": lev_sim,
        "address_shared_numeric_token_count": shared_numeric_count,
        "address_numeric_token_overlap": num_overlap,
        "address_primary_number_match": primary_match,
        "address_anchor_agreement": anchor_match,
    }
