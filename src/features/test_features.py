"""
Unit tests for the pairwise feature engineering module.

Tests all 20 required scenarios:
1. Identical names
2. Different names
3. Partial token overlap
4. Identical addresses
5. Partial address overlap
6. Shared numeric tokens
7. Same country
8. Different country
9. Missing address
10. Missing name
11. Unicode/Devanagari
12. Tamil
13. Kannada
14. French accented text
15. Source 2 candidate
16. Source 3 candidate
17. Multiple blocking rules
18. True match
19. Hard negative
20. No-match/singleton case
"""

import unittest
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.blocking.candidate_generator import CandidatePair
from src.features.pair_features import (
    build_feature_dataset,
    compute_pair_features,
)
from src.preprocessing.normalize import normalize_record


class TestPairFeatures(unittest.TestCase):

    # 1. Identical names
    def test_identical_names(self) -> None:
        s1 = normalize_record({"entity_id": "S1-1", "business_name": "Apex Summit Corp", "business_address": "1 Main St", "country": "US"})
        c2 = normalize_record({"entity_id": "S2-1", "business_name": "Apex Summit Corp", "business_address": "2 Oak Rd", "country": "US"})
        feats = compute_pair_features(s1, c2)
        self.assertEqual(feats["name_exact_match"], 1.0)
        self.assertEqual(feats["name_token_jaccard"], 1.0)
        self.assertEqual(feats["name_levenshtein_sim"], 1.0)
        self.assertEqual(feats["name_jaro_winkler"], 1.0)

    # 2. Different names
    def test_different_names(self) -> None:
        s1 = normalize_record({"entity_id": "S1-2", "business_name": "Apex Summit Corp", "business_address": "1 Main St", "country": "US"})
        c2 = normalize_record({"entity_id": "S2-2", "business_name": "Blue Horizon Bakery", "business_address": "1 Main St", "country": "US"})
        feats = compute_pair_features(s1, c2)
        self.assertEqual(feats["name_exact_match"], 0.0)
        self.assertEqual(feats["name_token_jaccard"], 0.0)
        self.assertLess(feats["name_jaro_winkler"], 0.6)

    # 3. Partial token overlap
    def test_partial_token_overlap(self) -> None:
        s1 = normalize_record({"entity_id": "S1-3", "business_name": "Apex Summit Global Logistics", "business_address": "1 St", "country": "US"})
        c2 = normalize_record({"entity_id": "S2-3", "business_name": "Apex Summit Express", "business_address": "1 St", "country": "US"})
        feats = compute_pair_features(s1, c2)
        self.assertEqual(feats["name_exact_match"], 0.0)
        self.assertGreater(feats["name_token_jaccard"], 0.0)
        self.assertLess(feats["name_token_jaccard"], 1.0)
        self.assertEqual(feats["name_shared_token_count"], 2.0)  # 'apex', 'summit'

    # 4. Identical addresses
    def test_identical_addresses(self) -> None:
        s1 = normalize_record({"entity_id": "S1-4", "business_name": "Company A", "business_address": "1795 Westchester Drive, High Point, NC", "country": "US"})
        c2 = normalize_record({"entity_id": "S2-4", "business_name": "Company B", "business_address": "1795 Westchester Drive, High Point, NC", "country": "US"})
        feats = compute_pair_features(s1, c2)
        self.assertEqual(feats["address_exact_match"], 1.0)
        self.assertEqual(feats["address_token_jaccard"], 1.0)
        self.assertEqual(feats["address_primary_number_match"], 1.0)

    # 5. Partial address overlap
    def test_partial_address_overlap(self) -> None:
        s1 = normalize_record({"entity_id": "S1-5", "business_name": "A", "business_address": "1795 Westchester Drive, High Point, NC", "country": "US"})
        c2 = normalize_record({"entity_id": "S2-5", "business_name": "A", "business_address": "Westchester Drive, High Point", "country": "US"})
        feats = compute_pair_features(s1, c2)
        self.assertEqual(feats["address_exact_match"], 0.0)
        self.assertGreater(feats["address_token_jaccard"], 0.0)
        self.assertLess(feats["address_token_jaccard"], 1.0)

    # 6. Shared numeric tokens
    def test_shared_numeric_tokens(self) -> None:
        s1 = normalize_record({"entity_id": "S1-6", "business_name": "A", "business_address": "1056 Belden Ave", "country": "US"})
        # 01056 normalizes to 1056
        c2 = normalize_record({"entity_id": "S2-6", "business_name": "A", "business_address": "01056 Belden Avenue", "country": "US"})
        feats = compute_pair_features(s1, c2)
        self.assertEqual(feats["address_shared_numeric_token_count"], 1.0)
        self.assertEqual(feats["address_primary_number_match"], 1.0)

    # 7. Same country
    def test_same_country(self) -> None:
        s1 = normalize_record({"entity_id": "S1-7", "business_name": "A", "business_address": "X", "country": "France"})
        c2 = normalize_record({"entity_id": "S2-7", "business_name": "A", "business_address": "X", "country": "France"})
        feats = compute_pair_features(s1, c2)
        self.assertEqual(feats["country_exact_match"], 1.0)
        self.assertEqual(feats["country_mismatch"], 0.0)

    # 8. Different country
    def test_different_country(self) -> None:
        s1 = normalize_record({"entity_id": "S1-8", "business_name": "A", "business_address": "X", "country": "US"})
        c2 = normalize_record({"entity_id": "S2-8", "business_name": "A", "business_address": "X", "country": "India"})
        feats = compute_pair_features(s1, c2)
        self.assertEqual(feats["country_exact_match"], 0.0)
        self.assertEqual(feats["country_mismatch"], 1.0)

    # 9. Missing address
    def test_missing_address(self) -> None:
        s1 = normalize_record({"entity_id": "S1-9", "business_name": "A", "business_address": "123 Main St", "country": "US"})
        c3 = normalize_record({"entity_id": "S3-9", "business_name": "A", "business_address": "", "country": "US"})
        feats = compute_pair_features(s1, c3)
        self.assertEqual(feats["address_missing_candidate"], 1.0)
        self.assertEqual(feats["address_missing_s1"], 0.0)
        self.assertEqual(feats["address_exact_match"], 0.0)
        self.assertEqual(feats["address_token_jaccard"], 0.0)

    # 10. Missing name
    def test_missing_name(self) -> None:
        s1 = normalize_record({"entity_id": "S1-10", "business_name": "", "business_address": "123 Main St", "country": "US"})
        c2 = normalize_record({"entity_id": "S2-10", "business_name": "Tech Co", "business_address": "123 Main St", "country": "US"})
        feats = compute_pair_features(s1, c2)
        self.assertEqual(feats["name_missing_s1"], 1.0)
        self.assertEqual(feats["name_missing_candidate"], 0.0)
        self.assertEqual(feats["name_exact_match"], 0.0)

    # 11. Unicode / Devanagari
    def test_unicode_devanagari(self) -> None:
        s1 = normalize_record({"entity_id": "S1-11", "business_name": "राम मार्केटिंग प्राइवेट लिमिटेड", "business_address": "नई दिल्ली", "country": "India"})
        c2 = normalize_record({"entity_id": "S2-11", "business_name": "राम मार्केटिंग प्राइवेट लिमिटेड", "business_address": "दिल्ली", "country": "India"})
        feats = compute_pair_features(s1, c2)
        self.assertEqual(feats["name_exact_match"], 1.0)
        self.assertEqual(feats["script_match"], 1.0)
        self.assertEqual(feats["is_cross_script"], 0.0)
        self.assertEqual(feats["has_indic_or_non_ascii"], 1.0)

    # 12. Tamil
    def test_tamil(self) -> None:
        s1 = normalize_record({"entity_id": "S1-12", "business_name": "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ்", "business_address": "சென்னை", "country": "India"})
        c2 = normalize_record({"entity_id": "S2-12", "business_name": "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ்", "business_address": "சென்னை", "country": "India"})
        feats = compute_pair_features(s1, c2)
        self.assertEqual(feats["name_exact_match"], 1.0)
        self.assertEqual(feats["has_indic_or_non_ascii"], 1.0)

    # 13. Kannada
    def test_kannada(self) -> None:
        s1 = normalize_record({"entity_id": "S1-13", "business_name": "ಸ್ಕೈ ಡೆವಲಪರ್ಸ್", "business_address": "ಬೆಂಗಳೂರು", "country": "India"})
        c2 = normalize_record({"entity_id": "S2-13", "business_name": "ಸ್ಕೈ ಡೆವಲಪರ್ಸ್", "business_address": "ಬೆಂಗಳೂರು", "country": "India"})
        feats = compute_pair_features(s1, c2)
        self.assertEqual(feats["name_exact_match"], 1.0)
        self.assertEqual(feats["has_indic_or_non_ascii"], 1.0)

    # 14. French accented text
    def test_french_accented_text(self) -> None:
        s1 = normalize_record({"entity_id": "S1-14", "business_name": "Amicale des École SARL", "business_address": "Paris", "country": "France"})
        c2 = normalize_record({"entity_id": "S2-14", "business_name": "Amicale des Ecole SARL", "business_address": "Paris", "country": "France"})
        feats = compute_pair_features(s1, c2)
        # Saran's preprocessing folds Latin accents so normalized names are identical
        self.assertEqual(feats["name_exact_match"], 1.0)
        self.assertEqual(feats["country_exact_match"], 1.0)

    # 15. Source 2 candidate
    def test_source2_candidate(self) -> None:
        s1 = normalize_record({"entity_id": "S1-15", "business_name": "A", "business_address": "B", "country": "US"})
        c2 = normalize_record({"entity_id": "S2-15", "business_name": "A", "business_address": "B", "country": "US"})
        feats = compute_pair_features(s1, c2, candidate_source="S2")
        self.assertEqual(feats["is_source2"], 1.0)
        self.assertEqual(feats["is_source3"], 0.0)

    # 16. Source 3 candidate
    def test_source3_candidate(self) -> None:
        s1 = normalize_record({"entity_id": "S1-16", "business_name": "A", "business_address": "B", "country": "US"})
        c3 = normalize_record({"entity_id": "S3-16", "business_name": "A", "business_address": "B", "country": "US"})
        feats = compute_pair_features(s1, c3, candidate_source="S3")
        self.assertEqual(feats["is_source2"], 0.0)
        self.assertEqual(feats["is_source3"], 1.0)

    # 17. Multiple blocking rules
    def test_multiple_blocking_rules(self) -> None:
        s1 = normalize_record({"entity_id": "S1-17", "business_name": "A", "business_address": "B", "country": "US"})
        c2 = normalize_record({"entity_id": "S2-17", "business_name": "A", "business_address": "B", "country": "US"})
        rules = {"exact_name", "name_country", "name_token"}
        feats = compute_pair_features(s1, c2, blocking_rules=rules)
        self.assertEqual(feats["block_exact_name"], 1.0)
        self.assertEqual(feats["block_name_country"], 1.0)
        self.assertEqual(feats["block_name_token"], 1.0)
        self.assertEqual(feats["block_address_anchor"], 0.0)
        self.assertEqual(feats["blocking_rule_count"], 3.0)

    # 18. True match
    def test_true_match(self) -> None:
        s1 = normalize_record({"entity_id": "S1-18", "business_name": "Red Ventures Private Limited", "business_address": "Jaipur Rajasthan", "country": "India"})
        c2 = normalize_record({"entity_id": "S2-18", "business_name": "Red Ventures Private", "business_address": "Jaipur Rajasthan", "country": "India"})
        gt = {"S1-18": {"S2-18"}}

        pairs = [CandidatePair("S1-18", "S2-18", "S2", {"name_token", "exact_address"})]
        X, meta, y = build_feature_dataset(pairs, {"S1-18": s1}, {"S2-18": c2}, ground_truth_map=gt)

        self.assertEqual(len(X), 1)
        self.assertEqual(y[0], 1)
        self.assertGreater(X.loc[0, "name_token_jaccard"], 0.5)

    # 19. Hard negative
    def test_hard_negative(self) -> None:
        # Same franchise name, same city, but different street address
        s1 = normalize_record({"entity_id": "S1-19", "business_name": "Subway Sandwiches", "business_address": "100 Broadway, New York, NY", "country": "US"})
        c2 = normalize_record({"entity_id": "S2-19", "business_name": "Subway Sandwiches", "business_address": "500 5th Ave, New York, NY", "country": "US"})
        gt = {"S1-19": set()}  # Not a match

        pairs = [CandidatePair("S1-19", "S2-19", "S2", {"exact_name", "name_country"})]
        X, meta, y = build_feature_dataset(pairs, {"S1-19": s1}, {"S2-19": c2}, ground_truth_map=gt)

        self.assertEqual(y[0], 0)
        self.assertEqual(X.loc[0, "name_exact_match"], 1.0)
        self.assertEqual(X.loc[0, "address_exact_match"], 0.0)
        self.assertEqual(X.loc[0, "name_exact_and_address_exact"], 0.0)

    # 20. No-match / singleton case
    def test_no_match_singleton(self) -> None:
        s1 = normalize_record({"entity_id": "S1-20", "business_name": "Unique Lone Bakery", "business_address": "Remote Road", "country": "US"})
        # No candidate pairs generated
        pairs = []
        X, meta, y = build_feature_dataset(pairs, {"S1-20": s1}, {}, ground_truth_map={"S1-20": set()})
        self.assertEqual(len(X), 0)
        self.assertEqual(len(meta), 0)
        self.assertEqual(len(y), 0)


if __name__ == "__main__":
    unittest.main()
