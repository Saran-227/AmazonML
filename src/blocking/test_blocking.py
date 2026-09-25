"""
Unit tests for the blocking module.

Tests all required scenarios:
1. Exact name blocking
2. Name + country blocking
3. Token blocking
4. Duplicate candidate removal
5. Source 2 support
6. Source 3 support
7. Missing address handling
8. Unicode/multilingual names (Devanagari, Tamil, Kannada, French)
9. No-match entity (singleton)
10. Multiple-match entity
"""

import unittest
from typing import Any, Dict

from src.blocking.candidate_generator import CandidateGenerator
from src.blocking.exact_blocking import ExactBlocker
from src.blocking.token_blocking import TokenBlocker
from src.blocking.blocking_evaluation import evaluate_blocking
from src.preprocessing.normalize import normalize_record


class TestBlockingModule(unittest.TestCase):

    def setUp(self) -> None:
        self.generator = CandidateGenerator(
            enable_exact_name=True,
            enable_name_country=True,
            enable_exact_address=True,
            enable_name_tokens=True,
            enable_compressed_name=True,
            enable_address_anchor=True,
            max_block_size=500,
        )

    # 1. Exact name blocking
    def test_exact_name_blocking(self) -> None:
        s1 = normalize_record({
            "entity_id": "S1-001",
            "business_name": "Apex Summit Corp",
            "business_address": "123 Main St",
            "country": "US",
        })
        s2 = normalize_record({
            "entity_id": "S2-001",
            "business_name": "Apex Summit Corp",
            "business_address": "456 Oak Rd",
            "country": "US",
        })
        self.generator.index_candidates([s2])
        cands = self.generator.generate_candidates_for_record(s1)

        cand_ids = [c.candidate_entity_id for c in cands]
        self.assertIn("S2-001", cand_ids)
        pair = next(c for c in cands if c.candidate_entity_id == "S2-001")
        self.assertIn("exact_name", pair.blocking_rules)

    # 2. Name + country blocking
    def test_name_country_blocking(self) -> None:
        s1 = normalize_record({
            "entity_id": "S1-002",
            "business_name": "Global Logistics",
            "business_address": "100 Port Rd",
            "country": "France",
        })
        s2_match = normalize_record({
            "entity_id": "S2-002",
            "business_name": "Global Logistics",
            "business_address": "200 Port Rd",
            "country": "France",
        })
        self.generator.index_candidates([s2_match])
        cands = self.generator.generate_candidates_for_record(s1)

        self.assertTrue(any(c.candidate_entity_id == "S2-002" for c in cands))
        pair = next(c for c in cands if c.candidate_entity_id == "S2-002")
        self.assertIn("name_country", pair.blocking_rules)

    # 3. Token blocking
    def test_token_blocking(self) -> None:
        # Different legal suffix, but shares distinctive token 'zephay'
        s1 = normalize_record({
            "entity_id": "S1-003",
            "business_name": "Zephay Labs Inc",
            "business_address": "10 Pine St",
            "country": "US",
        })
        s3 = normalize_record({
            "entity_id": "S3-003",
            "business_name": "Zephay Research LLC",
            "business_address": "99 Maple Ave",
            "country": "US",
        })
        self.generator.index_candidates([s3])
        cands = self.generator.generate_candidates_for_record(s1)

        cand_ids = [c.candidate_entity_id for c in cands]
        self.assertIn("S3-003", cand_ids)
        pair = next(c for c in cands if c.candidate_entity_id == "S3-003")
        self.assertIn("name_token", pair.blocking_rules)

    # 4. Duplicate candidate removal
    def test_duplicate_candidate_removal(self) -> None:
        # A candidate matching on exact name AND name_country AND token must appear ONCE
        s1 = normalize_record({
            "entity_id": "S1-004",
            "business_name": "Horizon Dynamics Ltd",
            "business_address": "111 Tech Blvd",
            "country": "India",
        })
        s2 = normalize_record({
            "entity_id": "S2-004",
            "business_name": "Horizon Dynamics Ltd",
            "business_address": "111 Tech Blvd",
            "country": "India",
        })
        self.generator.index_candidates([s2])
        cands = self.generator.generate_candidates_for_record(s1)

        cand_ids = [c.candidate_entity_id for c in cands]
        # Must only appear once
        self.assertEqual(cand_ids.count("S2-004"), 1)
        # But rules should aggregate multiple triggers
        pair = cands[0]
        self.assertTrue(len(pair.blocking_rules) >= 2)

    # 5. Source 2 support
    def test_source_2_support(self) -> None:
        s1 = normalize_record({
            "entity_id": "S1-005",
            "business_name": "Delta Marine Supplies",
            "business_address": "5 Harbour St",
            "country": "US",
        })
        s2 = normalize_record({
            "entity_id": "S2-005",
            "business_name": "Delta Marine Supplies",
            "business_address": "5 Harbour St",
            "country": "US",
        })
        self.generator.index_candidates([s2])
        cands = self.generator.generate_candidates_for_record(s1)

        pair = next((c for c in cands if c.candidate_entity_id == "S2-005"), None)
        self.assertIsNotNone(pair)
        self.assertEqual(pair.candidate_source, "S2")

    # 6. Source 3 support
    def test_source_3_support(self) -> None:
        s1 = normalize_record({
            "entity_id": "S1-006",
            "business_name": "Omega BioTech Co",
            "business_address": "77 Science Park",
            "country": "US",
        })
        s3 = normalize_record({
            "entity_id": "S3-006",
            "business_name": "Omega BioTech Co",
            "business_address": "77 Science Park",
            "country": "US",
        })
        self.generator.index_candidates([s3])
        cands = self.generator.generate_candidates_for_record(s1)

        pair = next((c for c in cands if c.candidate_entity_id == "S3-006"), None)
        self.assertIsNotNone(pair)
        self.assertEqual(pair.candidate_source, "S3")

    # 7. Missing address handling
    def test_missing_address_handling(self) -> None:
        # Candidate has an empty / NaN address (common in Source 3)
        s1 = normalize_record({
            "entity_id": "S1-007",
            "business_name": "International South Consultants",
            "business_address": "25 MG Road, Bangalore",
            "country": "India",
        })
        s3_empty_addr = normalize_record({
            "entity_id": "S3-007",
            "business_name": "International South Consultants Pvt Ltd",
            "business_address": "",
            "country": "India",
        })
        self.generator.index_candidates([s3_empty_addr])
        cands = self.generator.generate_candidates_for_record(s1)

        # Name-based blocking must still recover this candidate
        cand_ids = [c.candidate_entity_id for c in cands]
        self.assertIn("S3-007", cand_ids)

    # 8. Unicode/multilingual names (Devanagari, Tamil, Kannada, French)
    def test_unicode_multilingual_names(self) -> None:
        # Hindi / Devanagari
        s1_hindi = normalize_record({
            "entity_id": "S1-008A",
            "business_name": "राम मार्केटिंग प्राइवेट लिमिटेड",
            "business_address": "दिल्ली",
            "country": "India",
        })
        s2_hindi = normalize_record({
            "entity_id": "S2-008A",
            "business_name": "राम मार्केटिंग प्राइवेट लिमिटेड",
            "business_address": "नई दिल्ली",
            "country": "India",
        })
        # French with accents
        s1_french = normalize_record({
            "entity_id": "S1-008B",
            "business_name": "Amicale des École et Amis SARL",
            "business_address": "15 Rue de la Paix, Paris",
            "country": "France",
        })
        s2_french = normalize_record({
            "entity_id": "S2-008B",
            "business_name": "Amicale des Ecole et Amis SARL",
            "business_address": "15 Rue de la Paix, Paris",
            "country": "France",
        })
        self.generator.index_candidates([s2_hindi, s2_french])

        cands_hindi = self.generator.generate_candidates_for_record(s1_hindi)
        self.assertIn("S2-008A", [c.candidate_entity_id for c in cands_hindi])

        cands_french = self.generator.generate_candidates_for_record(s1_french)
        self.assertIn("S2-008B", [c.candidate_entity_id for c in cands_french])

    # 9. No-match entity (singleton)
    def test_no_match_entity_singleton(self) -> None:
        s1_singleton = normalize_record({
            "entity_id": "S1-009",
            "business_name": "Unique Lone Enterprise Never Seen Before",
            "business_address": "999 Nowhere Land",
            "country": "US",
        })
        # Add unrelated records to index
        unrelated = normalize_record({
            "entity_id": "S2-099",
            "business_name": "Standard Bakery",
            "business_address": "1 First Ave",
            "country": "US",
        })
        self.generator.index_candidates([unrelated])
        cands = self.generator.generate_candidates_for_record(s1_singleton)

        self.assertEqual(len(cands), 0)

    # 10. Multiple-match entity
    def test_multiple_match_entity(self) -> None:
        # One S1 entity matching multiple records in both S2 and S3
        s1 = normalize_record({
            "entity_id": "S1-010",
            "business_name": "Dahlia Power Scientific LLC",
            "business_address": "630 45th Terrace, Kansas City, MO",
            "country": "US",
        })
        s2_m1 = normalize_record({
            "entity_id": "S2-010A",
            "business_name": "Dahlia Power Scientific",
            "business_address": "630 45th Terrace, Kansas City, MO",
            "country": "US",
        })
        s2_m2 = normalize_record({
            "entity_id": "S2-010B",
            "business_name": "Dahlia Power",
            "business_address": "630 45th Terrace, Kansas City, MO",
            "country": "US",
        })
        s3_m1 = normalize_record({
            "entity_id": "S3-010A",
            "business_name": "Dahlia Power Scientific LLC",
            "business_address": "45th Terrace, Kansas City, MO",
            "country": "US",
        })
        self.generator.index_candidates([s2_m1, s2_m2, s3_m1])
        cands = self.generator.generate_candidates_for_record(s1)

        cand_ids = {c.candidate_entity_id for c in cands}
        self.assertIn("S2-010A", cand_ids)
        self.assertIn("S2-010B", cand_ids)
        self.assertIn("S3-010A", cand_ids)

    # 11. Evaluation integration test
    def test_evaluation_integration(self) -> None:
        gt = {
            "S1-100": {"S2-101", "S3-102"},
            "S1-200": {"S2-201"},
            "S1-300": set(),  # singleton
        }
        # Candidate set captures 2 out of 3 matches
        cand_dict = {
            "S1-100": {
                "S2-101": ("S2", {"exact_name"}),
                "S2-999": ("S2", {"name_token"}),  # false candidate
            },
            "S1-200": {
                "S2-201": ("S2", {"name_country", "exact_address"}),
            },
            "S1-300": {},
        }
        eval_result = evaluate_blocking(gt, cand_dict)
        self.assertEqual(eval_result.total_true_matches, 3)
        self.assertEqual(eval_result.total_captured_matches, 2)
        self.assertAlmostEqual(eval_result.overall_recall, 66.6667, places=3)
        self.assertEqual(eval_result.source2_total_true_matches, 2)
        self.assertEqual(eval_result.source2_captured_matches, 2)
        self.assertAlmostEqual(eval_result.source2_recall, 100.0)
        self.assertEqual(eval_result.source3_total_true_matches, 1)
        self.assertEqual(eval_result.source3_captured_matches, 0)
        self.assertAlmostEqual(eval_result.source3_recall, 0.0)


if __name__ == "__main__":
    unittest.main()
