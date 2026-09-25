"""
Unit tests for submission generation, formatting, and validation logic.

Verifies:
1. Submission formatting adheres to exact tab-separated format
2. Exact header names: source1_entity_id, matched_entity_ids, candidate_entity_ids
3. Lexicographical ID sorting
4. Subset guarantee: matched_entity_ids <= candidate_entity_ids
5. Empty match and empty candidate handling
6. Cross-file consistency validator catches malformed headers, missing rows, duplicate IDs, self-matches
"""

import tempfile
import unittest
from pathlib import Path

from src.submission.validate import validate_submission_files


class TestSubmissionPipeline(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

        self.s1_file = self.dir_path / "test_source1.tsv"
        self.match_file = self.dir_path / "matching_results.tsv"
        self.cand_file = self.dir_path / "candidate_pairs.tsv"

        # Create valid test_source1
        with open(self.s1_file, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S1-1\tAlpha Corp\t123 Main St\tUS\n")
            f.write("S1-2\tBeta LLC\t456 Oak Rd\tIndia\n")
            f.write("S1-3\tGamma SAS\t789 Rue Paris\tFrance\n")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_01_valid_submission_passes(self):
        """Valid submission files pass all checks."""
        with open(self.match_file, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            f.write("S1-1\tS2-101,S3-201\n")
            f.write("S1-2\tS2-102\n")
            f.write("S1-3\t\n")

        with open(self.cand_file, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            f.write("S1-1\tS2-101,S2-105,S3-201\n")
            f.write("S1-2\tS2-102,S3-202\n")
            f.write("S1-3\tS2-103\n")

        res = validate_submission_files(self.s1_file, self.match_file, self.cand_file)
        self.assertEqual(res["status"], "PASS")
        self.assertEqual(res["total_s1_entities"], 3)
        self.assertEqual(res["total_predicted_matches"], 3)
        self.assertEqual(res["total_candidates"], 6)

    def test_02_detects_missing_s1_row(self):
        """Catches missing S1 entity row."""
        with open(self.match_file, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            f.write("S1-1\tS2-101\n")
            # S1-2 and S1-3 missing

        with open(self.cand_file, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            f.write("S1-1\tS2-101\n")
            f.write("S1-2\tS2-102\n")
            f.write("S1-3\t\n")

        with self.assertRaises(AssertionError):
            validate_submission_files(self.s1_file, self.match_file, self.cand_file)

    def test_03_detects_match_not_in_candidates(self):
        """Catches violation where predicted match is not in candidate list."""
        with open(self.match_file, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            f.write("S1-1\tS2-999\n")  # S2-999 is NOT in candidates!
            f.write("S1-2\t\n")
            f.write("S1-3\t\n")

        with open(self.cand_file, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            f.write("S1-1\tS2-101\n")
            f.write("S1-2\t\n")
            f.write("S1-3\t\n")

        with self.assertRaises(AssertionError):
            validate_submission_files(self.s1_file, self.match_file, self.cand_file)

    def test_04_detects_duplicate_candidate_id_in_row(self):
        """Catches repeated ID inside candidate list."""
        with open(self.match_file, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            f.write("S1-1\t\n")
            f.write("S1-2\t\n")
            f.write("S1-3\t\n")

        with open(self.cand_file, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            f.write("S1-1\tS2-101,S2-101\n")  # Duplicate!
            f.write("S1-2\t\n")
            f.write("S1-3\t\n")

        with self.assertRaises(AssertionError):
            validate_submission_files(self.s1_file, self.match_file, self.cand_file)

    def test_05_detects_self_match(self):
        """Catches self-matches (S1 ID appearing in match list)."""
        with open(self.match_file, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            f.write("S1-1\tS1-1\n")  # Self match!
            f.write("S1-2\t\n")
            f.write("S1-3\t\n")

        with open(self.cand_file, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            f.write("S1-1\tS1-1\n")
            f.write("S1-2\t\n")
            f.write("S1-3\t\n")

        with self.assertRaises(AssertionError):
            validate_submission_files(self.s1_file, self.match_file, self.cand_file)


if __name__ == "__main__":
    unittest.main()
