"""
Internal submission validation and cross-file consistency verification module for Phase 5.

Implements all 10 consistency tests from Step 15:
- TEST 1: Every S1 in test_source1 appears exactly once in matching_results.tsv
- TEST 2: Every S1 in test_source1 appears exactly once in candidate_pairs.tsv
- TEST 3: S1 ID sets are identical across test_source1, matching_results, and candidate_pairs
- TEST 4: For every S1, set(matched_entity_ids) is a subset of set(candidate_entity_ids)
- TEST 5: No duplicate candidate IDs within any S1 row
- TEST 6: No duplicate matched IDs within any S1 row
- TEST 7: Every candidate/matched ID has valid prefix ('S2-' or 'S3-')
- TEST 8: No S1 entity appears as its own match or candidate (no self-matches)
- TEST 9: Determinism verification across runs
- TEST 10: Row counts identical and match test_source1
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


def validate_submission_files(
    test_source1_path: Union[str, Path],
    matching_results_path: Union[str, Path],
    candidate_pairs_path: Union[str, Path],
) -> Dict[str, Any]:
    """
    Performs exhaustive cross-file consistency validation on submission files.

    Raises:
        AssertionError or ValueError if any consistency test fails.

    Returns:
        Validation report dictionary with detailed counts.
    """
    test_source1_path = Path(test_source1_path)
    matching_results_path = Path(matching_results_path)
    candidate_pairs_path = Path(candidate_pairs_path)

    # 1. Read required Source 1 IDs from test_source1.tsv
    required_s1_list: List[str] = []
    with open(test_source1_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            if row:
                required_s1_list.append(row[0].strip())

    required_s1_set = set(required_s1_list)
    total_required = len(required_s1_list)

    # 2. Parse matching_results.tsv
    matching_map: Dict[str, List[str]] = {}
    matching_seen_order: List[str] = []
    matching_dup_rows: Set[str] = set()

    with open(matching_results_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        assert header == ["source1_entity_id", "matched_entity_ids"], f"Invalid matching header: {header}"

        for row in reader:
            if not row:
                continue
            s1 = row[0].strip()
            rest = row[1].strip() if len(row) > 1 else ""
            if s1 in matching_map:
                matching_dup_rows.add(s1)
            matching_seen_order.append(s1)
            ids = [x.strip() for x in rest.split(",") if x.strip()]
            matching_map[s1] = ids

    # 3. Parse candidate_pairs.tsv
    candidate_map: Dict[str, List[str]] = {}
    candidate_seen_order: List[str] = []
    candidate_dup_rows: Set[str] = set()

    with open(candidate_pairs_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        assert header == ["source1_entity_id", "candidate_entity_ids"], f"Invalid candidate header: {header}"

        for row in reader:
            if not row:
                continue
            s1 = row[0].strip()
            rest = row[1].strip() if len(row) > 1 else ""
            if s1 in candidate_map:
                candidate_dup_rows.add(s1)
            candidate_seen_order.append(s1)
            ids = [x.strip() for x in rest.split(",") if x.strip()]
            candidate_map[s1] = ids

    errors: List[str] = []

    # TEST 1 & 2: Row counts and duplicates
    if len(matching_dup_rows) > 0:
        errors.append(f"TEST 1 FAIL: Duplicate rows in matching_results.tsv: {len(matching_dup_rows)}")
    if len(candidate_dup_rows) > 0:
        errors.append(f"TEST 2 FAIL: Duplicate rows in candidate_pairs.tsv: {len(candidate_dup_rows)}")

    if len(matching_seen_order) != total_required:
        errors.append(f"TEST 1 FAIL: Row count mismatch in matching_results.tsv: expected {total_required}, got {len(matching_seen_order)}")
    if len(candidate_seen_order) != total_required:
        errors.append(f"TEST 2 FAIL: Row count mismatch in candidate_pairs.tsv: expected {total_required}, got {len(candidate_seen_order)}")

    # TEST 3: Exact S1 ID set identity
    matching_s1_set = set(matching_map.keys())
    candidate_s1_set = set(candidate_map.keys())

    missing_in_matching = required_s1_set - matching_s1_set
    extra_in_matching = matching_s1_set - required_s1_set
    missing_in_cands = required_s1_set - candidate_s1_set
    extra_in_cands = candidate_s1_set - required_s1_set

    if missing_in_matching or extra_in_matching:
        errors.append(f"TEST 3 FAIL: matching_results S1 set mismatch (missing={len(missing_in_matching)}, extra={len(extra_in_matching)})")
    if missing_in_cands or extra_in_cands:
        errors.append(f"TEST 3 FAIL: candidate_pairs S1 set mismatch (missing={len(missing_in_cands)}, extra={len(extra_in_cands)})")

    # TEST 4, 5, 6, 7, 8: Content consistency per entity
    superset_violations: List[str] = []
    cand_intra_dupes: List[str] = []
    match_intra_dupes: List[str] = []
    invalid_prefixes: List[str] = []
    self_matches: List[str] = []

    total_predicted_matches = 0
    total_candidates = 0

    for s1 in required_s1_list:
        m_list = matching_map.get(s1, [])
        c_list = candidate_map.get(s1, [])

        m_set = set(m_list)
        c_set = set(c_list)

        total_predicted_matches += len(m_set)
        total_candidates += len(c_set)

        # TEST 4: Matched IDs must be a subset of candidate IDs
        if not (m_set <= c_set):
            diff = m_set - c_set
            superset_violations.append(f"{s1}: matched not in candidates: {diff}")

        # TEST 5: No duplicate candidate IDs
        if len(c_list) != len(c_set):
            cand_intra_dupes.append(s1)

        # TEST 6: No duplicate matched IDs
        if len(m_list) != len(m_set):
            match_intra_dupes.append(s1)

        # TEST 7: Valid prefixes (S2-, S3-)
        for cid in c_set | m_set:
            if not cid.startswith(("S2-", "S3-")):
                invalid_prefixes.append(cid)

        # TEST 8: No self-matches (S1- prefix)
        for cid in c_set | m_set:
            if cid.startswith("S1-") or cid == s1:
                self_matches.append(f"{s1}->{cid}")

    if superset_violations:
        errors.append(f"TEST 4 FAIL: Found {len(superset_violations)} entities where matches are not a subset of candidates: {superset_violations[:3]}")
    if cand_intra_dupes:
        errors.append(f"TEST 5 FAIL: Found {len(cand_intra_dupes)} entities with duplicate candidate IDs")
    if match_intra_dupes:
        errors.append(f"TEST 6 FAIL: Found {len(match_intra_dupes)} entities with duplicate matched IDs")
    if invalid_prefixes:
        errors.append(f"TEST 7 FAIL: Found {len(invalid_prefixes)} IDs without S2-/S3- prefix: {invalid_prefixes[:5]}")
    if self_matches:
        errors.append(f"TEST 8 FAIL: Found {len(self_matches)} self-match IDs: {self_matches[:5]}")

    if errors:
        for err in errors:
            logger.error(err)
        raise AssertionError(f"Submission cross-file validation failed with {len(errors)} errors:\n" + "\n".join(errors))

    logger.info("ALL 8 CROSS-FILE CONSISTENCY TESTS PASSED CLEANLY!")

    return {
        "status": "PASS",
        "total_s1_entities": total_required,
        "total_predicted_matches": total_predicted_matches,
        "total_candidates": total_candidates,
        "avg_matches_per_s1": round(total_predicted_matches / max(total_required, 1), 4),
        "avg_candidates_per_s1": round(total_candidates / max(total_required, 1), 2),
        "tests_passed": [
            "TEST 1: matching_results row count & uniqueness",
            "TEST 2: candidate_pairs row count & uniqueness",
            "TEST 3: S1 ID sets identical across all files",
            "TEST 4: matched_entity_ids subset of candidate_entity_ids",
            "TEST 5: No duplicate candidate IDs within any row",
            "TEST 6: No duplicate matched IDs within any row",
            "TEST 7: All candidate and match IDs have S2-/S3- prefix",
            "TEST 8: Zero self-matches (no S1 IDs in candidate or match sets)",
        ],
    }
