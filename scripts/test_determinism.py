"""
Sprint Step 11: Programmatic Pipeline Determinism Verification.

Runs the production pipeline twice on identical input:
- Run 1 -> matching_results_run1.tsv, candidate_pairs_run1.tsv
- Run 2 -> matching_results_run2.tsv, candidate_pairs_run2.tsv

Verifies that SHA-256 cryptographic hashes are 100% bitwise IDENTICAL.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import pandas as pd

from src.blocking.candidate_generator import CandidateGenerator
from src.features.pair_features import compute_pair_features
from src.preprocessing.normalize import normalize_record

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_pipeline_deterministic(
    s1_records: list,
    candidate_records: dict,
    model: any,
    out_matching: Path,
    out_candidate: Path,
    threshold: float = 0.95,
):
    gen = CandidateGenerator(max_block_size=500)
    gen.index_candidates(candidate_records.values())

    predictions = {}
    for s1_rec in s1_records:
        s1_id = s1_rec["entity_id"]
        pairs = gen.generate_candidates_for_record(s1_rec, max_candidates=30)
        if not pairs:
            predictions[s1_id] = ([], [])
            continue

        feats = [
            compute_pair_features(
                s1_rec,
                candidate_records[p.candidate_entity_id],
                p.blocking_rules,
                p.candidate_source,
            )
            for p in pairs
        ]
        X = pd.DataFrame(feats)
        proba = model.predict_proba(X)[:, 1]

        c_ids = [p.candidate_entity_id for p in pairs]
        matched_cids = [c_ids[i] for i, p_val in enumerate(proba) if p_val >= threshold]

        sorted_cands = sorted(list(set(c_ids)))
        sorted_matches = sorted(list(set(matched_cids)))
        predictions[s1_id] = (sorted_cands, sorted_matches)

    with open(out_matching, "w", encoding="utf-8") as f_match, \
         open(out_candidate, "w", encoding="utf-8") as f_cand:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_rec in s1_records:
            s1_id = s1_rec["entity_id"]
            cands, matches = predictions.get(s1_id, ([], []))
            f_match.write(f"{s1_id}\t{','.join(matches)}\n")
            f_cand.write(f"{s1_id}\t{','.join(cands)}\n")


def main():
    test_dir = REPO_ROOT / "student_resource" / "dataset" / "test"
    det_dir = REPO_ROOT / "reports" / "determinism_audit"
    det_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading test sample (1,000 S1 records)...")
    s1_df = pd.read_csv(test_dir / "test_source1.tsv", sep="\t", keep_default_na=False, nrows=1000)
    s1_records = [normalize_record(r.to_dict()) for _, r in s1_df.iterrows()]

    logger.info("Loading candidate pool...")
    s2_df = pd.read_csv(test_dir / "test_source2.tsv", sep="\t", keep_default_na=False, nrows=5000)
    s3_df = pd.read_csv(test_dir / "test_source3.tsv", sep="\t", keep_default_na=False, nrows=5000)

    candidate_records = {}
    for _, r in s2_df.iterrows():
        candidate_records[r["entity_id"]] = normalize_record(r.to_dict())
    for _, r in s3_df.iterrows():
        candidate_records[r["entity_id"]] = normalize_record(r.to_dict())

    model_path = REPO_ROOT / "src" / "models" / "production_model.joblib"
    model = joblib.load(model_path)

    # RUN 1
    logger.info("Executing Run 1...")
    m_run1 = det_dir / "matching_results_run1.tsv"
    c_run1 = det_dir / "candidate_pairs_run1.tsv"
    run_pipeline_deterministic(s1_records, candidate_records, model, m_run1, c_run1)
    hash_m1 = compute_sha256(m_run1)
    hash_c1 = compute_sha256(c_run1)

    # RUN 2
    logger.info("Executing Run 2...")
    m_run2 = det_dir / "matching_results_run2.tsv"
    c_run2 = det_dir / "candidate_pairs_run2.tsv"
    run_pipeline_deterministic(s1_records, candidate_records, model, m_run2, c_run2)
    hash_m2 = compute_sha256(m_run2)
    hash_c2 = compute_sha256(c_run2)

    logger.info("Run 1 matching SHA-256:  %s", hash_m1)
    logger.info("Run 2 matching SHA-256:  %s", hash_m2)
    logger.info("Run 1 candidate SHA-256: %s", hash_c1)
    logger.info("Run 2 candidate SHA-256: %s", hash_c2)

    assert hash_m1 == hash_m2, f"Determinism failure in matching_results! {hash_m1} != {hash_m2}"
    assert hash_c1 == hash_c2, f"Determinism failure in candidate_pairs! {hash_c1} != {hash_c2}"

    logger.info("DETERMINISM AUDIT PASSED: 100% BITWISE IDENTICAL OUTPUTS.")

    # Save summary report
    report = {
        "status": "PASS",
        "matching_results_sha256_run1": hash_m1,
        "matching_results_sha256_run2": hash_m2,
        "matching_results_identical": (hash_m1 == hash_m2),
        "candidate_pairs_sha256_run1": hash_c1,
        "candidate_pairs_sha256_run2": hash_c2,
        "candidate_pairs_identical": (hash_c1 == hash_c2),
        "full_submission_matching_sha256": compute_sha256(REPO_ROOT / "student_resource" / "output" / "matching_results.tsv"),
        "full_submission_candidate_sha256": compute_sha256(REPO_ROOT / "student_resource" / "output" / "candidate_pairs.tsv"),
    }
    with open(det_dir / "determinism_report.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
