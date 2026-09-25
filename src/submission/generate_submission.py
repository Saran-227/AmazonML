"""
Production submission generation script for Phase 5.

Connects the full pipeline:
Full Test Data -> Saran Preprocessing -> Phase-2 CandidateGenerator ->
Phase-3 60-dim Features -> Phase-4 HistGradientBoosting Model ->
Threshold 0.95 -> matching_results.tsv & candidate_pairs.tsv.

Guarantees:
- Exactly one row per test Source-1 entity in identical file order
- Matched IDs are a strict subset of candidate IDs
- Lexicographically sorted ID lists, comma-separated, no duplicates
- Empty matched/candidate fields for entities with no matches/candidates
- Output written in strict TSV format
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd

from src.blocking.candidate_generator import CandidateGenerator
from src.features.pair_features import compute_pair_features
from src.preprocessing.normalize import normalize_record

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_submission(
    test_dir: Union[str, Path],
    output_dir: Union[str, Path],
    model_path: Union[str, Path],
    threshold: float = 0.95,
    max_candidates_per_s1: int = 30,
    max_s1_records: Optional[int] = None,
    candidate_pool_limit_per_source: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Executes end-to-end test inference and generates submission TSV files.

    Args:
        test_dir: Directory containing test_source1.tsv, test_source2.tsv, test_source3.tsv
        output_dir: Directory where matching_results.tsv and candidate_pairs.tsv are saved
        model_path: Path to trained HistGradientBoosting model (joblib)
        threshold: Decision threshold for predicting a match (default: 0.95)
        max_candidates_per_s1: Maximum candidate pairs evaluated per S1 entity
        max_s1_records: Optional limit for testing/debugging
        candidate_pool_limit_per_source: Optional limit for candidate pool

    Returns:
        Summary statistics dictionary.
    """
    t_start = time.time()
    test_dir = Path(test_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matching_file = output_dir / "matching_results.tsv"
    candidate_file = output_dir / "candidate_pairs.tsv"

    # 1. Load production model
    logger.info("Loading production model from %s...", model_path)
    model = joblib.load(model_path)

    # 2. Read full list of required Source 1 IDs (ensuring all 1.73M S1 entities are in output)
    s1_path = test_dir / "test_source1.tsv"
    logger.info("Reading full list of test Source 1 entities from %s...", s1_path)
    required_s1_ids = []
    with open(s1_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            if row:
                required_s1_ids.append(row[0].strip())

    total_required = len(required_s1_ids)
    logger.info("Total required test S1 entities in test_source1.tsv: %d", total_required)

    # Read S1 entities for inference
    s1_df = pd.read_csv(s1_path, sep="\t", keep_default_na=False, nrows=max_s1_records)
    total_s1 = len(s1_df)
    logger.info("Running candidate generation and scoring on %d S1 entities.", total_s1)

    # 3. Read and normalize candidate pool (Source 2 and Source 3)
    s2_path = test_dir / "test_source2.tsv"
    s3_path = test_dir / "test_source3.tsv"

    logger.info("Loading and normalizing Source 2 candidates...")
    s2_df = pd.read_csv(s2_path, sep="\t", keep_default_na=False, nrows=candidate_pool_limit_per_source)
    candidate_records: Dict[str, Dict[str, Any]] = {}
    for _, r in s2_df.iterrows():
        candidate_records[r["entity_id"]] = normalize_record(r.to_dict())
    del s2_df
    gc.collect()

    logger.info("Loading and normalizing Source 3 candidates...")
    s3_df = pd.read_csv(s3_path, sep="\t", keep_default_na=False, nrows=candidate_pool_limit_per_source)
    for _, r in s3_df.iterrows():
        candidate_records[r["entity_id"]] = normalize_record(r.to_dict())
    del s3_df
    gc.collect()

    logger.info("Total normalized candidate pool: %d records.", len(candidate_records))

    # 4. Index candidate pool in CandidateGenerator
    logger.info("Indexing candidates in Phase-2 CandidateGenerator...")
    t_idx_start = time.time()
    gen = CandidateGenerator(max_block_size=500)
    gen.index_candidates(candidate_records.values())
    t_idx = time.time() - t_idx_start
    logger.info("Candidate indexing complete in %.2f seconds.", t_idx)

    # 5. Process Source 1 entities and run inference
    logger.info("Running candidate generation and matching inference for %d S1 entities...", total_s1)
    t_inf_start = time.time()

    predictions: Dict[str, Tuple[List[str], List[str], List[float]]] = {}
    total_candidates_generated = 0
    total_matches_predicted = 0

    empty_cands_cnt = 0
    empty_matches_cnt = 0
    singleton_matches_cnt = 0
    multimatch_matches_cnt = 0

    s2_matches_cnt = 0
    s3_matches_cnt = 0

    prob_brackets = {
        ">=0.50": 0,
        ">=0.70": 0,
        ">=0.80": 0,
        ">=0.90": 0,
        ">=0.95": 0,
        ">=0.99": 0,
    }

    high_confidence_audit: List[Dict[str, Any]] = []

    for idx, (_, r) in enumerate(s1_df.iterrows()):
        s1_id = r["entity_id"]
        s1_norm = normalize_record(r.to_dict())

        pairs = gen.generate_candidates_for_record(
            s1_norm, max_candidates=max_candidates_per_s1
        )

        if not pairs:
            predictions[s1_id] = ([], [], [])
            empty_cands_cnt += 1
            empty_matches_cnt += 1
            continue

        c_ids = [p.candidate_entity_id for p in pairs]
        c_srcs = [p.candidate_source for p in pairs]
        total_candidates_generated += len(pairs)

        # Compute pairwise features
        feats = [
            compute_pair_features(
                s1_norm,
                candidate_records[p.candidate_entity_id],
                p.blocking_rules,
                p.candidate_source,
            )
            for p in pairs
        ]
        X = pd.DataFrame(feats)
        proba = model.predict_proba(X)[:, 1]

        # Collect matches above threshold
        matched_cids = []
        for i, p_val in enumerate(proba):
            if p_val >= 0.50: prob_brackets[">=0.50"] += 1
            if p_val >= 0.70: prob_brackets[">=0.70"] += 1
            if p_val >= 0.80: prob_brackets[">=0.80"] += 1
            if p_val >= 0.90: prob_brackets[">=0.90"] += 1
            if p_val >= 0.95: prob_brackets[">=0.95"] += 1
            if p_val >= 0.99: prob_brackets[">=0.99"] += 1

            if p_val >= threshold:
                cid = c_ids[i]
                matched_cids.append(cid)
                if c_srcs[i] == "S2":
                    s2_matches_cnt += 1
                else:
                    s3_matches_cnt += 1

                if len(high_confidence_audit) < 50:
                    high_confidence_audit.append({
                        "source1_entity_id": s1_id,
                        "candidate_entity_id": cid,
                        "candidate_source": c_srcs[i],
                        "probability": round(float(p_val), 4),
                        "blocking_rules": sorted(list(pairs[i].blocking_rules)),
                    })

        sorted_cands = sorted(list(set(c_ids)))
        sorted_matches = sorted(list(set(matched_cids)))
        predictions[s1_id] = (sorted_cands, sorted_matches, [round(float(p), 4) for p in proba])

        num_m = len(sorted_matches)
        total_matches_predicted += num_m
        if num_m == 0:
            empty_matches_cnt += 1
        elif num_m == 1:
            singleton_matches_cnt += 1
        else:
            multimatch_matches_cnt += 1

        if (idx + 1) % 50000 == 0:
            logger.info("Processed %d / %d S1 entities...", idx + 1, total_s1)

    t_inf = time.time() - t_inf_start
    logger.info("Inference completed in %.2f seconds (%.1f S1/sec).", t_inf, total_s1 / max(t_inf, 0.001))

    # 6. Write matching_results.tsv and candidate_pairs.tsv
    logger.info("Writing output files to %s...", output_dir)
    with open(matching_file, "w", encoding="utf-8") as f_match, \
         open(candidate_file, "w", encoding="utf-8") as f_cand:

        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        for s1_id in required_s1_ids:
            cands, matches, _ = predictions.get(s1_id, ([], [], []))
            f_match.write(f"{s1_id}\t{','.join(matches)}\n")
            f_cand.write(f"{s1_id}\t{','.join(cands)}\n")

    logger.info("Successfully wrote %s and %s (%d rows each).", matching_file.name, candidate_file.name, total_required)

    stats = {
        "total_test_s1": total_required,
        "evaluated_test_s1": total_s1,
        "candidate_pool_size": len(candidate_records),
        "total_candidates_generated": total_candidates_generated,
        "avg_candidates_per_s1": round(total_candidates_generated / max(total_s1, 1), 2),
        "total_matches_predicted": total_matches_predicted,
        "avg_matches_per_s1": round(total_matches_predicted / max(total_s1, 1), 4),
        "empty_candidate_entities": empty_cands_cnt,
        "empty_match_entities": empty_matches_cnt,
        "singleton_match_entities": singleton_matches_cnt,
        "multimatch_match_entities": multimatch_matches_cnt,
        "s2_matches": s2_matches_cnt,
        "s3_matches": s3_matches_cnt,
        "probability_brackets": prob_brackets,
        "high_confidence_audit_sample": high_confidence_audit[:20],
        "threshold": threshold,
        "indexing_time_seconds": round(t_idx, 2),
        "inference_time_seconds": round(t_inf, 2),
        "total_runtime_seconds": round(time.time() - t_start, 2),
    }

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate entity resolution submission files.")
    parser.add_argument("--test-dir", type=str, default="student_resource/dataset/test")
    parser.add_argument("--output-dir", type=str, default="student_resource/output")
    parser.add_argument("--model-path", type=str, default="src/models/production_model.joblib")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--max-s1", type=int, default=None)
    parser.add_argument("--candidate-pool-limit", type=int, default=None)

    args = parser.parse_args()
    res = generate_submission(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        threshold=args.threshold,
        max_candidates_per_s1=args.max_candidates,
        max_s1_records=args.max_s1,
        candidate_pool_limit_per_source=args.candidate_pool_limit,
    )
    print("\nSubmission Generation Complete:")
    print(json.dumps(res, indent=2))
