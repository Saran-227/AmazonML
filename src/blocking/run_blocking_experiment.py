"""
Deterministic baseline blocking experiment runner for Phase 2.

Evaluates multi-block candidate generation on a representative sample of Ground Truth,
integrating with Saran's preprocessing, measuring all candidate statistics, recall,
rule contributions, and failure categories.
"""

from __future__ import annotations

import csv
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.blocking.blocking_evaluation import (
    evaluate_blocking,
    format_evaluation_report,
)
from src.blocking.candidate_generator import CandidateGenerator
from src.preprocessing.normalize import normalize_record

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_experiment(
    sample_size: int = 2000,
    background_pool_size_per_source: int = 35000,
    max_block_size: int = 500,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Executes deterministic baseline blocking experiment.
    """
    t0 = time.time()
    logger.info("Starting baseline blocking experiment with %d S1 entities...", sample_size)

    repo_root = Path(__file__).resolve().parent.parent.parent
    dataset_dir = repo_root / "student_resource" / "dataset" / "train"

    gt_file = dataset_dir / "train_ground_truth.tsv"
    s1_file = dataset_dir / "train_source1.tsv"
    s2_file = dataset_dir / "train_source2.tsv"
    s3_file = dataset_dir / "train_source3.tsv"

    for f in (gt_file, s1_file, s2_file, s3_file):
        if not f.is_file():
            raise FileNotFoundError(f"Required dataset file not found: {f}")

    # 1. Load sample of Ground Truth (deterministic first N rows)
    logger.info("Loading Ground Truth sample...")
    gt_df = pd.read_csv(gt_file, sep="\t", keep_default_na=False, nrows=sample_size)
    gt_map: Dict[str, Set[str]] = {}
    all_true_matched_ids: Set[str] = set()

    for _, r in gt_df.iterrows():
        s1_id = r["source1_entity_id"]
        matched = [x.strip() for x in r["matched_entity_ids"].split(",") if x.strip()]
        gt_map[s1_id] = set(matched)
        all_true_matched_ids.update(matched)

    s1_target_ids = set(gt_map.keys())
    total_true_matches = sum(len(m) for m in gt_map.values())
    logger.info(
        "Loaded %d S1 entities with %d total true matches (%d unique matched IDs).",
        len(gt_map),
        total_true_matches,
        len(all_true_matched_ids),
    )

    # 2. Load and normalize S1 records using Saran's preprocessing
    logger.info("Loading and normalizing Source 1 records...")
    s1_df = pd.read_csv(s1_file, sep="\t", keep_default_na=False)
    s1_df_sampled = s1_df[s1_df["entity_id"].isin(s1_target_ids)]
    s1_records: Dict[str, Dict[str, Any]] = {}
    for _, r in s1_df_sampled.iterrows():
        s1_records[r["entity_id"]] = normalize_record(r.to_dict())
    del s1_df, s1_df_sampled
    gc.collect()

    # 3. Load and normalize candidate pool (Source 2 and Source 3)
    logger.info(
        "Loading candidate pool (background: %d per source + all true matches)...",
        background_pool_size_per_source,
    )
    s2_df = pd.read_csv(s2_file, sep="\t", keep_default_na=False, nrows=background_pool_size_per_source)
    s3_df = pd.read_csv(s3_file, sep="\t", keep_default_na=False, nrows=background_pool_size_per_source)

    candidate_records: Dict[str, Dict[str, Any]] = {}
    for _, r in s2_df.iterrows():
        candidate_records[r["entity_id"]] = normalize_record(r.to_dict())
    for _, r in s3_df.iterrows():
        candidate_records[r["entity_id"]] = normalize_record(r.to_dict())

    del s2_df, s3_df
    gc.collect()

    # Ensure all ground-truth target IDs are present in the candidate pool to measure true recall ceiling
    missing_matched = [mid for mid in all_true_matched_ids if mid not in candidate_records]
    if missing_matched:
        m_s2 = [m for m in missing_matched if m.startswith("S2-")]
        m_s3 = [m for m in missing_matched if m.startswith("S3-")]
        logger.info(
            "Fetching %d additional true match records from disk (%d S2, %d S3)...",
            len(missing_matched),
            len(m_s2),
            len(m_s3),
        )
        if m_s2:
            s2_extra = pd.read_csv(s2_file, sep="\t", keep_default_na=False)
            for _, r in s2_extra[s2_extra["entity_id"].isin(m_s2)].iterrows():
                candidate_records[r["entity_id"]] = normalize_record(r.to_dict())
            del s2_extra
            gc.collect()
        if m_s3:
            s3_extra = pd.read_csv(s3_file, sep="\t", keep_default_na=False)
            for _, r in s3_extra[s3_extra["entity_id"].isin(m_s3)].iterrows():
                candidate_records[r["entity_id"]] = normalize_record(r.to_dict())
            del s3_extra
            gc.collect()

    logger.info("Candidate pool ready: %d total records.", len(candidate_records))

    # 4. Initialize CandidateGenerator and index candidate pool
    logger.info("Indexing candidate pool in CandidateGenerator...")
    generator = CandidateGenerator(
        enable_exact_name=True,
        enable_name_country=True,
        enable_exact_address=True,
        enable_name_tokens=True,
        enable_compressed_name=True,
        enable_address_anchor=True,
        max_block_size=max_block_size,
        min_token_len=3,
    )
    generator.index_candidates(candidate_records.values())
    logger.info("Inverted indexes built: %s", generator.get_index_stats())

    # 5. Generate candidates for each S1 entity
    logger.info("Executing candidate blocking for all S1 entities...")
    candidates_by_s1: Dict[str, Dict[str, Tuple[str, Set[str]]]] = {}

    for s1_id, s1_rec in s1_records.items():
        cands = generator.generate_candidates_for_record(s1_rec)
        cand_dict: Dict[str, Tuple[str, Set[str]]] = {}
        for c in cands:
            cand_dict[c.candidate_entity_id] = (c.candidate_source, c.blocking_rules)
        candidates_by_s1[s1_id] = cand_dict

    # 6. Evaluate blocking performance
    logger.info("Evaluating blocking metrics and failure modes...")
    eval_result = evaluate_blocking(
        ground_truth=gt_map,
        candidates_by_s1=candidates_by_s1,
        s1_records=s1_records,
        candidate_records=candidate_records,
    )

    elapsed_time = time.time() - t0
    report_text = format_evaluation_report(eval_result)
    print("\n" + report_text + "\n")
    logger.info("Experiment completed in %.2f seconds.", elapsed_time)

    # 7. Record experiment log
    exp_log_file = repo_root / "student_resource" / "code" / "business_entity_resolution" / "experiments" / "experiment_log.csv"
    cd = eval_result.candidate_distribution

    log_entry = {
        "experiment_id": "EXP_002",
        "stage": "candidate_blocking",
        "description": "Baseline multi-block candidate generation (exact, country, token, domain, address anchor)",
        "status": "completed",
        "sample_size": sample_size,
        "overall_recall": eval_result.overall_recall,
        "source2_recall": eval_result.source2_recall,
        "source3_recall": eval_result.source3_recall,
        "total_candidate_pairs": cd.get("total_candidates", 0),
        "avg_candidates_per_s1": cd.get("mean", 0.0),
        "median_candidates": cd.get("median", 0.0),
        "p90_candidates": cd.get("p90", 0.0),
        "p95_candidates": cd.get("p95", 0.0),
        "p99_candidates": cd.get("p99", 0.0),
        "max_candidates": cd.get("max", 0),
        "runtime_seconds": round(elapsed_time, 2),
    }

    if exp_log_file.parent.is_dir():
        exists = exp_log_file.is_file()
        with open(exp_log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(log_entry.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(log_entry)
        logger.info("Logged experiment to %s", exp_log_file)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "blocking_baseline_report.txt"
        report_path.write_text(report_text, encoding="utf-8")
        json_path = output_dir / "blocking_baseline_metrics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(eval_result.to_dict(), f, indent=2)
        logger.info("Saved report to %s", report_path)

    return {
        "evaluation_result": eval_result,
        "elapsed_time": elapsed_time,
        "log_entry": log_entry,
    }


if __name__ == "__main__":
    run_experiment(sample_size=2000, background_pool_size_per_source=35000)
