"""
Deterministic pairwise feature engineering experiment runner for Phase 3 (EXP_003).

Executes end-to-end feature extraction on a representative sample of Phase-2 candidate pairs:
1. Loads deterministic S1 sample and candidate pool (S2 & S3)
2. Normalizes records using Saran's preprocessing
3. Generates candidate pairs using Phase-2 CandidateGenerator
4. Computes full pairwise feature matrix (X), metadata, and ground truth labels (y)
5. Validates feature data quality (NaNs, infinite values, dtypes, ranges)
6. Performs positive vs. hard-negative separation analysis
7. Measures feature redundancy and correlation
8. Analyzes multilingual and missing-address subsets
9. Logs results to experiment_log.csv and writes compact summary report
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

import numpy as np
import pandas as pd

from src.blocking.candidate_generator import CandidateGenerator, CandidatePair
from src.features.pair_features import build_feature_dataset, compute_pair_features
from src.preprocessing.normalize import normalize_record

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_feature_experiment(
    sample_size: int = 500,
    background_pool_size_per_source: int = 25000,
    max_block_size: int = 500,
) -> Dict[str, Any]:
    """
    Executes Phase 3 feature engineering experiment on a deterministic sample.
    """
    t_start = time.time()
    logger.info("Starting Phase 3 feature engineering experiment (S1 sample=%d)...", sample_size)

    dataset_dir = REPO_ROOT / "student_resource" / "dataset" / "train"
    gt_file = dataset_dir / "train_ground_truth.tsv"
    s1_file = dataset_dir / "train_source1.tsv"
    s2_file = dataset_dir / "train_source2.tsv"
    s3_file = dataset_dir / "train_source3.tsv"

    # 1. Load Ground Truth sample
    logger.info("Loading ground truth for %d S1 entities...", sample_size)
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
    logger.info("Target S1 entities: %d, Total True Matches: %d", len(s1_target_ids), total_true_matches)

    # 2. Load and normalize Source 1 records
    logger.info("Loading and normalizing Source 1 sample...")
    s1_df = pd.read_csv(s1_file, sep="\t", keep_default_na=False)
    s1_df_sample = s1_df[s1_df["entity_id"].isin(s1_target_ids)]
    s1_records: Dict[str, Dict[str, Any]] = {}
    for _, r in s1_df_sample.iterrows():
        s1_records[r["entity_id"]] = normalize_record(r.to_dict())
    del s1_df, s1_df_sample
    gc.collect()

    # 3. Load and normalize candidate pool (S2 and S3)
    logger.info("Loading candidate records (background pool + true matches)...")
    s2_df = pd.read_csv(s2_file, sep="\t", keep_default_na=False)
    s2_matches = s2_df[s2_df["entity_id"].isin(all_true_matched_ids)]
    s2_bg = s2_df[~s2_df["entity_id"].isin(all_true_matched_ids)].head(background_pool_size_per_source)
    s2_pool = pd.concat([s2_matches, s2_bg], ignore_index=True)
    del s2_df, s2_matches, s2_bg
    gc.collect()

    s3_df = pd.read_csv(s3_file, sep="\t", keep_default_na=False)
    s3_matches = s3_df[s3_df["entity_id"].isin(all_true_matched_ids)]
    s3_bg = s3_df[~s3_df["entity_id"].isin(all_true_matched_ids)].head(background_pool_size_per_source)
    s3_pool = pd.concat([s3_matches, s3_bg], ignore_index=True)
    del s3_df, s3_matches, s3_bg
    gc.collect()

    candidate_records: Dict[str, Dict[str, Any]] = {}
    for _, r in s2_pool.iterrows():
        candidate_records[r["entity_id"]] = normalize_record(r.to_dict())
    for _, r in s3_pool.iterrows():
        candidate_records[r["entity_id"]] = normalize_record(r.to_dict())
    del s2_pool, s3_pool
    gc.collect()

    logger.info("Normalized %d candidate pool records (S2 + S3).", len(candidate_records))

    # 4. Phase-2 Candidate Generation (Blocking)
    logger.info("Indexing candidates in Phase-2 CandidateGenerator...")
    gen = CandidateGenerator(
        enable_exact_name=True,
        enable_name_country=True,
        enable_exact_address=True,
        enable_name_tokens=True,
        enable_compressed_name=True,
        enable_address_anchor=True,
        max_block_size=max_block_size,
    )
    gen.index_candidates(candidate_records.values())

    logger.info("Generating candidate pairs for %d S1 records...", len(s1_records))
    t_gen_start = time.time()
    candidate_pairs: List[CandidatePair] = []
    for s1_rec in s1_records.values():
        pairs = gen.generate_candidates_for_record(s1_rec)
        candidate_pairs.extend(pairs)
    t_gen_elapsed = time.time() - t_gen_start

    num_pairs = len(candidate_pairs)
    logger.info("Generated %d candidate pairs in %.2f seconds.", num_pairs, t_gen_elapsed)

    # 5. Feature Engineering (Phase 3)
    logger.info("Extracting pairwise features for all %d candidate pairs...", num_pairs)
    t_feat_start = time.time()
    features_df, metadata_df, labels = build_feature_dataset(
        candidate_pairs=candidate_pairs,
        s1_records=s1_records,
        candidate_records=candidate_records,
        ground_truth_map=gt_map,
    )
    t_feat_elapsed = time.time() - t_feat_start
    throughput = num_pairs / max(t_feat_elapsed, 0.0001)
    logger.info(
        "Extracted features for %d pairs in %.2f seconds (%.0f pairs/sec).",
        num_pairs,
        t_feat_elapsed,
        throughput,
    )

    # 6. Quality Validation
    num_features = features_df.shape[1]
    feature_names = list(features_df.columns)
    pos_mask = (labels == 1)
    neg_mask = (labels == 0)
    num_pos = int(np.sum(pos_mask))
    num_neg = int(np.sum(neg_mask))
    blocking_recall = (num_pos / total_true_matches * 100) if total_true_matches > 0 else 0.0

    logger.info("Feature columns: %d", num_features)
    logger.info("Candidate pairs: %d | Positive: %d | Hard Negative: %d | Blocking Recall: %.2f%%",
                num_pairs, num_pos, num_neg, blocking_recall)

    # Missing / NaN / Inf validation
    null_counts = features_df.isnull().sum()
    total_nulls = int(null_counts.sum())
    inf_counts = int(np.isinf(features_df.values).sum())

    logger.info("Data Quality Check: Total NaNs/Nulls = %d, Total Infs = %d", total_nulls, inf_counts)

    # 7. Summary Statistics & Positive vs. Negative Separation Analysis
    stats_rows = []
    pos_df = features_df[pos_mask]
    neg_df = features_df[neg_mask]

    for col in feature_names:
        col_series = features_df[col]
        pos_series = pos_df[col]
        neg_series = neg_df[col]

        c_dtype = str(col_series.dtype)
        c_min = float(col_series.min())
        c_max = float(col_series.max())
        c_mean = float(col_series.mean())
        c_std = float(col_series.std())
        c_unique = int(col_series.nunique())

        pos_mean = float(pos_series.mean()) if len(pos_series) > 0 else 0.0
        neg_mean = float(neg_series.mean()) if len(neg_series) > 0 else 0.0
        pos_std = float(pos_series.std()) if len(pos_series) > 0 else 0.0
        neg_std = float(neg_series.std()) if len(neg_series) > 0 else 0.0

        # Mean separation / normalized difference
        pooled_std = np.sqrt(0.5 * (pos_std ** 2 + neg_std ** 2))
        cohen_d = (pos_mean - neg_mean) / pooled_std if pooled_std > 1e-6 else 0.0

        stats_rows.append({
            "feature": col,
            "dtype": c_dtype,
            "min": round(c_min, 4),
            "max": round(c_max, 4),
            "overall_mean": round(c_mean, 4),
            "overall_std": round(c_std, 4),
            "unique_values": c_unique,
            "pos_mean": round(pos_mean, 4),
            "neg_mean": round(neg_mean, 4),
            "mean_diff": round(pos_mean - neg_mean, 4),
            "cohen_d": round(cohen_d, 4),
        })

    stats_df = pd.DataFrame(stats_rows)

    # 8. Redundancy & Correlation Analysis
    corr_matrix = features_df.corr().abs()
    high_corr_pairs = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            f1 = feature_names[i]
            f2 = feature_names[j]
            r_val = corr_matrix.loc[f1, f2]
            if r_val >= 0.90:
                high_corr_pairs.append({
                    "feature_1": f1,
                    "feature_2": f2,
                    "correlation": round(float(r_val), 4),
                })

    # Constant or near-constant features
    constant_features = [r["feature"] for r in stats_rows if r["unique_values"] <= 1]
    near_constant_features = [r["feature"] for r in stats_rows if r["overall_std"] < 0.01 and r["unique_values"] > 1]

    # 9. Multilingual Subset Analysis
    is_non_latin = features_df["has_indic_or_non_ascii"] == 1.0
    num_non_latin = int(is_non_latin.sum())
    pos_non_latin = int((pos_mask & is_non_latin).sum())
    neg_non_latin = int((neg_mask & is_non_latin).sum())

    multilingual_stats = {
        "candidate_non_ascii_total": num_non_latin,
        "candidate_non_ascii_positives": pos_non_latin,
        "candidate_non_ascii_negatives": neg_non_latin,
        "script_match_pos_mean": round(float(pos_df["script_match"].mean()), 4),
        "script_match_neg_mean": round(float(neg_df["script_match"].mean()), 4),
        "cross_script_pos_mean": round(float(pos_df["is_cross_script"].mean()), 4),
        "cross_script_neg_mean": round(float(neg_df["is_cross_script"].mean()), 4),
    }

    # 10. Missing-Address Subset Analysis
    addr_missing = features_df["address_missing_candidate"] == 1.0
    num_addr_missing = int(addr_missing.sum())
    pos_addr_missing = int((pos_mask & addr_missing).sum())
    neg_addr_missing = int((neg_mask & addr_missing).sum())

    missing_addr_stats = {
        "address_missing_total": num_addr_missing,
        "address_missing_positives": pos_addr_missing,
        "address_missing_negatives": neg_addr_missing,
        "pos_addr_missing_pct": round(pos_addr_missing / max(num_pos, 1) * 100, 2),
        "neg_addr_missing_pct": round(neg_addr_missing / max(num_neg, 1) * 100, 2),
    }

    # 11. Multi-blocking Rule Breakdown
    rule_counts_pos = pos_df["blocking_rule_count"].value_counts().to_dict()
    rule_counts_neg = neg_df["blocking_rule_count"].value_counts().to_dict()

    total_runtime = time.time() - t_start

    results = {
        "experiment_id": "EXP_003",
        "stage": "pairwise_feature_engineering",
        "sample_size_s1": sample_size,
        "total_true_matches": total_true_matches,
        "candidate_pairs_total": num_pairs,
        "positive_pairs": num_pos,
        "negative_pairs": num_neg,
        "blocking_recall": round(blocking_recall, 2),
        "feature_count": num_features,
        "feature_names": feature_names,
        "null_count": total_nulls,
        "inf_count": inf_counts,
        "constant_features": constant_features,
        "near_constant_features": near_constant_features,
        "high_correlation_pairs": high_corr_pairs,
        "multilingual_stats": multilingual_stats,
        "missing_address_stats": missing_addr_stats,
        "rule_counts_pos": rule_counts_pos,
        "rule_counts_neg": rule_counts_neg,
        "throughput_pairs_per_sec": round(throughput, 1),
        "runtime_seconds": round(total_runtime, 2),
        "feature_stats": stats_rows,
    }

    # Write experiment log entry
    exp_log_path = REPO_ROOT / "student_resource" / "code" / "business_entity_resolution" / "experiments" / "experiment_log.csv"
    with open(exp_log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "EXP_003",
            "pairwise_features",
            "Comprehensive deterministic pairwise feature engineering (Groups A-H)",
            "completed",
            f"S1_samples={sample_size}",
            f"cands={num_pairs}",
            f"pos={num_pos}",
            f"neg={num_neg}",
            f"features={num_features}",
            f"throughput={throughput:.0f}_pairs/s",
            f"runtime={total_runtime:.2f}s",
            f"nulls={total_nulls}",
        ])
    logger.info("Logged EXP_003 to %s", exp_log_path)

    # Save summary report JSON
    report_path = REPO_ROOT / "student_resource" / "code" / "business_entity_resolution" / "reports" / "feature_engineering_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved feature engineering report to %s", report_path)

    return results


if __name__ == "__main__":
    results = run_feature_experiment(sample_size=500)
    print("\n" + "=" * 60)
    print("PHASE 3 FEATURE ENGINEERING EXPERIMENT COMPLETE (EXP_003)")
    print("=" * 60)
    print(f"Total Candidate Pairs: {results['candidate_pairs_total']}")
    print(f"Positive Matches (y=1): {results['positive_pairs']}")
    print(f"Hard Negatives (y=0):   {results['negative_pairs']}")
    print(f"Feature Count:          {results['feature_count']}")
    print(f"Nulls / NaNs / Infs:    {results['null_count']} / {results['inf_count']}")
    print(f"Throughput:             {results['throughput_pairs_per_sec']} pairs/sec")
    print(f"Total Runtime:          {results['runtime_seconds']} s")
    print("=" * 60)
