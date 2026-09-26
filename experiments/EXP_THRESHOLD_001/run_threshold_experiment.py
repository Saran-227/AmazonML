"""
Controlled Threshold Experiment: EXP_THRESHOLD_001.

Evaluates global and source-specific decision thresholds using the experimentally
improved candidate generator (EXP_BLOCKING_001) under strict grouped validation.

Configurations evaluated:
Global: 0.90, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97
Source-Specific:
- S2=0.92 / S3=0.95
- S2=0.93 / S3=0.95
- S2=0.94 / S3=0.95
- S2=0.95 / S3=0.95
- S2=0.95 / S3=0.94
- S2=0.95 / S3=0.96
Additional combinations:
- S2=0.93 / S3=0.94
- S2=0.94 / S3=0.96

Outputs:
1. Complete threshold comparison table
2. experiments/EXP_THRESHOLD_001/threshold_experiment_report.json
3. experiments/EXP_THRESHOLD_001/threshold_comparison_table.csv
4. Appends all configurations to experiments/experiment_log.csv
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
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from experiments.EXP_BLOCKING_001.run_blocking_experiment_001 import ExperimentalCandidateGenerator
from src.blocking.candidate_generator import CandidatePair
from src.evaluation.metrics import evaluate_entity_level
from src.evaluation.validation import split_grouped_dataset, verify_group_leakage
from src.features.pair_features import build_feature_dataset
from src.models.predict import predict_probabilities
from src.models.train import validate_feature_matrix
from src.preprocessing.normalize import normalize_record

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)


def predict_matches_source_specific(
    probabilities: np.ndarray,
    metadata_df: pd.DataFrame,
    threshold_s2: float,
    threshold_s3: float,
    all_s1_ids: Optional[Iterable[str]] = None,
) -> Dict[str, Set[str]]:
    """Translates probabilities into match sets using source-specific thresholds."""
    if len(probabilities) != len(metadata_df):
        raise ValueError("Length mismatch between probabilities and metadata.")

    pred_map: Dict[str, Set[str]] = {}
    if all_s1_ids is not None:
        for s1 in all_s1_ids:
            pred_map[s1] = set()

    s1_col = metadata_df["source1_entity_id"].values
    cand_col = metadata_df["candidate_entity_id"].values
    src_col = metadata_df["candidate_source"].values

    is_s2 = (src_col == "S2") | np.array([c.startswith("S2-") for c in cand_col])
    
    # Vectorized condition
    passes_threshold = np.where(
        is_s2,
        probabilities >= threshold_s2,
        probabilities >= threshold_s3,
    )
    
    pass_indices = np.where(passes_threshold)[0]
    for idx in pass_indices:
        s1 = s1_col[idx]
        c = cand_col[idx]
        if s1 not in pred_map:
            pred_map[s1] = set()
        pred_map[s1].add(c)

    return pred_map


def run_experiment(
    sample_size: int = 500,
    background_pool_size: int = 25000,
    val_ratio: float = 0.25,
    random_state: int = 42,
):
    exp_dir = REPO_ROOT / "experiments" / "EXP_THRESHOLD_001"
    exp_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    dataset_dir = REPO_ROOT / "student_resource" / "dataset" / "train"

    # 1. Load Ground Truth sample
    logger.info("Loading ground truth sample (%d S1)...", sample_size)
    gt_df = pd.read_csv(dataset_dir / "train_ground_truth.tsv", sep="\t", keep_default_na=False, nrows=sample_size)
    gt_map: Dict[str, Set[str]] = {}
    all_true_matched_ids: Set[str] = set()
    for _, r in gt_df.iterrows():
        s1_id = r["source1_entity_id"]
        matched = [x.strip() for x in r["matched_entity_ids"].split(",") if x.strip()]
        gt_map[s1_id] = set(matched)
        all_true_matched_ids.update(matched)

    s1_target_ids = set(gt_map.keys())

    # 2. Normalize S1 sample
    logger.info("Loading and normalizing Source 1 records...")
    s1_df = pd.read_csv(dataset_dir / "train_source1.tsv", sep="\t", keep_default_na=False)
    s1_sample = s1_df[s1_df["entity_id"].isin(s1_target_ids)]
    s1_records = {r["entity_id"]: normalize_record(r.to_dict()) for _, r in s1_sample.iterrows()}
    del s1_df, s1_sample
    gc.collect()

    # 3. Load Candidate Pool
    logger.info("Loading candidate records (background: %d per source + true matches)...", background_pool_size)
    s2_df = pd.read_csv(dataset_dir / "train_source2.tsv", sep="\t", keep_default_na=False)
    s2_matches = s2_df[s2_df["entity_id"].isin(all_true_matched_ids)]
    s2_bg = s2_df[~s2_df["entity_id"].isin(all_true_matched_ids)].head(background_pool_size)
    s2_pool = pd.concat([s2_matches, s2_bg], ignore_index=True)
    del s2_df, s2_matches, s2_bg
    gc.collect()

    s3_df = pd.read_csv(dataset_dir / "train_source3.tsv", sep="\t", keep_default_na=False)
    s3_matches = s3_df[s3_df["entity_id"].isin(all_true_matched_ids)]
    s3_bg = s3_df[~s3_df["entity_id"].isin(all_true_matched_ids)].head(background_pool_size)
    s3_pool = pd.concat([s3_matches, s3_bg], ignore_index=True)
    del s3_df, s3_matches, s3_bg
    gc.collect()

    candidate_records = {}
    for _, r in s2_pool.iterrows():
        candidate_records[r["entity_id"]] = normalize_record(r.to_dict())
    for _, r in s3_pool.iterrows():
        candidate_records[r["entity_id"]] = normalize_record(r.to_dict())
    del s2_pool, s3_pool
    gc.collect()

    # 4. Generate candidates with EXP_BLOCKING_001 generator
    logger.info("Generating candidate pairs with EXP_BLOCKING_001 candidate generator...")
    exp_gen = ExperimentalCandidateGenerator(
        max_block_size=500,
        enable_cross_script_address=True,
        enable_domain_tokens=True,
    )
    exp_gen.index_candidates(candidate_records.values())

    candidate_pairs: List[CandidatePair] = []
    for s1_rec in s1_records.values():
        pairs = exp_gen.generate_candidates_for_record(s1_rec)
        candidate_pairs.extend(pairs)

    logger.info("Generated %d candidate pairs.", len(candidate_pairs))

    # 5. Extract 60 pairwise features
    logger.info("Extracting pairwise features for all candidate pairs...")
    t_feat_start = time.time()
    features_df, metadata_df, labels = build_feature_dataset(
        candidate_pairs=candidate_pairs,
        s1_records=s1_records,
        candidate_records=candidate_records,
        ground_truth_map=gt_map,
    )
    validate_feature_matrix(features_df)
    t_feat = time.time() - t_feat_start
    logger.info("Feature extraction complete in %.2f seconds.", t_feat)

    # 6. Grouped Leakage-Safe Train/Val Split
    logger.info("Splitting dataset by source1_entity_id (val_ratio=%.2f)...", val_ratio)
    train_idx, val_idx, train_groups, val_groups = split_grouped_dataset(
        metadata_df=metadata_df,
        val_ratio=val_ratio,
        group_col="source1_entity_id",
        random_state=random_state,
    )
    verify_group_leakage(train_groups, val_groups)

    X_train = features_df.iloc[train_idx].reset_index(drop=True)
    y_train = labels[train_idx]
    meta_train = metadata_df.iloc[train_idx].reset_index(drop=True)

    X_val = features_df.iloc[val_idx].reset_index(drop=True)
    y_val = labels[val_idx]
    meta_val = metadata_df.iloc[val_idx].reset_index(drop=True)

    logger.info(
        "Train set: %d pairs (%d groups, pos=%d, neg=%d) | Val set: %d pairs (%d groups, pos=%d, neg=%d)",
        len(X_train), len(train_groups), int(y_train.sum()), int((y_train == 0).sum()),
        len(X_val), len(val_groups), int(y_val.sum()), int((y_val == 0).sum())
    )

    # 7. Train HistGradientBoostingClassifier model
    logger.info("Training HistGradientBoostingClassifier model...")
    t_train_start = time.time()
    model = HistGradientBoostingClassifier(
        class_weight="balanced",
        max_iter=150,
        max_leaf_nodes=31,
        learning_rate=0.1,
        min_samples_leaf=20,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    t_train = time.time() - t_train_start
    logger.info("Model trained in %.3f seconds.", t_train)

    # 8. Predict probabilities on validation fold
    logger.info("Predicting validation probabilities...")
    t_pred_start = time.time()
    val_probs, _ = predict_probabilities(model, X_val)
    t_pred = time.time() - t_pred_start

    # Validation ground truth map strictly for val groups
    val_gt_map = {s1: gt_map.get(s1, set()) for s1 in val_groups}
    val_all_s1_ids = sorted(list(val_groups))

    # Calculate False Positive Rate baseline denominator
    total_val_negatives = int((y_val == 0).sum())

    # 9. Configurations to Evaluate
    configs = [
        # Global thresholds
        {"name": "global_0.90", "type": "global", "th_s2": 0.90, "th_s3": 0.90},
        {"name": "global_0.92", "type": "global", "th_s2": 0.92, "th_s3": 0.92},
        {"name": "global_0.93", "type": "global", "th_s2": 0.93, "th_s3": 0.93},
        {"name": "global_0.94", "type": "global", "th_s2": 0.94, "th_s3": 0.94},
        {"name": "global_0.95", "type": "global", "th_s2": 0.95, "th_s3": 0.95},
        {"name": "global_0.96", "type": "global", "th_s2": 0.96, "th_s3": 0.96},
        {"name": "global_0.97", "type": "global", "th_s2": 0.97, "th_s3": 0.97},
        # Source-specific thresholds
        {"name": "src_S2_0.92_S3_0.95", "type": "source_specific", "th_s2": 0.92, "th_s3": 0.95},
        {"name": "src_S2_0.93_S3_0.95", "type": "source_specific", "th_s2": 0.93, "th_s3": 0.95},
        {"name": "src_S2_0.94_S3_0.95", "type": "source_specific", "th_s2": 0.94, "th_s3": 0.95},
        {"name": "src_S2_0.95_S3_0.95", "type": "source_specific", "th_s2": 0.95, "th_s3": 0.95},
        {"name": "src_S2_0.95_S3_0.94", "type": "source_specific", "th_s2": 0.95, "th_s3": 0.94},
        {"name": "src_S2_0.95_S3_0.96", "type": "source_specific", "th_s2": 0.95, "th_s3": 0.96},
        # High-potential fine-grained combinations
        {"name": "src_S2_0.93_S3_0.94", "type": "source_specific", "th_s2": 0.93, "th_s3": 0.94},
        {"name": "src_S2_0.94_S3_0.96", "type": "source_specific", "th_s2": 0.94, "th_s3": 0.96},
    ]

    results_table = []
    log_entries = []

    for cfg in configs:
        th2 = cfg["th_s2"]
        th3 = cfg["th_s3"]

        pred_map = predict_matches_source_specific(
            probabilities=val_probs,
            metadata_df=meta_val,
            threshold_s2=th2,
            threshold_s3=th3,
            all_s1_ids=val_all_s1_ids,
        )

        res = evaluate_entity_level(
            gt_map=val_gt_map,
            pred_map=pred_map,
            all_s1_ids=val_all_s1_ids,
        )

        # Pair-level false positive calculations
        cand_col = meta_val["candidate_entity_id"].values
        src_col = meta_val["candidate_source"].values
        is_s2 = (src_col == "S2") | np.array([c.startswith("S2-") for c in cand_col])
        pair_pred = np.where(is_s2, val_probs >= th2, val_probs >= th3)
        pair_fp = int(((pair_pred == 1) & (y_val == 0)).sum())
        pair_fpr = round(pair_fp / total_val_negatives * 100, 4) if total_val_negatives > 0 else 0.0

        # Entity-level false positive count
        entity_fp_count = 0
        for s1, preds in pred_map.items():
            t_set = val_gt_map.get(s1, set())
            if any(p not in t_set for p in preds):
                entity_fp_count += 1
        entity_fpr = round(entity_fp_count / len(val_all_s1_ids) * 100, 2)

        row = {
            "config_name": cfg["name"],
            "type": cfg["type"],
            "th_s2": th2,
            "th_s3": th3,
            "macro_f05": round(res["macro_f05"], 4),
            "macro_precision": round(res["macro_precision"], 4),
            "macro_recall": round(res["macro_recall"], 4),
            "micro_f05": round(res["micro_f05"], 4),
            "pair_fp_count": pair_fp,
            "pair_fpr_pct": pair_fpr,
            "entity_fpr_pct": entity_fpr,
            "predicted_matches": res["total_predicted_matches"],
            "zero_match_entities": res["empty_prediction_count"],
            "singleton_entities": res["singleton_prediction_count"],
            "multimatch_entities": res["multimatch_prediction_count"],
        }
        results_table.append(row)

        # Log entry for experiment_log.csv
        sub_exp_id = f"EXP_TH_{cfg['name']}"
        log_entry = [
            sub_exp_id,
            "f1d32ca",
            "2026-09-26",
            f"Controlled threshold eval: {cfg['name']} on EXP_BLOCKING_001 features",
            "EXP_BLOCKING_001 (8 rules)",
            len(candidate_pairs),
            "98.47% (Blocking)",
            60,
            "HistGradientBoostingClassifier",
            f"S2={th2}/S3={th3}" if th2 != th3 else str(th2),
            round(res["macro_precision"], 4),
            round(res["macro_recall"], 4),
            round(res["macro_f05"], 4),
            f"{t_train + t_pred:.2f}s",
            "~2.2GB",
            f"Macro_F05={res['macro_f05']:.4f}_P={res['macro_precision']:.4f}_R={res['macro_recall']:.4f}",
            "CANDIDATE_FOR_SELECTION" if res["macro_f05"] > 0.9572 else "REJECTED_LOWER_F05"
        ]
        log_entries.append(log_entry)

    # Sort results by Macro F0.5 descending, breaking ties by Precision descending
    results_table.sort(key=lambda r: (r["macro_f05"], r["macro_precision"]), reverse=True)
    best_config = results_table[0]

    elapsed_total = time.time() - t_start

    # Save CSV comparison table
    csv_path = exp_dir / "threshold_comparison_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results_table[0].keys()))
        writer.writeheader()
        writer.writerows(results_table)
    logger.info("Saved threshold comparison table to %s", csv_path)

    # Append to root experiment_log.csv
    log_csv = REPO_ROOT / "experiments" / "experiment_log.csv"
    with open(log_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for entry in log_entries:
            writer.writerow(entry)
    logger.info("Appended %d threshold configurations to %s", len(log_entries), log_csv)

    # Save JSON Report
    report_dict = {
        "experiment_id": "EXP_THRESHOLD_001",
        "date": "2026-09-26",
        "description": "Controlled Global and Source-Specific Threshold Optimization on EXP_BLOCKING_001 candidate features",
        "val_sample_size_s1": len(val_all_s1_ids),
        "val_candidate_pairs": len(X_val),
        "baseline_reference_macro_f05": 0.9572,
        "selected_configuration": best_config,
        "is_improvement_over_baseline": best_config["macro_f05"] > 0.9572,
        "delta_macro_f05": round(best_config["macro_f05"] - 0.9572, 4),
        "all_configurations_ranked": results_table,
        "runtime_seconds": round(elapsed_total, 2),
        "reproducibility": {
            "random_state": random_state,
            "val_ratio": val_ratio,
            "model": "HistGradientBoostingClassifier",
            "git_commit": "f1d32ca",
        }
    }
    report_json_path = exp_dir / "threshold_experiment_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    logger.info("Saved report to %s", report_json_path)

    # Print formatted comparative table
    print("\n" + "=" * 115)
    print("PHASE 5: CONTROLLED THRESHOLD EXPERIMENT EVALUATION TABLE")
    print("=" * 115)
    print(f"{'Config Name':<24} | {'Th_S2':<5} | {'Th_S3':<5} | {'Macro F0.5':<10} | {'Macro P':<8} | {'Macro R':<8} | {'Pair FPR%':<9} | {'Matches':<7} | {'Zero':<5} | {'Single':<6} | {'Multi':<5}")
    print("-" * 115)
    for r in results_table:
        is_best = " ⭐ BEST" if r == best_config else ""
        print(f"{r['config_name']:<24} | {r['th_s2']:<5.2f} | {r['th_s3']:<5.2f} | {r['macro_f05']:<10.4f} | {r['macro_precision']:<8.4f} | {r['macro_recall']:<8.4f} | {r['pair_fpr_pct']:<9.4f} | {r['predicted_matches']:<7} | {r['zero_match_entities']:<5} | {r['singleton_entities']:<6} | {r['multimatch_entities']:<5}{is_best}")
    print("=" * 115)
    print(f"\nDECISION SUMMARY:")
    print(f"- Frozen Baseline Reference Macro F0.5: 0.9572 (Global threshold 0.95)")
    print(f"- Optimal Configuration: {best_config['config_name']} (S2={best_config['th_s2']}, S3={best_config['th_s3']})")
    print(f"- Optimal Macro F0.5: {best_config['macro_f05']:.4f} (Change: {best_config['macro_f05'] - 0.9572:+.4f})")
    print(f"- Macro Precision: {best_config['macro_precision']:.4f}, Macro Recall: {best_config['macro_recall']:.4f}")
    print("=" * 115 + "\n")


if __name__ == "__main__":
    run_experiment()
