"""
Phase 4 Matching Model and Full Local Validation experiment runner.

Performs:
1. Deterministic candidate generation and feature extraction on representative S1 sample
2. Leakage-safe grouped train/validation split by source1_entity_id
3. Model 1 (Logistic Regression) baseline training and threshold optimization
4. Model 2 (HistGradientBoosting tree) training and threshold optimization
5. Official Entity-Level Macro F0.5 evaluation over 0.10-0.95 grid
6. Grouped 3-fold cross-validation sanity check
7. High-confidence false positive and false negative error analysis
8. Subgroup analyses: Singletons, Multi-match, Source-specific (S2 vs S3), Multilingual
9. Permutation feature importance analysis
10. Experiment logging for EXP_004 (LR) and EXP_005 (Tree)
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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.blocking.candidate_generator import CandidateGenerator, CandidatePair
from src.evaluation.metrics import evaluate_entity_level
from src.evaluation.validation import (
    iter_grouped_folds,
    split_grouped_dataset,
    verify_group_leakage,
)
from src.features.pair_features import build_feature_dataset
from src.models.predict import (
    build_prediction_dataframe,
    predict_matches_at_threshold,
    predict_probabilities,
)
from src.models.threshold import evaluate_threshold_grid, select_best_threshold
from src.models.train import (
    train_logistic_regression,
    train_tree_model,
    validate_feature_matrix,
)
from src.preprocessing.normalize import normalize_record

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_phase4_experiment(
    sample_size: int = 500,
    background_pool_size_per_source: int = 25000,
    val_ratio: float = 0.25,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Executes Phase 4 matching model and validation pipeline.
    """
    t_start = time.time()
    logger.info("Starting Phase 4 matching model experiment (S1 sample=%d)...", sample_size)

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

    # 4. Generate candidate pairs
    logger.info("Indexing candidates and generating Phase-2 candidate pairs...")
    gen = CandidateGenerator()
    gen.index_candidates(candidate_records.values())

    candidate_pairs: List[CandidatePair] = []
    for s1_rec in s1_records.values():
        pairs = gen.generate_candidates_for_record(s1_rec)
        candidate_pairs.extend(pairs)

    num_pairs = len(candidate_pairs)
    logger.info("Generated %d candidate pairs.", num_pairs)

    # 5. Extract pairwise features
    logger.info("Extracting pairwise features for %d candidate pairs...", num_pairs)
    features_df, metadata_df, labels = build_feature_dataset(
        candidate_pairs=candidate_pairs,
        s1_records=s1_records,
        candidate_records=candidate_records,
        ground_truth_map=gt_map,
    )
    feature_names = validate_feature_matrix(features_df)

    # 6. Leakage-safe Grouped Train/Validation Split
    logger.info("Performing leakage-safe grouped train/val split (val_ratio=%.2f)...", val_ratio)
    train_idx, val_idx, train_groups, val_groups = split_grouped_dataset(
        metadata_df=metadata_df,
        val_ratio=val_ratio,
        group_col="source1_entity_id",
        random_state=random_state,
    )

    # Explicit leakage verification assertion
    verify_group_leakage(train_groups, val_groups)
    assert len(train_groups & val_groups) == 0, "Fatal: Group leakage detected!"

    X_train = features_df.iloc[train_idx].reset_index(drop=True)
    y_train = labels[train_idx]
    meta_train = metadata_df.iloc[train_idx].reset_index(drop=True)

    X_val = features_df.iloc[val_idx].reset_index(drop=True)
    y_val = labels[val_idx]
    meta_val = metadata_df.iloc[val_idx].reset_index(drop=True)

    validate_feature_matrix(X_train, X_val)

    # Split statistics
    train_pos = int(np.sum(y_train == 1))
    train_neg = int(np.sum(y_train == 0))
    val_pos = int(np.sum(y_val == 1))
    val_neg = int(np.sum(y_val == 0))

    val_true_matches = sum(len(gt_map.get(s1, set())) for s1 in val_groups)
    val_blocking_recall = (val_pos / val_true_matches * 100) if val_true_matches > 0 else 0.0

    logger.info(
        "Split Stats: Train Groups=%d, Val Groups=%d | Train Cands=%d (Pos=%d, Neg=%d, %.2f%%) | "
        "Val Cands=%d (Pos=%d, Neg=%d, %.2f%%, Blocking Recall=%.2f%%)",
        len(train_groups), len(val_groups),
        len(X_train), train_pos, train_neg, (train_pos / len(X_train) * 100),
        len(X_val), val_pos, val_neg, (val_pos / len(X_val) * 100),
        val_blocking_recall,
    )

    # =========================================================================
    # 7. MODEL 1: LOGISTIC REGRESSION BASELINE (EXP_004)
    # =========================================================================
    logger.info("Training Model 1 (Logistic Regression)...")
    lr_model, lr_train_time = train_logistic_regression(
        X_train, y_train, class_weight="balanced", random_state=random_state
    )
    lr_proba_val, lr_pred_time = predict_probabilities(lr_model, X_val)

    logger.info("Evaluating Model 1 across threshold grid...")
    lr_grid_df = evaluate_threshold_grid(
        probabilities=lr_proba_val,
        metadata_df=meta_val,
        gt_map=gt_map,
        all_s1_ids=val_groups,
    )
    lr_best_th, lr_best_metrics, lr_tied, lr_tied_list = select_best_threshold(lr_grid_df)

    logger.info(
        "Model 1 (LR) Best Threshold: %.2f | Macro F0.5: %.4f | Macro P: %.4f | Macro R: %.4f | "
        "Micro P: %.4f | Micro R: %.4f | Matches: %d",
        lr_best_th,
        lr_best_metrics["macro_f05"],
        lr_best_metrics["macro_precision"],
        lr_best_metrics["macro_recall"],
        lr_best_metrics["micro_precision"],
        lr_best_metrics["micro_recall"],
        lr_best_metrics["predicted_matches"],
    )

    # =========================================================================
    # 8. MODEL 2: GRADIENT-BOOSTED DECISION TREE (EXP_005)
    # =========================================================================
    logger.info("Training Model 2 (HistGradientBoostingClassifier)...")
    tree_model, tree_train_time = train_tree_model(
        X_train, y_train, class_weight="balanced", random_state=random_state
    )
    tree_proba_val, tree_pred_time = predict_probabilities(tree_model, X_val)

    logger.info("Evaluating Model 2 across threshold grid...")
    tree_grid_df = evaluate_threshold_grid(
        probabilities=tree_proba_val,
        metadata_df=meta_val,
        gt_map=gt_map,
        all_s1_ids=val_groups,
    )
    tree_best_th, tree_best_metrics, tree_tied, tree_tied_list = select_best_threshold(tree_grid_df)

    logger.info(
        "Model 2 (Tree) Best Threshold: %.2f | Macro F0.5: %.4f | Macro P: %.4f | Macro R: %.4f | "
        "Micro P: %.4f | Micro R: %.4f | Matches: %d",
        tree_best_th,
        tree_best_metrics["macro_f05"],
        tree_best_metrics["macro_precision"],
        tree_best_metrics["macro_recall"],
        tree_best_metrics["micro_precision"],
        tree_best_metrics["micro_recall"],
        tree_best_metrics["predicted_matches"],
    )

    # Select best model for downstream deep-dive analysis
    chosen_model_name = "HistGradientBoosting" if tree_best_metrics["macro_f05"] >= lr_best_metrics["macro_f05"] else "LogisticRegression"
    chosen_proba_val = tree_proba_val if chosen_model_name == "HistGradientBoosting" else lr_proba_val
    chosen_best_th = tree_best_th if chosen_model_name == "HistGradientBoosting" else lr_best_th
    chosen_metrics = tree_best_metrics if chosen_model_name == "HistGradientBoosting" else lr_best_metrics

    # Build validation prediction dataframe
    val_pred_df = build_prediction_dataframe(
        metadata_df=meta_val,
        probabilities=chosen_proba_val,
        threshold=chosen_best_th,
        y_true=y_val,
    )

    # =========================================================================
    # 9. 3-FOLD GROUPED CROSS-VALIDATION SANITY CHECK
    # =========================================================================
    logger.info("Running 3-fold grouped cross-validation sanity check for tree model...")
    cv_fold_scores = []
    for fold_num, (tr_idx, va_idx, tr_grp, va_grp) in enumerate(
        iter_grouped_folds(metadata_df, n_splits=3, group_col="source1_entity_id")
    ):
        verify_group_leakage(tr_grp, va_grp)
        X_tr = features_df.iloc[tr_idx].reset_index(drop=True)
        y_tr = labels[tr_idx]
        X_va = features_df.iloc[va_idx].reset_index(drop=True)
        meta_va = metadata_df.iloc[va_idx].reset_index(drop=True)

        m_fold, _ = train_tree_model(X_tr, y_tr, class_weight="balanced", random_state=random_state)
        p_fold, _ = predict_probabilities(m_fold, X_va)

        grid_fold = evaluate_threshold_grid(p_fold, meta_va, gt_map, all_s1_ids=va_grp)
        _, fold_best, _, _ = select_best_threshold(grid_fold)
        cv_fold_scores.append(fold_best["macro_f05"])
        logger.info("Fold %d Macro F0.5: %.4f (best threshold: %.2f)", fold_num + 1, fold_best["macro_f05"], fold_best["threshold"])

    cv_mean = float(np.mean(cv_fold_scores))
    cv_std = float(np.std(cv_fold_scores))
    logger.info("Grouped 3-Fold CV Macro F0.5: %.4f +/- %.4f", cv_mean, cv_std)

    # =========================================================================
    # 10. ERROR ANALYSIS: FALSE POSITIVES & FALSE NEGATIVES
    # =========================================================================
    logger.info("Performing Error Analysis (False Positives and False Negatives)...")
    # False Positives: predicted = 1, true = 0
    fp_mask = (val_pred_df["predicted_match"] == 1) & (val_pred_df["true_match"] == 0)
    fp_df = val_pred_df[fp_mask].sort_values(by="predicted_probability", ascending=False)
    fp_examples = []
    for _, r in fp_df.head(10).iterrows():
        c_idx = r.name
        feat_dict = X_val.iloc[c_idx].to_dict()
        pair_rec = candidate_records.get(r["candidate_entity_id"], {})
        s1_rec = s1_records.get(r["source1_entity_id"], {})
        fp_examples.append({
            "source1_entity_id": r["source1_entity_id"],
            "candidate_entity_id": r["candidate_entity_id"],
            "candidate_source": r["candidate_source"],
            "predicted_probability": float(r["predicted_probability"]),
            "s1_name": s1_rec.get("business_name", ""),
            "cand_name": pair_rec.get("business_name", ""),
            "s1_address": s1_rec.get("address", ""),
            "cand_address": pair_rec.get("address", ""),
            "blocking_rules": [c for c in feat_dict if c.startswith("block_") and feat_dict[c] == 1.0],
            "key_features": {
                "name_token_jaccard": feat_dict.get("name_token_jaccard", 0.0),
                "name_jaro_winkler": feat_dict.get("name_jaro_winkler", 0.0),
                "address_token_overlap_coefficient": feat_dict.get("address_token_overlap_coefficient", 0.0),
                "address_anchor_agreement": feat_dict.get("address_anchor_agreement", 0.0),
                "blocking_rule_count": feat_dict.get("blocking_rule_count", 0.0),
            },
        })

    # False Negatives: predicted = 0, true = 1 (MODEL MISSES)
    fn_mask = (val_pred_df["predicted_match"] == 0) & (val_pred_df["true_match"] == 1)
    fn_df = val_pred_df[fn_mask].sort_values(by="predicted_probability", ascending=True)
    fn_examples = []
    fn_category_counts = {
        "address_variation": 0,
        "name_variation": 0,
        "cross_script_multilingual": 0,
        "missing_address": 0,
        "single_blocking_rule": 0,
        "other": 0,
    }

    for _, r in fn_df.iterrows():
        c_idx = r.name
        feat_dict = X_val.iloc[c_idx].to_dict()
        pair_rec = candidate_records.get(r["candidate_entity_id"], {})
        s1_rec = s1_records.get(r["source1_entity_id"], {})

        # Categorize
        if feat_dict.get("address_missing_candidate", 0.0) == 1.0:
            fn_category_counts["missing_address"] += 1
        elif feat_dict.get("is_cross_script", 0.0) == 1.0:
            fn_category_counts["cross_script_multilingual"] += 1
        elif feat_dict.get("address_token_overlap_coefficient", 0.0) < 0.2:
            fn_category_counts["address_variation"] += 1
        elif feat_dict.get("name_token_jaccard", 0.0) < 0.4:
            fn_category_counts["name_variation"] += 1
        elif feat_dict.get("blocking_rule_count", 0.0) <= 1.0:
            fn_category_counts["single_blocking_rule"] += 1
        else:
            fn_category_counts["other"] += 1

        if len(fn_examples) < 10:
            fn_examples.append({
                "source1_entity_id": r["source1_entity_id"],
                "candidate_entity_id": r["candidate_entity_id"],
                "candidate_source": r["candidate_source"],
                "predicted_probability": float(r["predicted_probability"]),
                "s1_name": s1_rec.get("business_name", ""),
                "cand_name": pair_rec.get("business_name", ""),
                "s1_address": s1_rec.get("address", ""),
                "cand_address": pair_rec.get("address", ""),
                "key_features": {
                    "name_token_jaccard": feat_dict.get("name_token_jaccard", 0.0),
                    "name_jaro_winkler": feat_dict.get("name_jaro_winkler", 0.0),
                    "address_token_overlap_coefficient": feat_dict.get("address_token_overlap_coefficient", 0.0),
                    "blocking_rule_count": feat_dict.get("blocking_rule_count", 0.0),
                    "is_cross_script": feat_dict.get("is_cross_script", 0.0),
                },
            })

    # Blocking misses vs Model misses
    val_captured_pos = val_pos
    val_model_misses = len(fn_df)
    val_blocking_misses = val_true_matches - val_captured_pos
    val_total_misses = val_blocking_misses + val_model_misses

    logger.info(
        "Miss breakdown: True Matches=%d | Captured by Blocker=%d | Blocking Misses=%d | "
        "Model Misses=%d | Total Misses=%d",
        val_true_matches, val_captured_pos, val_blocking_misses, val_model_misses, val_total_misses
    )

    # =========================================================================
    # 11. SUBGROUP ANALYSES
    # =========================================================================
    # A. Singleton / Empty-ground-truth analysis
    val_pred_map = predict_matches_at_threshold(chosen_proba_val, meta_val, chosen_best_th, all_s1_ids=val_groups)
    val_singleton_s1 = [s1 for s1 in val_groups if len(gt_map.get(s1, set())) == 0]
    val_singleton_correct = sum(1 for s1 in val_singleton_s1 if len(val_pred_map.get(s1, set())) == 0)
    val_singleton_fp = len(val_singleton_s1) - val_singleton_correct

    singleton_stats = {
        "true_singleton_count": len(val_singleton_s1),
        "correct_empty_predictions": val_singleton_correct,
        "false_positive_singleton_predictions": val_singleton_fp,
        "singleton_f05": 1.0 if len(val_singleton_s1) == 0 else round(val_singleton_correct / max(len(val_singleton_s1), 1), 4),
    }

    # B. Multi-match analysis (entities with 2+, 3+, 5+ true matches)
    multi_match_groups = {}
    for min_matches in [1, 2, 3, 5]:
        target_s1 = [s1 for s1 in val_groups if len(gt_map.get(s1, set())) >= min_matches]
        if target_s1:
            sub_res = evaluate_entity_level(
                gt_map={s1: gt_map.get(s1, set()) for s1 in target_s1},
                pred_map={s1: val_pred_map.get(s1, set()) for s1 in target_s1},
                all_s1_ids=target_s1,
            )
            avg_true = float(np.mean([len(gt_map.get(s1, set())) for s1 in target_s1]))
            avg_pred = float(np.mean([len(val_pred_map.get(s1, set())) for s1 in target_s1]))
            multi_match_groups[f"{min_matches}+_matches"] = {
                "entity_count": len(target_s1),
                "avg_true_matches": round(avg_true, 2),
                "avg_pred_matches": round(avg_pred, 2),
                "macro_f05": sub_res["macro_f05"],
                "macro_precision": sub_res["macro_precision"],
                "macro_recall": sub_res["macro_recall"],
            }

    # C. Source-specific analysis (S2 vs S3)
    source_stats = {}
    for src in ["S2", "S3"]:
        src_mask = meta_val["candidate_source"] == src
        src_df = meta_val[src_mask]
        src_proba = chosen_proba_val[src_mask]
        src_pred_map = predict_matches_at_threshold(src_proba, src_df, chosen_best_th)
        src_gt_map = {
            s1: {cid for cid in gt_map.get(s1, set()) if cid.startswith(src)}
            for s1 in val_groups
        }
        src_res = evaluate_entity_level(gt_map=src_gt_map, pred_map=src_pred_map, all_s1_ids=val_groups)
        source_stats[src] = {
            "candidate_count": int(src_mask.sum()),
            "positive_count": int(np.sum(y_val[src_mask] == 1)),
            "macro_f05": src_res["macro_f05"],
            "macro_precision": src_res["macro_precision"],
            "macro_recall": src_res["macro_recall"],
            "micro_precision": src_res["micro_precision"],
            "micro_recall": src_res["micro_recall"],
        }

    # D. Multilingual analysis
    multilingual_stats = {}
    cross_script_mask = X_val["is_cross_script"] == 1.0
    non_ascii_mask = X_val["has_indic_or_non_ascii"] == 1.0
    latin_only_mask = (~non_ascii_mask) & (~cross_script_mask)

    for subset_name, sub_mask in [
        ("latin_only", latin_only_mask),
        ("indic_or_non_ascii", non_ascii_mask),
        ("cross_script", cross_script_mask),
    ]:
        sub_cands = int(sub_mask.sum())
        sub_pos = int(np.sum(y_val[sub_mask] == 1))
        sub_preds = int(np.sum((chosen_proba_val[sub_mask] >= chosen_best_th)))
        sub_tp = int(np.sum((chosen_proba_val[sub_mask] >= chosen_best_th) & (y_val[sub_mask] == 1)))
        sub_p = (sub_tp / sub_preds) if sub_preds > 0 else 0.0
        sub_r = (sub_tp / sub_pos) if sub_pos > 0 else 0.0
        sub_denom = 0.25 * sub_p + sub_r
        sub_f05 = (1.25 * sub_p * sub_r) / sub_denom if sub_denom > 0 else 0.0

        multilingual_stats[subset_name] = {
            "candidate_count": sub_cands,
            "positive_count": sub_pos,
            "predicted_count": sub_preds,
            "true_positives": sub_tp,
            "precision": round(sub_p, 4),
            "recall": round(sub_r, 4),
            "f05": round(sub_f05, 4),
        }

    # =========================================================================
    # 12. FEATURE IMPORTANCE (Permutation Importance on Validation Set)
    # =========================================================================
    logger.info("Computing permutation feature importance on validation set...")
    # Sample up to 5,000 validation candidates for fast permutation importance
    perm_sample_size = min(5000, len(X_val))
    rng = np.random.RandomState(random_state)
    perm_indices = rng.choice(len(X_val), size=perm_sample_size, replace=False)

    perm_res = permutation_importance(
        tree_model,
        X_val.iloc[perm_indices],
        y_val[perm_indices],
        n_repeats=3,
        random_state=random_state,
        scoring="roc_auc",
    )
    importances = perm_res.importances_mean
    sorted_feat_idx = np.argsort(importances)[::-1]

    top_features = []
    for i in sorted_feat_idx[:25]:
        top_features.append({
            "feature": feature_names[i],
            "importance_mean": round(float(importances[i]), 5),
            "importance_std": round(float(perm_res.importances_std[i]), 5),
        })

    # =========================================================================
    # 13. DETERMINISM CHECK
    # =========================================================================
    p1, _ = predict_probabilities(tree_model, X_val.head(100))
    p2, _ = predict_probabilities(tree_model, X_val.head(100))
    determinism_check = bool(np.array_equal(p1, p2))
    logger.info("Determinism Check: %s", "PASSED (Identical outputs)" if determinism_check else "FAILED")

    total_runtime = time.time() - t_start

    # Compile complete results dictionary
    results = {
        "experiment_ids": ["EXP_004", "EXP_005"],
        "stage": "matching_model_and_local_validation",
        "sample_size_s1": sample_size,
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "candidate_counts": {
            "total": num_pairs,
            "train": len(X_train),
            "val": len(X_val),
            "train_positive": train_pos,
            "train_negative": train_neg,
            "val_positive": val_pos,
            "val_negative": val_neg,
        },
        "models": {
            "logistic_regression": {
                "experiment_id": "EXP_004",
                "training_time": round(lr_train_time, 3),
                "prediction_time": round(lr_pred_time, 3),
                "best_threshold": lr_best_th,
                "best_metrics": lr_best_metrics,
                "threshold_grid": lr_grid_df.to_dict(orient="records"),
            },
            "tree_hist_gradient_boosting": {
                "experiment_id": "EXP_005",
                "training_time": round(tree_train_time, 3),
                "prediction_time": round(tree_pred_time, 3),
                "best_threshold": tree_best_th,
                "best_metrics": tree_best_metrics,
                "threshold_grid": tree_grid_df.to_dict(orient="records"),
            },
        },
        "chosen_model": chosen_model_name,
        "cv_results_tree": {
            "folds_macro_f05": [round(x, 4) for x in cv_fold_scores],
            "mean_macro_f05": round(cv_mean, 4),
            "std_macro_f05": round(cv_std, 4),
        },
        "miss_breakdown": {
            "val_true_matches": val_true_matches,
            "val_captured_by_blocking": val_captured_pos,
            "blocking_misses": val_blocking_misses,
            "model_misses": val_model_misses,
            "total_misses": val_total_misses,
            "model_miss_categories": fn_category_counts,
        },
        "error_analysis": {
            "false_positive_examples": fp_examples,
            "false_negative_examples": fn_examples,
        },
        "subgroup_analysis": {
            "singleton": singleton_stats,
            "multi_match": multi_match_groups,
            "source_specific": source_stats,
            "multilingual": multilingual_stats,
        },
        "top_features": top_features,
        "determinism_check": determinism_check,
        "runtime_seconds": round(total_runtime, 2),
    }

    # Append EXP_004 and EXP_005 to experiment_log.csv
    exp_log_path = REPO_ROOT / "student_resource" / "code" / "business_entity_resolution" / "experiments" / "experiment_log.csv"
    with open(exp_log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "EXP_004",
            "matching_model",
            "Logistic Regression baseline with StandardScaler and balanced class weighting",
            "completed",
            f"train_groups={len(train_groups)}",
            f"val_groups={len(val_groups)}",
            f"best_th={lr_best_th}",
            f"macro_f05={lr_best_metrics['macro_f05']}",
            f"macro_p={lr_best_metrics['macro_precision']}",
            f"macro_r={lr_best_metrics['macro_recall']}",
            f"runtime={lr_train_time + lr_pred_time:.2f}s",
        ])
        writer.writerow([
            "EXP_005",
            "matching_model",
            "HistGradientBoostingClassifier tree model with balanced class weighting",
            "completed",
            f"train_groups={len(train_groups)}",
            f"val_groups={len(val_groups)}",
            f"best_th={tree_best_th}",
            f"macro_f05={tree_best_metrics['macro_f05']}",
            f"macro_p={tree_best_metrics['macro_precision']}",
            f"macro_r={tree_best_metrics['macro_recall']}",
            f"cv_mean_f05={cv_mean:.4f}",
            f"runtime={tree_train_time + tree_pred_time:.2f}s",
        ])
    logger.info("Logged EXP_004 and EXP_005 to %s", exp_log_path)

    # Save detailed JSON report
    report_path = REPO_ROOT / "student_resource" / "code" / "business_entity_resolution" / "reports" / "matching_model_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved matching model report to %s", report_path)

    return results


if __name__ == "__main__":
    results = run_phase4_experiment(sample_size=500)
    print("\n" + "=" * 60)
    print("PHASE 4 MATCHING MODEL EXPERIMENT COMPLETE (EXP_004 & EXP_005)")
    print("=" * 60)
    lr = results["models"]["logistic_regression"]
    tree = results["models"]["tree_hist_gradient_boosting"]
    print(f"Model 1 (Logistic Regression): Best Threshold={lr['best_threshold']}, Macro F0.5={lr['best_metrics']['macro_f05']:.4f}")
    print(f"Model 2 (HistGradientBoosting): Best Threshold={tree['best_threshold']}, Macro F0.5={tree['best_metrics']['macro_f05']:.4f}")
    print(f"3-Fold Grouped CV Macro F0.5: {results['cv_results_tree']['mean_macro_f05']:.4f} +/- {results['cv_results_tree']['std_macro_f05']:.4f}")
    print(f"Total Runtime: {results['runtime_seconds']} s")
    print("=" * 60)
