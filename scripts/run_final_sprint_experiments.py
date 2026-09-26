"""
Comprehensive Final Sprint Experiments:
1. Blocking Rule Evaluation (Baseline 6-rule vs Domain-Split 7-rule vs EXP_BLOCKING_001 8-rule)
2. Downstream Model Evaluation on Candidate Sets (Macro F0.5, Precision, Recall)
3. Production Threshold Optimization (Global 0.90 - 0.98 and Source-Specific Grid)
4. Feature Ablation (Baseline 60 features vs 60 + Cross-Script Address Interaction)

Produces:
- reports/final_blocking_comparison.csv
- reports/final_threshold_optimization.csv
- reports/final_feature_ablation.csv
- reports/final_sprint_experiments.json
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

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from experiments.EXP_BLOCKING_001.run_blocking_experiment_001 import ExperimentalCandidateGenerator
from src.blocking.candidate_generator import CandidateGenerator, CandidatePair
from src.evaluation.metrics import evaluate_entity_level
from src.evaluation.validation import split_grouped_dataset, verify_group_leakage
from src.features.pair_features import build_feature_dataset
from src.models.predict import predict_matches_at_threshold, predict_probabilities
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
    passes = np.where(is_s2, probabilities >= threshold_s2, probabilities >= threshold_s3)

    for idx in np.where(passes)[0]:
        s1 = s1_col[idx]
        c = cand_col[idx]
        if s1 not in pred_map:
            pred_map[s1] = set()
        pred_map[s1].add(c)

    return pred_map


def run_experiments():
    reports_dir = REPO_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = REPO_ROOT / "student_resource" / "dataset" / "train"

    sample_size_s1 = 1000
    bg_pool_size = 35000

    logger.info("Loading ground truth (sample size = %d S1)...", sample_size_s1)
    gt_df = pd.read_csv(dataset_dir / "train_ground_truth.tsv", sep="\t", keep_default_na=False, nrows=sample_size_s1)
    gt_map: Dict[str, Set[str]] = {}
    all_true_matched_ids: Set[str] = set()
    for _, r in gt_df.iterrows():
        s1_id = r["source1_entity_id"]
        matched = [x.strip() for x in r["matched_entity_ids"].split(",") if x.strip()]
        gt_map[s1_id] = set(matched)
        all_true_matched_ids.update(matched)

    s1_target_ids = set(gt_map.keys())

    logger.info("Loading and normalizing Source-1 records...")
    s1_df = pd.read_csv(dataset_dir / "train_source1.tsv", sep="\t", keep_default_na=False)
    s1_sample = s1_df[s1_df["entity_id"].isin(s1_target_ids)]
    s1_records = {r["entity_id"]: normalize_record(r.to_dict()) for _, r in s1_sample.iterrows()}
    del s1_df, s1_sample
    gc.collect()

    logger.info("Loading candidate pool (S2 and S3, bg_size=%d + all true matches)...", bg_pool_size)
    s2_df = pd.read_csv(dataset_dir / "train_source2.tsv", sep="\t", keep_default_na=False)
    s2_matches = s2_df[s2_df["entity_id"].isin(all_true_matched_ids)]
    s2_bg = s2_df[~s2_df["entity_id"].isin(all_true_matched_ids)].head(bg_pool_size)
    s2_pool = pd.concat([s2_matches, s2_bg], ignore_index=True)
    del s2_df, s2_matches, s2_bg
    gc.collect()

    s3_df = pd.read_csv(dataset_dir / "train_source3.tsv", sep="\t", keep_default_na=False)
    s3_matches = s3_df[s3_df["entity_id"].isin(all_true_matched_ids)]
    s3_bg = s3_df[~s3_df["entity_id"].isin(all_true_matched_ids)].head(bg_pool_size)
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

    total_true_matches = sum(len(v) for v in gt_map.values())
    total_s2_matches = sum(len([m for m in v if m.startswith("S2-")]) for v in gt_map.values())
    total_s3_matches = sum(len([m for m in v if m.startswith("S3-")]) for v in gt_map.values())

    # =========================================================================
    # PART 1: BLOCKING COMPARISON
    # =========================================================================
    logger.info("--- PART 1: EVALUATING BLOCKING CONFIGURATIONS ---")
    blocking_configs = [
        {
            "name": "baseline_6rule",
            "description": "Production Baseline (6 rules)",
            "generator": CandidateGenerator(max_block_size=500),
        },
        {
            "name": "domain_tokens_7rule",
            "description": "Baseline 6 rules + domain_subtoken splitting",
            "generator": ExperimentalCandidateGenerator(
                max_block_size=500, enable_cross_script_address=False, enable_domain_tokens=True
            ),
        },
        {
            "name": "exp_blocking_001_8rule",
            "description": "EXP_BLOCKING_001 (8 rules: +domain +cross_script_address)",
            "generator": ExperimentalCandidateGenerator(
                max_block_size=500, enable_cross_script_address=True, enable_domain_tokens=True
            ),
        },
    ]

    blocking_results = []
    generated_pairs_dict = {}

    for bconf in blocking_configs:
        gen = bconf["generator"]
        t0 = time.time()
        gen.index_candidates(candidate_records.values())
        t_index = time.time() - t0

        t1 = time.time()
        pairs: List[CandidatePair] = []
        captured = 0
        captured_s2 = 0
        captured_s3 = 0
        counts_per_s1 = []

        for s1_id, s1_rec in s1_records.items():
            s1_pairs = gen.generate_candidates_for_record(s1_rec)
            pairs.extend(s1_pairs)
            counts_per_s1.append(len(s1_pairs))
            cand_ids = {p.candidate_entity_id for p in s1_pairs}
            true_set = gt_map.get(s1_id, set())
            hits = cand_ids & true_set
            captured += len(hits)
            captured_s2 += len([h for h in hits if h.startswith("S2-")])
            captured_s3 += len([h for h in hits if h.startswith("S3-")])

        t_gen = time.time() - t1
        generated_pairs_dict[bconf["name"]] = pairs

        counts_arr = np.array(counts_per_s1)
        res = {
            "config_name": bconf["name"],
            "description": bconf["description"],
            "total_candidates": len(pairs),
            "recall_overall_pct": round(captured / total_true_matches * 100, 4),
            "recall_s2_pct": round(captured_s2 / total_s2_matches * 100, 4),
            "recall_s3_pct": round(captured_s3 / total_s3_matches * 100, 4),
            "captured_matches": captured,
            "missed_matches": total_true_matches - captured,
            "avg_cands_per_s1": round(float(np.mean(counts_arr)), 2),
            "median_cands_per_s1": round(float(np.median(counts_arr)), 2),
            "p95_cands_per_s1": round(float(np.percentile(counts_arr, 95)), 2),
            "max_cands_per_s1": int(np.max(counts_arr)),
            "indexing_time_sec": round(t_index, 2),
            "generation_time_sec": round(t_gen, 2),
            "total_time_sec": round(t_index + t_gen, 2),
        }
        blocking_results.append(res)
        logger.info(
            "[%s] Recall=%.2f%% (S2=%.2f%%, S3=%.2f%%) | Pairs=%d | Avg/S1=%.1f | Time=%.2fs",
            bconf["name"], res["recall_overall_pct"], res["recall_s2_pct"],
            res["recall_s3_pct"], res["total_candidates"], res["avg_cands_per_s1"], res["total_time_sec"]
        )

    # Save blocking comparison CSV
    with open(reports_dir / "final_blocking_comparison.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(blocking_results[0].keys()))
        writer.writeheader()
        writer.writerows(blocking_results)

    # =========================================================================
    # PART 2: DOWNSTREAM PIPELINE & THRESHOLD OPTIMIZATION (ON PRODUCTION BASELINE POOL)
    # =========================================================================
    logger.info("--- PART 2: DOWNSTREAM PIPELINE ON BASELINE CANDIDATES ---")
    base_pairs = generated_pairs_dict["baseline_6rule"]

    logger.info("Extracting 60 pairwise features on %d baseline pairs...", len(base_pairs))
    t0 = time.time()
    features_df, metadata_df, labels = build_feature_dataset(
        candidate_pairs=base_pairs,
        s1_records=s1_records,
        candidate_records=candidate_records,
        ground_truth_map=gt_map,
    )
    validate_feature_matrix(features_df)
    t_feat = time.time() - t0
    logger.info("Feature extraction complete in %.2f seconds.", t_feat)

    # Leakage-safe grouped split
    train_idx, val_idx, train_groups, val_groups = split_grouped_dataset(
        metadata_df=metadata_df, val_ratio=0.25, group_col="source1_entity_id", random_state=42
    )
    verify_group_leakage(train_groups, val_groups)

    X_train = features_df.iloc[train_idx].reset_index(drop=True)
    y_train = labels[train_idx]
    meta_train = metadata_df.iloc[train_idx].reset_index(drop=True)

    X_val = features_df.iloc[val_idx].reset_index(drop=True)
    y_val = labels[val_idx]
    meta_val = metadata_df.iloc[val_idx].reset_index(drop=True)

    logger.info("Training HistGradientBoostingClassifier on train split (%d pairs, %d groups)...", len(X_train), len(train_groups))
    t0 = time.time()
    model = HistGradientBoostingClassifier(
        class_weight="balanced",
        max_iter=150,
        max_leaf_nodes=31,
        learning_rate=0.1,
        min_samples_leaf=20,
        random_state=42,
    )
    model.fit(X_train, y_train)
    t_train = time.time() - t0
    logger.info("Model training complete in %.2f seconds.", t_train)

    val_probs, _ = predict_probabilities(model, X_val)
    val_gt_map = {s1: gt_map.get(s1, set()) for s1 in val_groups}
    val_s1_ids = sorted(list(val_groups))

    # Evaluate Threshold Grid
    threshold_grid = [
        # Global
        {"name": "global_0.90", "type": "global", "th_s2": 0.90, "th_s3": 0.90},
        {"name": "global_0.92", "type": "global", "th_s2": 0.92, "th_s3": 0.92},
        {"name": "global_0.93", "type": "global", "th_s2": 0.93, "th_s3": 0.93},
        {"name": "global_0.94", "type": "global", "th_s2": 0.94, "th_s3": 0.94},
        {"name": "global_0.95", "type": "global", "th_s2": 0.95, "th_s3": 0.95},
        {"name": "global_0.96", "type": "global", "th_s2": 0.96, "th_s3": 0.96},
        {"name": "global_0.97", "type": "global", "th_s2": 0.97, "th_s3": 0.97},
        {"name": "global_0.98", "type": "global", "th_s2": 0.98, "th_s3": 0.98},
        # Source-specific
        {"name": "src_S2_0.94_S3_0.95", "type": "source_specific", "th_s2": 0.94, "th_s3": 0.95},
        {"name": "src_S2_0.95_S3_0.94", "type": "source_specific", "th_s2": 0.95, "th_s3": 0.94},
        {"name": "src_S2_0.95_S3_0.96", "type": "source_specific", "th_s2": 0.95, "th_s3": 0.96},
        {"name": "src_S2_0.96_S3_0.95", "type": "source_specific", "th_s2": 0.96, "th_s3": 0.95},
        {"name": "src_S2_0.92_S3_0.95", "type": "source_specific", "th_s2": 0.92, "th_s3": 0.95},
        {"name": "src_S2_0.93_S3_0.95", "type": "source_specific", "th_s2": 0.93, "th_s3": 0.95},
    ]

    th_results = []
    for cfg in threshold_grid:
        pred_map = predict_matches_source_specific(
            val_probs, meta_val, cfg["th_s2"], cfg["th_s3"], all_s1_ids=val_s1_ids
        )
        metrics = evaluate_entity_level(val_gt_map, pred_map)

        # Count false positives at pair level
        pred_pairs = {(s1, c) for s1, cands in pred_map.items() for c in cands}
        true_pairs = {(s1, c) for s1, cands in val_gt_map.items() for c in cands}
        fp_count = len(pred_pairs - true_pairs)
        tp_count = len(pred_pairs & true_pairs)
        total_pred = len(pred_pairs)

        match_lens = [len(cands) for cands in pred_map.values()]
        zero_cnt = sum(1 for l in match_lens if l == 0)
        single_cnt = sum(1 for l in match_lens if l == 1)
        multi_cnt = sum(1 for l in match_lens if l > 1)

        th_res = {
            "config_name": cfg["name"],
            "type": cfg["type"],
            "th_s2": cfg["th_s2"],
            "th_s3": cfg["th_s3"],
            "macro_f05": metrics["macro_f05"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "micro_f05": metrics["micro_f05"],
            "micro_precision": metrics["micro_precision"],
            "micro_recall": metrics["micro_recall"],
            "pair_fp_count": fp_count,
            "pair_tp_count": tp_count,
            "predicted_matches": total_pred,
            "zero_match_entities": zero_cnt,
            "singleton_entities": single_cnt,
            "multimatch_entities": multi_cnt,
        }
        th_results.append(th_res)
        logger.info(
            "[%s] Macro F0.5=%.4f | Prec=%.4f | Rec=%.4f | FP=%d | Pred=%d",
            cfg["name"], metrics["macro_f05"], metrics["macro_precision"], metrics["macro_recall"], fp_count, total_pred
        )

    # Sort by macro_f05 descending
    th_results.sort(key=lambda x: x["macro_f05"], reverse=True)
    with open(reports_dir / "final_threshold_optimization.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(th_results[0].keys()))
        writer.writeheader()
        writer.writerows(th_results)

    # =========================================================================
    # PART 3: DOWNSTREAM EVALUATION ON THE OTHER BLOCKERS
    # =========================================================================
    logger.info("--- PART 3: DOWNSTREAM MACRO F0.5 EVALUATION ACROSS BLOCKERS AT THRESHOLD 0.95 ---")
    blocker_downstream_eval = []

    for bname in ["baseline_6rule", "domain_tokens_7rule", "exp_blocking_001_8rule"]:
        logger.info("Evaluating blocker: %s", bname)
        b_pairs = generated_pairs_dict[bname]
        b_feat_df, b_meta_df, b_labels = build_feature_dataset(
            candidate_pairs=b_pairs,
            s1_records=s1_records,
            candidate_records=candidate_records,
            ground_truth_map=gt_map,
        )
        validate_feature_matrix(b_feat_df)

        b_train_idx, b_val_idx, b_tr_grp, b_val_grp = split_grouped_dataset(
            metadata_df=b_meta_df, val_ratio=0.25, group_col="source1_entity_id", random_state=42
        )
        verify_group_leakage(b_tr_grp, b_val_grp)

        b_X_tr = b_feat_df.iloc[b_train_idx].reset_index(drop=True)
        b_y_tr = b_labels[b_train_idx]
        b_X_vl = b_feat_df.iloc[b_val_idx].reset_index(drop=True)
        b_meta_vl = b_meta_df.iloc[b_val_idx].reset_index(drop=True)

        b_model = HistGradientBoostingClassifier(
            class_weight="balanced", max_iter=150, max_leaf_nodes=31, learning_rate=0.1, min_samples_leaf=20, random_state=42
        )
        b_model.fit(b_X_tr, b_y_tr)
        b_probs, _ = predict_probabilities(b_model, b_X_vl)

        b_val_gt = {s1: gt_map.get(s1, set()) for s1 in b_val_grp}
        b_pred_map = predict_matches_at_threshold(b_probs, b_meta_vl, threshold=0.95, all_s1_ids=sorted(list(b_val_grp)))
        b_met = evaluate_entity_level(b_val_gt, b_pred_map)

        blocker_downstream_eval.append({
            "blocker": bname,
            "macro_f05": b_met["macro_f05"],
            "macro_precision": b_met["macro_precision"],
            "macro_recall": b_met["macro_recall"],
            "micro_f05": b_met["micro_f05"],
        })
        logger.info(
            "Blocker %s -> Macro F0.5 = %.4f (Prec=%.4f, Rec=%.4f)",
            bname, b_met["macro_f05"], b_met["macro_precision"], b_met["macro_recall"]
        )

    # =========================================================================
    # PART 4: FEATURE ABLATION (Cross-Script Interaction)
    # =========================================================================
    logger.info("--- PART 4: FEATURE ABLATION EXPERIMENT ---")
    # Base feature matrix: 60 features
    # Candidate feature: cross_script_address_agreement = is_cross_script * addr_jaccard
    X_train_augmented = X_train.copy()
    X_val_augmented = X_val.copy()

    X_train_augmented["feat_cross_script_addr_jaccard"] = (
        X_train["is_cross_script"] * X_train["address_token_jaccard"]
    )
    X_val_augmented["feat_cross_script_addr_jaccard"] = (
        X_val["is_cross_script"] * X_val["address_token_jaccard"]
    )
    validate_feature_matrix(X_train_augmented, X_val_augmented)

    model_aug = HistGradientBoostingClassifier(
        class_weight="balanced", max_iter=150, max_leaf_nodes=31, learning_rate=0.1, min_samples_leaf=20, random_state=42
    )
    model_aug.fit(X_train_augmented, y_train)
    val_probs_aug, _ = predict_probabilities(model_aug, X_val_augmented)
    pred_map_aug = predict_matches_at_threshold(val_probs_aug, meta_val, threshold=0.95, all_s1_ids=val_s1_ids)
    metrics_aug = evaluate_entity_level(val_gt_map, pred_map_aug)

    # Base at 0.95:
    base_res_95 = next(r for r in th_results if r["config_name"] == "global_0.95")

    ablation_rows = [
        {
            "configuration": "Baseline 60 Features",
            "num_features": 60,
            "macro_f05": base_res_95["macro_f05"],
            "macro_precision": base_res_95["macro_precision"],
            "macro_recall": base_res_95["macro_recall"],
            "decision": "ACCEPTED_PRODUCTION",
        },
        {
            "configuration": "Baseline 60 Features + CrossScript-Address Interaction",
            "num_features": 61,
            "macro_f05": metrics_aug["macro_f05"],
            "macro_precision": metrics_aug["macro_precision"],
            "macro_recall": metrics_aug["macro_recall"],
            "decision": "REJECT" if metrics_aug["macro_f05"] <= base_res_95["macro_f05"] else "CANDIDATE",
        },
    ]

    with open(reports_dir / "final_feature_ablation.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ablation_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ablation_rows)

    # =========================================================================
    # PART 5: SAVE MASTER EXPERIMENT JSON
    # =========================================================================
    master_results = {
        "sample_size_s1": sample_size_s1,
        "candidate_pool_size": len(candidate_records),
        "blocking_comparison": blocking_results,
        "blocker_downstream_evaluation": blocker_downstream_eval,
        "threshold_optimization": th_results,
        "feature_ablation": ablation_rows,
        "best_overall_configuration": {
            "blocking": "baseline_6rule",
            "features": "60_features_baseline",
            "model": "HistGradientBoostingClassifier",
            "threshold": 0.95,
            "macro_f05": base_res_95["macro_f05"],
            "macro_precision": base_res_95["macro_precision"],
            "macro_recall": base_res_95["macro_recall"],
        },
    }

    with open(reports_dir / "final_sprint_experiments.json", "w", encoding="utf-8") as f:
        json.dump(master_results, f, indent=2)
    logger.info("Saved final sprint experiments to %s", reports_dir / "final_sprint_experiments.json")


if __name__ == "__main__":
    run_experiments()
