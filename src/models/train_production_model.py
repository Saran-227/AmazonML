"""
Production model training script for Phase 5.

Trains the final HistGradientBoostingClassifier model using labeled training data
spanning both US and India jurisdictions:
1. Loads representative S1 sample and true match candidate pool
2. Normalizes records with Saran's preprocessing
3. Generates candidate pairs using Phase-2 CandidateGenerator
4. Extracts 60-dimensional pairwise features and binary ground-truth labels
5. Validates feature matrix (no IDs, no NaN, no inf, identical 60 columns)
6. Fits HistGradientBoostingClassifier with balanced class weighting
7. Serializes trained model and feature metadata for high-speed test inference
"""

from __future__ import annotations

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

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.blocking.candidate_generator import CandidateGenerator
from src.features.pair_features import build_feature_dataset
from src.models.train import validate_feature_matrix
from src.preprocessing.normalize import normalize_record

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PRODUCTION_MODEL_PATH = REPO_ROOT / "src" / "models" / "production_model.joblib"
METADATA_PATH = REPO_ROOT / "src" / "models" / "model_metadata.json"


def train_production_model(
    sample_size_per_country: int = 1000,
    background_pool_per_source: int = 30000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Trains and saves the final production entity matching model.
    """
    t_start = time.time()
    logger.info("Starting production model training pipeline...")

    dataset_dir = REPO_ROOT / "student_resource" / "dataset" / "train"
    gt_file = dataset_dir / "train_ground_truth.tsv"
    s1_file = dataset_dir / "train_source1.tsv"
    s2_file = dataset_dir / "train_source2.tsv"
    s3_file = dataset_dir / "train_source3.tsv"

    # 1. Load Ground Truth map
    logger.info("Reading ground truth mapping...")
    gt_df = pd.read_csv(gt_file, sep="\t", keep_default_na=False)
    gt_map: Dict[str, Set[str]] = {}
    for _, r in gt_df.iterrows():
        s1_id = r["source1_entity_id"]
        matched = [x.strip() for x in r["matched_entity_ids"].split(",") if x.strip()]
        gt_map[s1_id] = set(matched)

    # 2. Sample balanced S1 records across India and US
    logger.info("Sampling Source 1 records across India and US...")
    s1_df = pd.read_csv(s1_file, sep="\t", keep_default_na=False)
    s1_in = s1_df[s1_df["country"].str.lower() == "india"].head(sample_size_per_country)
    s1_us = s1_df[s1_df["country"].str.lower() == "us"].head(sample_size_per_country)
    s1_sample = pd.concat([s1_in, s1_us], ignore_index=True)
    del s1_df, s1_in, s1_us
    gc.collect()

    s1_sample_ids = set(s1_sample["entity_id"])
    target_gt_map = {s1: gt_map.get(s1, set()) for s1 in s1_sample_ids}
    all_true_ids = set().union(*target_gt_map.values())
    total_true_matches = sum(len(v) for v in target_gt_map.values())

    logger.info("Sampled %d S1 entities (%d true matches).", len(s1_sample), total_true_matches)

    # Normalize S1 records
    s1_records: Dict[str, Dict[str, Any]] = {}
    for _, r in s1_sample.iterrows():
        s1_records[r["entity_id"]] = normalize_record(r.to_dict())
    del s1_sample
    gc.collect()

    # 3. Load candidate pool (true matches + background)
    logger.info("Loading candidate records from Source 2 and Source 3...")
    s2_df = pd.read_csv(s2_file, sep="\t", keep_default_na=False)
    s2_matches = s2_df[s2_df["entity_id"].isin(all_true_ids)]
    s2_bg = s2_df[~s2_df["entity_id"].isin(all_true_ids)].head(background_pool_per_source)
    s2_pool = pd.concat([s2_matches, s2_bg], ignore_index=True)
    del s2_df, s2_matches, s2_bg
    gc.collect()

    s3_df = pd.read_csv(s3_file, sep="\t", keep_default_na=False)
    s3_matches = s3_df[s3_df["entity_id"].isin(all_true_ids)]
    s3_bg = s3_df[~s3_df["entity_id"].isin(all_true_ids)].head(background_pool_per_source)
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

    logger.info("Indexed candidate pool: %d records.", len(candidate_records))

    # 4. Generate candidate pairs using Phase-2 CandidateGenerator
    logger.info("Running CandidateGenerator on training sample...")
    gen = CandidateGenerator()
    gen.index_candidates(candidate_records.values())

    candidate_pairs = []
    for s1 in s1_records.values():
        pairs = gen.generate_candidates_for_record(s1)
        candidate_pairs.extend(pairs)

    num_pairs = len(candidate_pairs)
    logger.info("Generated %d candidate pairs for training.", num_pairs)

    # 5. Extract 60-dimensional features and ground-truth labels
    logger.info("Extracting pairwise features for all %d candidate pairs...", num_pairs)
    features_df, metadata_df, labels = build_feature_dataset(
        candidate_pairs=candidate_pairs,
        s1_records=s1_records,
        candidate_records=candidate_records,
        ground_truth_map=target_gt_map,
    )

    feature_names = validate_feature_matrix(features_df)
    pos_count = int(np.sum(labels == 1))
    neg_count = int(np.sum(labels == 0))
    logger.info(
        "Feature matrix shape: %s | Positives: %d (%.2f%%) | Negatives: %d",
        features_df.shape, pos_count, (pos_count / num_pairs * 100), neg_count
    )

    # 6. Fit HistGradientBoostingClassifier
    logger.info("Training HistGradientBoostingClassifier (trees=150, class_weight='balanced')...")
    t_train_start = time.time()
    model = HistGradientBoostingClassifier(
        class_weight="balanced",
        max_iter=150,
        max_leaf_nodes=31,
        learning_rate=0.1,
        min_samples_leaf=20,
        random_state=random_state,
    )
    model.fit(features_df, labels)
    t_train = time.time() - t_train_start
    logger.info("Model fitted successfully in %.3f seconds.", t_train)

    # 7. Serialize model and metadata
    logger.info("Saving production model to %s...", PRODUCTION_MODEL_PATH)
    joblib.dump(model, PRODUCTION_MODEL_PATH)

    metadata = {
        "model_type": "HistGradientBoostingClassifier",
        "training_samples": num_pairs,
        "positive_pairs": pos_count,
        "negative_pairs": neg_count,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "default_threshold": 0.95,
        "training_time_seconds": round(t_train, 3),
        "total_pipeline_time_seconds": round(time.time() - t_start, 2),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Production model and metadata saved successfully!")
    return metadata


if __name__ == "__main__":
    train_production_model()
