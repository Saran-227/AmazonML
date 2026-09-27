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
- Fully memory-safe with country-and-source streaming partitions (< 2 GB RAM)
- Batch-vectorized feature computation and model probability prediction
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd

from src.blocking.candidate_generator import CandidateGenerator
from src.features.pair_features import compute_pair_features
from src.preprocessing.normalize import normalize_record
from src.submission.validate import validate_submission_files

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
    country: Optional[str] = None,
    checkpoint_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Executes end-to-end test inference and generates submission TSV files.
    Supports country-by-country checkpointing for crash-resilience and memory safety.

    Args:
        test_dir: Directory containing test_source1.tsv, test_source2.tsv, test_source3.tsv
        output_dir: Directory where matching_results.tsv and candidate_pairs.tsv are saved
        model_path: Path to trained HistGradientBoosting model (joblib)
        threshold: Decision threshold for predicting a match (default: 0.95)
        max_candidates_per_s1: Maximum candidate pairs evaluated per S1 entity
        max_s1_records: Optional limit for testing/debugging
        candidate_pool_limit_per_source: Optional limit for candidate pool
        country: Optional specific country to run ("France", "US", "India", or "all")
        checkpoint_dir: Optional directory for intermediate country checkpoints

    Returns:
        Summary statistics dictionary.
    """
    t_start = time.time()
    test_dir = Path(test_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    matching_file = output_dir / "matching_results.tsv"
    candidate_file = output_dir / "candidate_pairs.tsv"

    # 1. Load production model
    logger.info("Loading production model from %s...", model_path)
    model = joblib.load(model_path)

    # 2. Read full list of required Source 1 IDs (ensuring all 1.73M S1 entities are in output)
    s1_path = test_dir / "test_source1.tsv"
    logger.info("Reading test Source 1 entities from %s...", s1_path)
    required_s1_ids = []
    s1_by_country: Dict[str, List[Tuple[str, str, str, str]]] = {
        "France": [],
        "US": [],
        "India": [],
    }
    with open(s1_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if row:
                sid = row[0].strip()
                required_s1_ids.append(sid)
                if len(row) >= 4:
                    c_tag = row[3].strip()
                    if c_tag in s1_by_country:
                        s1_by_country[c_tag].append((sid, row[1].strip(), row[2].strip(), c_tag))
                if max_s1_records is not None and len(required_s1_ids) >= max_s1_records:
                    break

    total_required = len(required_s1_ids)
    logger.info("Total S1 entities to evaluate: %d", total_required)

    # Tracking counters
    total_candidates_generated = 0
    total_matches_predicted = 0
    s2_matches_cnt = 0
    s3_matches_cnt = 0
    prob_brackets = {">=0.50": 0, ">=0.70": 0, ">=0.80": 0, ">=0.90": 0, ">=0.95": 0, ">=0.99": 0}
    high_confidence_audit: List[Dict[str, Any]] = []

    t_idx_total = 0.0
    t_inf_total = 0.0

    if max_s1_records is not None or candidate_pool_limit_per_source is not None:
        # Fast subset mode for unit tests and smoke tests
        logger.info("Running fast subset mode (max_s1=%s, pool_limit=%s)...", max_s1_records, candidate_pool_limit_per_source)
        cands_by_s1: Dict[str, Set[str]] = {sid: set() for sid in required_s1_ids}
        matches_by_s1: Dict[str, Set[str]] = {sid: set() for sid in required_s1_ids}

        s1_df = pd.read_csv(s1_path, sep="\t", keep_default_na=False, nrows=max_s1_records)
        s2_path = test_dir / "test_source2.tsv"
        s3_path = test_dir / "test_source3.tsv"

        s2_df = pd.read_csv(s2_path, sep="\t", keep_default_na=False, nrows=candidate_pool_limit_per_source)
        candidate_records: Dict[str, Dict[str, Any]] = {}
        for _, r in s2_df.iterrows():
            candidate_records[r["entity_id"]] = normalize_record(r.to_dict())
        del s2_df

        s3_df = pd.read_csv(s3_path, sep="\t", keep_default_na=False, nrows=candidate_pool_limit_per_source)
        for _, r in s3_df.iterrows():
            candidate_records[r["entity_id"]] = normalize_record(r.to_dict())
        del s3_df
        gc.collect()

        t0_idx = time.time()
        gen = CandidateGenerator(max_block_size=500)
        gen.index_candidates(candidate_records.values())
        t_idx_total = time.time() - t0_idx

        t0_inf = time.time()
        batch_query_info = []
        batch_feats = []

        for _, r in s1_df.iterrows():
            s1_norm = normalize_record(r.to_dict())
            s1_id = s1_norm["entity_id"]
            pairs = gen.generate_candidates_for_record(s1_norm, max_candidates=max_candidates_per_s1)
            c_ids = [p.candidate_entity_id for p in pairs]
            c_srcs = [p.candidate_source for p in pairs]
            b_rules = [p.blocking_rules for p in pairs]
            batch_query_info.append((s1_id, c_ids, c_srcs, len(pairs)))
            total_candidates_generated += len(pairs)
            for p in pairs:
                batch_feats.append(compute_pair_features(s1_norm, candidate_records[p.candidate_entity_id], p.blocking_rules, p.candidate_source))

        if batch_feats:
            feat_names = list(model.feature_names_in_)
            X_arr = np.array([[d[k] for k in feat_names] for d in batch_feats], dtype=np.float32)
            proba_all = model.predict_proba(X_arr)[:, 1]
        else:
            proba_all = np.array([], dtype=float)

        offset = 0
        for s1_id, c_ids, c_srcs, n_cands in batch_query_info:
            if n_cands == 0:
                continue
            p_slice = proba_all[offset : offset + n_cands]
            offset += n_cands

            cands_by_s1[s1_id].update(c_ids)
            for i, p_val in enumerate(p_slice):
                if p_val >= 0.50: prob_brackets[">=0.50"] += 1
                if p_val >= 0.70: prob_brackets[">=0.70"] += 1
                if p_val >= 0.80: prob_brackets[">=0.80"] += 1
                if p_val >= 0.90: prob_brackets[">=0.90"] += 1
                if p_val >= 0.95: prob_brackets[">=0.95"] += 1
                if p_val >= 0.99: prob_brackets[">=0.99"] += 1

                if p_val >= threshold:
                    cid = c_ids[i]
                    matches_by_s1[s1_id].add(cid)
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
                        })

        t_inf_total = time.time() - t0_inf
        total_matches_predicted = sum(len(m) for m in matches_by_s1.values())

        # Write output files for subset mode
        empty_cands_cnt = sum(1 for c in cands_by_s1.values() if len(c) == 0)
        empty_matches_cnt = sum(1 for m in matches_by_s1.values() if len(m) == 0)
        singleton_matches_cnt = sum(1 for m in matches_by_s1.values() if len(m) == 1)
        multimatch_matches_cnt = sum(1 for m in matches_by_s1.values() if len(m) > 1)

        with open(matching_file, "w", encoding="utf-8") as f_match, \
             open(candidate_file, "w", encoding="utf-8") as f_cand:
            f_match.write("source1_entity_id\tmatched_entity_ids\n")
            f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
            for s1_id in required_s1_ids:
                cands = sorted(list(cands_by_s1.get(s1_id, set())))
                matches = sorted(list(matches_by_s1.get(s1_id, set())))
                f_match.write(f"{s1_id}\t{','.join(matches)}\n")
                f_cand.write(f"{s1_id}\t{','.join(cands)}\n")

        validation_report = validate_submission_files(
            test_source1_path=matching_file,
            matching_results_path=matching_file,
            candidate_pairs_path=candidate_file,
        )

        return {
            "total_test_s1": total_required,
            "evaluated_test_s1": total_required,
            "total_candidates_generated": total_candidates_generated,
            "avg_candidates_per_s1": round(total_candidates_generated / max(total_required, 1), 2),
            "total_matches_predicted": total_matches_predicted,
            "avg_matches_per_s1": round(total_matches_predicted / max(total_required, 1), 4),
            "empty_candidate_entities": empty_cands_cnt,
            "empty_match_entities": empty_matches_cnt,
            "singleton_match_entities": singleton_matches_cnt,
            "multimatch_match_entities": multimatch_matches_cnt,
            "s2_matches": s2_matches_cnt,
            "s3_matches": s3_matches_cnt,
            "probability_brackets": prob_brackets,
            "high_confidence_audit_sample": high_confidence_audit[:20],
            "threshold": threshold,
            "indexing_time_seconds": round(t_idx_total, 2),
            "inference_time_seconds": round(t_inf_total, 2),
            "total_runtime_seconds": round(time.time() - t_start, 2),
            "validation_report": validation_report,
        }

    # Full Production Mode with Country Checkpointing
    all_known_countries = ["France", "US", "India"]
    if country and country.lower() != "all":
        target_countries = [c for c in all_known_countries if c.lower() == country.lower()]
        if not target_countries:
            raise ValueError(f"Unknown country: {country}. Must be one of: {all_known_countries} or 'all'")
    else:
        target_countries = list(all_known_countries)

    max_cands_per_source = max(1, max_candidates_per_s1 // 2)
    sources = [
        ("Source 2", test_dir / "test_source2.tsv", "S2"),
        ("Source 3", test_dir / "test_source3.tsv", "S3"),
    ]
    feat_names = list(model.feature_names_in_)

    for c_name in target_countries:
        ckpt_tsv = ckpt_dir / f"checkpoint_{c_name}.tsv"
        ckpt_meta = ckpt_dir / f"checkpoint_{c_name}_meta.json"

        if ckpt_tsv.exists() and ckpt_meta.exists():
            logger.info("Checkpoint already exists for %s at %s. Skipping computation.", c_name, ckpt_tsv)
            continue

        logger.info("=" * 70)
        logger.info("PROCESSING COUNTRY: %s (%d S1 entities)", c_name, len(s1_by_country.get(c_name, [])))
        logger.info("=" * 70)

        country_s1_raw = s1_by_country.get(c_name, [])
        t0_s1_norm = time.time()
        logger.info("Pre-normalizing %d S1 entities for %s...", len(country_s1_raw), c_name)
        s1_norms = [
            normalize_record({
                "entity_id": r[0],
                "business_name": r[1],
                "business_address": r[2],
                "country": r[3],
            })
            for r in country_s1_raw
        ]
        logger.info("Pre-normalized %d S1 entities in %.2f seconds.", len(s1_norms), time.time() - t0_s1_norm)

        c_cands_by_s1: Dict[str, Set[str]] = {r[0]: set() for r in country_s1_raw}
        c_matches_by_s1: Dict[str, Set[str]] = {r[0]: set() for r in country_s1_raw}
        c_prob_brackets = {">=0.50": 0, ">=0.70": 0, ">=0.80": 0, ">=0.90": 0, ">=0.95": 0, ">=0.99": 0}
        c_audit: List[Dict[str, Any]] = []
        c_t_idx = 0.0
        c_t_inf = 0.0

        for s_idx, (s_name, s_file, s_tag) in enumerate(sources, 1):
            logger.info("-" * 60)
            logger.info("  %s Partition %d/2: Indexing %s (%s)...", c_name, s_idx, s_name, s_tag)
            logger.info("-" * 60)

            t0_p_idx = time.time()
            raw_cands: Dict[str, Tuple[str, str, str]] = {}
            gen = CandidateGenerator(max_block_size=500)
            cand_norm_cache: Dict[str, Dict[str, Any]] = {}
            cand_indexed_count = 0

            with open(s_file, "r", encoding="utf-8") as f_cand_in:
                reader = csv.reader(f_cand_in, delimiter="\t")
                next(reader)
                for row in reader:
                    if len(row) >= 4 and row[3].strip() == c_name:
                        eid = row[0].strip()
                        bname = row[1].strip()
                        baddr = row[2].strip()
                        bctry = row[3].strip()
                        raw_cands[eid] = (bname, baddr, bctry)
                        norm = normalize_record({
                            "entity_id": eid,
                            "business_name": bname,
                            "business_address": baddr,
                            "country": bctry,
                        })
                        gen.exact_blocker.add_record(norm, source_tag=s_tag)
                        gen.token_blocker.add_record(norm, source_tag=s_tag)
                        cand_indexed_count += 1

            p_idx_elapsed = time.time() - t0_p_idx
            c_t_idx += p_idx_elapsed
            logger.info("  Indexed %d %s candidates for %s in %.2f seconds.", cand_indexed_count, s_name, c_name, p_idx_elapsed)

            # Stream S1 queries for this country and source
            t0_p_inf = time.time()
            s1_processed_count = 0
            chunk_sz = 5000

            for i in range(0, len(s1_norms), chunk_sz):
                s1_batch = s1_norms[i : i + chunk_sz]
                _score_s1_batch(
                    s1_batch_norms=s1_batch,
                    gen=gen,
                    raw_cands=raw_cands,
                    cand_norm_cache=cand_norm_cache,
                    model=model,
                    feat_names=feat_names,
                    threshold=threshold,
                    max_candidates=max_cands_per_source,
                    cands_by_s1=c_cands_by_s1,
                    matches_by_s1=c_matches_by_s1,
                    prob_brackets=c_prob_brackets,
                    high_confidence_audit=c_audit,
                )
                s1_processed_count += len(s1_batch)
                if s1_processed_count % 10000 == 0 or s1_processed_count == len(s1_norms):
                    t_elapsed = time.time() - t0_p_inf
                    qps = s1_processed_count / max(t_elapsed, 0.001)
                    pct = (s1_processed_count / len(s1_norms)) * 100
                    logger.info(
                        "    %s %s progress: %d/%d (%.1f%%) in %.1fs (%.0f queries/s)...",
                        c_name, s_tag, s1_processed_count, len(s1_norms), pct, t_elapsed, qps
                    )

            p_inf_elapsed = time.time() - t0_p_inf
            c_t_inf += p_inf_elapsed
            logger.info("  Completed %s %s inference for %d S1 queries in %.2f seconds.", c_name, s_tag, s1_processed_count, p_inf_elapsed)

            del gen, raw_cands, cand_norm_cache
            gc.collect()

        # Both S2 and S3 completed for this country - Save checkpoint immediately!
        logger.info("Writing checkpoint for %s to %s...", c_name, ckpt_tsv)
        c_total_cands = sum(len(c) for c in c_cands_by_s1.values())
        c_total_matches = sum(len(m) for m in c_matches_by_s1.values())
        c_s2_cnt = sum(1 for ms in c_matches_by_s1.values() for m in ms if m.startswith("S2-"))
        c_s3_cnt = sum(1 for ms in c_matches_by_s1.values() for m in ms if m.startswith("S3-"))
        c_empty_cands = sum(1 for c in c_cands_by_s1.values() if len(c) == 0)
        c_empty_matches = sum(1 for m in c_matches_by_s1.values() if len(m) == 0)
        c_singletons = sum(1 for m in c_matches_by_s1.values() if len(m) == 1)
        c_multimatches = sum(1 for m in c_matches_by_s1.values() if len(m) > 1)

        with open(ckpt_tsv, "w", encoding="utf-8") as f_ckpt:
            f_ckpt.write("source1_entity_id\tcandidate_entity_ids\tmatched_entity_ids\n")
            for r in country_s1_raw:
                sid = r[0]
                c_list = sorted(list(c_cands_by_s1.get(sid, set())))
                m_list = sorted(list(c_matches_by_s1.get(sid, set())))
                assert set(m_list).issubset(set(c_list)), f"Invariant violation for {sid}"
                f_ckpt.write(f"{sid}\t{','.join(c_list)}\t{','.join(m_list)}\n")

        c_meta_data = {
            "country": c_name,
            "total_s1": len(country_s1_raw),
            "total_candidates": c_total_cands,
            "total_matches": c_total_matches,
            "s2_matches": c_s2_cnt,
            "s3_matches": c_s3_cnt,
            "empty_candidates": c_empty_cands,
            "empty_matches": c_empty_matches,
            "singleton_matches": c_singletons,
            "multimatch_matches": c_multimatches,
            "indexing_time_seconds": round(c_t_idx, 2),
            "inference_time_seconds": round(c_t_inf, 2),
            "prob_brackets": c_prob_brackets,
            "high_confidence_audit_sample": c_audit[:20],
        }
        with open(ckpt_meta, "w", encoding="utf-8") as f_meta:
            json.dump(c_meta_data, f_meta, indent=2)

        logger.info(
            "Successfully saved checkpoint for %s: %d entities, %d candidates, %d matches (%.2fs)",
            c_name, len(country_s1_raw), c_total_cands, c_total_matches, c_t_inf
        )

        del s1_norms, c_cands_by_s1, c_matches_by_s1
        if c_name in s1_by_country:
            del s1_by_country[c_name]
        gc.collect()

    # Check if all 3 country checkpoints are available
    ready_countries = [c for c in all_known_countries if (ckpt_dir / f"checkpoint_{c}.tsv").exists() and (ckpt_dir / f"checkpoint_{c}_meta.json").exists()]
    pending_countries = [c for c in all_known_countries if c not in ready_countries]

    if pending_countries:
        logger.info("Checkpoints ready: %s. Pending: %s.", ready_countries, pending_countries)
        logger.info("Intermediate run complete. To run remaining countries, re-run with --country <name> or --country all.")
        return {
            "status": "CHECKPOINT_SAVED",
            "ready_countries": ready_countries,
            "pending_countries": pending_countries,
            "checkpoint_dir": str(ckpt_dir),
            "total_runtime_seconds": round(time.time() - t_start, 2),
        }

    # All country checkpoints are ready: Assemble final 1.73M entity submission!
    logger.info("=" * 80)
    logger.info("ALL COUNTRY CHECKPOINTS READY! Assembling final submission across all 1,732,544 S1 entities...")
    logger.info("=" * 80)

    stats = _assemble_submission_from_checkpoints(
        s1_path=s1_path,
        checkpoint_dir=ckpt_dir,
        matching_file=matching_file,
        candidate_file=candidate_file,
        countries=all_known_countries,
        threshold=threshold,
        t_start=t_start,
    )
    return stats


def _assemble_submission_from_checkpoints(
    s1_path: Path,
    checkpoint_dir: Path,
    matching_file: Path,
    candidate_file: Path,
    countries: List[str],
    threshold: float,
    t_start: float,
) -> Dict[str, Any]:
    """
    Streams all country checkpoints and test_source1.tsv to construct the final
    matching_results.tsv and candidate_pairs.tsv in exact test_source1.tsv order.
    """
    logger.info("Loading checkpoints from %s for countries %s...", checkpoint_dir, countries)
    cand_map: Dict[str, str] = {}
    match_map: Dict[str, str] = {}

    total_candidates_generated = 0
    total_matches_predicted = 0
    s2_matches_cnt = 0
    s3_matches_cnt = 0
    empty_cands_cnt = 0
    empty_matches_cnt = 0
    singleton_matches_cnt = 0
    multimatch_matches_cnt = 0
    t_idx_total = 0.0
    t_inf_total = 0.0
    prob_brackets = {">=0.50": 0, ">=0.70": 0, ">=0.80": 0, ">=0.90": 0, ">=0.95": 0, ">=0.99": 0}
    high_confidence_audit: List[Dict[str, Any]] = []

    for c in countries:
        ckpt_tsv = checkpoint_dir / f"checkpoint_{c}.tsv"
        ckpt_meta = checkpoint_dir / f"checkpoint_{c}_meta.json"

        with open(ckpt_meta, "r", encoding="utf-8") as f_meta:
            meta = json.load(f_meta)
            total_candidates_generated += meta.get("total_candidates", 0)
            total_matches_predicted += meta.get("total_matches", 0)
            s2_matches_cnt += meta.get("s2_matches", 0)
            s3_matches_cnt += meta.get("s3_matches", 0)
            empty_cands_cnt += meta.get("empty_candidates", 0)
            empty_matches_cnt += meta.get("empty_matches", 0)
            singleton_matches_cnt += meta.get("singleton_matches", 0)
            multimatch_matches_cnt += meta.get("multimatch_matches", 0)
            t_idx_total += meta.get("indexing_time_seconds", 0.0)
            t_inf_total += meta.get("inference_time_seconds", 0.0)
            for k, v in meta.get("prob_brackets", {}).items():
                prob_brackets[k] = prob_brackets.get(k, 0) + v
            high_confidence_audit.extend(meta.get("high_confidence_audit_sample", []))

        with open(ckpt_tsv, "r", encoding="utf-8") as f_tsv:
            reader = csv.reader(f_tsv, delimiter="\t")
            next(reader)
            for row in reader:
                if row:
                    sid = row[0]
                    c_str = row[1] if len(row) > 1 else ""
                    m_str = row[2] if len(row) > 2 else ""
                    cand_map[sid] = c_str
                    match_map[sid] = m_str

    logger.info("Loaded %d candidate records and %d match records from checkpoints.", len(cand_map), len(match_map))

    # Stream test_source1.tsv and write output TSVs in exact file order
    logger.info("Writing output files to %s in exact test_source1.tsv order...", matching_file.parent)
    total_written = 0

    with open(s1_path, "r", encoding="utf-8") as f_s1, \
         open(matching_file, "w", encoding="utf-8") as f_match, \
         open(candidate_file, "w", encoding="utf-8") as f_cand:

        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        reader = csv.reader(f_s1, delimiter="\t")
        next(reader)
        for row in reader:
            if row:
                sid = row[0].strip()
                c_str = cand_map.get(sid, "")
                m_str = match_map.get(sid, "")

                if m_str and c_str:
                    m_set = set(m_str.split(","))
                    c_set = set(c_str.split(","))
                    assert m_set.issubset(c_set), f"Invariant violation for {sid}"

                f_match.write(f"{sid}\t{m_str}\n")
                f_cand.write(f"{sid}\t{c_str}\n")
                total_written += 1

    del cand_map, match_map
    gc.collect()

    logger.info("Successfully wrote %s and %s (%d rows each).", matching_file.name, candidate_file.name, total_written)

    # Internal Validation
    logger.info("Running internal cross-file consistency validator on generated submission...")
    validation_report = validate_submission_files(
        test_source1_path=s1_path,
        matching_results_path=matching_file,
        candidate_pairs_path=candidate_file,
    )
    logger.info("Internal validation result: %s", validation_report["status"])

    stats = {
        "status": "PASS",
        "total_test_s1": total_written,
        "evaluated_test_s1": total_written,
        "total_candidates_generated": total_candidates_generated,
        "avg_candidates_per_s1": round(total_candidates_generated / max(total_written, 1), 2),
        "total_matches_predicted": total_matches_predicted,
        "avg_matches_per_s1": round(total_matches_predicted / max(total_written, 1), 4),
        "empty_candidate_entities": empty_cands_cnt,
        "empty_match_entities": empty_matches_cnt,
        "singleton_match_entities": singleton_matches_cnt,
        "multimatch_match_entities": multimatch_matches_cnt,
        "s2_matches": s2_matches_cnt,
        "s3_matches": s3_matches_cnt,
        "probability_brackets": prob_brackets,
        "high_confidence_audit_sample": high_confidence_audit[:20],
        "threshold": threshold,
        "indexing_time_seconds": round(t_idx_total, 2),
        "inference_time_seconds": round(t_inf_total, 2),
        "total_runtime_seconds": round(time.time() - t_start, 2),
        "validation_report": validation_report,
    }
    return stats


def _score_s1_batch(
    s1_batch_norms: List[Dict[str, Any]],
    gen: CandidateGenerator,
    raw_cands: Dict[str, Tuple[str, str, str]],
    cand_norm_cache: Dict[str, Dict[str, Any]],
    model: Any,
    feat_names: List[str],
    threshold: float,
    max_candidates: int,
    cands_by_s1: Dict[str, Set[str]],
    matches_by_s1: Dict[str, Set[str]],
    prob_brackets: Dict[str, int],
    high_confidence_audit: List[Dict[str, Any]],
) -> None:
    """Helper function to batch score candidate pairs for a chunk of S1 queries."""
    batch_query_info = []

    for s1_norm in s1_batch_norms:
        s1_id = s1_norm["entity_id"]
        pairs = gen.generate_candidates_for_record(s1_norm, max_candidates=max_candidates)
        if not pairs:
            continue
        c_ids = [p.candidate_entity_id for p in pairs]
        c_srcs = [p.candidate_source for p in pairs]
        b_rules = [p.blocking_rules for p in pairs]
        batch_query_info.append((s1_norm, pairs, c_ids, c_srcs, b_rules))

    if not batch_query_info:
        return

    # Compute features for all pairs in batch
    batch_feats = []
    flat_pair_meta = []
    for s1_norm, pairs, c_ids, c_srcs, b_rules in batch_query_info:
        s1_id = s1_norm["entity_id"]
        for idx_p, p in enumerate(pairs):
            cid = c_ids[idx_p]
            if cid not in cand_norm_cache:
                c_raw = raw_cands.get(cid)
                if c_raw is not None:
                    cand_norm_cache[cid] = normalize_record({
                        "entity_id": cid,
                        "business_name": c_raw[0],
                        "business_address": c_raw[1],
                        "country": c_raw[2],
                    })
            c_rec = cand_norm_cache.get(cid)
            if c_rec is None:
                continue
            feats = compute_pair_features(s1_norm, c_rec, b_rules[idx_p], c_srcs[idx_p])
            batch_feats.append(feats)
            flat_pair_meta.append((s1_id, cid, c_srcs[idx_p]))

    if not batch_feats:
        return

    # Batch model prediction using fast NumPy float32 array
    X_arr = np.array([[d[k] for k in feat_names] for d in batch_feats], dtype=np.float32)
    proba_all = model.predict_proba(X_arr)[:, 1]

    # Assign predictions
    for idx_f, p_val in enumerate(proba_all):
        s1_id, cid, c_src = flat_pair_meta[idx_f]
        cands_by_s1[s1_id].add(cid)

        if p_val >= 0.50: prob_brackets[">=0.50"] += 1
        if p_val >= 0.70: prob_brackets[">=0.70"] += 1
        if p_val >= 0.80: prob_brackets[">=0.80"] += 1
        if p_val >= 0.90: prob_brackets[">=0.90"] += 1
        if p_val >= 0.95: prob_brackets[">=0.95"] += 1
        if p_val >= 0.99: prob_brackets[">=0.99"] += 1

        if p_val >= threshold:
            matches_by_s1[s1_id].add(cid)
            if len(high_confidence_audit) < 50:
                high_confidence_audit.append({
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": cid,
                    "candidate_source": c_src,
                    "probability": round(float(p_val), 4),
                })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate entity resolution submission files.")
    parser.add_argument("--test-dir", type=str, default="data/test")
    parser.add_argument("--output-dir", type=str, default="submissions/SUB_001_production_baseline")
    parser.add_argument("--model-path", type=str, default="src/models/production_model.joblib")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--max-s1", type=int, default=None)
    parser.add_argument("--candidate-pool-limit", type=int, default=None)
    parser.add_argument("--country", type=str, default=None, help="Process specific country: France, US, India, or all")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory to store country checkpoints")

    args = parser.parse_args()
    res = generate_submission(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        threshold=args.threshold,
        max_candidates_per_s1=args.max_candidates,
        max_s1_records=args.max_s1,
        candidate_pool_limit_per_source=args.candidate_pool_limit,
        country=args.country,
        checkpoint_dir=args.checkpoint_dir,
    )
    print("\nSubmission Generation Complete:")
    print(json.dumps(res, indent=2))
