"""
Master Accelerated Runner for India Partition & Final Phase 7 Assembly.

1. Concurrently executes 3 worker processes across India's 809,986 S1 queries.
   - Worker 0: [0, 270000)      -> checkpoint_India_part0.tsv
   - Worker 1: [270000, 540000)  -> checkpoint_India_part1.tsv
   - Worker 2: [540000, 809986)  -> checkpoint_India_part2.tsv
2. Merges intermediate parts into unified checkpoint_India.tsv & checkpoint_India_meta.json.
3. Automatically triggers final Phase 7 assembly, validation, and reporting.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_india_parallel")

CKPT_DIR = REPO_ROOT / "submissions" / "SUB_001_production_baseline" / "checkpoints"


def run_parallel_india():
    t_start = time.time()
    logger.info("=" * 80)
    logger.info("STARTING ACCELERATED PARALLEL INDIA EXECUTION (3 WORKERS)")
    logger.info("=" * 80)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    reports_dir = REPO_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Check if unified checkpoint_India.tsv already exists
    final_india_tsv = CKPT_DIR / "checkpoint_India.tsv"
    final_india_meta = CKPT_DIR / "checkpoint_India_meta.json"

    if final_india_tsv.exists() and final_india_meta.exists():
        logger.info("Unified checkpoint_India.tsv already exists! Proceeding directly to final assembly.")
    else:
        # Define 3 slices for India (total 809,986)
        slices = [
            ("part0", 0, 270000),
            ("part1", 270000, 540000),
            ("part2", 540000, 809986),
        ]

        procs: List[subprocess.Popen] = []
        log_files = []

        for part_name, s_start, s_end in slices:
            part_tsv = CKPT_DIR / f"checkpoint_India_{part_name}.tsv"
            part_meta = CKPT_DIR / f"checkpoint_India_{part_name}_meta.json"
            if part_tsv.exists() and part_meta.exists():
                logger.info("Slice %s already complete (%s).", part_name, part_tsv)
                continue

            log_path = reports_dir / f"india_{part_name}.log"
            f_log = open(log_path, "w", encoding="utf-8")
            log_files.append(f_log)

            cmd = [
                sys.executable,
                "-m", "src.submission.generate_submission",
                "--country", "India",
                "--s1-slice-start", str(s_start),
                "--s1-slice-end", str(s_end),
                "--part-name", part_name,
            ]
            logger.info("Launching Worker %s: range [%d, %d) -> %s", part_name, s_start, s_end, log_path)
            proc = subprocess.Popen(
                cmd,
                stdout=f_log,
                stderr=subprocess.STDOUT,
                cwd=str(REPO_ROOT),
            )
            procs.append(proc)

        # Wait for all workers to complete
        logger.info("All %d active worker processes launched. Monitoring progress...", len(procs))
        while procs:
            time.sleep(15)
            finished = []
            for p in procs:
                ret = p.poll()
                if ret is not None:
                    finished.append(p)
                    if ret != 0:
                        logger.error("A worker process failed with exit code %d!", ret)
                        raise RuntimeError(f"Worker process failed with exit code {ret}")
            for f in finished:
                procs.remove(f)
            if procs:
                logger.info("Waiting for %d remaining worker process(es)... (elapsed: %.1fs)", len(procs), time.time() - t_start)

        for f_log in log_files:
            f_log.close()

        logger.info("All 3 India slice workers completed successfully!")

        # Merge slices into unified checkpoint_India.tsv
        logger.info("Merging 3 India slice checkpoints into %s...", final_india_tsv)
        total_s1 = 0
        total_cands = 0
        total_matches = 0
        s2_matches = 0
        s3_matches = 0
        empty_cands = 0
        empty_matches = 0
        singleton_matches = 0
        multimatch_matches = 0
        idx_time = 0.0
        inf_time = 0.0
        prob_brackets: Dict[str, int] = {}
        audit_sample: List[Dict[str, Any]] = []

        with open(final_india_tsv, "w", encoding="utf-8") as f_out:
            f_out.write("source1_entity_id\tcandidate_entity_ids\tmatched_entity_ids\n")
            for part_name, _, _ in slices:
                part_tsv = CKPT_DIR / f"checkpoint_India_{part_name}.tsv"
                part_meta_path = CKPT_DIR / f"checkpoint_India_{part_name}_meta.json"

                with open(part_meta_path, "r", encoding="utf-8") as fm:
                    pm = json.load(fm)
                    total_s1 += pm.get("total_s1", 0)
                    total_cands += pm.get("total_candidates", 0)
                    total_matches += pm.get("total_matches", 0)
                    s2_matches += pm.get("s2_matches", 0)
                    s3_matches += pm.get("s3_matches", 0)
                    empty_cands += pm.get("empty_candidates", 0)
                    empty_matches += pm.get("empty_matches", 0)
                    singleton_matches += pm.get("singleton_matches", 0)
                    multimatch_matches += pm.get("multimatch_matches", 0)
                    idx_time = max(idx_time, pm.get("indexing_time_seconds", 0.0))
                    inf_time = max(inf_time, pm.get("inference_time_seconds", 0.0))
                    for k, v in pm.get("prob_brackets", {}).items():
                        prob_brackets[k] = prob_brackets.get(k, 0) + v
                    audit_sample.extend(pm.get("high_confidence_audit_sample", []))

                with open(part_tsv, "r", encoding="utf-8") as ft:
                    r = csv.reader(ft, delimiter="\t")
                    next(r)  # skip header
                    for row in r:
                        if row:
                            f_out.write(f"{row[0]}\t{row[1] if len(row) > 1 else ''}\t{row[2] if len(row) > 2 else ''}\n")

        india_meta = {
            "country": "India",
            "total_s1": total_s1,
            "total_candidates": total_cands,
            "total_matches": total_matches,
            "s2_matches": s2_matches,
            "s3_matches": s3_matches,
            "empty_candidates": empty_cands,
            "empty_matches": empty_matches,
            "singleton_matches": singleton_matches,
            "multimatch_matches": multimatch_matches,
            "indexing_time_seconds": round(idx_time, 2),
            "inference_time_seconds": round(inf_time, 2),
            "prob_brackets": prob_brackets,
            "high_confidence_audit_sample": audit_sample[:20],
        }
        with open(final_india_meta, "w", encoding="utf-8") as fm_out:
            json.dump(india_meta, fm_out, indent=2)

        logger.info(
            "Successfully saved unified checkpoint for India: %d entities, %d candidates, %d matches (wall time: %.2fs)",
            total_s1, total_cands, total_matches, time.time() - t_start
        )

    # 4. Final Assembly & Validation Across All Countries
    logger.info("=" * 80)
    logger.info("ALL THREE COUNTRY CHECKPOINTS READY (France, US, India)!")
    logger.info("EXECUTING FINAL PHASE 7 ASSEMBLY, VALIDATION & REPORTING...")
    logger.info("=" * 80)

    from scripts.run_phase7_integration import run_phase7
    final_report = run_phase7(country="all")
    logger.info("=" * 80)
    logger.info("ALL TASKS COMPLETED SUCCESSFULLY! SUBMISSION IS READY FOR REVIEW.")
    logger.info("=" * 80)
    return final_report


if __name__ == "__main__":
    run_parallel_india()
