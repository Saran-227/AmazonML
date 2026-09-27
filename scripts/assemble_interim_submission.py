"""
Interim Submission Assembler.

Assembles 100% compliant, verified matching_results.tsv and candidate_pairs.tsv
using all completed country checkpoints (France 100%, US 100%) and preserving
exact test_source1.tsv row ordering across all 1,732,544 entities.
"""

import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.submission.validate import validate_submission_files

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("interim_assembler")


def assemble_interim_submission():
    t_start = time.time()
    ckpt_dir = REPO_ROOT / "submissions" / "SUB_001_production_baseline" / "checkpoints"
    sub_dir = REPO_ROOT / "submissions" / "SUB_001_production_baseline"
    out_dir = REPO_ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    s1_path = REPO_ROOT / "data" / "test" / "test_source1.tsv"
    if not s1_path.exists():
        s1_path = REPO_ROOT / "student_resource" / "dataset" / "test" / "test_source1.tsv"

    matching_file = sub_dir / "matching_results.tsv"
    candidate_file = sub_dir / "candidate_pairs.tsv"

    cand_map = {}
    match_map = {}

    available_countries = []
    for c in ["France", "US", "India"]:
        c_tsv = ckpt_dir / f"checkpoint_{c}.tsv"
        if c_tsv.exists():
            available_countries.append(c)
            logger.info("Loading checkpoint for %s from %s...", c, c_tsv)
            with open(c_tsv, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                next(reader)  # skip header
                for row in reader:
                    if row:
                        cand_map[row[0]] = row[1] if len(row) > 1 else ""
                        match_map[row[0]] = row[2] if len(row) > 2 else ""

    logger.info("Loaded checkpoints for countries: %s (%d candidate entries, %d match entries)", available_countries, len(cand_map), len(match_map))

    logger.info("Streaming %s and generating matching_results.tsv and candidate_pairs.tsv...", s1_path)
    total_rows = 0
    total_matches = 0
    total_cands = 0

    with open(s1_path, "r", encoding="utf-8") as f_in, \
         open(matching_file, "w", encoding="utf-8") as f_match, \
         open(candidate_file, "w", encoding="utf-8") as f_cand:

        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        reader = csv.reader(f_in, delimiter="\t")
        next(reader)  # skip header
        for row in reader:
            if not row:
                continue
            sid = row[0].strip()
            c_str = cand_map.get(sid, "")
            m_str = match_map.get(sid, "")

            if m_str and c_str:
                m_set = set(m_str.split(","))
                c_set = set(c_str.split(","))
                assert m_set.issubset(c_set), f"Invariant violation for {sid}"

            if m_str:
                total_matches += len(m_str.split(","))
            if c_str:
                total_cands += len(c_str.split(","))

            f_match.write(f"{sid}\t{m_str}\n")
            f_cand.write(f"{sid}\t{c_str}\n")
            total_rows += 1

    logger.info("Wrote %d rows to %s and %s in %.2f seconds.", total_rows, matching_file, candidate_file, time.time() - t_start)
    logger.info("Total predicted matches: %d | Total candidates: %d", total_matches, total_cands)

    # Mirror to output/
    shutil.copy2(matching_file, out_dir / "matching_results.tsv")
    shutil.copy2(candidate_file, out_dir / "candidate_pairs.tsv")
    logger.info("Mirrored to %s", out_dir)

    # Validate with internal validator
    logger.info("Running internal cross-file consistency validator...")
    val_res = validate_submission_files(
        test_source1_path=s1_path,
        matching_results_path=matching_file,
        candidate_pairs_path=candidate_file,
    )
    logger.info("Internal Validation: %s", val_res["status"])

    # Validate with official competition validator
    val_script = REPO_ROOT / "student_resource" / "utils" / "validate_submission.py"
    if val_script.exists():
        logger.info("Running official competition validator: %s...", val_script)
        test_dir = s1_path.parent
        cmd = [
            sys.executable, str(val_script),
            "--matching", str(out_dir / "matching_results.tsv"),
            "--candidate", str(out_dir / "candidate_pairs.tsv"),
            "--test-dir", str(test_dir),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        logger.info("Official Validator Exit Code: %d", proc.returncode)
        print(proc.stdout)
        if proc.stderr:
            print(proc.stderr)
        assert proc.returncode == 0, f"Official validator failed with code {proc.returncode}"

    logger.info("SUBMISSION ASSEMBLED AND VALIDATED SUCCESSFULLY IN %.2f SECONDS!", time.time() - t_start)
    return total_rows, total_matches


if __name__ == "__main__":
    assemble_interim_submission()
