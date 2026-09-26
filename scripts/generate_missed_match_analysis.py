"""
Comprehensive Error Analysis of the 213 Missed True Matches (Phase 4).

Extracts and classifies every missed match from the 2,000 S1 benchmark into a fine-grained
taxonomy, providing concrete example patterns, candidate blocking solutions,
and candidate explosion risk assessments.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.blocking.candidate_generator import CandidateGenerator
from src.preprocessing.normalize import normalize_record
from src.blocking.blocking_evaluation import detect_script

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


def classify_missed_pair(s1: Dict[str, Any], cand: Dict[str, Any]) -> str:
    s1_name = s1.get("business_name", "")
    s1_norm = s1.get("normalized_name", "")
    c_name = cand.get("business_name", "")
    c_norm = cand.get("normalized_name", "")

    s1_addr = s1.get("business_address", "")
    s1_norm_addr = s1.get("normalized_address", "")
    c_addr = cand.get("business_address", "")
    c_norm_addr = cand.get("normalized_address", "")

    s1_script = detect_script(s1_name)
    c_script = detect_script(c_name)

    s1_toks = set(s1.get("name_tokens", []))
    c_toks = set(cand.get("name_tokens", []))

    s1_addr_toks = set(s1.get("address_tokens", []))
    c_addr_toks = set(cand.get("address_tokens", []))

    # 1. Cross-script: different writing systems (Latin vs Indic/Devanagari/Tamil)
    if (s1_script != c_script) and (s1_script != "latin" or c_script != "latin"):
        return "cross-script"

    # 2. Domain concatenation / URL format (.com, .in, no spaces in candidate)
    if any(ext in c_name.lower() for ext in [".com", ".in", ".org", ".co", ".net"]) or (
        len(c_norm.split()) == 1 and len(s1_norm.split()) > 1 and c_norm in s1_norm.replace(" ", "")
    ):
        return "domain concatenation"

    # 3. Missing address with disparate/transliterated name
    if not c_norm_addr and not (s1_toks & c_toks):
        return "missing address"

    # 4. Numeric variation in address (e.g. 1056-1060 vs 1056, or Roman numerals)
    s1_nums = {t for t in s1_addr_toks if t.isdigit()}
    c_nums = {t for t in c_addr_toks if t.isdigit()}
    if s1_nums and c_nums and not (s1_nums & c_nums) and (s1_addr_toks & c_addr_toks):
        return "numeric variation"

    # 5. Address formatting / ordering (tokens present but not captured by anchor)
    if not (s1_toks & c_toks) and len(s1_addr_toks & c_addr_toks) >= 2:
        return "address formatting"

    # 6. Abbreviation / Acronym (e.g. ABC vs American Business Corp)
    s1_initials = "".join([w[0] for w in s1_norm.split() if w])
    c_initials = "".join([w[0] for w in c_norm.split() if w])
    if (len(c_norm) <= 4 and c_norm in s1_initials) or (len(s1_norm) <= 4 and s1_norm in c_initials):
        return "abbreviation"

    # 7. Disparate Alias (completely different corporate names with little overlap)
    if not (s1_toks & c_toks):
        return "alias"

    # 8. Token ordering variation
    if s1_toks == c_toks and s1_norm != c_norm:
        return "token ordering"

    # 9. Transliteration (Latin phonetic representations of Hindi/Indian names)
    if s1_script == "latin" and c_script == "latin" and not (s1_toks & c_toks):
        # Check if phonetic / letter overlap is high
        return "transliteration"

    # 10. Spelling / typo variation
    return "spelling variation"


def main():
    dataset_dir = REPO_ROOT / "student_resource" / "dataset" / "train"
    reports_dir = REPO_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading 2,000 S1 benchmark dataset...")
    gt_df = pd.read_csv(dataset_dir / "train_ground_truth.tsv", sep="\t", keep_default_na=False, nrows=2000)
    gt_map: Dict[str, Set[str]] = {}
    all_true_matched_ids: Set[str] = set()

    for _, r in gt_df.iterrows():
        s1_id = r["source1_entity_id"]
        matched = [x.strip() for x in r["matched_entity_ids"].split(",") if x.strip()]
        gt_map[s1_id] = set(matched)
        all_true_matched_ids.update(matched)

    s1_df = pd.read_csv(dataset_dir / "train_source1.tsv", sep="\t", keep_default_na=False)
    s1_sampled = s1_df[s1_df["entity_id"].isin(gt_map.keys())]
    s1_records = {r["entity_id"]: normalize_record(r.to_dict()) for _, r in s1_sampled.iterrows()}

    logger.info("Loading candidate pool...")
    s2_df = pd.read_csv(dataset_dir / "train_source2.tsv", sep="\t", keep_default_na=False, nrows=35000)
    s3_df = pd.read_csv(dataset_dir / "train_source3.tsv", sep="\t", keep_default_na=False, nrows=35000)

    candidate_records = {}
    for _, r in s2_df.iterrows():
        candidate_records[r["entity_id"]] = normalize_record(r.to_dict())
    for _, r in s3_df.iterrows():
        candidate_records[r["entity_id"]] = normalize_record(r.to_dict())

    missing_matched = [mid for mid in all_true_matched_ids if mid not in candidate_records]
    m_s2 = [m for m in missing_matched if m.startswith("S2-")]
    m_s3 = [m for m in missing_matched if m.startswith("S3-")]
    if m_s2:
        s2_extra = pd.read_csv(dataset_dir / "train_source2.tsv", sep="\t", keep_default_na=False)
        for _, r in s2_extra[s2_extra["entity_id"].isin(m_s2)].iterrows():
            candidate_records[r["entity_id"]] = normalize_record(r.to_dict())
    if m_s3:
        s3_extra = pd.read_csv(dataset_dir / "train_source3.tsv", sep="\t", keep_default_na=False)
        for _, r in s3_extra[s3_extra["entity_id"].isin(m_s3)].iterrows():
            candidate_records[r["entity_id"]] = normalize_record(r.to_dict())

    logger.info("Indexing and generating baseline candidates...")
    generator = CandidateGenerator(max_block_size=500)
    generator.index_candidates(candidate_records.values())

    missed_pairs: List[Dict[str, Any]] = []
    total_true_matches = sum(len(v) for v in gt_map.values())
    captured_matches = 0

    for s1_id, s1_rec in s1_records.items():
        cands = generator.generate_candidates_for_record(s1_rec)
        cand_ids = {c.candidate_entity_id for c in cands}
        true_set = gt_map.get(s1_id, set())
        captured_matches += len(cand_ids & true_set)
        for mid in true_set - cand_ids:
            cand_rec = candidate_records.get(mid, {})
            cat = classify_missed_pair(s1_rec, cand_rec)
            missed_pairs.append({
                "source1_id": s1_id,
                "candidate_id": mid,
                "category": cat,
                "s1_name": s1_rec.get("business_name", ""),
                "s1_norm_name": s1_rec.get("normalized_name", ""),
                "s1_address": s1_rec.get("business_address", ""),
                "s1_norm_addr": s1_rec.get("normalized_address", ""),
                "s1_country": s1_rec.get("country", ""),
                "cand_name": cand_rec.get("business_name", ""),
                "cand_norm_name": cand_rec.get("normalized_name", ""),
                "cand_address": cand_rec.get("business_address", ""),
                "cand_norm_addr": cand_rec.get("normalized_address", ""),
                "cand_country": cand_rec.get("country", ""),
            })

    total_missed = len(missed_pairs)
    logger.info("Total Missed Matches: %d (Captured %d / %d = %.2f%%)",
                total_missed, captured_matches, total_true_matches, captured_matches / total_true_matches * 100)

    category_counts = Counter(p["category"] for p in missed_pairs)

    # Solution and risk mapping per category
    TAXONOMY_META = {
        "cross-script": {
            "solution": "Cross-script address-only fallback blocker: index non-Latin records by distinctive address token pairs + country",
            "risk": "Low to Moderate: Restricting address pair blocking to records where at least one entity is non-Latin prevents candidate explosion on Latin records."
        },
        "alias": {
            "solution": "Shared street number + 2 distinct address tokens anchor fallback",
            "risk": "Moderate: Requires exact street number match and 2 locality tokens to prevent dense city over-generation."
        },
        "domain concatenation": {
            "solution": "Sub-token domain splitting (strip .com/.in and split on embedded capital/numeric boundaries) or relaxed compressed name prefix",
            "risk": "Low: Cleanly handles URL artifacts without adding unconstrained candidates."
        },
        "spelling variation": {
            "solution": "3-gram char blocking for short distinctive tokens or Levenshtein prefix blocker (fuzzy blocker)",
            "risk": "Moderate: Must require min token length >= 4 and cap posting list size at 200."
        },
        "address formatting": {
            "solution": "Dual-token address co-occurrence blocking (without requiring street number)",
            "risk": "High if uncapped: Must filter common city names like 'Delhi', 'New York' using IDF/stopwords."
        },
        "missing address": {
            "solution": "High-fidelity name-only 3-gram char blocking for entities with empty address",
            "risk": "Moderate: Strictly applied only when address is empty and name is distinctive."
        },
        "numeric variation": {
            "solution": "Range number expansion (e.g. '1056-1060' expanded to 1056 and 1060)",
            "risk": "Low: Handled during address normalization."
        },
        "abbreviation": {
            "solution": "Acronym index: match single word acronyms against initials of multi-word business names",
            "risk": "Moderate: Must require country agreement and address token overlap."
        },
        "transliteration": {
            "solution": "Phonetic Soundex / Double Metaphone keys on distinctive name tokens",
            "risk": "High if uncapped: Many Indian surnames produce identical phonetic codes; requires country + address filter."
        },
        "token ordering": {
            "solution": "Unordered token set conjunction (already largely handled by name_token)",
            "risk": "Very Low."
        },
    }

    category_rows = []
    for cat, count in category_counts.most_common():
        pct = round(count / total_missed * 100, 2)
        ex = next(p for p in missed_pairs if p["category"] == cat)
        ex_str = f"S1: '{ex['s1_name']}' [{ex['s1_norm_addr'][:30]}] vs Cand: '{ex['cand_name']}' [{ex['cand_norm_addr'][:30]}]"
        meta = TAXONOMY_META.get(cat, {"solution": "Fuzzy fallback", "risk": "Medium"})
        category_rows.append({
            "category": cat,
            "count": count,
            "percentage": pct,
            "example_pattern": ex_str,
            "possible_blocking_solution": meta["solution"],
            "risk_of_candidate_explosion": meta["risk"]
        })

    # Save CSV
    csv_path = reports_dir / "blocking_missed_match_analysis.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "count", "percentage", "example_pattern", "possible_blocking_solution", "risk_of_candidate_explosion"])
        writer.writeheader()
        writer.writerows(category_rows)
    logger.info("Saved CSV analysis to %s", csv_path)

    # Save JSON
    json_path = reports_dir / "blocking_missed_match_analysis.json"
    analysis_data = {
        "benchmark_sample_size": len(s1_records),
        "total_true_matches": total_true_matches,
        "captured_true_matches": captured_matches,
        "overall_baseline_recall": round(captured_matches / total_true_matches * 100, 4),
        "total_missed_matches": total_missed,
        "category_breakdown": category_rows,
        "sample_missed_matches_details": missed_pairs[:25]
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analysis_data, f, indent=2, ensure_ascii=False)
    logger.info("Saved JSON analysis to %s", json_path)

    # Print summary table
    print("\n" + "=" * 100)
    print("PHASE 4: MISSED MATCH ERROR ANALYSIS BREAKDOWN (213 MISSED MATCHES)")
    print("=" * 100)
    print(f"{'Category':<22} | {'Count':<6} | {'%':<6} | {'Explosion Risk':<15} | {'Proposed Solution'}")
    print("-" * 100)
    for r in category_rows:
        print(f"{r['category']:<22} | {r['count']:<6} | {r['percentage']:<6.2f} | {r['risk_of_candidate_explosion'][:15]:<15} | {r['possible_blocking_solution'][:50]}...")
    print("=" * 100)


if __name__ == "__main__":
    main()
