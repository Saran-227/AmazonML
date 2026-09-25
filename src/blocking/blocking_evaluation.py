"""
Blocking evaluation and failure analysis module for Business Entity Resolution.

Evaluates candidate generation performance against Ground Truth:
- Overall recall = captured true matches / total true matches
- Source 2 and Source 3 separate recall
- Candidate set size statistics (mean, median, P90, P95, P99, min, max)
- Blocking-rule contribution (generated candidates, captured matches, unique matches)
- Failure mode categorization of missed true matches
"""

from __future__ import annotations

import logging
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.evaluation.metrics import (
    calculate_blocking_recall,
    calculate_candidate_distribution,
)

logger = logging.getLogger(__name__)


@dataclass
class RuleContribution:
    rule_name: str
    total_candidates_generated: int = 0
    true_matches_captured: int = 0
    unique_true_matches_captured: int = 0
    capture_percentage: float = 0.0


@dataclass
class BlockingEvaluationResult:
    total_source1_entities: int = 0
    total_true_matches: int = 0
    total_captured_matches: int = 0
    overall_recall: float = 0.0

    # Source-specific recall
    source2_total_true_matches: int = 0
    source2_captured_matches: int = 0
    source2_recall: float = 0.0

    source3_total_true_matches: int = 0
    source3_captured_matches: int = 0
    source3_recall: float = 0.0

    # Candidate statistics
    candidate_distribution: Dict[str, float] = field(default_factory=dict)
    source2_candidate_distribution: Dict[str, float] = field(default_factory=dict)
    source3_candidate_distribution: Dict[str, float] = field(default_factory=dict)

    # Rule contributions
    rule_contributions: List[RuleContribution] = field(default_factory=list)

    # Failure analysis
    failure_categories: Dict[str, int] = field(default_factory=dict)
    sample_missed_matches: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_source1_entities": self.total_source1_entities,
            "total_true_matches": self.total_true_matches,
            "total_captured_matches": self.total_captured_matches,
            "overall_recall": round(self.overall_recall, 4),
            "source2_total_true_matches": self.source2_total_true_matches,
            "source2_captured_matches": self.source2_captured_matches,
            "source2_recall": round(self.source2_recall, 4),
            "source3_total_true_matches": self.source3_total_true_matches,
            "source3_captured_matches": self.source3_captured_matches,
            "source3_recall": round(self.source3_recall, 4),
            "candidate_distribution": self.candidate_distribution,
            "source2_candidate_distribution": self.source2_candidate_distribution,
            "source3_candidate_distribution": self.source3_candidate_distribution,
            "rule_contributions": [asdict(r) for r in self.rule_contributions],
            "failure_categories": self.failure_categories,
            "sample_missed_matches": self.sample_missed_matches,
        }


def detect_script(text: str) -> str:
    """Classifies text into primary script family: 'latin', 'devanagari', 'tamil', 'kannada', or 'other'."""
    counts = Counter()
    for ch in text:
        if ch.isalpha():
            name = unicodedata.name(ch, "").lower()
            if "devanagari" in name:
                counts["devanagari"] += 1
            elif "tamil" in name:
                counts["tamil"] += 1
            elif "kannada" in name:
                counts["kannada"] += 1
            elif "latin" in name:
                counts["latin"] += 1
            else:
                counts["other"] += 1
    return counts.most_common(1)[0][0] if counts else "latin"


def categorize_missed_match(
    s1_rec: Dict[str, Any], cand_rec: Dict[str, Any]
) -> str:
    """Categorizes the primary failure mode of a missed true match."""
    s1_name = s1_rec.get("normalized_name", "")
    c_name = cand_rec.get("normalized_name", "")
    s1_addr = s1_rec.get("normalized_address", "")
    c_addr = cand_rec.get("normalized_address", "")

    # 1. Missing address in candidate
    if not c_addr and not s1_name:
        return "missing_address_and_name"
    if not c_addr and s1_addr:
        # Check if names have any overlap
        s1_toks = set(s1_rec.get("name_tokens", []))
        c_toks = set(cand_rec.get("name_tokens", []))
        if not (s1_toks & c_toks):
            return "missing_address_with_divergent_name"

    # 2. Cross-script / Multilingual mismatch
    s1_script = detect_script(s1_rec.get("business_name", ""))
    c_script = detect_script(cand_rec.get("business_name", ""))
    if s1_script != c_script and s1_script != "latin" or c_script != "latin":
        return "cross_script_multilingual"

    # 3. Domain-name or concatenated token
    if ".com" in cand_rec.get("business_name", "").lower() or (
        len(c_name.split()) == 1 and len(s1_name.split()) > 1 and c_name in s1_name.replace(" ", "")
    ):
        return "domain_concatenation"

    # 4. Completely synthetic alias / disparate names
    s1_toks = set(s1_rec.get("name_tokens", []))
    c_toks = set(cand_rec.get("name_tokens", []))
    if not (s1_toks & c_toks):
        return "synthetic_or_disparate_alias"

    # 5. Minor typo / spelling variation
    return "spelling_or_token_variation"


def evaluate_blocking(
    ground_truth: Dict[str, Set[str]],
    candidates_by_s1: Dict[str, Dict[str, Tuple[str, Set[str]]]],
    s1_records: Optional[Dict[str, Dict[str, Any]]] = None,
    candidate_records: Optional[Dict[str, Dict[str, Any]]] = None,
) -> BlockingEvaluationResult:
    """
    Evaluates blocking recall, candidate statistics, rule contribution, and failure modes.

    Args:
        ground_truth: Mapping from source1_entity_id -> Set of true matched entity IDs.
        candidates_by_s1: Mapping from source1_entity_id -> Dict[candidate_id, (source_tag, set_of_rules)].
        s1_records: Optional mapping of S1 normalized records (for failure analysis).
        candidate_records: Optional mapping of candidate normalized records (for failure analysis).
    """
    total_true = 0
    total_captured = 0
    s2_true = 0
    s2_captured = 0
    s3_true = 0
    s3_captured = 0

    all_cand_counts: List[int] = []
    s2_cand_counts: List[int] = []
    s3_cand_counts: List[int] = []

    # Rule contribution tracking
    rule_generated_counts: Dict[str, int] = defaultdict(int)
    rule_captured_matches: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    captured_by_any_rule: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    # Missed matches tracking
    missed_pairs: List[Tuple[str, str]] = []

    for s1_id, true_matches in ground_truth.items():
        total_true += len(true_matches)
        for mid in true_matches:
            if mid.startswith("S2-"):
                s2_true += 1
            elif mid.startswith("S3-"):
                s3_true += 1

        cand_dict = candidates_by_s1.get(s1_id, {})
        all_cand_counts.append(len(cand_dict))

        s2_cands = [cid for cid, (src, _) in cand_dict.items() if src == "S2" or cid.startswith("S2-")]
        s3_cands = [cid for cid, (src, _) in cand_dict.items() if src == "S3" or cid.startswith("S3-")]
        s2_cand_counts.append(len(s2_cands))
        s3_cand_counts.append(len(s3_cands))

        # Check hits
        for cid, (src, rules) in cand_dict.items():
            for r in rules:
                rule_generated_counts[r] += 1
            if cid in true_matches:
                pair = (s1_id, cid)
                captured_by_any_rule[pair].update(rules)
                for r in rules:
                    rule_captured_matches[r].add(pair)

        # Captured true matches for this S1
        captured_for_this_s1 = set(cand_dict.keys()) & true_matches
        total_captured += len(captured_for_this_s1)

        for mid in captured_for_this_s1:
            if mid.startswith("S2-"):
                s2_captured += 1
            elif mid.startswith("S3-"):
                s3_captured += 1

        for mid in true_matches - set(cand_dict.keys()):
            missed_pairs.append((s1_id, mid))

    # Calculate unique rule hits
    unique_rule_hits: Dict[str, int] = defaultdict(int)
    for pair, rules in captured_by_any_rule.items():
        if len(rules) == 1:
            r = next(iter(rules))
            unique_rule_hits[r] += 1

    all_rules = sorted(list(set(rule_generated_counts.keys()) | set(rule_captured_matches.keys())))
    rule_contribs = []
    for r in all_rules:
        hits = len(rule_captured_matches[r])
        uniq = unique_rule_hits[r]
        pct = (hits / total_true * 100.0) if total_true > 0 else 0.0
        rule_contribs.append(
            RuleContribution(
                rule_name=r,
                total_candidates_generated=rule_generated_counts[r],
                true_matches_captured=hits,
                unique_true_matches_captured=uniq,
                capture_percentage=round(pct, 2),
            )
        )

    # Sort rule contributions by true matches captured descending
    rule_contribs.sort(key=lambda rc: rc.true_matches_captured, reverse=True)

    # Failure mode categorization
    failure_counts: Dict[str, int] = Counter()
    sample_missed: List[Dict[str, Any]] = []

    if s1_records and candidate_records:
        for s1_id, mid in missed_pairs:
            s1_rec = s1_records.get(s1_id, {})
            cand_rec = candidate_records.get(mid, {})
            cat = categorize_missed_match(s1_rec, cand_rec)
            failure_counts[cat] += 1

            if len(sample_missed) < 15:
                sample_missed.append({
                    "source1_id": s1_id,
                    "candidate_id": mid,
                    "failure_category": cat,
                    "s1_name": s1_rec.get("business_name", ""),
                    "s1_normalized_name": s1_rec.get("normalized_name", ""),
                    "s1_address": s1_rec.get("business_address", ""),
                    "s1_normalized_address": s1_rec.get("normalized_address", ""),
                    "candidate_name": cand_rec.get("business_name", ""),
                    "candidate_normalized_name": cand_rec.get("normalized_name", ""),
                    "candidate_address": cand_rec.get("business_address", ""),
                    "candidate_normalized_address": cand_rec.get("normalized_address", ""),
                })

    return BlockingEvaluationResult(
        total_source1_entities=len(ground_truth),
        total_true_matches=total_true,
        total_captured_matches=total_captured,
        overall_recall=round(calculate_blocking_recall(total_captured, total_true), 4),
        source2_total_true_matches=s2_true,
        source2_captured_matches=s2_captured,
        source2_recall=round(calculate_blocking_recall(s2_captured, s2_true), 4),
        source3_total_true_matches=s3_true,
        source3_captured_matches=s3_captured,
        source3_recall=round(calculate_blocking_recall(s3_captured, s3_true), 4),
        candidate_distribution=calculate_candidate_distribution(all_cand_counts),
        source2_candidate_distribution=calculate_candidate_distribution(s2_cand_counts),
        source3_candidate_distribution=calculate_candidate_distribution(s3_cand_counts),
        rule_contributions=rule_contribs,
        failure_categories=dict(failure_counts),
        sample_missed_matches=sample_missed,
    )


def format_evaluation_report(result: BlockingEvaluationResult) -> str:
    """Formats an evaluation result into a comprehensive, readable report."""
    cd = result.candidate_distribution
    s2_cd = result.source2_candidate_distribution
    s3_cd = result.source3_candidate_distribution

    lines = [
        "=" * 85,
        "PHASE 2: CANDIDATE BLOCKING EVALUATION REPORT",
        "=" * 85,
        "",
        "1. BLOCKING RECALL SUMMARY",
        "-" * 85,
        f"Evaluated Source-1 Entities:       {result.total_source1_entities:,d}",
        f"Total Ground-Truth True Matches:   {result.total_true_matches:,d}",
        f"Captured True Matches:             {result.total_captured_matches:,d}",
        f"OVERALL BLOCKING RECALL:           {result.overall_recall:.2f}%",
        "",
        f"Source 2 Matches (True / Captured): {result.source2_total_true_matches:,d} / {result.source2_captured_matches:,d} ({result.source2_recall:.2f}%)",
        f"Source 3 Matches (True / Captured): {result.source3_total_true_matches:,d} / {result.source3_captured_matches:,d} ({result.source3_recall:.2f}%)",
        "",
        "2. CANDIDATE SET SIZE DISTRIBUTION (PER S1 ENTITY)",
        "-" * 85,
        f"{'Metric':<25} | {'All Sources':<15} | {'Source 2':<15} | {'Source 3':<15}",
        "-" * 85,
        f"{'Total Candidate Pairs':<25} | {cd.get('total_candidates', 0):<15,d} | {s2_cd.get('total_candidates', 0):<15,d} | {s3_cd.get('total_candidates', 0):<15,d}",
        f"{'Average (Mean)':<25} | {cd.get('mean', 0.0):<15.2f} | {s2_cd.get('mean', 0.0):<15.2f} | {s3_cd.get('mean', 0.0):<15.2f}",
        f"{'Median (P50)':<25} | {cd.get('median', 0.0):<15.1f} | {s2_cd.get('median', 0.0):<15.1f} | {s3_cd.get('median', 0.0):<15.1f}",
        f"{'90th Percentile (P90)':<25} | {cd.get('p90', 0.0):<15.1f} | {s2_cd.get('p90', 0.0):<15.1f} | {s3_cd.get('p90', 0.0):<15.1f}",
        f"{'95th Percentile (P95)':<25} | {cd.get('p95', 0.0):<15.1f} | {s2_cd.get('p95', 0.0):<15.1f} | {s3_cd.get('p95', 0.0):<15.1f}",
        f"{'99th Percentile (P99)':<25} | {cd.get('p99', 0.0):<15.1f} | {s2_cd.get('p99', 0.0):<15.1f} | {s3_cd.get('p99', 0.0):<15.1f}",
        f"{'Minimum Candidates':<25} | {cd.get('min', 0):<15} | {s2_cd.get('min', 0):<15} | {s3_cd.get('min', 0):<15}",
        f"{'Maximum Candidates':<25} | {cd.get('max', 0):<15} | {s2_cd.get('max', 0):<15} | {s3_cd.get('max', 0):<15}",
        "",
        "3. BLOCKING RULE CONTRIBUTION BREAKDOWN",
        "-" * 85,
        f"{'Blocking Rule':<20} | {'Pairs Generated':<17} | {'Matches Captured':<18} | {'Recall%':<10} | {'Unique Hits'}",
        "-" * 85,
    ]

    for rc in result.rule_contributions:
        lines.append(
            f"{rc.rule_name:<20} | {rc.total_candidates_generated:<17,d} | {rc.true_matches_captured:<18,d} | {rc.capture_percentage:<10.2f} | {rc.unique_true_matches_captured:,d}"
        )

    lines.extend([
        "",
        "4. FAILURE ANALYSIS OF MISSED MATCHES",
        "-" * 85,
        f"Total Missed Matches: {result.total_true_matches - result.total_captured_matches:,d}",
    ])

    for cat, cnt in sorted(result.failure_categories.items(), key=lambda x: x[1], reverse=True):
        pct = (cnt / (result.total_true_matches - result.total_captured_matches) * 100.0) if (result.total_true_matches - result.total_captured_matches) > 0 else 0.0
        lines.append(f"  * {cat:<35}: {cnt:<5,d} ({pct:.1f}%)")

    lines.append("=" * 85)
    return "\n".join(lines)
