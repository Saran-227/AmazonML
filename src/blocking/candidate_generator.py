"""
Candidate generator module for Business Entity Resolution.

Orchestrates multi-block union candidate generation across Source 2 and Source 3:
- Exact name blocking
- Exact name + country blocking
- Distinctive name token blocking
- Domain-compressed name blocking
- Exact address blocking
- Address anchor blocking (number + locality)

Removes duplicates, preserves candidate source metadata, and tracks which blocking
rules generated each candidate pair.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

from src.blocking.exact_blocking import ExactBlocker
from src.blocking.token_blocking import TokenBlocker

logger = logging.getLogger(__name__)


@dataclass
class CandidatePair:
    """Represents a generated candidate pair between Source 1 and a candidate entity."""
    source1_entity_id: str
    candidate_entity_id: str
    candidate_source: str  # 'S2', 'S3', or custom tag
    blocking_rules: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source1_entity_id": self.source1_entity_id,
            "candidate_entity_id": self.candidate_entity_id,
            "candidate_source": self.candidate_source,
            "blocking_rules": sorted(list(self.blocking_rules)),
        }


class CandidateGenerator:
    """
    Unified multi-block candidate generator.
    Indexes target sources (Source 2 and/or Source 3) and executes multi-rule
    candidate generation for Source 1 reference records.
    """

    def __init__(
        self,
        enable_exact_name: bool = True,
        enable_name_country: bool = True,
        enable_exact_address: bool = True,
        enable_name_tokens: bool = True,
        enable_compressed_name: bool = True,
        enable_address_anchor: bool = True,
        max_block_size: int = 500,
        min_token_len: int = 3,
        stopwords: Optional[Set[str]] = None,
    ) -> None:
        self.exact_blocker = ExactBlocker(
            enable_exact_name=enable_exact_name,
            enable_name_country=enable_name_country,
            enable_exact_address=enable_exact_address,
        )
        self.token_blocker = TokenBlocker(
            enable_name_tokens=enable_name_tokens,
            enable_compressed_name=enable_compressed_name,
            enable_address_anchor=enable_address_anchor,
            max_block_size=max_block_size,
            min_token_len=min_token_len,
            stopwords=stopwords,
        )
        self.total_indexed_records = 0

    def index_candidates(
        self,
        records: Iterable[Dict[str, Any]],
        source_tag: Optional[str] = None,
    ) -> int:
        """
        Indexes candidate records into all underlying inverted indexes.
        Automatically infers source_tag from entity_id prefix ('S2' or 'S3') if not provided.
        Returns the number of indexed records.
        """
        count = 0
        for rec in records:
            eid = rec.get("entity_id", "")
            if not eid:
                continue

            tag = source_tag
            if not tag:
                if eid.startswith("S2-"):
                    tag = "S2"
                elif eid.startswith("S3-"):
                    tag = "S3"
                else:
                    tag = "UNKNOWN"

            self.exact_blocker.add_record(rec, source_tag=tag)
            self.token_blocker.add_record(rec, source_tag=tag)
            count += 1

        self.total_indexed_records += count
        return count

    def generate_candidates_for_record(
        self,
        query_record: Dict[str, Any],
        max_candidates: Optional[int] = None,
    ) -> List[CandidatePair]:
        """
        Generates deduplicated candidate pairs for a single Source 1 record.
        
        Merges hits from both exact and token inverted indexes, tracking all rules
        that contributed to each candidate.
        """
        s1_id = query_record.get("entity_id", "")
        if not s1_id:
            return []

        merged: Dict[str, Tuple[str, Set[str]]] = {}

        # 1. Exact blocking hits
        exact_hits = self.exact_blocker.get_candidates(query_record)
        for cid, (src, rules) in exact_hits.items():
            if cid not in merged:
                merged[cid] = (src, set(rules))
            else:
                merged[cid][1].update(rules)

        # 2. Token / anchor blocking hits
        token_hits = self.token_blocker.get_candidates(query_record)
        for cid, (src, rules) in token_hits.items():
            if cid not in merged:
                merged[cid] = (src, set(rules))
            else:
                merged[cid][1].update(rules)

        pairs = [
            CandidatePair(
                source1_entity_id=s1_id,
                candidate_entity_id=cid,
                candidate_source=src,
                blocking_rules=rules,
            )
            for cid, (src, rules) in merged.items()
        ]

        if max_candidates is not None and len(pairs) > max_candidates:
            # Sort prioritized by number of matching rules (descending)
            pairs.sort(key=lambda p: len(p.blocking_rules), reverse=True)
            pairs = pairs[:max_candidates]

        return pairs

    def iter_candidate_pairs(
        self,
        query_records: Iterable[Dict[str, Any]],
        max_candidates_per_record: Optional[int] = None,
    ) -> Iterator[CandidatePair]:
        """
        Streams candidate pairs for an iterable of Source 1 records.
        """
        for q_rec in query_records:
            pairs = self.generate_candidates_for_record(
                q_rec, max_candidates=max_candidates_per_record
            )
            for p in pairs:
                yield p

    def clear(self) -> None:
        """Clears all indexed candidate data."""
        self.exact_blocker.clear()
        self.token_blocker.clear()
        self.total_indexed_records = 0

    def get_index_stats(self) -> Dict[str, Any]:
        """Returns consolidated inverted index statistics."""
        stats = {
            "total_indexed_records": self.total_indexed_records,
            **self.exact_blocker.get_stats(),
            **self.token_blocker.get_stats(),
        }
        return stats
