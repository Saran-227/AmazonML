"""
Token blocking module for Business Entity Resolution.

Implements inverted indexes for token-based and anchor-based matching:
- Distinctive name tokens + country
- Domain-compressed name + country
- Address anchor keys (country + street number + locality token)

Includes posting list capping / frequency protection to prevent candidate explosion.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from src.blocking.blocking_keys import (
    DEFAULT_ADDRESS_STOPWORDS,
    DEFAULT_LEGAL_STOPWORDS,
    extract_address_anchor_keys,
    extract_compressed_name_keys,
    extract_name_tokens_keys,
)


class TokenBlocker:
    """
    Manages token and anchor inverted indexes for business records.
    Supports distinctive name token indexing, compressed name matching,
    and number-anchored address locality indexing with size-capping protection.
    """

    def __init__(
        self,
        enable_name_tokens: bool = True,
        enable_compressed_name: bool = True,
        enable_address_anchor: bool = True,
        max_block_size: int = 500,
        min_token_len: int = 3,
        stopwords: Optional[Set[str]] = None,
        addr_stopwords: Optional[Set[str]] = None,
    ) -> None:
        self.enable_name_tokens = enable_name_tokens
        self.enable_compressed_name = enable_compressed_name
        self.enable_address_anchor = enable_address_anchor
        self.max_block_size = max_block_size
        self.min_token_len = min_token_len
        self.stopwords = stopwords or DEFAULT_LEGAL_STOPWORDS
        self.addr_stopwords = addr_stopwords or DEFAULT_ADDRESS_STOPWORDS

        # Inverted index mappings: key -> List[Tuple[entity_id, source_tag]]
        self.name_token_index: Dict[Tuple[str, ...], List[Tuple[str, str]]] = defaultdict(list)
        self.compressed_name_index: Dict[Tuple[str, ...], List[Tuple[str, str]]] = defaultdict(list)
        self.address_anchor_index: Dict[Tuple[str, str, str], List[Tuple[str, str]]] = defaultdict(list)

    def add_record(self, record: Dict[str, Any], source_tag: str = "") -> None:
        """Indexes a candidate record into the active token inverted indexes."""
        entity_id = record.get("entity_id", "")
        if not entity_id:
            return

        item = (entity_id, source_tag)

        if self.enable_name_tokens:
            token_keys = extract_name_tokens_keys(
                record,
                min_len=self.min_token_len,
                stopwords=self.stopwords,
                with_country=True,
            )
            for k in token_keys:
                self.name_token_index[k].append(item)

        if self.enable_compressed_name:
            comp_keys = extract_compressed_name_keys(
                record,
                min_len=5,
                stopwords=self.stopwords,
                with_country=True,
            )
            for k in comp_keys:
                self.compressed_name_index[k].append(item)

        if self.enable_address_anchor:
            anchor_keys = extract_address_anchor_keys(
                record,
                addr_stopwords=self.addr_stopwords,
                min_word_len=4,
                max_words_per_number=3,
            )
            for k in anchor_keys:
                self.address_anchor_index[k].append(item)

    def get_candidates(self, query_record: Dict[str, Any]) -> Dict[str, Tuple[str, Set[str]]]:
        """
        Retrieves matching candidates for a query record.
        Skips posting lists that exceed max_block_size to protect against candidate explosion.
        
        Returns:
            Dict mapping candidate_entity_id -> (source_tag, set_of_matching_rules)
        """
        candidates: Dict[str, Tuple[str, Set[str]]] = {}

        def record_hit(cid: str, src: str, rule: str) -> None:
            if cid not in candidates:
                candidates[cid] = (src, {rule})
            else:
                candidates[cid][1].add(rule)

        if self.enable_name_tokens:
            token_keys = extract_name_tokens_keys(
                query_record,
                min_len=self.min_token_len,
                stopwords=self.stopwords,
                with_country=True,
            )
            for k in token_keys:
                post = self.name_token_index.get(k)
                if post and len(post) <= self.max_block_size:
                    for cid, src in post:
                        record_hit(cid, src, "name_token")

        if self.enable_compressed_name:
            comp_keys = extract_compressed_name_keys(
                query_record,
                min_len=5,
                stopwords=self.stopwords,
                with_country=True,
            )
            for k in comp_keys:
                post = self.compressed_name_index.get(k)
                if post and len(post) <= self.max_block_size:
                    for cid, src in post:
                        record_hit(cid, src, "compressed_name")

        if self.enable_address_anchor:
            anchor_keys = extract_address_anchor_keys(
                query_record,
                addr_stopwords=self.addr_stopwords,
                min_word_len=4,
                max_words_per_number=3,
            )
            for k in anchor_keys:
                post = self.address_anchor_index.get(k)
                if post and len(post) <= self.max_block_size:
                    for cid, src in post:
                        record_hit(cid, src, "address_anchor")

        return candidates

    def clear(self) -> None:
        """Clears all inverted indexes."""
        self.name_token_index.clear()
        self.compressed_name_index.clear()
        self.address_anchor_index.clear()

    def get_stats(self) -> Dict[str, int]:
        """Returns size statistics of the inverted indexes."""
        return {
            "name_token_keys": len(self.name_token_index),
            "compressed_name_keys": len(self.compressed_name_index),
            "address_anchor_keys": len(self.address_anchor_index),
        }
