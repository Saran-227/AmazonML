"""
Fuzzy blocking module for Business Entity Resolution.

Provides character n-gram and prefix blocking capabilities for handling
typos, minor misspellings, and morphological variations.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple


def get_character_ngrams(text: str, n: int = 3) -> List[str]:
    """Generates character n-grams from cleaned text."""
    clean = text.replace(" ", "")
    if len(clean) < n:
        return [clean] if clean else []
    return [clean[i : i + n] for i in range(len(clean) - n + 1)]


class FuzzyBlocker:
    """
    Inverted index for character n-gram and prefix-based fuzzy candidate generation.
    """

    def __init__(
        self,
        ngram_size: int = 3,
        min_prefix_len: int = 4,
        max_block_size: int = 500,
    ) -> None:
        self.ngram_size = ngram_size
        self.min_prefix_len = min_prefix_len
        self.max_block_size = max_block_size
        self.prefix_index: Dict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)

    def add_record(self, record: Dict[str, Any], source_tag: str = "") -> None:
        """Indexes name prefixes paired with country."""
        entity_id = record.get("entity_id", "")
        name = record.get("normalized_name", "").strip()
        country = record.get("normalized_country", "").strip()
        if not entity_id or not name:
            return

        if len(name) >= self.min_prefix_len:
            prefix = name[: self.min_prefix_len]
            self.prefix_index[(prefix, country)].append((entity_id, source_tag))

    def get_candidates(self, query_record: Dict[str, Any]) -> Dict[str, Tuple[str, Set[str]]]:
        """Retrieves candidates sharing the name prefix."""
        name = query_record.get("normalized_name", "").strip()
        country = query_record.get("normalized_country", "").strip()
        candidates: Dict[str, Tuple[str, Set[str]]] = {}

        if len(name) >= self.min_prefix_len:
            prefix = name[: self.min_prefix_len]
            post = self.prefix_index.get((prefix, country))
            if post and len(post) <= self.max_block_size:
                for cid, src in post:
                    candidates[cid] = (src, {"name_prefix"})

        return candidates

    def clear(self) -> None:
        self.prefix_index.clear()
