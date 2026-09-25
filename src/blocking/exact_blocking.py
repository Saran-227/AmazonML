"""
Exact blocking module for Business Entity Resolution.

Implements inverted indexes for exact matches:
- Exact normalized name
- Exact normalized name + country
- Exact normalized address + country
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from src.blocking.blocking_keys import (
    extract_exact_address_key,
    extract_exact_name_key,
    extract_name_country_key,
)


class ExactBlocker:
    """
    Manages exact-match inverted indexes for business records.
    Provides fast, deterministic lookups for identical names and addresses.
    """

    def __init__(
        self,
        enable_exact_name: bool = True,
        enable_name_country: bool = True,
        enable_exact_address: bool = True,
    ) -> None:
        self.enable_exact_name = enable_exact_name
        self.enable_name_country = enable_name_country
        self.enable_exact_address = enable_exact_address

        # Inverted index mappings: key -> List[Tuple[entity_id, source_tag]]
        self.exact_name_index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.name_country_index: Dict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)
        self.exact_address_index: Dict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)

    def add_record(self, record: Dict[str, Any], source_tag: str = "") -> None:
        """Indexes a candidate record into the active exact inverted indexes."""
        entity_id = record.get("entity_id", "")
        if not entity_id:
            return

        item = (entity_id, source_tag)

        if self.enable_exact_name:
            k_name = extract_exact_name_key(record)
            if k_name:
                self.exact_name_index[k_name].append(item)

        if self.enable_name_country:
            k_nc = extract_name_country_key(record)
            if k_nc:
                self.name_country_index[k_nc].append(item)

        if self.enable_exact_address:
            k_addr = extract_exact_address_key(record)
            if k_addr:
                self.exact_address_index[k_addr].append(item)

    def get_candidates(self, query_record: Dict[str, Any]) -> Dict[str, Tuple[str, Set[str]]]:
        """
        Retrieves matching candidates for a query record (e.g. from Source 1).
        
        Returns:
            Dict mapping candidate_entity_id -> (source_tag, set_of_matching_rules)
        """
        candidates: Dict[str, Tuple[str, Set[str]]] = {}

        def record_hit(cid: str, src: str, rule: str) -> None:
            if cid not in candidates:
                candidates[cid] = (src, {rule})
            else:
                candidates[cid][1].add(rule)

        if self.enable_exact_name:
            k_name = extract_exact_name_key(query_record)
            if k_name and k_name in self.exact_name_index:
                for cid, src in self.exact_name_index[k_name]:
                    record_hit(cid, src, "exact_name")

        if self.enable_name_country:
            k_nc = extract_name_country_key(query_record)
            if k_nc and k_nc in self.name_country_index:
                for cid, src in self.name_country_index[k_nc]:
                    record_hit(cid, src, "name_country")

        if self.enable_exact_address:
            k_addr = extract_exact_address_key(query_record)
            if k_addr and k_addr in self.exact_address_index:
                for cid, src in self.exact_address_index[k_addr]:
                    record_hit(cid, src, "exact_address")

        return candidates

    def clear(self) -> None:
        """Clears all inverted indexes."""
        self.exact_name_index.clear()
        self.name_country_index.clear()
        self.exact_address_index.clear()

    def get_stats(self) -> Dict[str, int]:
        """Returns size statistics of the inverted indexes."""
        return {
            "exact_name_keys": len(self.exact_name_index),
            "name_country_keys": len(self.name_country_index),
            "exact_address_keys": len(self.exact_address_index),
        }
