"""
Master pairwise feature generator module for Business Entity Resolution.

Combines all feature groups into a unified numerical feature vector for each candidate pair:
- Group A: Name features
- Group B: Address features
- Group C: Country features
- Group D: Explicit missingness indicators
- Group E: Blocking rule indicators
- Group F: Candidate source indicators
- Group G: Script and multilingual features
- Group H: Combined evidence interaction features

Also separates feature matrix (X) construction from label generation (y).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from src.blocking.candidate_generator import CandidatePair
from src.features.address_features import compute_address_features
from src.features.blocking_features import compute_blocking_and_source_features
from src.features.country_features import compute_country_features
from src.features.name_features import compute_name_features
from src.features.script_features import compute_script_features


def compute_pair_features(
    s1_rec: Dict[str, Any],
    cand_rec: Dict[str, Any],
    blocking_rules: Optional[Iterable[str]] = None,
    candidate_source: str = "",
) -> Dict[str, float]:
    """
    Computes complete numerical feature dictionary for a single candidate pair.
    Deterministic, pure-function, zero side effects.
    """
    # Group A: Name features
    feat_name = compute_name_features(s1_rec, cand_rec)

    # Group B: Address features
    feat_addr = compute_address_features(s1_rec, cand_rec)

    # Group C: Country features
    feat_country = compute_country_features(s1_rec, cand_rec)

    # Group D: Explicit missingness indicators
    n1 = s1_rec.get("normalized_name", "") or ""
    n2 = cand_rec.get("normalized_name", "") or ""
    a1 = s1_rec.get("normalized_address", "") or ""
    a2 = cand_rec.get("normalized_address", "") or ""

    feat_missing = {
        "name_missing_s1": 1.0 if not n1 else 0.0,
        "name_missing_candidate": 1.0 if not n2 else 0.0,
        "address_missing_s1": 1.0 if not a1 else 0.0,
        "address_missing_candidate": 1.0 if not a2 else 0.0,
    }

    # Group E & F: Blocking rules & Candidate source indicators
    cid = cand_rec.get("entity_id", "")
    feat_blocking = compute_blocking_and_source_features(
        blocking_rules=blocking_rules,
        candidate_source=candidate_source,
        candidate_id=cid,
    )

    # Group G: Script / Multilingual features
    feat_script = compute_script_features(s1_rec, cand_rec)

    # Group H: Combined evidence interactions
    name_jaccard = feat_name["name_token_jaccard"]
    addr_jaccard = feat_addr["address_token_jaccard"]
    name_exact = feat_name["name_exact_match"]
    addr_exact = feat_addr["address_exact_match"]
    country_exact = feat_country["country_exact_match"]
    primary_num_match = feat_addr["address_primary_number_match"]
    anchor_match = feat_addr["address_anchor_agreement"]
    name_overlap = feat_name["name_token_overlap_coefficient"]

    feat_combined = {
        "name_x_address_sim": round(name_jaccard * addr_jaccard, 4),
        "name_exact_and_address_exact": 1.0 if (name_exact == 1.0 and addr_exact == 1.0) else 0.0,
        "name_exact_and_country_match": 1.0 if (name_exact == 1.0 and country_exact == 1.0) else 0.0,
        "high_name_sim_and_number_match": 1.0 if (name_jaccard >= 0.5 and primary_num_match == 1.0) else 0.0,
        "address_sim_and_country_match": round(addr_jaccard * country_exact, 4),
        "name_overlap_and_address_anchor": round(name_overlap * anchor_match, 4),
    }

    # Merge all groups in deterministic order
    all_features: Dict[str, float] = {}
    all_features.update(feat_name)
    all_features.update(feat_addr)
    all_features.update(feat_country)
    all_features.update(feat_missing)
    all_features.update(feat_blocking)
    all_features.update(feat_script)
    all_features.update(feat_combined)

    return all_features


def build_feature_dataset(
    candidate_pairs: Iterable[Union[CandidatePair, Dict[str, Any]]],
    s1_records: Dict[str, Dict[str, Any]],
    candidate_records: Dict[str, Dict[str, Any]],
    ground_truth_map: Optional[Dict[str, Set[str]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[np.ndarray]]:
    """
    Constructs numerical feature DataFrame (X), identifier metadata DataFrame,
    and optional binary ground-truth label array (y).

    Returns:
        features_df: pd.DataFrame (numerical columns only)
        metadata_df: pd.DataFrame (source1_entity_id, candidate_entity_id, candidate_source)
        labels: Optional[np.ndarray] (1 for true match, 0 for non-match)
    """
    rows_features: List[Dict[str, float]] = []
    rows_metadata: List[Dict[str, str]] = []
    labels_list: Optional[List[int]] = [] if ground_truth_map is not None else None

    for pair in candidate_pairs:
        if isinstance(pair, CandidatePair):
            s1_id = pair.source1_entity_id
            c_id = pair.candidate_entity_id
            c_src = pair.candidate_source
            rules = pair.blocking_rules
        elif isinstance(pair, dict):
            s1_id = pair.get("source1_entity_id", "")
            c_id = pair.get("candidate_entity_id", "")
            c_src = pair.get("candidate_source", "")
            rules = set(pair.get("blocking_rules", []))
        else:
            continue

        s1_rec = s1_records.get(s1_id)
        c_rec = candidate_records.get(c_id)

        if not s1_rec or not c_rec:
            continue

        feat_dict = compute_pair_features(
            s1_rec=s1_rec,
            cand_rec=c_rec,
            blocking_rules=rules,
            candidate_source=c_src,
        )

        rows_features.append(feat_dict)
        rows_metadata.append({
            "source1_entity_id": s1_id,
            "candidate_entity_id": c_id,
            "candidate_source": c_src,
        })

        if labels_list is not None:
            true_set = ground_truth_map.get(s1_id, set())
            labels_list.append(1 if c_id in true_set else 0)

    features_df = pd.DataFrame(rows_features)
    metadata_df = pd.DataFrame(rows_metadata)
    labels = np.array(labels_list, dtype=np.int32) if labels_list is not None else None

    return features_df, metadata_df, labels
