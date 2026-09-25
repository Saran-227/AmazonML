"""
Features package for Amazon ML Challenge Business Entity Resolution.
Exposes modular feature extractors for names, addresses, countries, script analysis,
blocking rules, and unified pairwise feature generation.
"""

from src.features.name_features import compute_name_features
from src.features.address_features import compute_address_features
from src.features.country_features import compute_country_features
from src.features.script_features import compute_script_features
from src.features.blocking_features import compute_blocking_and_source_features
from src.features.pair_features import (
    compute_pair_features,
    build_feature_dataset,
)
from src.features.feature_utils import (
    levenshtein_distance,
    normalized_levenshtein_similarity,
    jaro_similarity,
    jaro_winkler_similarity,
    token_jaccard,
    token_overlap_coefficient,
    character_ngram_jaccard,
    extract_numeric_tokens,
    compute_script_ratios,
)

__all__ = [
    "compute_name_features",
    "compute_address_features",
    "compute_country_features",
    "compute_script_features",
    "compute_blocking_and_source_features",
    "compute_pair_features",
    "build_feature_dataset",
    "levenshtein_distance",
    "normalized_levenshtein_similarity",
    "jaro_similarity",
    "jaro_winkler_similarity",
    "token_jaccard",
    "token_overlap_coefficient",
    "character_ngram_jaccard",
    "extract_numeric_tokens",
    "compute_script_ratios",
]
