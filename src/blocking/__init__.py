"""
Blocking package for Amazon ML Challenge Business Entity Resolution.
Exposes modular blocking key extractors, exact blockers, token blockers,
unified candidate generation, and evaluation tools.
"""

from src.blocking.blocking_keys import (
    DEFAULT_ADDRESS_STOPWORDS,
    DEFAULT_LEGAL_STOPWORDS,
    extract_address_anchor_keys,
    extract_compressed_name_keys,
    extract_exact_address_key,
    extract_exact_name_key,
    extract_name_country_key,
    extract_name_tokens_keys,
    normalize_number_token,
)
from src.blocking.exact_blocking import ExactBlocker
from src.blocking.token_blocking import TokenBlocker
from src.blocking.candidate_generator import CandidateGenerator, CandidatePair
from src.blocking.blocking_evaluation import (
    BlockingEvaluationResult,
    RuleContribution,
    evaluate_blocking,
    format_evaluation_report,
)

__all__ = [
    "DEFAULT_ADDRESS_STOPWORDS",
    "DEFAULT_LEGAL_STOPWORDS",
    "extract_exact_name_key",
    "extract_name_country_key",
    "extract_name_tokens_keys",
    "extract_compressed_name_keys",
    "extract_exact_address_key",
    "extract_address_anchor_keys",
    "normalize_number_token",
    "ExactBlocker",
    "TokenBlocker",
    "CandidateGenerator",
    "CandidatePair",
    "BlockingEvaluationResult",
    "RuleContribution",
    "evaluate_blocking",
    "format_evaluation_report",
]
