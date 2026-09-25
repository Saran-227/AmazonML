"""
Preprocessing package for Amazon ML Challenge Business Entity Resolution.
Exposes modular normalizers, tokenizers, and high-level batch/streaming pipelines.
"""

from src.preprocessing.name_normalizer import (
    normalize_business_name,
    tokenize_name,
)
from src.preprocessing.address_normalizer import (
    normalize_business_address,
    tokenize_address,
)
from src.preprocessing.country_normalizer import (
    normalize_country,
)
from src.preprocessing.normalize import (
    normalize_record,
    normalize_dataframe,
    iter_normalized_records,
    process_file_in_chunks,
)
from src.preprocessing.text_utils import (
    remove_latin_accents,
    clean_multilingual_text,
    tokenize_text,
)

__all__ = [
    "normalize_business_name",
    "tokenize_name",
    "normalize_business_address",
    "tokenize_address",
    "normalize_country",
    "normalize_record",
    "normalize_dataframe",
    "iter_normalized_records",
    "process_file_in_chunks",
    "remove_latin_accents",
    "clean_multilingual_text",
    "tokenize_text",
]
