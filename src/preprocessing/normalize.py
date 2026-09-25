"""
High-level normalization pipeline for business entity records.
Provides clean boundaries, preserves raw values, and supports memory-efficient
streaming and chunked processing for large datasets.
"""

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Union
import pandas as pd

from src.preprocessing.name_normalizer import normalize_business_name, tokenize_name
from src.preprocessing.address_normalizer import (
    normalize_business_address,
    tokenize_address,
)
from src.preprocessing.country_normalizer import normalize_country


def normalize_record(record: Dict[str, Any], include_tokens: bool = True) -> Dict[str, Any]:
    """
    Normalizes a single business record dictionary.
    
    Preserves all original fields and adds derived fields:
      - normalized_name
      - normalized_address
      - normalized_country
      - name_tokens (if include_tokens=True)
      - address_tokens (if include_tokens=True)

    Original fields (business_name, business_address, country, entity_id) are NEVER overwritten.
    Returns a new dictionary without mutating the input.
    """
    raw_name = record.get("business_name", "")
    raw_address = record.get("business_address", "")
    raw_country = record.get("country", "")

    norm_name = normalize_business_name(raw_name)
    norm_address = normalize_business_address(raw_address)
    norm_country = normalize_country(raw_country)

    normalized: Dict[str, Any] = {
        "entity_id": record.get("entity_id", ""),
        "business_name": raw_name,
        "business_address": raw_address,
        "country": raw_country,
        "normalized_name": norm_name,
        "normalized_address": norm_address,
        "normalized_country": norm_country,
    }

    if include_tokens:
        normalized["name_tokens"] = tokenize_name(norm_name)
        normalized["address_tokens"] = tokenize_address(norm_address)

    # Preserve any other non-standard metadata keys present in the record
    for k, v in record.items():
        if k not in normalized:
            normalized[k] = v

    return normalized


def normalize_dataframe(
    df: pd.DataFrame, include_tokens: bool = True, copy: bool = True
) -> pd.DataFrame:
    """
    Normalizes a pandas DataFrame (or chunk) containing business records.
    
    Preserves raw columns (business_name, business_address, country) and adds:
      - normalized_name
      - normalized_address
      - normalized_country
      - name_tokens (optional, default True)
      - address_tokens (optional, default True)

    Designed for high throughput using fast list comprehensions over column arrays.
    """
    if copy:
        df = df.copy()

    # Safely handle missing columns
    name_col = df["business_name"] if "business_name" in df.columns else [""] * len(df)
    addr_col = df["business_address"] if "business_address" in df.columns else [""] * len(df)
    country_col = df["country"] if "country" in df.columns else [""] * len(df)

    norm_names = [normalize_business_name(x) for x in name_col]
    norm_addrs = [normalize_business_address(x) for x in addr_col]
    norm_countries = [normalize_country(x) for x in country_col]

    df["normalized_name"] = norm_names
    df["normalized_address"] = norm_addrs
    df["normalized_country"] = norm_countries

    if include_tokens:
        df["name_tokens"] = [tokenize_name(x) for x in norm_names]
        df["address_tokens"] = [tokenize_address(x) for x in norm_addrs]

    return df


def iter_normalized_records(
    source: Union[str, Path, Iterable[Dict[str, Any]]],
    sep: str = "\t",
    include_tokens: bool = True,
) -> Iterator[Dict[str, Any]]:
    """
    Memory-efficient generator yielding normalized records one by one.
    Accepts either a TSV/CSV filepath or an iterable of record dictionaries.
    Streams large files line-by-line without loading entire datasets into memory.
    """
    if isinstance(source, (str, Path)):
        filepath = Path(source)
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter=sep)
            try:
                header = next(reader)
            except StopIteration:
                return

            for row in reader:
                if not row:
                    continue
                record = {col: val for col, val in zip(header, row)}
                yield normalize_record(record, include_tokens=include_tokens)
    else:
        for record in source:
            yield normalize_record(record, include_tokens=include_tokens)


def process_file_in_chunks(
    filepath: Union[str, Path],
    sep: str = "\t",
    chunksize: int = 50000,
    include_tokens: bool = True,
) -> Iterator[pd.DataFrame]:
    """
    Reads a large TSV/CSV file in chunks using pandas and yields normalized DataFrames.
    Avoids memory exhaustion when processing files >500MB.
    """
    filepath = Path(filepath)
    with pd.read_csv(
        filepath,
        sep=sep,
        chunksize=chunksize,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    ) as reader:
        for chunk in reader:
            yield normalize_dataframe(chunk, include_tokens=include_tokens, copy=False)
