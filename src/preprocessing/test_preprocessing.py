"""
Unit tests for the preprocessing module.
Validates business name, address, country normalization, tokenization,
record and DataFrame normalization, and streaming functions.
"""

import math
import unittest
import pandas as pd

from src.preprocessing.name_normalizer import (
    normalize_business_name,
    tokenize_name,
)
from src.preprocessing.address_normalizer import (
    normalize_business_address,
    tokenize_address,
)
from src.preprocessing.country_normalizer import normalize_country
from src.preprocessing.normalize import (
    normalize_record,
    normalize_dataframe,
    iter_normalized_records,
)
from src.preprocessing.text_utils import remove_latin_accents


class TestNameNormalizer(unittest.TestCase):
    def test_standard_names(self):
        self.assertEqual(normalize_business_name("Orelee's Barbershop"), "orelees barbershop")
        self.assertEqual(normalize_business_name(" Prime Money, Inc. "), "prime money inc")
        self.assertEqual(normalize_business_name("B+ Retail Inc"), "b retail inc")

    def test_capitalization_and_whitespace(self):
        self.assertEqual(normalize_business_name("PAYNE-ENRTPRMISES"), "payne enrtprmises")
        self.assertEqual(normalize_business_name("payne-enrtprmises"), "payne enrtprmises")
        self.assertEqual(normalize_business_name("  -- Holloway Peak Inc Seafood  "), "holloway peak inc seafood")
        self.assertEqual(normalize_business_name("Hendricks and    Flowers Inc"), "hendricks and flowers inc")

    def test_ampersand_expansion(self):
        self.assertEqual(normalize_business_name("AT&T"), "at and t")
        self.assertEqual(normalize_business_name("Johnson & Johnson"), "johnson and johnson")

    def test_missing_and_null_values(self):
        self.assertEqual(normalize_business_name(None), "")
        self.assertEqual(normalize_business_name(""), "")
        self.assertEqual(normalize_business_name("   "), "")
        self.assertEqual(normalize_business_name(float("nan")), "")
        self.assertEqual(normalize_business_name("null"), "")
        self.assertEqual(normalize_business_name("none"), "")

    def test_latin_accents_folded(self):
        self.assertEqual(normalize_business_name("LLC Moncada Léarning Center"), "llc moncada learning center")
        self.assertEqual(normalize_business_name("Dréxkor"), "drexkor")
        self.assertEqual(normalize_business_name("Payne Énterprises"), "payne enterprises")
        self.assertEqual(normalize_business_name("Lumay Bóral"), "lumay boral")

    def test_indic_scripts_preserved(self):
        # Hindi / Devanagari
        self.assertEqual(
            normalize_business_name("राम मार्केटिंग प्राइवेट लिमिटेड"),
            "राम मार्केटिंग प्राइवेट लिमिटेड",
        )
        # Tamil
        self.assertEqual(
            normalize_business_name("ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி"),
            "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி",
        )
        # Kannada
        self.assertEqual(
            normalize_business_name("ಕರ್ನಾಟಕ ಎಂಟರ್‌ಪ್ರೈಸಸ್"),
            "ಕರ್ನಾಟಕ ಎಂಟರ್ ಪ್ರೈಸಸ್",
        )

    def test_name_tokenization(self):
        self.assertEqual(tokenize_name("Prime Money Inc"), ["prime", "money", "inc"])
        self.assertEqual(tokenize_name("Orelee's Barbershop"), ["orelees", "barbershop"])
        self.assertEqual(tokenize_name(None), [])
        self.assertEqual(tokenize_name(""), [])


class TestAddressNormalizer(unittest.TestCase):
    def test_standard_addresses(self):
        self.assertEqual(
            normalize_business_address("1795 Westchester Drive, High Point, NC"),
            "1795 westchester drive high point nc",
        )
        self.assertEqual(
            normalize_business_address("2100 Cameron Drive, Unit APARTMENT G, Dundalk, MD"),
            "2100 cameron drive unit apartment g dundalk md",
        )

    def test_numbers_and_postal_codes_preserved(self):
        # Preserves number ranges, PO Box, postal codes with leading zeros intact
        self.assertEqual(
            normalize_business_address("1056-1060 BELDEN AVE, PO BOX 8807, AKRON, OH 02138"),
            "1056 1060 belden ave po box 8807 akron oh 02138",
        )
        self.assertEqual(
            normalize_business_address("Af-684, Nandgram Near Mother India Public School. Ph. 989, 9487203"),
            "af 684 nandgram near mother india public school ph 989 9487203",
        )

    def test_strip_literal_null_token(self):
        # Raw dataset often contains literal 'null' string concatenated from missing address fields
        self.assertEqual(
            normalize_business_address("45ND TERRACE, null, KANSAS CITY, MO"),
            "45nd terrace kansas city mo",
        )
        self.assertEqual(
            normalize_business_address("067 PRODUCTION CT, NULL, INDEPENDENCE, KY"),
            "067 production ct independence ky",
        )

    def test_missing_and_null_addresses(self):
        self.assertEqual(normalize_business_address(None), "")
        self.assertEqual(normalize_business_address(""), "")
        self.assertEqual(normalize_business_address("   "), "")
        self.assertEqual(normalize_business_address(float("nan")), "")

    def test_multilingual_addresses(self):
        addr = "Door No 183, 41St Cross, 22Nd Main 9Th Block Jayanagar, Bengaluru Urban, Bangalore, ಕರ್ನಾಟಕ"
        expected = "door no 183 41st cross 22nd main 9th block jayanagar bengaluru urban bangalore ಕರ್ನಾಟಕ"
        self.assertEqual(normalize_business_address(addr), expected)

    def test_address_tokenization(self):
        tokens = tokenize_address("1795 Westchester Drive, High Point, NC")
        self.assertEqual(tokens, ["1795", "westchester", "drive", "high", "point", "nc"])
        self.assertEqual(tokenize_address(None), [])
        self.assertEqual(tokenize_address(""), [])


class TestCountryNormalizer(unittest.TestCase):
    def test_country_normalization(self):
        self.assertEqual(normalize_country("US"), "us")
        self.assertEqual(normalize_country("  India  "), "india")
        self.assertEqual(normalize_country("FRANCE"), "france")
        self.assertEqual(normalize_country("u.s.a."), "u s a")
        self.assertEqual(normalize_country(None), "")
        self.assertEqual(normalize_country(""), "")
        self.assertEqual(normalize_country(float("nan")), "")


class TestRecordAndDataframeNormalization(unittest.TestCase):
    def test_normalize_record_preserves_raw(self):
        raw_record = {
            "entity_id": "S1-925783039",
            "business_name": "Orelee's Barbershop",
            "business_address": "1795 Westchester Drive, High Point, NC",
            "country": "US",
        }
        record_copy = dict(raw_record)
        normalized = normalize_record(raw_record, include_tokens=True)

        # Raw record was not mutated
        self.assertEqual(raw_record, record_copy)

        # Raw values are preserved in output
        self.assertEqual(normalized["entity_id"], "S1-925783039")
        self.assertEqual(normalized["business_name"], "Orelee's Barbershop")
        self.assertEqual(normalized["business_address"], "1795 Westchester Drive, High Point, NC")
        self.assertEqual(normalized["country"], "US")

        # Derived fields exist
        self.assertEqual(normalized["normalized_name"], "orelees barbershop")
        self.assertEqual(normalized["normalized_address"], "1795 westchester drive high point nc")
        self.assertEqual(normalized["normalized_country"], "us")
        self.assertEqual(normalized["name_tokens"], ["orelees", "barbershop"])
        self.assertEqual(
            normalized["address_tokens"],
            ["1795", "westchester", "drive", "high", "point", "nc"],
        )

    def test_normalize_dataframe(self):
        data = {
            "entity_id": ["S1-1", "S1-2"],
            "business_name": ["Prime Money, Inc.", "LLC Moncada Léarning Center"],
            "business_address": ["17560 Ellis Road, Tahlequah, OK", ""],
            "country": ["US", "US"],
        }
        df = pd.DataFrame(data)
        norm_df = normalize_dataframe(df)

        # Check raw preserved
        self.assertEqual(norm_df["business_name"].iloc[0], "Prime Money, Inc.")
        self.assertEqual(norm_df["business_address"].iloc[0], "17560 Ellis Road, Tahlequah, OK")
        # Check derived
        self.assertEqual(norm_df["normalized_name"].iloc[0], "prime money inc")
        self.assertEqual(norm_df["normalized_name"].iloc[1], "llc moncada learning center")
        self.assertEqual(norm_df["normalized_address"].iloc[0], "17560 ellis road tahlequah ok")
        self.assertEqual(norm_df["normalized_address"].iloc[1], "")
        self.assertEqual(norm_df["name_tokens"].iloc[0], ["prime", "money", "inc"])

    def test_iter_normalized_records(self):
        records = [
            {
                "entity_id": "S1-1",
                "business_name": "B+ Retail Inc",
                "business_address": "1712 Montebello Avenue, Phoenix, AZ",
                "country": "US",
            }
        ]
        results = list(iter_normalized_records(records))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["normalized_name"], "b retail inc")


class TestIdempotency(unittest.TestCase):
    def test_idempotent_normalization(self):
        name = "Orelee's Barbershop & B+ Retail Inc"
        norm1 = normalize_business_name(name)
        norm2 = normalize_business_name(norm1)
        self.assertEqual(norm1, norm2)

        addr = "1056-1060 BELDEN AVE, PO BOX 8807, AKRON, OH"
        norm_addr1 = normalize_business_address(addr)
        norm_addr2 = normalize_business_address(norm_addr1)
        self.assertEqual(norm_addr1, norm_addr2)


class TestRealDataIntegration(unittest.TestCase):
    def test_stream_first_100_from_train_source1(self):
        generator = iter_normalized_records("data/train/train_source1.tsv")
        records = []
        for i, rec in enumerate(generator):
            if i >= 100:
                break
            records.append(rec)

        self.assertEqual(len(records), 100)
        self.assertIn("entity_id", records[0])
        self.assertIn("business_name", records[0])
        self.assertIn("business_address", records[0])
        self.assertIn("country", records[0])
        self.assertIn("normalized_name", records[0])
        self.assertIn("normalized_address", records[0])
        self.assertIn("normalized_country", records[0])
        self.assertIn("name_tokens", records[0])
        self.assertIn("address_tokens", records[0])

        # Verify first row matches known values ("Orelee's Barbershop")
        self.assertEqual(records[0]["normalized_name"], "orelees barbershop")
        self.assertEqual(records[0]["name_tokens"], ["orelees", "barbershop"])

    def test_chunked_processing_source2(self):
        from src.preprocessing.normalize import process_file_in_chunks

        chunks = process_file_in_chunks("data/train/train_source2.tsv", chunksize=100)
        first_chunk = next(chunks)
        self.assertEqual(len(first_chunk), 100)
        self.assertIn("normalized_name", first_chunk.columns)
        self.assertIn("normalized_address", first_chunk.columns)
        self.assertIn("normalized_country", first_chunk.columns)
        self.assertIn("name_tokens", first_chunk.columns)
        self.assertIn("address_tokens", first_chunk.columns)

        # Check Hindi row normalization in S2 row 0
        self.assertEqual(first_chunk["normalized_name"].iloc[0], "राम मार्केटिंग प्राइवेट लिमिटेड")

    def test_chunked_processing_source3(self):
        from src.preprocessing.normalize import process_file_in_chunks

        chunks = process_file_in_chunks("data/train/train_source3.tsv", chunksize=100)
        first_chunk = next(chunks)
        self.assertEqual(len(first_chunk), 100)
        # Check accented row in S3 row 2 (LLC Moncada Léarning Center)
        self.assertEqual(first_chunk["normalized_name"].iloc[2], "llc moncada learning center")


if __name__ == "__main__":
    unittest.main()
