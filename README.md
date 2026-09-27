# Amazon ML Challenge 2026 — Business Entity Resolution

[![Tests](https://img.shields.io/badge/unit%20tests-73%2F73%20passed-brightgreen.svg)]()
[![Model](https://img.shields.io/badge/model-HistGradientBoosting-blue.svg)]()
[![Macro F0.5](https://img.shields.io/badge/Macro%20F0.5-0.9572-success.svg)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-informational.svg)]()

Production-grade, end-to-end machine learning system developed for the **Amazon ML Challenge 2026 — Business Entity Resolution**. The system resolves ambiguous, noisy, multilingual query records from a primary source (**Source 1**) to their corresponding commercial entity records across two reference catalogues (**Source 2** and **Source 3**).

---

## 1. Problem Statement

Business Entity Resolution (BER) is the record-linkage task of establishing whether two disparate business records refer to the same real-world commercial entity. In large-scale e-commerce catalogues, business records originate from different data vendors, government registries, and user submissions, resulting in:
- **Orthographic Discrepancies**: Abbreviations, legal suffixes (`Pvt Ltd` vs `Private Limited`, `LLC`, `Corp`, `SARL`), and typographical noise.
- **Address Formatting Variations**: Missing suite numbers, localized landmarks, non-standard postal codes.
- **Multilingual & Cross-Script Representations**: Latin English vs native Indic scripts (Devanagari, Odia, Bengali, Tamil, Kannada) and European accented characters (French `é`, `è`, `ç`, `à`).
- **Open-Set Jurisdictions**: Training data spanning India and the US, while test data introduces France.
- **Cardinality**: A query record may have zero, one, or multiple valid matches across Source 2 and Source 3.

---

## 2. Dataset Layout

The dataset resides under `data/train/` and `data/test/`:

| Split | Path | Records | Description |
| :--- | :--- | :--- | :--- |
| **Train** | `data/train/train_source1.tsv` | ~2,000 queries | Benchmark queries with full ground truth |
| **Train** | `data/train/train_source2.tsv` | ~4.88M records | Secondary reference catalogue 1 |
| **Train** | `data/train/train_source3.tsv` | ~5.08M records | Secondary reference catalogue 2 |
| **Train** | `data/train/train_ground_truth.tsv` | Aligned with S1 | True match mappings |
| **Test** | `data/test/test_source1.tsv` | **1,732,544 queries** | Official evaluation queries (India, US, France) |
| **Test** | `data/test/test_source2.tsv` | **4,887,273 records** | Reference candidate catalogue 1 |
| **Test** | `data/test/test_source3.tsv` | **5,082,316 records** | Reference candidate catalogue 2 |

---

## 3. Overall Pipeline Architecture

The system operates as a five-stage modular pipeline:

```
[Raw TSV Records (S1, S2, S3)]
              │
              ▼
[1. Unicode-Safe Preprocessing & Normalization]
              │
              ▼
[2. Inverted-Index Multi-Rule Blocking] ─── (Candidate Pruning <= 30)
              │
              ▼
[3. 60-Dimensional Pairwise Feature Extraction]
              │
              ▼
[4. HistGradientBoosting Classification & Probability Calibration]
              │
              ▼
[5. High-Precision Thresholding (tau = 0.95) & TSV Formatting]
              │
              ▼
[matching_results.tsv & candidate_pairs.tsv]
```

---

## 4. Pipeline Stages

### A. Preprocessing (`src/preprocessing/`)
- **Unicode Canonical Decomposition & Folding**: NFC normalization with selective accent stripping for Latin characters (converting French accents e.g., `é`, `ç` to ASCII) while strictly preserving native Indic code points (Devanagari, Tamil, Bengali, Odia).
- **Legal Suffix Standardization**: Canonical mapping of legal entities (`pvt ltd`, `private limited`, `llc`, `inc`, `corp`, `sarl`, `gmbh`).
- **Address & Country Normalization**: Standardized directional terms, street abbreviations, and ISO alpha-2 country codes.

### B. Multi-Rule Inverted-Index Blocking (`src/blocking/`)
To scale across $>1.7 \times 10^{13}$ potential pairs, candidate pairs are retrieved via 6 complementary blocking rules over inverted indexes:
1. `exact_name`: Exact normalized business name match.
2. `name_country`: Normalized business name + normalized country code match.
3. `exact_address`: Exact normalized address match.
4. `name_token`: Common distinctive name tokens with inverted-list pruning (frequency cap: 500).
5. `compressed_name`: Alphanumeric characters with whitespace/punctuation stripped.
6. `address_anchor`: Primary street number + leading address token anchor.

Candidates are ranked by rule agreement count and pruned to `max_candidates_per_s1 = 30` to bound memory and feature computation.

### C. Pairwise Feature Engineering (`src/features/`)
Each candidate pair is represented by **60 numerical pairwise features** across 8 distinct groups:
- **Group A (Name)**: Exact match, token Jaccard, overlap coefficient, Levenshtein similarity, Jaro-Winkler, char n-gram Jaccard, length ratios.
- **Group B (Address)**: Exact match, token Jaccard, overlap, Levenshtein, numeric token overlap, primary street number match, anchor agreement.
- **Group C (Country)**: Country exact match, country mismatch indicator, missingness flags.
- **Group D (Missingness)**: Explicit indicator flags for missing names, addresses, or countries.
- **Group E (Blocking Rules)**: One-hot binary indicators for triggering each blocking rule + total rule count.
- **Group F (Source Indicators)**: One-hot flags for Source 2 vs Source 3 reference origin.
- **Group G (Script & Multilingual)**: Latin character ratio, non-Latin ratio, script match indicator, cross-script indicator, Indic character presence.
- **Group H (Interaction Features)**: Name similarity $\times$ address similarity, exact name $\wedge$ exact address, name overlap $\wedge$ address anchor.

### D. Machine Learning Model (`src/models/`)
- **Classifier**: `HistGradientBoostingClassifier` (scikit-learn).
- **Hyperparameters**:
  - `class_weight="balanced"` (handles extreme positive/negative imbalance)
  - `max_iter=150`
  - `max_leaf_nodes=31`
  - `learning_rate=0.1`
  - `min_samples_leaf=20`
  - `random_state=42`
- **Trained Artifact**: `src/models/production_model.joblib` (~302 KB).

### E. Decision Threshold & Evaluation Metric
- **Official Evaluation Metric**: Entity-Level Macro $F_{0.5}$ ($\beta = 0.5$):
  $$F_{0.5} = \frac{(1 + 0.5^2) \cdot \text{Precision} \cdot \text{Recall}}{0.5^2 \cdot \text{Precision} + \text{Recall}} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$$
  Because $\beta = 0.5$, precision errors are penalized four times as heavily as recall errors.
- **Decision Threshold**: $\tau = 0.95$. Empirically calibrated to maximize Macro $F_{0.5}$ by driving precision to $>0.99$.

---

## 5. Production Inference & Memory Optimization

To evaluate all **1,732,544 test Source 1 queries** against the **9,969,589 reference candidates** within standard RAM limits (< 4.5 GB):
- Test entities and reference candidates are partitioned strictly by country (`France`, `US`, `India`) and catalog source (`S2`, `S3`).
- Candidates are loaded and indexed partition-by-partition with temporary disk caching.
- S1 queries are processed in streaming batches of 5,000 queries.
- Feature computation and probability prediction are batch-vectorized.
- Memory usage remains strictly below 2.0 GB throughout full test execution.

---

## 6. Official Submission Format

Final outputs strictly adhere to the competition schema:
- `matching_results.tsv`:
  ```tsv
  source1_entity_id	matched_entity_ids
  S1-123456789	S2-987654321,S3-555666777
  S1-234567890	
  ```
- `candidate_pairs.tsv`:
  ```tsv
  source1_entity_id	candidate_entity_ids
  S1-123456789	S2-987654321,S2-111222333,S3-555666777
  S1-234567890	S2-444555666
  ```

### Structural Guarantees
1. Exactly 1,732,544 rows matching `test_source1.tsv` in identical row order.
2. Comma-separated, lexicographically sorted candidate and match IDs.
3. Strict subset guarantee: $\hat{\mathcal{M}}(e_1) \subseteq \mathcal{C}(e_1)$.
4. Zero self-matches, zero duplicate IDs within any row.
5. Empty strings for entities with zero candidates or zero matches.

---

## 7. How to Reproduce

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
python -m unittest discover -s src -p "test_*.py" -v
```
Expected: `73 tests passing (100% OK)`.

### 3. Train Production Model
```bash
python src/models/train_production_model.py
```
Outputs model to `src/models/production_model.joblib`.

### 4. Verify Pipeline Determinism
```bash
python scripts/test_determinism.py
```
Verifies 100% bitwise identical cryptographic hashes across independent runs.

### 5. Generate Full Test Submission
```bash
python src/submission/generate_submission.py
```
Writes final validated files to `submissions/SUB_001_production_baseline/`.

### 6. Validate Submission Files
```bash
python -c "
from src.submission.validate import validate_submission_files
validate_submission_files(
    'data/test/test_source1.tsv',
    'submissions/SUB_001_production_baseline/matching_results.tsv',
    'submissions/SUB_001_production_baseline/candidate_pairs.tsv',
)
"
```

---

## 8. Empirical Performance Summary

| Metric | Result |
| :--- | :--- |
| **Blocking Recall** | **96.90%** (S2: 96.10%, S3: 97.63%) |
| **Validation Macro Precision** | **0.9918** |
| **Validation Macro Recall** | **0.9240** |
| **Entity-Level Macro $F_{0.5}$** | **0.9572** |
| **3-Fold Grouped CV Macro $F_{0.5}$** | **0.9601 ± 0.0048** |
| **Deterministic SHA-256 Match** | **100% Bitwise Identical** |
| **Internal Validator** | **8 / 8 Consistency Tests Passed** |