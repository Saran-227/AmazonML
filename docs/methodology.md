# Methodology Summary: Amazon ML Challenge 2026

## Business Entity Resolution

This document summarizes the core technical methodology, architectural pipeline, and empirical results for the Amazon ML Challenge 2026 Business Entity Resolution system. For complete derivations, deep architectural details, and full statistical logs, refer to [final_methodology.md](final_methodology.md).

---

### 1. Problem & Metric Formulation

The task resolves noisy, multilingual business query records from **Source 1** against reference entity records in **Source 2** and **Source 3**. Each Source 1 entity may link to zero, one, or multiple entities in Source 2 and Source 3.

The official evaluation metric is **Entity-Level Macro $F_{0.5}$** ($\beta = 0.5$):

$$F_{0.5} = \frac{(1 + 0.5^2) \cdot \text{Precision} \cdot \text{Recall}}{0.5^2 \cdot \text{Precision} + \text{Recall}} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$$

Because $\beta = 0.5$, precision errors are penalized four times as heavily as recall errors. A false positive severely harms the score, necessitating a high decision threshold and conservative prediction strategy.

---

### 2. End-to-End Architecture

The system is structured as a five-stage modular pipeline:

```
[Raw Records (S1, S2, S3)]
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
```

---

### 3. Key Pipeline Components

#### A. Preprocessing & Normalization (`src/preprocessing/`)
- **Unicode Decomposition & Canonical Folding**: NFC normalization with selective accent stripping for Latin characters (converting French accents e.g., `é`, `ç` to ASCII) while strictly preserving native Indic code points (Devanagari, Tamil, Bengali, Odia).
- **Legal Suffix Standardization**: Canonical mapping of legal entities (`pvt ltd`, `private limited`, `llc`, `inc`, `corp`, `sarl`, `gmbh`).
- **Address & Country Normalization**: Standardized directional terms, street abbreviations, and ISO alpha-2 country codes.

#### B. Multi-Rule Inverted-Index Blocking (`src/blocking/`)
To avoid evaluating $>1.7 \times 10^{13}$ pairwise comparisons, candidate pairs are retrieved via 6 complementary blocking rules over inverted indexes:
1. `exact_name`: Exact normalized business name match.
2. `name_country`: Normalized business name + normalized country code match.
3. `exact_address`: Exact normalized address match.
4. `name_token`: Common distinctive name tokens with inverted-list pruning (frequency cap: 500).
5. `compressed_name`: Alphanumeric characters with whitespace/punctuation stripped.
6. `address_anchor`: Primary street number + leading address token anchor.

Candidates are ranked by rule agreement count and pruned to `max_candidates_per_s1 = 30` to bound memory and feature computation.

#### C. Pairwise Feature Engineering (`src/features/`)
Each candidate pair is represented by **60 numerical features** across 8 distinct groups:
- **Group A (Name)**: Exact match, token Jaccard, overlap coefficient, Levenshtein similarity, Jaro-Winkler, char n-gram Jaccard, length ratios.
- **Group B (Address)**: Exact match, token Jaccard, overlap, Levenshtein, numeric token overlap, primary street number match, anchor agreement.
- **Group C (Country)**: Country exact match, country mismatch indicator, missingness flags.
- **Group D (Missingness)**: Explicit indicator flags for missing names, addresses, or countries.
- **Group E (Blocking Rules)**: One-hot binary indicators for triggering each blocking rule + total rule count.
- **Group F (Source Indicators)**: One-hot flags for Source 2 vs Source 3 reference origin.
- **Group G (Script & Multilingual)**: Latin character ratio, non-Latin ratio, script match indicator, cross-script indicator, Indic character presence.
- **Group H (Interaction Features)**: Name similarity $\times$ address similarity, exact name $\wedge$ exact address, name overlap $\wedge$ address anchor.

#### D. Machine Learning Model (`src/models/`)
- **Classifier**: `HistGradientBoostingClassifier` (scikit-learn).
- **Hyperparameters**:
  - `class_weight="balanced"` (handles extreme positive/negative imbalance)
  - `max_iter=150`
  - `max_leaf_nodes=31`
  - `learning_rate=0.1`
  - `min_samples_leaf=20`
  - `random_state=42`
- **Trained Artifact**: `src/models/production_model.joblib` (~302 KB).

#### E. High-Precision Decision Thresholding (`src/submission/`)
- **Decision Threshold**: $\tau = 0.95$.
- Calibrated to maximize Macro $F_{0.5}$ by driving precision to $>0.99$ while capturing high-confidence true matches.
- Multi-match entities are naturally supported if multiple candidates independently clear the threshold.

---

### 4. Empirical Results & Validation

| Metric | Result |
| :--- | :--- |
| **Blocking Recall** | **96.90%** (S2: 96.10%, S3: 97.63%) |
| **Macro Precision** | **0.9918** |
| **Macro Recall** | **0.9240** |
| **Entity-Level Macro $F_{0.5}$** | **0.9572** |
| **Grouped 3-Fold CV Macro $F_{0.5}$** | **0.9601 ± 0.0048** |
| **Unit Test Suite** | **73 / 73 tests passing (100%)** |
| **Determinism Audit** | **Identical bitwise cryptographic SHA-256 output** |

---

### 5. Memory-Conscious Production Inference

To process all **1,732,544 test Source 1 entities** against the **9,969,589 reference candidates** within physical RAM bounds (< 4.5 GB):
- Test entities are strictly partitioned by country (`France`, `US`, `India`).
- Reference candidates are loaded, normalized, and indexed per country partition.
- Inference is streamed in batches through vectorized candidate generation and batch model probability prediction.
- Intermediate results are indexed in memory and formatted in exact `test_source1.tsv` row sequence into `matching_results.tsv` and `candidate_pairs.tsv`.
