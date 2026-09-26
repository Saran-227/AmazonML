# Amazon ML Challenge 2026 — Business Entity Resolution

[![Tests](https://img.shields.io/badge/Regression_Tests-73%2F73_PASS-brightgreen.svg)]()
[![Validator](https://img.shields.io/badge/Official_Validator-PASS_100%25-brightgreen.svg)]()
[![Metric](https://img.shields.io/badge/Macro_F0.5-0.9572-blue.svg)]()

Production entity resolution pipeline linking noisy, multilingual business query records (**Source 1**) to their corresponding entries across two disparate reference catalogues (**Source 2** and **Source 3**).

---

## Architecture Overview

```
RAW AMAZON DATA
      ↓
PREPROCESSING (Unicode NFC/NFKD accent-folding, legal suffix canonicalization, tokenization)
      ↓
BLOCKING (6 deterministic rules: exact name, name+country, address, distinctive tokens, compressed name, address anchor)
      ↓
FEATURE ENGINEERING (60 numerical pairwise features: string sim, token Jaccard, script ratios, interactions)
      ↓
ML CLASSIFIER (HistGradientBoosting with balanced class weights)
      ↓
THRESHOLDING (Global 0.95 decision boundary optimized for Entity-Level Macro F0.5)
      ↓
SUBMISSION FILES (matching_results.tsv + candidate_pairs.tsv)
      ↓
VALIDATION (Official Amazon Validator with --check-ids + internal 10-rule cross-file validator)
```

---

## Core Specifications

- **WHAT**: Large-scale Business Entity Resolution across multilingual, cross-script catalogues.
- **WHY**: Precision-heavy optimization for **Entity-Level Macro $F_{0.5}$** ($\beta = 0.5$, penalizing precision errors 4× more than recall).
- **WHAT BLOCKING**: 6 Deterministic Inverted-Index Rules with dynamic posting pruning at 500 entries (Recall: **96.90%**, Candidates: 192.35 cands/S1).
- **WHAT FEATURES**: 60 dense numerical pairwise features (Groups A–H: Name, Address, Country, Missingness, Blocking, Source, Script, Interactions).
- **WHAT MODEL**: `HistGradientBoostingClassifier` trained under zero-leakage grouped splitting.
- **WHAT THRESHOLD**: **0.95** (Global). Precision = **0.9918**, Recall = **0.9240**, Macro $F_{0.5}$ = **0.9572**.

---

## Quickstart & Commands

### 1. Run Complete Test Suite
Executes all 73 regression tests across preprocessing, blocking, features, models, evaluation, and submission:
```bash
python3 -m unittest discover -s src -p "test_*.py" -v
```

### 2. Run Canonical Production Pipeline
Executes the full pipeline, updates `submissions/SUB_FINAL/`, and syncs to `student_resource/output/`:
```bash
python3 scripts/run_final_pipeline.py
```

### 3. Run Official Amazon Submission Validator
Validates format, row counts, subset constraints, and verifies all 9.9M candidate IDs:
```bash
python3 student_resource/utils/validate_submission.py \
  --matching student_resource/output/matching_results.tsv \
  --candidate student_resource/output/candidate_pairs.tsv \
  --test-dir student_resource/dataset/test \
  --check-ids
```

### 4. Verify Bitwise Pipeline Determinism
Validates that consecutive execution runs produce bitwise identical cryptographic hashes (SHA-256):
```bash
python3 scripts/test_determinism.py
```

---

## Key Performance Benchmarks

| Metric | Measured Result |
| :--- | :--- |
| **Blocking Recall** | **96.90%** overall (S2: 96.10%, S3: 97.63%) |
| **Entity-Level Macro $F_{0.5}$** | **0.9572** (3-Fold Grouped CV: **0.9601 ± 0.0048**) |
| **Macro Precision** | **0.9918** |
| **Macro Recall** | **0.9240** |
| **Regression Tests** | **73 / 73 PASS (100% OK)** |
| **Official Validator** | **PASS** (0 invalid S1 IDs, 0 invalid candidate IDs, 0 invalid match IDs) |
| **End-to-End Pipeline Runtime** | **~110.57 seconds** |
| **Peak RAM Footprint** | **~2.25 GB** (commodity single-machine execution) |

---

## Repository Structure

```
AmazonML/
├── docs/
│   └── final_methodology.md         # Comprehensive 20-topic technical methodology report
├── experiments/
│   ├── EXP_BASELINE/                # Frozen production baseline benchmark (0.9572 F0.5)
│   ├── EXP_BLOCKING_001/            # 8-rule blocking experiment report (rejected)
│   ├── EXP_THRESHOLD_001/           # 15-configuration threshold grid search report
│   └── experiment_log.csv           # Experiment audit log
├── reports/
│   ├── final_repository_audit.md    # Initial repository audit
│   ├── final_blocking_error_analysis.csv / .md  # 213 missed matches taxonomy
│   ├── final_blocking_comparison.csv # 6-rule vs 7-rule vs 8-rule comparison
│   ├── final_threshold_optimization.csv # Threshold search (0.90 - 0.98)
│   ├── final_feature_ablation.csv   # Feature ablation results
│   ├── final_submission_statistics.json / .md   # Complete test set prediction breakdown
│   ├── france_open_set_audit.json / .csv        # Open-set France compatibility audit
│   ├── determinism_audit/           # Cryptographic bitwise determinism reports
│   └── phase7_integration_report.json / .csv    # Pipeline timing and memory benchmark
├── scripts/
│   ├── run_final_pipeline.py        # Master production pipeline runner
│   ├── test_determinism.py          # Determinism audit script
│   ├── run_france_audit.py          # France open-set audit script
│   ├── run_final_sprint_experiments.py # Sprint experiment runner
│   └── run_final_sprint_step1.py    # Error analysis generator
├── src/
│   ├── preprocessing/               # Saran's Unicode-safe text & address normalizer
│   ├── blocking/                    # 6-rule deterministic inverted index candidate generator
│   ├── features/                    # 60 dense numerical pairwise feature generators
│   ├── models/                      # HistGradientBoosting classifier & production model
│   ├── evaluation/                  # Entity-level Macro F0.5 & leakage-safe group splitting
│   └── submission/                  # Submission output writer & 10-rule cross-file validator
├── student_resource/output/         # Mirrored final submission TSV files ready for upload
└── submissions/
    ├── SUB_001_production_baseline/ # Archived baseline submission
    └── SUB_FINAL/                   # Final competition submission TSVs & metadata
```

---

## Submission Artifacts

The final, verified competition submission files are located in:
- `submissions/SUB_FINAL/`
  - `matching_results.tsv` (SHA-256: `620fd51b8a7b2121ef8806bf140bd10098d71c61d921475436c00bd615208154`)
  - `candidate_pairs.tsv` (SHA-256: `a0238be43df7d88e89d29e49b0fe537db990cd5fd509948b324244ceecee3974`)
  - `submission_metadata.json`
- Mirrored in `student_resource/output/` for direct competition upload.