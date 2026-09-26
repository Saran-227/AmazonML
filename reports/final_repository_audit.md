# Final Repository Audit Report

**Date**: 2026-09-26  
**Auditor**: Asim (ML & Pipeline Engineering Lead)  
**Branch**: `asim` (synchronized with `origin/asim`)  
**Commit**: `6decf02`

---

## 1. Executive Summary & Git State

- **Active Branch**: `asim`
- **Remote Tracking**: `origin/asim` (up to date, clean working tree)
- **Latest Commit**: `6decf02 - feat: complete final sprint with methodology, error analysis, validation, and determinism verification`
- **Saran Branch Alignment**:
  - `origin/saran` head commit is `775272f - Implement business name and address preprocessing`.
  - `origin/saran..asim` contains 11 commits (+10,630 lines added, −4 lines changed across 65 files).
  - `asim..origin/saran` is **empty** (0 diverging commits on Saran's branch).
  - Saran can fast-forward merge `asim` cleanly without conflicts.

---

## 2. Inventory of Existing Code & Artifacts

### A. Source Code (`src/`)
1. **`src/preprocessing/`** (6 modules, 1 test file):
   - Saran's Unicode-safe normalization: `clean_multilingual_text`, `clean_business_name`, `clean_address`, `clean_country`, `normalize_record`.
   - Guaranteed normalization contract: preserves raw keys, emits cleaned `normalized_*` strings and token lists.
   - Tested in `test_preprocessing.py` (18 unit tests, 100% pass).
2. **`src/blocking/`** (6 modules, 1 test file):
   - 6 deterministic blocking rules: `exact_name`, `name_country`, `exact_address`, `name_token`, `compressed_name`, `address_anchor`.
   - Inverted indexing with dynamic pruning at `max_block_size = 500`.
   - Production candidate generator: `CandidateGenerator`.
   - Tested in `test_blocking.py` (10 unit tests, 100% pass).
3. **`src/features/`** (8 modules, 1 test file):
   - 60 dense numerical pairwise features across Groups A–H: Name (14), Address (16), Country (4), Missingness (4), Blocking (7), Source (2), Script (7), Interactions (6).
   - Validated feature matrix: zero NaN, zero Inf, 100% float64, identical schema.
   - Tested in `test_features.py` (19 unit tests, 100% pass).
4. **`src/models/`** (6 modules, 1 test file):
   - `HistGradientBoostingClassifier` with balanced class weights.
   - Pre-trained serialized production model: `src/models/production_model.joblib` (308 KB) and `model_metadata.json`.
   - Bounded probability prediction and thresholding engine.
   - Tested in `test_models.py` (6 unit tests, 100% pass).
5. **`src/evaluation/`** (3 modules, 1 test file):
   - Official competition metric: Entity-Level Macro $F_{0.5}$ with correct empty/singleton/multi-match handling.
   - Grouped leakage-safe validation and `verify_group_leakage()`.
   - Tested in `test_evaluation.py` (12 unit tests, 100% pass).
6. **`src/submission/`** (4 modules, 1 test file):
   - Submission generator: `generate_submission()`.
   - 10-rule cross-file validator: `validate_submission_files()`.
   - Tested in `test_submission.py` (8 unit tests, 100% pass).

### B. Production Scripts (`scripts/`)
- `scripts/run_phase7_integration.py`: End-to-end integration and timing benchmark runner.
- `scripts/run_final_sprint_experiments.py`: Controlled blocking, threshold grid search (0.90–0.98), and feature ablation runner.
- `scripts/run_final_sprint_step1.py`: Fine-grained error analysis classifier for the 213 missed matches.
- `scripts/run_france_audit.py`: Open-set France compatibility auditor (0 hardcoded branches).
- `scripts/test_determinism.py`: Programmatic dual-run SHA-256 bitwise determinism verifier.

### C. Experiments & Reports (`experiments/`, `reports/`)
- `experiments/EXP_BASELINE/baseline_report.json`: Frozen baseline benchmark (96.90% recall, 0.9572 Macro F0.5).
- `experiments/EXP_BLOCKING_001/`: 8-rule experimental blocker report (98.47% recall, 0.9499 Macro F0.5; rejected due to precision drop).
- `experiments/EXP_THRESHOLD_001/`: 15-configuration threshold grid search report.
- `experiments/experiment_log.csv`: Consolidated audit log of all experiments.
- `reports/final_blocking_error_analysis.csv` & `.md`: Taxonomy breakdown of the 213 missed matches.
- `reports/final_blocking_comparison.csv`: Head-to-head comparison of baseline 6-rule, 7-rule domain, and 8-rule blocker.
- `reports/final_threshold_optimization.csv`: Validation metrics across thresholds 0.90–0.98.
- `reports/final_feature_ablation.csv`: 60 vs 61 feature ablation results.
- `reports/france_open_set_audit.json` & `.csv`: Open-set France structural analysis.
- `reports/determinism_audit/determinism_report.json`: Cryptographic verification of bitwise identical outputs.
- `reports/phase7_integration_report.json` & `.csv`: Integration timing and memory benchmark.

### D. Documentation (`docs/`)
- `docs/final_methodology.md`: Comprehensive 17-section technical methodology report covering problem formulation, preprocessing, blocking, features, HGB model, threshold selection, multiple matches, error analysis, France audit, and validation.

### E. Existing Submissions & Outputs
- `submissions/SUB_001_production_baseline/`:
  - `matching_results.tsv` (23 MB, 1,732,545 rows, SHA-256: `620fd51b8a7b2121ef8806bf140bd10098d71c61d921475436c00bd615208154`)
  - `candidate_pairs.tsv` (26 MB, 1,732,545 rows, SHA-256: `a0238be43df7d88e89d29e49b0fe537db990cd5fd509948b324244ceecee3974`)
  - `submission_metadata.json`
- `student_resource/output/`:
  - Bit-for-bit identical copies of `matching_results.tsv` and `candidate_pairs.tsv`.

---

## 3. Dataset Location & Volume

The dataset resides in `student_resource/dataset/`:
- **Training Set (`dataset/train/`)**:
  - `train_source1.tsv`: 200 MB
  - `train_source2.tsv`: 467 MB
  - `train_source3.tsv`: 480 MB
  - `train_ground_truth.tsv`: 121 MB
- **Test Set (`dataset/test/`)**:
  - `test_source1.tsv`: 167 MB (1,732,544 query entities)
  - `test_source2.tsv`: 486 MB (4,887,273 candidate entities)
  - `test_source3.tsv`: 483 MB (5,082,316 candidate entities)

---

## 4. Git Ignore Rules Inspection

Inspection of `.gitignore` confirms:
- `*.tsv` is ignored (protects datasets and multi-hundred-MB submission outputs from accidental commit).
- `*.csv` is ignored by default (report CSVs are force-added explicitly where appropriate).
- `student_resource/` is ignored by default.
- `output/`, `*.joblib`, `*.pkl` are ignored.
- Python caches (`__pycache__/`, `*.pyc`) and OS artifacts (`.DS_Store`) are properly ignored.

---

## 5. Script Analysis: Canonical Production Pipeline

- `scripts/run_phase7_integration.py` successfully connects the full pipeline and benchmarked Phase 7.
- To meet the Phase 2 requirement for ONE canonical, standalone command that executes the complete end-to-end flow from dataset to `submissions/SUB_FINAL/` with strict CLI flags, a dedicated `scripts/run_final_pipeline.py` will serve as the master entrypoint, utilizing the existing modular components in `src/` without duplicating logic.

---

## 6. What Is Missing for Final Submission Handoff

1. Canonical master entrypoint script: `scripts/run_final_pipeline.py`.
2. Final versioned submission directory: `submissions/SUB_FINAL/` containing `matching_results.tsv`, `candidate_pairs.tsv`, and complete `submission_metadata.json`.
3. Final submission statistics report: `reports/final_submission_statistics.json` and `reports/final_submission_statistics.md`.
4. Final determinism report: `reports/determinism_audit/final_determinism_report.json`.
5. Updated `docs/final_methodology.md` incorporating all 20 enumerated topics.
6. Practical, concise `README.md` for Saran and judges.
7. Final handoff documentation with exact PowerShell commands for Saran.

---
*Audit Completed Successfully.*
