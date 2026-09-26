# Amazon ML Challenge 2026 — Final Submission Statistics Report

- **Submission ID**: `SUB_FINAL`
- **Git Commit**: `6decf02`
- **Generated Timestamp**: 2026-09-26 14:43:33

## 1. Test Dataset Census

- **Test Source 1 (Query Entities)**: 1,732,544
- **Test Source 2 (Catalogue Entities)**: 4,887,273
- **Test Source 3 (Catalogue Entities)**: 5,082,316
- **Total Candidate Pool**: 9,969,589

## 2. Match Prediction Distribution

| Metric | Count | Percentage |
| :--- | :---: | :---: |
| **Total Predicted Matches** | **1,807** | 100.0% |
| Zero-Match Entities | 1,731,693 | 99.95% |
| Singleton Matches | 565 | 0.0326% |
| Multi-Match Entities | 286 | 0.0165% |
| Source-2 Matches | 1,023 | 56.61% |
| Source-3 Matches | 784 | 43.39% |

## 3. Open-Set France Census

- **France S1 Entities**: 259,452 (14.98% of query pool)
- **France S2 Entities**: 703,378 (14.39% of S2 pool)
- **France S3 Entities**: 731,615 (14.40% of S3 pool)
- *Note*: Test ground truth is withheld by the organizers; test-set accuracy is strictly unmeasurable and is not fabricated.

## 4. Official Validator Result

```
ML Challenge 2026 — submission validator
  test dir: /Users/asimmaji/Desktop/AmazonMl/student_resource/dataset/test
  required S1 entities: 1732544
  valid S2/S3 match IDs: 9969589
  matching_results.tsv: 1732544 rows
  candidate_pairs.tsv: 1732544 rows
PASS — no blocking issues found. Safe to submit.
```
