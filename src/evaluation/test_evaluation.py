"""
Unit tests for the official entity-level evaluation metrics and grouped validation split.

Covers all 8 official metric tests specified in Phase 4 requirements:
- Test 1: Full match singleton
- Test 2: False negative singleton
- Test 3: True empty predicted empty
- Test 4: False positive on empty ground truth
- Test 5: Full match multi-match
- Test 6: Partial recall multi-match
- Test 7: Precision penalty over-prediction
- Test 8: Entity-level macro averaging vs candidate-level
- Test 9: Group leakage programmatic assertion
- Test 10: Disjoint grouped train/val split
"""

import unittest
import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    calculate_entity_f05,
    evaluate_entity_level,
)
from src.evaluation.validation import (
    split_grouped_dataset,
    verify_group_leakage,
)


class TestEvaluationMetrics(unittest.TestCase):

    def test_01_identical_match(self):
        """TEST 1: true = {A}, predicted = {A} -> P=1, R=1, F0.5=1"""
        p, r, f05 = calculate_entity_f05(true_set={"A"}, pred_set={"A"})
        self.assertAlmostEqual(p, 1.0)
        self.assertAlmostEqual(r, 1.0)
        self.assertAlmostEqual(f05, 1.0)

    def test_02_empty_predicted_with_true(self):
        """TEST 2: true = {A}, predicted = {} -> F0.5=0"""
        p, r, f05 = calculate_entity_f05(true_set={"A"}, pred_set=set())
        self.assertAlmostEqual(r, 0.0)
        self.assertAlmostEqual(f05, 0.0)

    def test_03_empty_true_and_empty_pred(self):
        """TEST 3: true = {}, predicted = {} -> F0.5=1"""
        p, r, f05 = calculate_entity_f05(true_set=set(), pred_set=set())
        self.assertAlmostEqual(p, 1.0)
        self.assertAlmostEqual(r, 1.0)
        self.assertAlmostEqual(f05, 1.0)

    def test_04_empty_true_with_predicted(self):
        """TEST 4: true = {}, predicted = {A} -> F0.5=0"""
        p, r, f05 = calculate_entity_f05(true_set=set(), pred_set={"A"})
        self.assertAlmostEqual(p, 0.0)
        self.assertAlmostEqual(f05, 0.0)

    def test_05_multi_match_identical(self):
        """TEST 5: true = {A,B,C}, predicted = {A,B,C} -> F0.5=1"""
        p, r, f05 = calculate_entity_f05(true_set={"A", "B", "C"}, pred_set={"A", "B", "C"})
        self.assertAlmostEqual(p, 1.0)
        self.assertAlmostEqual(r, 1.0)
        self.assertAlmostEqual(f05, 1.0)

    def test_06_partial_recall_multi_match(self):
        """TEST 6: true = {A,B,C}, predicted = {A} -> P=1, R=1/3, F0.5 < 1"""
        p, r, f05 = calculate_entity_f05(true_set={"A", "B", "C"}, pred_set={"A"})
        self.assertAlmostEqual(p, 1.0)
        self.assertAlmostEqual(r, 1.0 / 3.0)
        expected_f05 = (1.25 * 1.0 * (1.0 / 3.0)) / (0.25 * 1.0 + (1.0 / 3.0))
        self.assertAlmostEqual(f05, expected_f05)
        self.assertLess(f05, 1.0)

    def test_07_precision_penalty_over_prediction(self):
        """TEST 7: true = {A}, predicted = {A,B,C} -> P=1/3, R=1, F0.5 reflects precision penalty"""
        p, r, f05 = calculate_entity_f05(true_set={"A"}, pred_set={"A", "B", "C"})
        self.assertAlmostEqual(p, 1.0 / 3.0)
        self.assertAlmostEqual(r, 1.0)
        # In F0.5, beta=0.5, weighting precision higher than recall.
        expected_f05 = (1.25 * (1.0 / 3.0) * 1.0) / (0.25 * (1.0 / 3.0) + 1.0)
        self.assertAlmostEqual(f05, expected_f05)
        # Precision penalty should make F0.5 significantly lower than 0.5
        self.assertLess(f05, 0.5)

    def test_08_entity_level_macro_averaging(self):
        """TEST 8: Multiple S1 entities with different match counts. Macro average per S1 entity."""
        gt_map = {
            "S1_1": {"A"},
            "S1_2": {"B", "C"},
            "S1_3": set(),
        }
        pred_map = {
            "S1_1": {"A"},       # F0.5 = 1.0
            "S1_2": {"B"},       # P=1, R=0.5 -> F0.5 = (1.25 * 1 * 0.5)/(0.25*1 + 0.5) = 0.625/0.75 = 0.8333
            "S1_3": set(),       # F0.5 = 1.0
        }
        metrics = evaluate_entity_level(gt_map=gt_map, pred_map=pred_map)
        f05_1 = 1.0
        f05_2 = (1.25 * 1.0 * 0.5) / (0.25 * 1.0 + 0.5)
        f05_3 = 1.0
        expected_macro_f05 = (f05_1 + f05_2 + f05_3) / 3.0
        self.assertAlmostEqual(metrics["macro_f05"], round(expected_macro_f05, 4))
        self.assertEqual(metrics["entity_count"], 3)
        self.assertEqual(metrics["empty_prediction_count"], 1)
        self.assertEqual(metrics["singleton_prediction_count"], 2)

    def test_09_verify_group_leakage_exception(self):
        """TEST 9: verify_group_leakage raises ValueError if any entity ID overlaps."""
        train_s1 = {"S1_1", "S1_2", "S1_3"}
        val_s1_clean = {"S1_4", "S1_5"}
        # Should not raise
        verify_group_leakage(train_s1, val_s1_clean)

        val_s1_leaky = {"S1_3", "S1_4"}
        with self.assertRaises(ValueError):
            verify_group_leakage(train_s1, val_s1_leaky)

    def test_10_grouped_train_val_split_disjoint(self):
        """TEST 10: split_grouped_dataset guarantees 0 overlap between train and val groups."""
        meta_df = pd.DataFrame({
            "source1_entity_id": ["S1_1", "S1_1", "S1_2", "S1_2", "S1_3", "S1_4", "S1_4", "S1_5"],
            "candidate_entity_id": ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"],
        })
        train_idx, val_idx, train_groups, val_groups = split_grouped_dataset(
            meta_df, val_ratio=0.4, random_state=42
        )
        self.assertEqual(len(train_groups & val_groups), 0)
        self.assertEqual(len(train_idx) + len(val_idx), len(meta_df))
        self.assertEqual(len(set(train_idx) & set(val_idx)), 0)


if __name__ == "__main__":
    unittest.main()
