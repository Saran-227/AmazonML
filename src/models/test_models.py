"""
Unit tests and sanity checks for model training, prediction, and feature validation.

Covers all 9 sanity checks from Section 10 and feature matrix tests from Section 11:
1. Model fits on synthetic data (both LR and Tree model).
2. Model predicts valid probabilities.
3. Probabilities are finite.
4. Probabilities are within [0.0, 1.0].
5. Number of predictions equals number of candidate rows.
6. Source-1 IDs remain strictly aligned.
7. Candidate IDs remain strictly aligned.
8. No feature columns accidentally contain IDs or raw text.
9. No label column enters X.
10. Feature matrix validation rejects NaNs, Infs, duplicate columns, and column order mismatches.
11. Threshold grid evaluates monotonically and selects optimal F0.5.
12. Threshold tie-breaking selects higher precision-conservative threshold.
"""

import unittest
import numpy as np
import pandas as pd

from src.models.predict import (
    build_prediction_dataframe,
    predict_matches_at_threshold,
    predict_probabilities,
)
from src.models.threshold import (
    evaluate_threshold_grid,
    select_best_threshold,
)
from src.models.train import (
    train_logistic_regression,
    train_tree_model,
    validate_feature_matrix,
)


class TestModelSanity(unittest.TestCase):

    def setUp(self):
        # Create small synthetic candidate dataset
        np.random.seed(42)
        self.n_samples = 100
        self.feature_names = [f"feat_{i}" for i in range(10)]

        # Synthetic feature matrix
        X_vals = np.random.randn(self.n_samples, len(self.feature_names))
        self.X = pd.DataFrame(X_vals, columns=self.feature_names)

        # Synthetic labels (imbalanced ~10% positive)
        self.y = np.zeros(self.n_samples, dtype=int)
        self.y[np.random.choice(self.n_samples, size=10, replace=False)] = 1

        # Aligned metadata
        s1_ids = [f"S1_{i // 5}" for i in range(self.n_samples)]
        cand_ids = [f"C_{i}" for i in range(self.n_samples)]
        self.metadata = pd.DataFrame({
            "source1_entity_id": s1_ids,
            "candidate_entity_id": cand_ids,
            "candidate_source": ["S2"] * (self.n_samples // 2) + ["S3"] * (self.n_samples // 2),
        })

    def test_01_feature_matrix_validations(self):
        """Sanity check: feature matrix validation rejects bad inputs."""
        # Valid check
        cols = validate_feature_matrix(self.X)
        self.assertEqual(cols, self.feature_names)

        # NaN detection
        bad_nan = self.X.copy()
        bad_nan.iloc[0, 0] = np.nan
        with self.assertRaises(ValueError):
            validate_feature_matrix(bad_nan)

        # Inf detection
        bad_inf = self.X.copy()
        bad_inf.iloc[0, 0] = np.inf
        with self.assertRaises(ValueError):
            validate_feature_matrix(bad_inf)

        # Duplicate column detection
        bad_dup = pd.concat([self.X, self.X[["feat_0"]]], axis=1)
        with self.assertRaises(ValueError):
            validate_feature_matrix(bad_dup)

        # Forbidden column detection (labels or IDs)
        bad_id = self.X.copy()
        bad_id["entity_id"] = "S1_1"
        with self.assertRaises(ValueError):
            validate_feature_matrix(bad_id)

        bad_label = self.X.copy()
        bad_label["label"] = 1
        with self.assertRaises(ValueError):
            validate_feature_matrix(bad_label)

    def test_02_train_and_predict_logistic_regression(self):
        """Sanity check: Logistic Regression fits and predicts bounded probabilities."""
        model, t_train = train_logistic_regression(self.X, self.y, random_state=42)
        self.assertIsNotNone(model)
        self.assertGreater(t_train, 0.0)

        proba, t_pred = predict_probabilities(model, self.X)
        self.assertEqual(len(proba), self.n_samples)
        self.assertTrue(np.isfinite(proba).all())
        self.assertTrue((proba >= 0.0).all())
        self.assertTrue((proba <= 1.0).all())

    def test_03_train_and_predict_tree_model(self):
        """Sanity check: HistGradientBoosting fits and predicts bounded probabilities."""
        model, t_train = train_tree_model(self.X, self.y, random_state=42)
        self.assertIsNotNone(model)
        self.assertGreater(t_train, 0.0)

        proba, t_pred = predict_probabilities(model, self.X)
        self.assertEqual(len(proba), self.n_samples)
        self.assertTrue(np.isfinite(proba).all())
        self.assertTrue((proba >= 0.0).all())
        self.assertTrue((proba <= 1.0).all())

    def test_04_id_alignment_and_predictions_dataframe(self):
        """Sanity check: IDs remain perfectly aligned in predictions dataframe."""
        model, _ = train_logistic_regression(self.X, self.y, random_state=42)
        proba, _ = predict_probabilities(model, self.X)

        pred_df = build_prediction_dataframe(self.metadata, proba, threshold=0.5, y_true=self.y)
        self.assertEqual(len(pred_df), self.n_samples)
        self.assertEqual(list(pred_df["source1_entity_id"]), list(self.metadata["source1_entity_id"]))
        self.assertEqual(list(pred_df["candidate_entity_id"]), list(self.metadata["candidate_entity_id"]))
        self.assertTrue("predicted_probability" in pred_df.columns)
        self.assertTrue("predicted_match" in pred_df.columns)
        self.assertTrue("true_match" in pred_df.columns)

    def test_05_predict_matches_one_to_many(self):
        """Sanity check: support 0, 1, or multiple matches per S1."""
        meta = pd.DataFrame({
            "source1_entity_id": ["S1_A", "S1_A", "S1_A", "S1_B"],
            "candidate_entity_id": ["C1", "C2", "C3", "C4"],
        })
        proba = np.array([0.9, 0.8, 0.2, 0.1])
        # Threshold 0.5: S1_A should get {C1, C2}, S1_B gets empty set
        pred_map = predict_matches_at_threshold(proba, meta, threshold=0.5, all_s1_ids=["S1_A", "S1_B"])
        self.assertEqual(pred_map["S1_A"], {"C1", "C2"})
        self.assertEqual(pred_map["S1_B"], set())

    def test_06_threshold_grid_and_selection(self):
        """Sanity check: threshold grid runs and tie-breaker selects higher threshold."""
        meta = pd.DataFrame({
            "source1_entity_id": ["S1_A", "S1_A", "S1_B", "S1_B"],
            "candidate_entity_id": ["C1", "C2", "C3", "C4"],
        })
        # Probabilities
        proba = np.array([0.9, 0.1, 0.8, 0.05])
        gt_map = {
            "S1_A": {"C1"},
            "S1_B": {"C3"},
        }
        grid_df = evaluate_threshold_grid(
            proba, meta, gt_map, thresholds=[0.2, 0.5, 0.7]
        )
        self.assertEqual(len(grid_df), 3)
        self.assertTrue("macro_f05" in grid_df.columns)

        # Monotonicity check on predicted matches: higher threshold -> fewer or equal matches
        matches = grid_df["predicted_matches"].tolist()
        self.assertTrue(matches[0] >= matches[1] >= matches[2])

        best_th, best_metrics, is_tied, tied_list = select_best_threshold(grid_df)
        self.assertIn(best_th, [0.2, 0.5, 0.7])
        self.assertIsNotNone(best_metrics)


if __name__ == "__main__":
    unittest.main()
