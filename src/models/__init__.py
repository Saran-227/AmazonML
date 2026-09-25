"""Models package exports."""

from src.models.predict import (
    build_prediction_dataframe,
    predict_matches_at_threshold,
    predict_probabilities,
)
from src.models.threshold import (
    DEFAULT_THRESHOLDS,
    evaluate_threshold_grid,
    select_best_threshold,
)
from src.models.train import (
    train_logistic_regression,
    train_tree_model,
    validate_feature_matrix,
)

__all__ = [
    "train_logistic_regression",
    "train_tree_model",
    "validate_feature_matrix",
    "predict_probabilities",
    "predict_matches_at_threshold",
    "build_prediction_dataframe",
    "evaluate_threshold_grid",
    "select_best_threshold",
    "DEFAULT_THRESHOLDS",
]
