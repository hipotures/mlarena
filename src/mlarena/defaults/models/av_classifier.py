"""
Adversarial Validation Classifier Model

Standard MLArena model for binary classification to detect distribution shift
between training and test sets. Trains AutoGluon classifier to distinguish
train (label=0) from test (label=1) samples.

Interface:
    train(train_df, val_df, config, artifacts) -> (predictor, summary)
    predict(model, test_df, config, artifacts) -> predictions
"""

from typing import Any, Dict, Optional, Tuple

import pandas as pd
from autogluon.tabular import TabularPredictor


def train(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    config: Dict[str, Any],
    artifacts: Optional[Any] = None,
) -> Tuple[TabularPredictor, Dict[str, Any]]:
    """
    Train binary classifier for adversarial validation.

    Args:
        train_df: Combined train+test data with __is_test__ label
        val_df: Not used (AV uses all data for training)
        config: Minimal config dict with keys:
            - presets: AutoGluon preset (default: "medium_quality_faster_train")
            - time_limit: Training time in seconds (default: 600)
            - included_model_types: List of model types or None (default: None)
            - label: Target column name (default: "__is_test__")
            - problem_type: Problem type (default: "binary")
            - eval_metric: Evaluation metric (default: "roc_auc")
            - output_path: Model save path (default: "av_model")
        artifacts: Optional artifacts dict

    Returns:
        Tuple of:
            - predictor: Trained TabularPredictor
            - summary: Dict with av_auc and av_rows
    """
    # Extract config with defaults
    presets = config.get("presets", "medium_quality_faster_train")
    time_limit = config.get("time_limit", 600)
    included_model_types = config.get("included_model_types")
    problem_type = config.get("problem_type", "binary")
    eval_metric = config.get("eval_metric", "roc_auc")
    label = config.get("label", "__is_test__")
    output_path = config.get("output_path", "av_model")

    # Create TabularPredictor
    predictor = TabularPredictor(
        label=label,
        problem_type=problem_type,
        eval_metric=eval_metric,
        path=str(output_path),
        verbosity=2,
    )

    # Prepare fit_kwargs
    fit_kwargs = {"presets": presets, "time_limit": time_limit}
    if included_model_types:
        fit_kwargs["included_model_types"] = included_model_types

    # Train
    predictor.fit(train_df, **fit_kwargs)

    # Extract score from leaderboard
    leaderboard = predictor.leaderboard(silent=True)
    auc = None
    if "score_val" in leaderboard.columns:
        auc = float(leaderboard["score_val"].max())

    summary = {"av_auc": auc, "av_rows": len(train_df)}
    return predictor, summary


def predict(
    model: TabularPredictor,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
    artifacts: Optional[Any] = None,
) -> pd.Series:
    """
    Predict probabilities for test set being from test distribution.

    Args:
        model: Trained TabularPredictor
        test_df: Data to predict on (without __is_test__ label)
        config: Config dict (not used)
        artifacts: Optional artifacts dict

    Returns:
        Series of probabilities for class 1 (is_test)
    """
    # Return probabilities for class 1 (is_test)
    proba = model.predict_proba(test_df, as_multiclass=False)
    if isinstance(proba, pd.DataFrame):
        return proba[1]  # Column for class 1
    return proba
