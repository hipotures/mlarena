"""
AutoGluon model with external dataset support.

This model demonstrates how to use the external dataset (orig_df) provided
by the `external_dataset` preprocessing module.

The model merges train + orig datasets before training, allowing AutoGluon
to learn from both Kaggle competition data and external/original datasets.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from autogluon.tabular import TabularPredictor

from kaggle_tools.config import ModelConfig


def train(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    config: ModelConfig,
    artifacts: Optional[Dict[str, Any]] = None,
) -> Tuple[TabularPredictor, Dict[str, Any]]:
    """
    Train AutoGluon model with optional external dataset merging.

    If artifacts contains 'orig_df', it will be merged with train_df before training.

    Args:
        train_df: Kaggle training data (preprocessed)
        val_df: Validation data (optional, not used)
        config: Model configuration
        artifacts: Optional dict containing:
            - orig_df: External dataset (from external_dataset preprocessing module)
            - sample_weight: Sample weights (from imbalance_handler or av_weights)

    Returns:
        Tuple of (predictor, training_summary)
    """
    # Extract config
    target_column = config.dataset.target
    preset = config.hyperparameters.get("preset", "medium")
    time_limit = config.hyperparameters.get("time_limit", 300)
    use_gpu = config.hyperparameters.get("use_gpu", False)

    # Check for orig_df in artifacts
    orig_df = None
    sample_weight = None
    merged_rows = 0

    if artifacts:
        orig_df = artifacts.get('orig_df')
        sample_weight = artifacts.get('sample_weight')

    # Merge train + orig if available
    if orig_df is not None:
        print(f"[AutoGluon with Orig] Merging external dataset:")
        print(f"  Kaggle train rows: {len(train_df):,}")
        print(f"  External rows:     {len(orig_df):,}")

        # Concatenate train + orig
        train_df = pd.concat([train_df, orig_df], ignore_index=True)
        merged_rows = len(orig_df)

        print(f"  Merged total:      {len(train_df):,}")
    else:
        print("[AutoGluon with Orig] No external dataset found, using Kaggle data only")

    # Remove ID column if present
    id_column = config.dataset.id_column
    ignored_columns = config.dataset.ignored_columns.copy() if config.dataset.ignored_columns else []

    if id_column and id_column in train_df.columns:
        ignored_columns.append(id_column)

    # Drop ignored columns
    features = train_df.drop(columns=ignored_columns, errors='ignore')

    # Ensure target is present
    if target_column not in features.columns:
        raise ValueError(f"Target column '{target_column}' not found in training data")

    # Handle sample weights (if provided)
    sample_weight_col = None
    if sample_weight is not None and len(sample_weight) == len(features):
        sample_weight_col = "sample_weight"
        features[sample_weight_col] = sample_weight["sample_weight"].values
        print(f"[AutoGluon with Orig] Using sample weights: {sample_weight_col}")

    # Train model
    predictor = TabularPredictor(
        label=target_column,
        path=str(config.system.model_path),
        eval_metric=config.dataset.eval_metric if hasattr(config.dataset, 'eval_metric') else None,
        problem_type=config.dataset.problem_type if hasattr(config.dataset, 'problem_type') else None,
        sample_weight=sample_weight_col,
    )

    # Fit model
    predictor.fit(
        train_data=features,
        presets=preset,
        time_limit=time_limit,
        num_gpus=1 if use_gpu else 0,
        verbosity=2,
    )

    # Get best model score
    leaderboard = predictor.leaderboard(silent=True)
    best_score = leaderboard["score_val"].iloc[0] if len(leaderboard) > 0 else None

    # Build training summary
    training_summary = {
        "local_cv": float(best_score) if best_score is not None else None,
        "best_score": float(best_score) if best_score is not None else None,
        "model_path": str(config.system.model_path),
        "used_orig": orig_df is not None,
        "orig_rows": merged_rows,
        "total_train_rows": len(train_df),
        "preset": preset,
        "time_limit": time_limit,
        "use_gpu": use_gpu,
    }

    return predictor, training_summary
