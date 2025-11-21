"""
Experiment 1: Enhanced Feature Engineering (Tier 1)

Strategy:
- Log transformations for skewed distributions
- Yeo-Johnson power transformations
- Critical DTI-based features
- Payment capacity metrics
- Interest cost analysis

Expected improvement: +0.03-0.05 AUC
Time budget: ~1-1.5 hours
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from autogluon.tabular import TabularPredictor

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PREP_DIR = PROJECT_ROOT / "code" / "preprocessing"
sys.path.insert(0, str(PREP_DIR))

from fe_tier1 import add_tier1_features  # noqa: E402

from kaggle_tools.config_models import ModelConfig  # noqa: E402

EXPERIMENT_NAME = "exp01_tier1_features"


def get_default_config() -> Dict[str, Any]:
    """Configuration optimized for 1-1.5h runtime."""
    return {
        "hyperparameters": {
            "presets": "best_quality",  # Best quality with stacking
            "time_limit": 5400,  # 1.5 hours
            "num_bag_folds": 5,  # Balanced CV
            "num_stack_levels": 1,  # Single stacking layer
        },
        "model": {
            "eval_metric": "roc_auc",
            "problem_type": "binary",
            "leaderboard_rows": 30,
        },
        "preprocessing": {
            "feature_set": "tier1_critical",
        }
    }


def preprocess(train_df: pd.DataFrame, config: ModelConfig, is_train: bool = True) -> pd.DataFrame:
    """
    Apply Tier 1 feature engineering.

    Args:
        train_df: Input dataframe
        config: Model configuration
        is_train: Whether this is training data (unused, kept for compatibility)

    Returns:
        DataFrame with enriched features
    """
    print(f"[{EXPERIMENT_NAME}] Applying Tier 1 feature engineering...")
    enriched = add_tier1_features(train_df)

    print(f"[{EXPERIMENT_NAME}] Original features: {len(train_df.columns)}")
    print(f"[{EXPERIMENT_NAME}] Enriched features: {len(enriched.columns)}")
    print(f"[{EXPERIMENT_NAME}] New features: {len(enriched.columns) - len(train_df.columns)}")

    return enriched


def train(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    config: ModelConfig,
    artifacts: Optional[Dict[str, Any]] = None,
) -> Tuple[TabularPredictor, Dict[str, Any]]:
    """
    Train AutoGluon model with Tier 1 features.

    Args:
        train_df: Training data (already preprocessed)
        val_df: Validation data (optional, unused for AutoGluon)
        config: Model configuration
        artifacts: Additional artifacts from previous steps

    Returns:
        Tuple of (trained_predictor, training_metadata)
    """
    target_col = config.dataset.target
    hparams = config.hyperparameters

    print(f"\n{'='*60}")
    print(f"[{EXPERIMENT_NAME}] Training Configuration")
    print(f"{'='*60}")
    print(f"Training samples: {len(train_df)}")
    print(f"Features: {len(train_df.columns) - 1}")  # -1 for target
    print(f"Target: {target_col}")
    print(f"Preset: {getattr(hparams, 'presets', 'best_quality')}")
    print(f"Time limit: {getattr(hparams, 'time_limit', 3600)} seconds")
    print(f"Bag folds: {getattr(hparams, 'num_bag_folds', 5)}")
    print(f"Stack levels: {getattr(hparams, 'num_stack_levels', 1)}")
    print(f"{'='*60}\n")

    # Setup predictor
    predictor = TabularPredictor(
        label=target_col,
        eval_metric='roc_auc',
        problem_type='binary',
        path=str(PROJECT_ROOT / "AutogluonModels" / EXPERIMENT_NAME),
    )

    # Train with best_quality preset
    predictor.fit(
        train_data=train_df,
        presets=getattr(hparams, 'presets', 'best_quality'),
        time_limit=getattr(hparams, 'time_limit', 5400),
        num_bag_folds=getattr(hparams, 'num_bag_folds', 5),
        num_stack_levels=getattr(hparams, 'num_stack_levels', 1),
    )

    # Get training results
    leaderboard = predictor.leaderboard(train_df, silent=True)
    best_model = leaderboard.iloc[0]

    metadata = {
        "experiment": EXPERIMENT_NAME,
        "best_model": best_model['model'],
        "best_score_val": best_model['score_val'],
        "num_models": len(leaderboard),
        "feature_count": len(train_df.columns) - 1,
    }

    print(f"\n[{EXPERIMENT_NAME}] Training completed!")
    print(f"Best model: {metadata['best_model']}")
    print(f"Best validation ROC-AUC: {metadata['best_score_val']:.5f}")
    print(f"Total models trained: {metadata['num_models']}")

    return predictor, metadata


def predict(
    model: TabularPredictor,
    test_df: pd.DataFrame,
    config: ModelConfig,
    artifacts: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Generate predictions using trained model.

    Args:
        model: Trained TabularPredictor
        test_df: Test data (already preprocessed)
        config: Model configuration
        artifacts: Additional artifacts

    Returns:
        DataFrame with columns: [id, target]
    """
    print(f"[{EXPERIMENT_NAME}] Generating predictions on {len(test_df)} samples...")

    # Get probabilities for positive class
    pred_proba = model.predict_proba(test_df, as_multiclass=False)

    print(f"[{EXPERIMENT_NAME}] Predictions generated. Mean: {pred_proba.mean():.4f}")

    # Create submission DataFrame
    id_col = config.dataset.id_column
    target_col = config.dataset.target

    submission_df = pd.DataFrame({
        id_col: test_df[id_col],
        target_col: pred_proba
    })

    return submission_df


# Export functions for model runner
__all__ = ['get_default_config', 'preprocess', 'train', 'predict']
