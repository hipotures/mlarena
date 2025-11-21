"""
Experiment 2: Advanced Encoding + Interactions

Strategy:
- Target Encoding with CV (prevents leakage)
- Weight of Evidence encoding
- Polynomial features (degree 2)
- Cross-feature interactions
- Risk-adjusted metrics
- Builds on Tier 1 features

Expected improvement: +0.02-0.04 AUC
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

from fe_tier2_encoding import add_tier2_features  # noqa: E402

from kaggle_tools.config_models import ModelConfig  # noqa: E402

EXPERIMENT_NAME = "exp02_tier2_encoding"

# Global artifacts for encoding
_ENCODING_ARTIFACTS = {}


def get_default_config() -> Dict[str, Any]:
    """Configuration optimized for 1-1.5h runtime."""
    return {
        "hyperparameters": {
            "presets": "best_quality",
            "time_limit": 5400,  # 1.5 hours
            "num_bag_folds": 5,
            "num_stack_levels": 1,
        },
        "model": {
            "eval_metric": "roc_auc",
            "problem_type": "binary",
            "leaderboard_rows": 30,
        },
        "preprocessing": {
            "feature_set": "tier2_encoding",
            "include_tier1": True,
        }
    }


def preprocess(train_df: pd.DataFrame, config: ModelConfig, is_train: bool = True) -> pd.DataFrame:
    """
    Apply Tier 2 feature engineering with encodings.

    Note: For training, this stores encoding artifacts globally.
    For testing, it uses those artifacts.

    Args:
        train_df: Input dataframe
        config: Model configuration
        is_train: Whether this is training data

    Returns:
        DataFrame with enriched features
    """
    global _ENCODING_ARTIFACTS

    

    target_col = config.dataset.target

    print(f"[{EXPERIMENT_NAME}] Applying Tier 2 encoding features...")
    print(f"[{EXPERIMENT_NAME}] Is training: {is_train}")

    if is_train:
        # Training: fit encoders and store artifacts
        enriched, _, artifacts = add_tier2_features(
            train_df=train_df,
            test_df=None,
            target_col=target_col,
            include_tier1=True
        )
        _ENCODING_ARTIFACTS = artifacts
        print(f"[{EXPERIMENT_NAME}] Fitted and stored encoding artifacts")

    else:
        # Testing: use stored encoders
        print(f"[{EXPERIMENT_NAME}] Using stored encoding artifacts for test data")

        # Need a dummy train_df to fit encoders (they need target)
        # Actually, we should apply stored encoders directly
        # This is a limitation - we need to refactor for test-only application

        # Workaround: Apply tier1, then manually apply stored encoders
        from fe_tier1 import add_tier1_features

        enriched = add_tier1_features(train_df)

        # Apply stored encoders
        if 'target_encoder' in _ENCODING_ARTIFACTS:
            enriched = _ENCODING_ARTIFACTS['target_encoder'].transform(enriched)

        if 'woe_encoder' in _ENCODING_ARTIFACTS:
            enriched = _ENCODING_ARTIFACTS['woe_encoder'].transform(enriched)

        # Apply polynomial features if present
        if 'poly_transformer' in _ENCODING_ARTIFACTS:
            poly = _ENCODING_ARTIFACTS['poly_transformer']
            poly_cols = ['credit_score', 'debt_to_income_ratio', 'annual_income']
            poly_cols = [c for c in poly_cols if c in enriched.columns]

            if poly_cols:
                poly_transformed = poly.transform(enriched[poly_cols])
                poly_names = poly.get_feature_names_out(poly_cols)

                for i, name in enumerate(poly_names):
                    if '^' in name or ' ' in name:
                        enriched[f'poly_{name}'] = poly_transformed[:, i]

        # Apply manual interactions (these don't need fitting)
        if 'annual_income' in enriched.columns and 'credit_score' in enriched.columns:
            enriched['income_credit_power'] = (enriched['annual_income'] * enriched['credit_score']) / 100000

        if 'loan_amount' in enriched.columns and 'interest_rate' in enriched.columns:
            enriched['loan_cost_indicator'] = enriched['loan_amount'] * enriched['interest_rate']

        if 'credit_score' in enriched.columns and 'debt_to_income_ratio' in enriched.columns:
            enriched['credit_risk_score'] = enriched['credit_score'] / (enriched['debt_to_income_ratio'] + 0.01)

        if 'interest_rate' in enriched.columns and 'credit_score' in enriched.columns:
            enriched['risk_adjusted_return'] = (enriched['interest_rate'] * 100) / (enriched['credit_score'] / 10)

        if 'grade_subgrade' in enriched.columns:
            enriched['grade'] = enriched['grade_subgrade'].str[0]
            grade_map = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}
            enriched['grade_numeric'] = enriched['grade'].map(grade_map).fillna(0)
            enriched['subgrade_num'] = enriched['grade_subgrade'].str[1:].astype(float)

    print(f"[{EXPERIMENT_NAME}] Feature engineering completed")
    print(f"[{EXPERIMENT_NAME}] Final features: {len(enriched.columns)}")

    return enriched


def train(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    config: ModelConfig,
    artifacts: Optional[Dict[str, Any]] = None,
) -> Tuple[TabularPredictor, Dict[str, Any]]:
    """
    Train AutoGluon model with Tier 2 features.

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
    print(f"Features: {len(train_df.columns) - 1}")
    print(f"Target: {target_col}")
    print(f"Preset: {getattr(hparams, 'presets', 'best_quality')}")
    print(f"Time limit: {getattr(hparams, 'time_limit', 3600)} seconds")
    print(f"{'='*60}\n")

    # Setup predictor
    predictor = TabularPredictor(
        label=target_col,
        eval_metric='roc_auc',
        problem_type='binary',
        path=str(PROJECT_ROOT / "AutogluonModels" / EXPERIMENT_NAME),
    )

    # Train
    predictor.fit(
        train_data=train_df,
        presets=getattr(hparams, 'presets', 'best_quality'),
        time_limit=getattr(hparams, 'time_limit', 5400),
        num_bag_folds=getattr(hparams, 'num_bag_folds', 5),
        num_stack_levels=getattr(hparams, 'num_stack_levels', 1),
    )

    # Get results
    leaderboard = predictor.leaderboard(train_df, silent=True)
    best_model = leaderboard.iloc[0]

    metadata = {
        "experiment": EXPERIMENT_NAME,
        "best_model": best_model['model'],
        "best_score_val": best_model['score_val'],
        "num_models": len(leaderboard),
        "feature_count": len(train_df.columns) - 1,
        "encoding_artifacts": list(_ENCODING_ARTIFACTS.keys()),
    }

    print(f"\n[{EXPERIMENT_NAME}] Training completed!")
    print(f"Best model: {metadata['best_model']}")
    print(f"Best validation ROC-AUC: {metadata['best_score_val']:.5f}")
    print(f"Encoding artifacts: {metadata['encoding_artifacts']}")

    return predictor, metadata


def predict(
    model: TabularPredictor,
    test_df: pd.DataFrame,
    config: ModelConfig,
    artifacts: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Generate predictions using trained model."""
    print(f"[{EXPERIMENT_NAME}] Generating predictions on {len(test_df)} samples...")

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


__all__ = ['get_default_config', 'preprocess', 'train', 'predict']
