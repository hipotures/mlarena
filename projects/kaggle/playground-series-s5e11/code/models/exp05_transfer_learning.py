"""
Experiment 5: Transfer Learning with Original Dataset

Strategy:
- Pre-train model on original dataset (20k samples)
- Use predictions as meta-features for competition data
- Statistical augmentation (industry averages by loan_purpose, grade)
- AutoGluon with enriched features
- Uses Tier 1 features

Expected improvement: +0.005-0.01 AUC
Time budget: ~1-1.5 hours

NOTE: Requires original dataset at: data/loan_dataset_20000.csv
Download from: https://www.kaggle.com/datasets/nabihazahid/loan-prediction-dataset-2025
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

warnings.filterwarnings('ignore')

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PREP_DIR = PROJECT_ROOT / "code" / "preprocessing"
sys.path.insert(0, str(PREP_DIR))

from fe_tier1 import add_tier1_features  # noqa: E402

from kaggle_tools.config_models import ModelConfig  # noqa: E402

EXPERIMENT_NAME = "exp05_transfer_learning"

# Paths
ORIGINAL_DATASET_PATH = DATA_DIR / "loan_dataset_20000.csv"

# Global storage
_PRETRAINED_MODEL = None
_INDUSTRY_STATS = None


def get_default_config() -> Dict[str, Any]:
    """Configuration for transfer learning experiment."""
    return {
        "hyperparameters": {
            "presets": "best_quality",
            "time_limit": 5400,  # 1.5 hours total (including pre-training)
            "pretrain_time_limit": 1800,  # 30 min for pre-training
            "finetune_time_limit": 3600,  # 1 hour for fine-tuning
            "num_bag_folds": 5,
            "num_stack_levels": 1,
        },
        "model": {
            "eval_metric": "roc_auc",
            "problem_type": "binary",
        },
        "preprocessing": {
            "feature_set": "tier1_with_transfer",
            "use_original_dataset": True,
        }
    }


def load_and_map_original_dataset() -> pd.DataFrame:
    """
    Load and map original dataset features to competition schema.

    Returns:
        DataFrame with mapped features
    """
    if not ORIGINAL_DATASET_PATH.exists():
        print(f"[{EXPERIMENT_NAME}] WARNING: Original dataset not found at {ORIGINAL_DATASET_PATH}")
        print(f"[{EXPERIMENT_NAME}] Download from: https://www.kaggle.com/datasets/nabihazahid/loan-prediction-dataset-2025")
        print(f"[{EXPERIMENT_NAME}] Proceeding without transfer learning...")
        return None

    print(f"[{EXPERIMENT_NAME}] Loading original dataset from {ORIGINAL_DATASET_PATH}")
    original_df = pd.read_csv(ORIGINAL_DATASET_PATH)

    print(f"[{EXPERIMENT_NAME}] Original dataset shape: {original_df.shape}")
    print(f"[{EXPERIMENT_NAME}] Original columns: {list(original_df.columns)}")

    # Map column names if needed (original dataset may have different names)
    # Assuming original has same structure - adjust mapping as needed
    if 'loan_paid_back' not in original_df.columns:
        if 'Loan_Status' in original_df.columns:
            original_df['loan_paid_back'] = (original_df['Loan_Status'] == 'Y').astype(int)
        elif 'loan_status' in original_df.columns:
            original_df['loan_paid_back'] = original_df['loan_status']

    return original_df


def compute_industry_statistics(df: pd.DataFrame, target_col: str = 'loan_paid_back') -> Dict[str, pd.DataFrame]:
    """
    Compute industry-level statistics from original dataset.

    Statistics computed:
    - Mean default rate by loan_purpose
    - Mean default rate by grade
    - Mean default rate by employment_status
    - Median income by purpose
    - Median loan amount by grade

    Returns:
        Dictionary of statistics DataFrames
    """
    stats = {}

    print(f"[{EXPERIMENT_NAME}] Computing industry statistics...")

    # Default rate by loan purpose
    if 'loan_purpose' in df.columns:
        purpose_stats = df.groupby('loan_purpose').agg({
            target_col: ['mean', 'count'],
            'annual_income': 'median',
            'loan_amount': 'median',
        }).reset_index()
        purpose_stats.columns = ['loan_purpose', 'default_rate', 'count', 'median_income', 'median_loan']
        stats['purpose'] = purpose_stats
        print(f"[{EXPERIMENT_NAME}]   Loan purpose stats: {len(purpose_stats)} categories")

    # Default rate by grade
    if 'grade_subgrade' in df.columns:
        df['grade'] = df['grade_subgrade'].str[0]
        grade_stats = df.groupby('grade').agg({
            target_col: ['mean', 'count'],
            'interest_rate': 'median',
        }).reset_index()
        grade_stats.columns = ['grade', 'default_rate', 'count', 'median_interest_rate']
        stats['grade'] = grade_stats
        print(f"[{EXPERIMENT_NAME}]   Grade stats: {len(grade_stats)} grades")

    # Default rate by employment status
    if 'employment_status' in df.columns:
        employment_stats = df.groupby('employment_status').agg({
            target_col: ['mean', 'count'],
        }).reset_index()
        employment_stats.columns = ['employment_status', 'default_rate', 'count']
        stats['employment'] = employment_stats
        print(f"[{EXPERIMENT_NAME}]   Employment stats: {len(employment_stats)} statuses")

    return stats


def add_transfer_features(
    df: pd.DataFrame,
    pretrained_model: Optional[TabularPredictor] = None,
    industry_stats: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Add transfer learning features to dataframe.

    Args:
        df: Input dataframe
        pretrained_model: Pre-trained model on original data
        industry_stats: Industry statistics from original data

    Returns:
        DataFrame with transfer features
    """
    enriched = df.copy()

    # 1. Add predictions from pre-trained model as meta-feature
    if pretrained_model is not None:
        print(f"[{EXPERIMENT_NAME}]   Adding pre-trained model predictions as meta-feature...")
        try:
            pretrain_pred = pretrained_model.predict_proba(enriched, as_multiclass=False)
            enriched['pretrain_pred'] = pretrain_pred
            print(f"[{EXPERIMENT_NAME}]   Pre-trained predictions: mean={pretrain_pred.mean():.4f}")
        except Exception as e:
            print(f"[{EXPERIMENT_NAME}]   Warning: Could not generate pre-trained predictions: {e}")

    # 2. Add industry statistics as features
    if industry_stats is not None:
        # Purpose-based stats
        if 'purpose' in industry_stats and 'loan_purpose' in enriched.columns:
            print(f"[{EXPERIMENT_NAME}]   Merging loan purpose industry stats...")
            enriched = enriched.merge(
                industry_stats['purpose'][['loan_purpose', 'default_rate', 'median_income', 'median_loan']],
                on='loan_purpose',
                how='left',
                suffixes=('', '_industry_purpose')
            )
            enriched.rename(columns={
                'default_rate': 'industry_default_rate_purpose',
                'median_income': 'industry_median_income_purpose',
                'median_loan': 'industry_median_loan_purpose',
            }, inplace=True)

        # Grade-based stats
        if 'grade' in industry_stats and 'grade_subgrade' in enriched.columns:
            print(f"[{EXPERIMENT_NAME}]   Merging grade industry stats...")
            enriched['grade'] = enriched['grade_subgrade'].str[0]
            enriched = enriched.merge(
                industry_stats['grade'][['grade', 'default_rate', 'median_interest_rate']],
                on='grade',
                how='left',
                suffixes=('', '_industry_grade')
            )
            enriched.rename(columns={
                'default_rate': 'industry_default_rate_grade',
                'median_interest_rate': 'industry_median_interest_rate',
            }, inplace=True)

        # Employment-based stats
        if 'employment' in industry_stats and 'employment_status' in enriched.columns:
            print(f"[{EXPERIMENT_NAME}]   Merging employment industry stats...")
            enriched = enriched.merge(
                industry_stats['employment'][['employment_status', 'default_rate']],
                on='employment_status',
                how='left',
                suffixes=('', '_industry_employment')
            )
            enriched.rename(columns={
                'default_rate': 'industry_default_rate_employment',
            }, inplace=True)

        # Deviation features (how much does this loan deviate from industry norms?)
        if 'industry_median_income_purpose' in enriched.columns and 'annual_income' in enriched.columns:
            enriched['income_vs_industry'] = enriched['annual_income'] / (enriched['industry_median_income_purpose'] + 1)

        if 'industry_median_loan_purpose' in enriched.columns and 'loan_amount' in enriched.columns:
            enriched['loan_vs_industry'] = enriched['loan_amount'] / (enriched['industry_median_loan_purpose'] + 1)

        if 'industry_median_interest_rate' in enriched.columns and 'interest_rate' in enriched.columns:
            enriched['rate_vs_industry'] = enriched['interest_rate'] - enriched['industry_median_interest_rate']

    return enriched


def preprocess(train_df: pd.DataFrame, config: ModelConfig, is_train: bool = True) -> pd.DataFrame:
    """Apply Tier 1 features + transfer learning features."""
    global _PRETRAINED_MODEL, _INDUSTRY_STATS

    print(f"[{EXPERIMENT_NAME}] Applying transfer learning features...")

    # Start with Tier 1 features
    enriched = add_tier1_features(train_df)

    # Add transfer features
    if is_train:
        # For training, we need to pre-train model first
        # This will be done in train() function
        # Here we just apply industry stats if available
        if _INDUSTRY_STATS is not None:
            enriched = add_transfer_features(enriched, None, _INDUSTRY_STATS)
    else:
        # For testing, use pre-trained model and stats
        enriched = add_transfer_features(enriched, _PRETRAINED_MODEL, _INDUSTRY_STATS)

    return enriched


def train(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    config: ModelConfig,
    artifacts: Optional[Dict[str, Any]] = None,
) -> Tuple[TabularPredictor, Dict[str, Any]]:
    """
    Train model with transfer learning.

    Steps:
    1. Load original dataset
    2. Compute industry statistics
    3. Pre-train model on original dataset
    4. Add transfer features to competition data
    5. Fine-tune model on competition data
    """
    global _PRETRAINED_MODEL, _INDUSTRY_STATS

    

    target_col = config.dataset.target
    hparams = config.hyperparameters

    print(f"\n{'='*60}")
    print(f"[{EXPERIMENT_NAME}] Transfer Learning Pipeline")
    print(f"{'='*60}\n")

    # Step 1: Load original dataset
    original_df = load_and_map_original_dataset()

    if original_df is None:
        print(f"[{EXPERIMENT_NAME}] Skipping transfer learning, training baseline...")
        # Fall back to regular training
        predictor = TabularPredictor(
            label=target_col,
            eval_metric='roc_auc',
            problem_type='binary',
            path=str(PROJECT_ROOT / "AutogluonModels" / EXPERIMENT_NAME),
        )
        predictor.fit(
            train_data=train_df,
            presets='best_quality',
            time_limit=getattr(hparams, 'time_limit', 5400),
        )
        return predictor, {"experiment": EXPERIMENT_NAME, "transfer_learning": False}

    # Step 2: Compute industry statistics
    _INDUSTRY_STATS = compute_industry_statistics(original_df, target_col)

    # Step 3: Pre-train on original dataset
    print(f"\n[{EXPERIMENT_NAME}] Step 3: Pre-training on original dataset...")
    print(f"[{EXPERIMENT_NAME}] Original dataset size: {len(original_df)}")

    # Apply Tier 1 features to original dataset
    original_enriched = add_tier1_features(original_df)

    pretrain_predictor = TabularPredictor(
        label=target_col,
        eval_metric='roc_auc',
        problem_type='binary',
        path=str(PROJECT_ROOT / "AutogluonModels" / f"{EXPERIMENT_NAME}_pretrain"),
    )

    pretrain_predictor.fit(
        train_data=original_enriched,
        presets='medium_quality',  # Use medium quality for speed
        time_limit=getattr(hparams, 'pretrain_time_limit', 1800),  # 30 minutes
    )

    _PRETRAINED_MODEL = pretrain_predictor

    pretrain_leaderboard = pretrain_predictor.leaderboard(original_enriched, silent=True)
    pretrain_score = pretrain_leaderboard.iloc[0]['score_val']

    print(f"[{EXPERIMENT_NAME}] Pre-training completed. Best score: {pretrain_score:.5f}")

    # Step 4: Add transfer features to competition data
    print(f"\n[{EXPERIMENT_NAME}] Step 4: Adding transfer features to competition data...")

    # Add industry stats
    train_enriched = add_transfer_features(train_df, None, _INDUSTRY_STATS)

    # Generate meta-features from pre-trained model
    print(f"[{EXPERIMENT_NAME}] Generating meta-features from pre-trained model...")
    pretrain_pred = pretrain_predictor.predict_proba(train_enriched, as_multiclass=False)
    train_enriched['pretrain_pred'] = pretrain_pred

    print(f"[{EXPERIMENT_NAME}] Transfer features added. New feature count: {len(train_enriched.columns)}")

    # Step 5: Fine-tune on competition data
    print(f"\n[{EXPERIMENT_NAME}] Step 5: Fine-tuning on competition data...")

    final_predictor = TabularPredictor(
        label=target_col,
        eval_metric='roc_auc',
        problem_type='binary',
        path=str(PROJECT_ROOT / "AutogluonModels" / EXPERIMENT_NAME),
    )

    final_predictor.fit(
        train_data=train_enriched,
        presets='best_quality',
        time_limit=getattr(hparams, 'finetune_time_limit', 3600),  # 1 hour
        num_bag_folds=getattr(hparams, 'num_bag_folds', 5),
        num_stack_levels=getattr(hparams, 'num_stack_levels', 1),
    )

    leaderboard = final_predictor.leaderboard(train_enriched, silent=True)
    best_model = leaderboard.iloc[0]

    metadata = {
        "experiment": EXPERIMENT_NAME,
        "transfer_learning": True,
        "original_dataset_size": len(original_df),
        "pretrain_score": pretrain_score,
        "finetune_best_model": best_model['model'],
        "finetune_score": best_model['score_val'],
        "industry_stats_categories": list(_INDUSTRY_STATS.keys()),
        "feature_count": len(train_enriched.columns) - 1,
    }

    print(f"\n[{EXPERIMENT_NAME}] Training completed!")
    print(f"  Pre-train score: {pretrain_score:.5f}")
    print(f"  Fine-tune score: {best_model['score_val']:.5f}")
    print(f"  Improvement: {best_model['score_val'] - pretrain_score:+.5f}")

    return final_predictor, metadata


def predict(
    model: TabularPredictor,
    test_df: pd.DataFrame,
    config: ModelConfig,
    artifacts: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Generate predictions with transfer learning features."""
    print(f"[{EXPERIMENT_NAME}] Generating predictions with transfer features...")

    # Add transfer features to test data
    test_enriched = add_transfer_features(test_df, _PRETRAINED_MODEL, _INDUSTRY_STATS)

    predictions = model.predict_proba(test_enriched, as_multiclass=False)

    print(f"[{EXPERIMENT_NAME}] Predictions generated. Mean: {predictions.mean():.4f}")

    return predictions


__all__ = ['get_default_config', 'preprocess', 'train', 'predict']
