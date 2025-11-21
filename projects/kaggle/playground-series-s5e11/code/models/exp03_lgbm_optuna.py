"""
Experiment 3: LightGBM with Optuna Hyperparameter Tuning

Strategy:
- Custom LightGBM with class_weight='balanced'
- Optuna Bayesian optimization (50 trials)
- Stratified K-Fold CV (5 folds)
- Early stopping
- Feature importance analysis
- Uses Tier 1 features

Expected improvement: +0.01-0.02 AUC
Time budget: ~1-1.5 hours (including tuning)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PREP_DIR = PROJECT_ROOT / "code" / "preprocessing"
sys.path.insert(0, str(PREP_DIR))

from fe_tier1 import add_tier1_features  # noqa: E402

from kaggle_tools.config_models import ModelConfig  # noqa: E402

EXPERIMENT_NAME = "exp03_lgbm_optuna"

# Global storage for best model
_BEST_MODEL = None
_FEATURE_IMPORTANCE = None


def get_default_config() -> Dict[str, Any]:
    """Configuration for LightGBM with Optuna."""
    return {
        "hyperparameters": {
            "n_trials": 50,  # Optuna trials
            "n_folds": 5,  # CV folds
            "early_stopping_rounds": 50,
            "verbose_eval": 100,
        },
        "model": {
            "eval_metric": "roc_auc",
            "problem_type": "binary",
        },
        "preprocessing": {
            "feature_set": "tier1_critical",
        }
    }


def preprocess(train_df: pd.DataFrame, config: ModelConfig, is_train: bool = True) -> pd.DataFrame:
    """Apply Tier 1 feature engineering and encode categoricals for LightGBM."""
    print(f"[{EXPERIMENT_NAME}] Applying Tier 1 features...")
    processed_df = add_tier1_features(train_df)

    # Convert object columns to category dtype for LightGBM
    for col in processed_df.select_dtypes(include=['object']).columns:
        processed_df[col] = processed_df[col].astype('category')

    return processed_df


def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, n_folds: int = 5) -> float:
    """
    Optuna objective function for LightGBM hyperparameter tuning.

    Args:
        trial: Optuna trial object
        X: Feature matrix
        y: Target vector
        n_folds: Number of CV folds

    Returns:
        Mean ROC-AUC across folds
    """
    # Suggest hyperparameters
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'class_weight': 'balanced',  # Handle imbalance
        'random_state': 42,

        # Learning rate and iterations
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),

        # Regularization
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),

        # Sampling
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 7),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }

    # Cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Train model
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ]
        )

        # Predict and score
        y_pred = model.predict(X_val)
        score = roc_auc_score(y_val, y_pred)
        fold_scores.append(score)

    mean_score = np.mean(fold_scores)
    return mean_score


def train(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    config: ModelConfig,
    artifacts: Optional[Dict[str, Any]] = None,
) -> Tuple[lgb.Booster, Dict[str, Any]]:
    """
    Train LightGBM with Optuna hyperparameter tuning.

    Args:
        train_df: Training data (already preprocessed)
        val_df: Validation data (unused, we use CV)
        config: Model configuration
        artifacts: Additional artifacts

    Returns:
        Tuple of (trained_model, training_metadata)
    """
    global _BEST_MODEL, _FEATURE_IMPORTANCE

    

    target_col = config.dataset.target
    hparams = config.hyperparameters

    # Separate features and target
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]

    print(f"\n{'='*60}")
    print(f"[{EXPERIMENT_NAME}] Training Configuration")
    print(f"{'='*60}")
    print(f"Training samples: {len(X)}")
    print(f"Features: {len(X.columns)}")
    print(f"Target: {target_col}")
    print(f"Optuna trials: {getattr(hparams, 'n_trials', 50)}")
    print(f"CV folds: {getattr(hparams, 'n_folds', 5)}")
    print(f"{'='*60}\n")

    # Run Optuna optimization
    print(f"[{EXPERIMENT_NAME}] Starting Optuna hyperparameter optimization...")
    study = optuna.create_study(
        direction='maximize',
        study_name=EXPERIMENT_NAME,
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    study.optimize(
        lambda trial: objective(
            trial,
            X,
            y,
            n_folds=getattr(hparams, 'n_folds', 5)
        ),
        n_trials=getattr(hparams, 'n_trials', 50),
        show_progress_bar=True,
    )

    print(f"\n[{EXPERIMENT_NAME}] Optuna optimization completed!")
    print(f"Best trial ROC-AUC: {study.best_value:.5f}")
    print(f"Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Train final model with best parameters on full training data
    print(f"\n[{EXPERIMENT_NAME}] Training final model with best parameters...")

    best_params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'class_weight': 'balanced',
        'random_state': 42,
        **study.best_params
    }

    # Use stratified split for validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(skf.split(X, y))

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Train with early stopping
    final_model = lgb.train(
        best_params,
        train_data,
        num_boost_round=2000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=100),
        ]
    )

    # Calculate final scores
    train_pred = final_model.predict(X_train)
    val_pred = final_model.predict(X_val)
    train_auc = roc_auc_score(y_train, train_pred)
    val_auc = roc_auc_score(y_val, val_pred)

    print(f"\n[{EXPERIMENT_NAME}] Final model performance:")
    print(f"  Train ROC-AUC: {train_auc:.5f}")
    print(f"  Valid ROC-AUC: {val_auc:.5f}")
    print(f"  Best iteration: {final_model.best_iteration}")

    # Feature importance
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': final_model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)

    print(f"\n[{EXPERIMENT_NAME}] Top 15 important features:")
    print(importance_df.head(15).to_string(index=False))

    # Store globally
    _BEST_MODEL = final_model
    _FEATURE_IMPORTANCE = importance_df

    metadata = {
        "experiment": EXPERIMENT_NAME,
        "best_trial_score": study.best_value,
        "final_train_auc": train_auc,
        "final_val_auc": val_auc,
        "best_iteration": final_model.best_iteration,
        "n_trials": len(study.trials),
        "best_params": study.best_params,
        "feature_count": len(X.columns),
    }

    return final_model, metadata


def predict(
    model: lgb.Booster,
    test_df: pd.DataFrame,
    config: ModelConfig,
    artifacts: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Generate predictions using trained LightGBM model."""
    

    target_col = config.dataset.target

    # Remove target if present
    X_test = test_df.drop(columns=[target_col], errors='ignore')

    print(f"[{EXPERIMENT_NAME}] Generating predictions on {len(X_test)} samples...")

    pred_proba = model.predict(X_test)

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
