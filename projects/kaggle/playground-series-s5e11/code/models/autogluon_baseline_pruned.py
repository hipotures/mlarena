"""
AutoGluon with MINIMAL feature engineering + automatic feature pruning.

Philosophy: Start with original features + only essential transforms (log, basic ratios),
then let AutoGluon's feature pruning automatically remove harmful features.

Key difference from autogluon_features_pruned:
- autogluon_features_pruned: ALL rich features → prune bad ones
- autogluon_baseline_pruned: MINIMAL features → prune bad ones

This tests whether starting minimal and pruning is better than starting rich and pruning.

Expected result: Potentially better than 0.92434 if rich features were adding noise.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import sys

MODEL_DIR = Path(__file__).parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

from kaggle_tools.config_models import ModelConfig

VARIANT_NAME = "baseline-pruned"


def get_default_config() -> Dict[str, Any]:
    """
    Extended time limit because feature pruning roughly doubles training time.
    """
    return {
        "hyperparameters": {
            "presets": "best_quality",
            "time_limit": 28800,  # 8 hours
            "num_bag_folds": 5,
            "num_stack_levels": 1,
            "use_gpu": False,
        },
        "model": {
            "leaderboard_rows": 30,
        },
        "feature_prune": {
            "enabled": True,
            # Core pruning control
            "force_prune": True,           # Force all models to use pruned features
            "time_limit": None,            # Auto-calculated as 30% of total time

            # Data sampling
            "max_train_samples": 50000,    # Max training rows for pruning model
            "min_fi_samples": 10000,       # Min validation rows for feature importance

            # Pruning thresholds
            "prune_threshold": "noise",    # 'noise' or float - importance threshold
            "prune_ratio": 0.05,           # 5% worst features per round

            # Stopping criteria
            "stopping_round": 50,          # Stop after N rounds without improvement
            "min_improvement": 1e-5,       # Minimum relative score improvement
            "max_fits": None,              # Max model fits (None = unlimited)

            # Reproducibility
            "seed": 42,
            "raise_exception": False,      # Don't crash on errors
        },
    }


def _drop_ignored(df: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    """Drop ignored columns (id, etc.) before training."""
    drop_cols = set(config.dataset.ignored_columns + [config.dataset.id_column])
    drop_cols.discard(config.dataset.target)
    return df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    MINIMAL feature engineering - only essential transforms.

    Strategy: Start with clean base + let pruning remove what doesn't help.
    """
    enriched = df.copy()

    # Log transforms (stabilize distributions)
    enriched["log_annual_income"] = np.log1p(enriched["annual_income"])
    enriched["log_loan_amount"] = np.log1p(enriched["loan_amount"])

    # Single key ratio
    enriched["loan_to_income_ratio"] = enriched["loan_amount"] / (enriched["annual_income"] + 1)

    # NO buckets, NO interactions, NO flags
    return enriched


def preprocess(df: pd.DataFrame, config: ModelConfig, is_train: bool = True) -> pd.DataFrame:
    """Apply minimal feature engineering - pruning happens during fit()."""
    return _engineer_features(df)


def train(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> Tuple[TabularPredictor, Dict[str, Any]]:
    """
    Train with automatic feature pruning enabled.

    Feature pruning uses Recursive Feature Elimination (RFE) with
    Permutation Feature Importance to identify and remove features
    that hurt model performance.
    """
    print(f"[{VARIANT_NAME}] Training with MINIMAL features + AUTOMATIC PRUNING")

    # Show column info BEFORE dropping
    print(f"[{VARIANT_NAME}] Columns summary:")
    print(f"  - Target column: '{config.dataset.target}'")
    print(f"  - Ignored columns (will be dropped): {config.dataset.ignored_columns}")
    print(f"  - ID column (will be dropped): '{config.dataset.id_column}'")
    print(f"  - Total columns before drop: {len(train_df.columns)}")

    # Drop ignored columns (id, etc.) before training
    features = _drop_ignored(train_df, config)
    train_data = features.copy()
    train_data[config.dataset.target] = train_df[config.dataset.target]

    # Show features AFTER dropping
    feature_cols = [col for col in features.columns if col != config.dataset.target]
    print(f"  - Feature count after drop: {len(feature_cols)}")
    print(f"  - Feature names: {feature_cols}")
    print(f"[{VARIANT_NAME}] AutoGluon will automatically prune harmful features from above list")

    # Prepare validation data if provided
    tuning_data = None
    if val_df is not None:
        val_features = _drop_ignored(val_df, config)
        tuning_data = val_features.copy()
        tuning_data[config.dataset.target] = val_df[config.dataset.target]

    # Get hyperparameters from config
    hyper_cfg = config.hyperparameters if hasattr(config, 'hyperparameters') else {}
    presets = getattr(hyper_cfg, 'presets', 'best_quality')
    time_limit = getattr(hyper_cfg, 'time_limit', 43200)
    num_bag_folds = getattr(hyper_cfg, 'num_bag_folds', 5)
    num_stack_levels = getattr(hyper_cfg, 'num_stack_levels', 1)
    use_gpu = getattr(hyper_cfg, 'use_gpu', False)
    excluded_models = getattr(hyper_cfg, 'excluded_models', None)
    included_models = getattr(hyper_cfg, 'included_model_types', None)

    # Get feature pruning config
    prune_cfg = getattr(config, 'feature_prune', {})
    if isinstance(prune_cfg, dict):
        force_prune = prune_cfg.get('force_prune', True)
        prune_time_limit = prune_cfg.get('time_limit', None)
        max_train_samples = prune_cfg.get('max_train_samples', 50000)
        min_fi_samples = prune_cfg.get('min_fi_samples', 10000)
        prune_threshold = prune_cfg.get('prune_threshold', 'noise')
        prune_ratio = prune_cfg.get('prune_ratio', 0.05)
        stopping_round = prune_cfg.get('stopping_round', 50)
        min_improvement = prune_cfg.get('min_improvement', 1e-5)
        max_fits = prune_cfg.get('max_fits', None)
        seed = prune_cfg.get('seed', 42)
        raise_exception = prune_cfg.get('raise_exception', False)
    else:
        force_prune = getattr(prune_cfg, 'force_prune', True)
        prune_time_limit = getattr(prune_cfg, 'time_limit', None)
        max_train_samples = getattr(prune_cfg, 'max_train_samples', 50000)
        min_fi_samples = getattr(prune_cfg, 'min_fi_samples', 10000)
        prune_threshold = getattr(prune_cfg, 'prune_threshold', 'noise')
        prune_ratio = getattr(prune_cfg, 'prune_ratio', 0.05)
        stopping_round = getattr(prune_cfg, 'stopping_round', 50)
        min_improvement = getattr(prune_cfg, 'min_improvement', 1e-5)
        max_fits = getattr(prune_cfg, 'max_fits', None)
        seed = getattr(prune_cfg, 'seed', 42)
        raise_exception = getattr(prune_cfg, 'raise_exception', False)

    # Calculate time limit for pruning (30% of total if not specified)
    if prune_time_limit is None:
        prune_time_limit = int(time_limit * 0.3)

    # Build feature_prune_kwargs with all parameters
    # NOTE: Do NOT include 'time_limit' here - it causes duplicate argument error
    feature_prune_kwargs = {
        'force_prune': force_prune,
        'max_train_samples': max_train_samples,
        'min_fi_samples': min_fi_samples,
        'prune_threshold': prune_threshold,
        'prune_ratio': prune_ratio,
        'stopping_round': stopping_round,
        'min_improvement': min_improvement,
        'max_fits': max_fits,
        'seed': seed,
        'raise_exception': raise_exception,
    }

    print(f"[{VARIANT_NAME}] Config: presets={presets}, time_limit={time_limit}s")
    print(f"[{VARIANT_NAME}] Bagging: {num_bag_folds} folds, {num_stack_levels} stack levels")
    print(f"[{VARIANT_NAME}] Excluded model types: {excluded_models if excluded_models else 'None (all models enabled)'}")
    print(f"[{VARIANT_NAME}] Feature pruning:")
    print(f"  - force_prune: {force_prune}")
    print(f"  - prune_threshold: {prune_threshold}")
    print(f"  - prune_ratio: {prune_ratio} (remove {prune_ratio*100:.1f}% worst features per round)")
    print(f"  - stopping_round: {stopping_round} (max rounds without improvement)")
    print(f"  - min_improvement: {min_improvement}")
    print(f"  - max_train_samples: {max_train_samples}, min_fi_samples: {min_fi_samples}")
    print(f"  - max_fits: {max_fits}, seed: {seed}")

    # Create predictor
    predictor = TabularPredictor(
        label=config.dataset.target,
        path=str(config.system.model_path),
        problem_type=config.dataset.problem_type,
        eval_metric=config.dataset.metric,
    )

    # Fit with feature pruning
    fit_kwargs = {
        'train_data': train_data,
        'presets': presets,
        'time_limit': time_limit,
        'num_bag_folds': num_bag_folds,
        'num_stack_levels': num_stack_levels,
        'num_cpus': 16,  # Total CPUs for predictor
        'num_gpus': 1 if use_gpu else 0,  # Total GPUs for predictor
        'dynamic_stacking': False,  # Disable DyStack check, keep stacking enabled
        'feature_prune_kwargs': feature_prune_kwargs,
    }

    # With GPU, use sequential folding to avoid splitting 1 GPU across parallel folds
    if use_gpu:
        fit_kwargs['ag_args_ensemble'] = {'fold_fitting_strategy': 'sequential_local'}

    if tuning_data is not None:
        fit_kwargs['tuning_data'] = tuning_data

    if included_models:
        fit_kwargs['included_model_types'] = included_models

    if excluded_models:
        fit_kwargs['excluded_model_types'] = excluded_models

    predictor.fit(**fit_kwargs)

    # Log results
    print(f"\n[{VARIANT_NAME}] Training complete!")
    print(f"[{VARIANT_NAME}] Best model: {predictor.model_best}")

    # Try to get feature info from pruned models
    try:
        leaderboard = predictor.leaderboard(silent=True)
        print(f"\n[{VARIANT_NAME}] Top 5 models:")
        print(leaderboard.head())

        # Check if any models have "_Prune" suffix (indicates pruning was used)
        pruned_models = [m for m in leaderboard['model'].tolist() if '_Prune' in m]
        if pruned_models:
            print(f"\n[{VARIANT_NAME}] Pruned models found: {pruned_models}")
    except Exception as e:
        print(f"[{VARIANT_NAME}] Could not get leaderboard: {e}")

    model_cfg = config.model if hasattr(config, 'model') else {}
    leaderboard_rows = getattr(model_cfg, 'leaderboard_rows', 20)

    return predictor, {"leaderboard_rows": leaderboard_rows}


def predict(
    model: TabularPredictor,
    test_df: pd.DataFrame,
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> pd.DataFrame:
    """Generate predictions using the trained (and pruned) model."""
    # Apply same feature engineering
    test_features = _engineer_features(test_df)

    # Get predictions (probabilities for binary classification)
    if config.dataset.problem_type == "binary":
        predictions = model.predict_proba(test_features, as_multiclass=False)
    else:
        predictions = model.predict(test_features)

    return pd.DataFrame({config.dataset.target: predictions})


# For direct testing
if __name__ == "__main__":
    print("This model uses MINIMAL features + automatic feature pruning.")
    print("Run via experiment_manager.py with template: best-cpu-baseline-pruned")
