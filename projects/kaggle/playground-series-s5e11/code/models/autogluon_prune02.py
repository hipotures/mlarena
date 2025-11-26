"""
AutoGluon with COMPREHENSIVE feature engineering from Kaggle notebook + automatic feature pruning.

Philosophy: Create ALL features from high-performing notebook (s5e11-single-xgboost-advanced-fe.ipynb),
then let AutoGluon's feature pruning automatically remove harmful features.

Features:
- 16 new numeric features (ratios, debt metrics, risk scoring, log transforms, grade parsing)
- ~20 categorical numeric (factorized) versions of all numeric features
- ~12 two-way interactions between categorical features
- ~100+ frequency encodings (count encoding) for all categorical features

Total: ~150-200 features before pruning → ~50-100 features after pruning

Expected result: Match or exceed notebook's 0.927 local CV with AutoGluon ensemble + pruning.
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

VARIANT_NAME = "prune02"


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
    COMPREHENSIVE feature engineering from Kaggle notebook.

    Creates ~150-200 features including:
    - 16 new numeric features
    - Categorical numeric (factorized) versions
    - Two-way interactions
    - Frequency encoding (count encoding)

    Strategy: Create everything, let pruning remove noise.
    """
    enriched = df.copy()

    # =============================================================================
    # STEP 1: Enhanced Financial Features (16 new numeric features)
    # =============================================================================

    # Core affordability
    enriched['income_loan_ratio'] = enriched['annual_income'] / (enriched['loan_amount'] + 1)
    enriched['loan_to_income'] = enriched['loan_amount'] / (enriched['annual_income'] + 1)

    # Debt metrics
    enriched['total_debt'] = enriched['debt_to_income_ratio'] * enriched['annual_income']
    enriched['available_income'] = enriched['annual_income'] * (1 - enriched['debt_to_income_ratio'])
    enriched['debt_burden'] = enriched['debt_to_income_ratio'] * enriched['loan_amount']

    # Payment analysis
    enriched['monthly_payment'] = enriched['loan_amount'] * enriched['interest_rate'] / 1200
    enriched['payment_to_income'] = enriched['monthly_payment'] / (enriched['annual_income'] / 12 + 1)
    enriched['affordability'] = enriched['available_income'] / (enriched['loan_amount'] + 1)

    # Risk scoring
    enriched['default_risk'] = (
        enriched['debt_to_income_ratio'] * 0.40 +
        (850 - enriched['credit_score']) / 850 * 0.35 +
        enriched['interest_rate'] / 100 * 0.25
    )

    # Credit analysis
    enriched['credit_utilization'] = enriched['credit_score'] * (1 - enriched['debt_to_income_ratio'])
    enriched['credit_interest_product'] = enriched['credit_score'] * enriched['interest_rate'] / 100

    # Log transformations
    enriched['annual_income_log'] = np.log1p(enriched['annual_income'])
    enriched['loan_amount_log'] = np.log1p(enriched['loan_amount'])

    # Grade parsing
    enriched['grade_letter'] = enriched['grade_subgrade'].str[0]
    enriched['grade_number'] = enriched['grade_subgrade'].str[1].astype(int)
    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    enriched['grade_rank'] = enriched['grade_letter'].map(grade_map)

    # =============================================================================
    # STEP 2: Define base categorical and numeric features
    # =============================================================================

    CATS_BASE = ['gender', 'marital_status', 'education_level', 'employment_status',
                 'loan_purpose', 'grade_subgrade']
    NUMS_BASE = ['annual_income', 'debt_to_income_ratio', 'credit_score',
                 'loan_amount', 'interest_rate']

    NEW_FEATURES = ['income_loan_ratio', 'loan_to_income', 'total_debt',
                    'available_income', 'debt_burden', 'monthly_payment',
                    'payment_to_income', 'affordability', 'default_risk',
                    'credit_utilization', 'credit_interest_product',
                    'annual_income_log', 'loan_amount_log', 'grade_number', 'grade_rank']

    NUMS = NUMS_BASE + NEW_FEATURES
    CATS = CATS_BASE + ['grade_letter']

    # =============================================================================
    # STEP 3: Categorical Numeric Features (Factorized)
    # =============================================================================

    CATS_NUM = []
    for c in NUMS:
        n = f"{c}_cat"
        CATS_NUM.append(n)
        enriched[n], _ = enriched[c].factorize()
        enriched[n] = enriched[n].astype('int32')

    # =============================================================================
    # STEP 4: Two-Way Interactions
    # =============================================================================

    # Strategic important pairs from notebook
    important_pairs = [
        ('employment_status', 'grade_subgrade'),
        ('employment_status', 'education_level'),
        ('employment_status', 'loan_purpose'),
        ('grade_subgrade', 'loan_purpose'),
        ('grade_subgrade', 'education_level'),
        ('marital_status', 'employment_status'),
    ]

    # Add numeric_cat interactions
    for num_cat in ['credit_score_cat', 'debt_to_income_ratio_cat', 'interest_rate_cat']:
        for cat in ['employment_status', 'grade_subgrade']:
            important_pairs.append((num_cat, cat))

    CATS_INTER = []
    for c1, c2 in important_pairs:
        name = f"{c1}_{c2}"
        if c1 in enriched.columns and c2 in enriched.columns:
            enriched[name] = enriched[c1].astype(str) + '_' + enriched[c2].astype(str)
            CATS_INTER.append(name)

    # =============================================================================
    # STEP 5: Count Encoding (Frequency Encoding)
    # =============================================================================

    ALL_CATS = CATS + CATS_NUM + CATS_INTER

    for c in ALL_CATS:
        # Use transform to avoid merge (more efficient)
        # This counts frequency of each category value
        enriched[f"CE_{c}"] = enriched.groupby(c)[c].transform('count')

    return enriched


def preprocess(df: pd.DataFrame, config: ModelConfig, is_train: bool = True) -> pd.DataFrame:
    """Apply comprehensive feature engineering - pruning happens during fit()."""
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
    print(f"[{VARIANT_NAME}] Training with COMPREHENSIVE features + AUTOMATIC PRUNING")

    # Show column info BEFORE dropping
    print(f"[{VARIANT_NAME}] Columns summary:")
    print(f"  - Target column: '{config.dataset.target}'")
    print(f"  - Ignored columns (dropped from TRAINING only): {config.dataset.ignored_columns}")
    print(f"  - ID column (dropped from TRAINING only): '{config.dataset.id_column}'")
    print(f"  - Total columns before drop: {len(train_df.columns)}")

    # Drop ignored columns (id, etc.) before training
    features = _drop_ignored(train_df, config)
    train_data = features.copy()
    train_data[config.dataset.target] = train_df[config.dataset.target]

    # Show features AFTER dropping
    feature_cols = [col for col in features.columns if col != config.dataset.target]
    print(f"  - Feature count after drop: {len(feature_cols)}")
    print(f"[{VARIANT_NAME}] AutoGluon will automatically prune harmful features from {len(feature_cols)} total features")

    # Prepare validation data if provided
    tuning_data = None
    if val_df is not None:
        val_features = _drop_ignored(val_df, config)
        tuning_data = val_features.copy()
        tuning_data[config.dataset.target] = val_df[config.dataset.target]

    # Get hyperparameters from config
    hyper_cfg = config.hyperparameters if hasattr(config, 'hyperparameters') else {}
    presets = getattr(hyper_cfg, 'presets', 'best_quality')
    time_limit = getattr(hyper_cfg, 'time_limit', 28800)
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
    feature_prune_kwargs = {
        'force_prune': force_prune,
        'feature_prune_time_limit': prune_time_limit,
        'n_train_subsample': max_train_samples,
        'n_fi_subsample': min_fi_samples,  # Correct parameter name
        'prune_threshold': prune_threshold,
        'prune_ratio': prune_ratio,
        'stopping_round': stopping_round,  # Correct parameter name
        'min_improvement': min_improvement,
        'max_fits': max_fits,
        'seed': seed,
        'raise_exception': raise_exception,
    }

    print(f"[{VARIANT_NAME}] Config: presets={presets}, time_limit={time_limit}s")
    print(f"[{VARIANT_NAME}] Bagging: {num_bag_folds} folds, {num_stack_levels} stack levels")
    print(f"[{VARIANT_NAME}] Excluded model types: {excluded_models if excluded_models else 'None (all models enabled)'}")
    print(f"[{VARIANT_NAME}] Feature pruning:")
    print(f"  - feature_prune_time_limit: {prune_time_limit}s (~{int(prune_time_limit/time_limit*100)}% of total)")
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
    # Apply same feature engineering (keeps 'id' - AutoGluon ignores it automatically)
    test_features = _engineer_features(test_df)

    # Get predictions (probabilities for binary classification)
    # AutoGluon automatically ignores columns it didn't train on (like 'id')
    if config.dataset.problem_type == "binary":
        predictions = model.predict_proba(test_features, as_multiclass=False)
    else:
        predictions = model.predict(test_features)

    # Return DataFrame with ID and predictions
    result = pd.DataFrame()
    result[config.dataset.id_column] = test_df[config.dataset.id_column]
    result[config.dataset.target] = predictions
    return result


# For direct testing
if __name__ == "__main__":
    print("This model uses COMPREHENSIVE features from Kaggle notebook + automatic feature pruning.")
    print("Run via experiment_manager.py with template: best-cpu-prune02 or best-gpu-prune02")
