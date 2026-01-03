"""
AutoGluon model with external dataset support.

This model demonstrates how to use the external dataset (orig_df) provided
by the `external_dataset` preprocessing module.

The model merges train + orig datasets before training, allowing AutoGluon
to learn from both Kaggle competition data and external/original datasets.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

from kaggle_tools.config_models import ModelConfig

# Suppress C++ compiler warnings and Python warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


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
            - config.dataset.sample_weight_strategy: Optional[str]
                'auto_weight' - AutoGluon auto-balancing
                'balance_weight' - Equal class weights
                custom column name - Use specific column from train_df
                None - Use weights from artifacts (legacy behavior)
            - config.dataset.weight_evaluation: Optional[bool]
                True - Use sample weights for evaluation metrics
                False - Ignore weights for evaluation
                None - Auto-detect (True for explicit weights, False for auto/balance)
        artifacts: Optional dict containing:
            - orig_df: External dataset (from external_dataset preprocessing module)
            - sample_weight: Sample weights (from imbalance_handler or av_weights)

    Returns:
        Tuple of (predictor, training_summary)
    """
    target_column = config.dataset.target
    seed = getattr(config.system, "random_seed", 42)
    np.random.seed(seed)
    preset = config.hyperparameters.presets or "medium"
    time_limit = config.hyperparameters.time_limit or 300
    use_gpu = bool(config.hyperparameters.use_gpu)

    # Check for orig_df in artifacts
    orig_df = None
    sample_weight = None
    merged_rows = 0

    if artifacts:
        orig_df = artifacts.get("orig_df")
        sample_weight = artifacts.get("sample_weight")

    base_train_rows = len(train_df)

    # Merge train + orig if available
    merge_orig = True
    if getattr(config, "model", None) and isinstance(config.model, dict):
        merge_orig = config.model.get("merge_orig", True)

    if not merge_orig:
        print("[AutoGluon with Orig] merge_orig disabled; skipping external dataset merge")
        orig_df = None

    if orig_df is not None:
        if target_column not in orig_df.columns:
            print(
                f"[AutoGluon with Orig] External dataset missing target '{target_column}', skipping merge"
            )
            orig_df = None
        else:
            # Drop rows with missing target (can't train on unlabeled rows)
            orig_before = len(orig_df)
            orig_df = orig_df.dropna(subset=[target_column])
            dropped = orig_before - len(orig_df)
            if dropped:
                print(f"[AutoGluon with Orig] Dropped {dropped:,} external rows with missing target")

    if orig_df is not None:
        print(f"[AutoGluon with Orig] Merging external dataset:")
        print(f"  Kaggle train rows: {base_train_rows:,}")
        print(f"  External rows:     {len(orig_df):,}")

        # Concatenate train + orig
        train_df = pd.concat([train_df, orig_df], ignore_index=True)
        merged_rows = int(len(orig_df))

        print(f"  Merged total:      {len(train_df):,}")
    else:
        print("[AutoGluon with Orig] No external dataset found, using Kaggle data only")

    # Drop ignored columns (keep target)
    drop_cols = set((config.dataset.ignored_columns or []) + [config.dataset.id_column])
    drop_cols.discard(target_column)
    train_data = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns], errors="ignore")

    if target_column not in train_data.columns:
        raise ValueError(f"Target column '{target_column}' not found in training data")

    # Handle sample weights (if provided) and (optionally) extend for external rows.
    # Preprocessing weights are typically a single-column DataFrame (column name may vary).
    sample_weight_param = None  # Value passed to TabularPredictor(sample_weight=...)

    # Check for config override (sample_weight_strategy: 'auto_weight', 'balance_weight', or custom column)
    sample_weight_strategy = config.dataset.sample_weight_strategy
    print(f"[DEBUG] sample_weight_strategy from config: {sample_weight_strategy}")
    print(f"[DEBUG] sample_weight from artifacts: {type(sample_weight).__name__ if sample_weight is not None else 'None'}")
    if sample_weight is not None and hasattr(sample_weight, 'shape'):
        print(f"[DEBUG] sample_weight shape: {sample_weight.shape}")

    if sample_weight_strategy:
        # User explicitly configured sample_weight_strategy in template
        if sample_weight_strategy in ["auto_weight", "balance_weight"]:
            # Special AutoGluon strategies - pass directly to TabularPredictor
            sample_weight_param = sample_weight_strategy
            print(f"[AutoGluon Sample Weights] Using strategy: {sample_weight_strategy}")
        else:
            # Custom column name - assume it's already in train_data
            if sample_weight_strategy in train_data.columns:
                sample_weight_param = sample_weight_strategy
                print(f"[AutoGluon Sample Weights] Using custom column: {sample_weight_strategy}")
            else:
                print(f"[AutoGluon Sample Weights] Warning: column '{sample_weight_strategy}' not found, ignoring")
    elif sample_weight is not None:
        # Legacy behavior: weights from artifacts (preprocessing modules like adversarial_validation)
        weights_series: Optional[pd.Series] = None
        if isinstance(sample_weight, pd.Series):
            weights_series = sample_weight
        elif isinstance(sample_weight, pd.DataFrame):
            if not sample_weight.empty:
                # Prefer common conventions, otherwise take the first column (MLArena requirement).
                if "__sample_weight__" in sample_weight.columns:
                    weights_series = sample_weight["__sample_weight__"]
                elif "sample_weight" in sample_weight.columns:
                    weights_series = sample_weight["sample_weight"]
                else:
                    weights_series = sample_weight.iloc[:, 0]
        else:
            try:
                weights_series = pd.Series(sample_weight)
            except Exception:
                weights_series = None

        if weights_series is not None:
            weights = pd.to_numeric(weights_series, errors="coerce").reset_index(drop=True).astype(float)
            if weights.isna().any():
                if weights.notna().any():
                    weights = weights.fillna(float(weights.mean()))
                else:
                    weights = weights.fillna(1.0)
            expected_rows = len(train_data)

            # If model merges train+orig, but weights are only for Kaggle train rows,
            # extend with a neutral fill value (mean keeps overall scale consistent).
            if merged_rows and len(weights) == base_train_rows:
                fill_value = float(weights.mean()) if weights.notna().any() else 1.0
                weights = pd.concat(
                    [weights, pd.Series([fill_value] * merged_rows)],
                    ignore_index=True,
                )

            if len(weights) == expected_rows:
                sample_weight_param = "__sample_weight__"
                train_data[sample_weight_param] = weights.values
                print(f"[AutoGluon Sample Weights] Using sample weights from artifacts: {sample_weight_param}")
            else:
                print(
                    f"[AutoGluon Sample Weights] Ignoring sample weights: expected {expected_rows:,} rows, got {len(weights):,}"
                )

    # Determine weight_evaluation setting
    # Priority: config.dataset.weight_evaluation > auto-detect based on sample_weight_param
    weight_evaluation_param = config.dataset.weight_evaluation
    if weight_evaluation_param is None:
        # Auto-detect: True if using explicit weights (not 'auto_weight'/'balance_weight')
        if sample_weight_param and sample_weight_param not in ["auto_weight", "balance_weight"]:
            weight_evaluation_param = True
        else:
            weight_evaluation_param = False

    if sample_weight_param:
        print(f"[AutoGluon Sample Weights] weight_evaluation={weight_evaluation_param}")
        if weight_evaluation_param and sample_weight_param in ["auto_weight", "balance_weight"]:
            print(f"[AutoGluon Sample Weights] WARNING: weight_evaluation=True with {sample_weight_param} is not recommended by AutoGluon docs")

    # Train model
    # Get verbosity from config or default to 2 (standard logging)
    verbosity = getattr(config.hyperparameters, 'verbosity', 2)

    predictor = TabularPredictor(
        label=target_column,
        path=str(config.system.model_path),
        eval_metric=config.dataset.metric,
        problem_type=config.dataset.problem_type,
        sample_weight=sample_weight_param,
        weight_evaluation=weight_evaluation_param,
        verbosity=verbosity,
    )

    fit_kwargs = {
        "presets": preset,
        "time_limit": time_limit,
        "num_gpus": 1 if use_gpu else 0,
    }
    if config.hyperparameters.excluded_models:
        fit_kwargs["excluded_model_types"] = config.hyperparameters.excluded_models
    included_models = getattr(config.hyperparameters, "included_model_types", None)
    if included_models:
        fit_kwargs["included_model_types"] = included_models
    fit_args = getattr(config.hyperparameters, "fit_args", None)
    if fit_args is not None and not isinstance(fit_args, dict):
        raise ValueError("fit_args must be a dict")

    hyper_dict = config.hyperparameters.model_dump(exclude_none=True)
    if use_gpu and not hyper_dict.get("ag_args_fit"):
        # Reserve GPU per model to avoid Ray launching multiple GPU tasks on one card.
        fit_kwargs["ag_args_fit"] = {"num_gpus": 1}
    if fit_args:
        fit_kwargs.update(fit_args)

    # NEW: Add HPO support
    hpo_tune_kwargs = getattr(config.hyperparameters, "hyperparameter_tune_kwargs", None)
    search_space_dict = getattr(config.hyperparameters, "search_space", None)

    if hpo_tune_kwargs:
        # Enable HPO
        fit_kwargs["hyperparameter_tune_kwargs"] = hpo_tune_kwargs
        print(f"[AutoGluon HPO] Enabled with {hpo_tune_kwargs['num_trials']} trials")
        print(f"[AutoGluon HPO] Scheduler: {hpo_tune_kwargs['scheduler']}, Searcher: {hpo_tune_kwargs['searcher']}")

    if search_space_dict:
        # Convert YAML search space to autogluon.common.space objects
        from mlarena.utils.hpo_space import parse_search_space

        converted_space = parse_search_space(search_space_dict)

        # Only apply search spaces for included models (if specified)
        if included_models:
            filtered_space = {
                model: params for model, params in converted_space.items() if model in included_models
            }
            converted_space = filtered_space

        # Set hyperparameters with search spaces
        fit_kwargs["hyperparameters"] = converted_space

        print(f"[AutoGluon HPO] Search spaces defined for: {list(converted_space.keys())}")
        for model_type in converted_space:
            print(f"[AutoGluon HPO]   {model_type}: {len(converted_space[model_type])} parameters")

    # Forward any model-specific hyperparameters (e.g., NN_TORCH, FASTAI) to AutoGluon.
    known_keys = {
        "presets",
        "time_limit",
        "use_gpu",
        "excluded_models",
        "included_model_types",
        "preset",
        "hyperparameter_tune_kwargs",  # NEW
        "search_space",  # NEW
        "fit_args",
    }
    model_hparams = {k: v for k, v in hyper_dict.items() if k not in known_keys}

    # Merge model-specific hyperparameters with search spaces
    if model_hparams:
        if "hyperparameters" in fit_kwargs:
            # Deep merge: model_hparams can add non-HPO models or static params
            existing = fit_kwargs["hyperparameters"]
            for model_type, params in model_hparams.items():
                if model_type not in existing:
                    existing[model_type] = params
                elif isinstance(existing[model_type], dict) and isinstance(params, dict):
                    existing[model_type].update(params)
        else:
            fit_kwargs["hyperparameters"] = model_hparams

    predictor.fit(train_data, **fit_kwargs)

    # Get best model info
    best_model_name = predictor.model_best
    best_score = predictor.model_info(best_model_name).get("val_score")

    # Get leaderboard for saving to file (called once here, not in model.py)
    leaderboard = predictor.leaderboard(silent=True)

    # Build training summary
    training_summary = {
        "local_cv_score": float(best_score) if best_score is not None else None,
        "best_model": best_model_name,
        "leaderboard": leaderboard,
        "model_path": str(config.system.model_path),
        "used_orig": orig_df is not None,
        "orig_rows": merged_rows,
        "total_train_rows": len(train_df),
        "preset": preset,
        "time_limit": time_limit,
        "use_gpu": use_gpu,
    }

    return predictor, training_summary
