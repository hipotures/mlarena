"""
AutoGluon model with external dataset support.

This model demonstrates how to use the external dataset (orig_df) provided
by the `external_dataset` preprocessing module.

The model merges train + orig datasets before training, allowing AutoGluon
to learn from both Kaggle competition data and external/original datasets.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pandas as pd
from autogluon.tabular import TabularPredictor

from kaggle_tools.config_models import ModelConfig


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
    target_column = config.dataset.target
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
    sample_weight_col = None
    if sample_weight is not None:
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
                sample_weight_col = "__sample_weight__"
                train_data[sample_weight_col] = weights.values
                print(f"[AutoGluon with Orig] Using sample weights: {sample_weight_col}")
            else:
                print(
                    f"[AutoGluon with Orig] Ignoring sample weights: expected {expected_rows:,} rows, got {len(weights):,}"
                )

    # Train model
    predictor = TabularPredictor(
        label=target_column,
        path=str(config.system.model_path),
        eval_metric=config.dataset.metric,
        problem_type=config.dataset.problem_type,
        sample_weight=sample_weight_col,
        weight_evaluation=True if sample_weight_col else False,  # Weighted metrics when sample_weight is used
        verbosity=2,
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
    hyper_dict = config.hyperparameters.model_dump(exclude_none=True)
    known_keys = {
        "presets",
        "time_limit",
        "use_gpu",
        "excluded_models",
        "included_model_types",
        "preset",
        "hyperparameter_tune_kwargs",  # NEW
        "search_space",  # NEW
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

    # Get best model score
    leaderboard = predictor.leaderboard(train_data, silent=True)
    best_score = leaderboard["score_val"].iloc[0] if not leaderboard.empty and "score_val" in leaderboard else None

    # Build training summary
    training_summary = {
        "local_cv": float(best_score) if best_score is not None else None,
        "model_path": str(config.system.model_path),
        "used_orig": orig_df is not None,
        "orig_rows": merged_rows,
        "total_train_rows": len(train_df),
        "preset": preset,
        "time_limit": time_limit,
        "use_gpu": use_gpu,
    }

    return predictor, training_summary
