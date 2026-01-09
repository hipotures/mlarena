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
from rich.console import Console

from kaggle_tools.config_models import ModelConfig

console = Console()

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
    Train AutoGluon model with optional external dataset merging and tuning data.

    If artifacts contains 'orig_df', it will be merged with train_df before training.
    If artifacts contains 'tuning_df', it will be passed to AutoGluon for hyperparameter tuning.

    Args:
        train_df: Kaggle training data (preprocessed)
        val_df: Validation data (DEPRECATED, use tuning_df in artifacts instead)
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
            - tuning_df: Validation data for hyperparameter tuning (from train_fraction module)
            - eval_df: Offline evaluation data (for leaderboard only, NOT passed to fit)

    Returns:
        Tuple of (predictor, training_summary)
    """
    target_column = config.dataset.target
    seed = getattr(config.system, "random_seed", 42)
    np.random.seed(seed)
    preset = config.hyperparameters.presets or "medium"
    time_limit = config.hyperparameters.time_limit or 300
    use_gpu = bool(config.hyperparameters.use_gpu)

    # Check for orig_df, sample_weight, tuning_df, and eval_df in artifacts
    orig_df = None
    sample_weight = None
    tuning_df = None
    eval_df = None
    merged_rows = 0

    if artifacts:
        orig_df = artifacts.get("orig_df")
        sample_weight = artifacts.get("sample_weight")
        tuning_df = artifacts.get("tuning_df")
        eval_df = artifacts.get("eval_df")

    base_train_rows = len(train_df)

    # Merge train + orig if available
    merge_orig = True
    if getattr(config, "model", None) and isinstance(config.model, dict):
        merge_orig = config.model.get("merge_orig", True)

    if not merge_orig:
        orig_df = None

    if orig_df is not None:
        if target_column not in orig_df.columns:
            orig_df = None
        else:
            # Drop rows with missing target (can't train on unlabeled rows)
            orig_df = orig_df.dropna(subset=[target_column])

    if orig_df is not None:
        # Concatenate train + orig
        train_df = pd.concat([train_df, orig_df], ignore_index=True)
        merged_rows = int(len(orig_df))

    # Drop ignored columns (keep target)
    drop_cols = set((config.dataset.ignored_columns or []) + [config.dataset.id_column])
    drop_cols.discard(target_column)
    train_data = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns], errors="ignore")

    if target_column not in train_data.columns:
        raise ValueError(f"Target column '{target_column}' not found in training data")

    # Prepare tuning_data (same preprocessing as train_data)
    tuning_data = None
    if tuning_df is not None:
        tuning_data = tuning_df.drop(columns=[c for c in drop_cols if c in tuning_df.columns], errors="ignore")

        if target_column not in tuning_data.columns:
            console.print(f"[yellow]⚠[/yellow] [bold]Tuning:[/bold] target '{target_column}' not in tuning_df, [red]ignoring[/red]")
            tuning_data = None
        else:
            console.print(f"[green]✓[/green] [bold]Tuning:[/bold] Using tuning_data with [cyan]{len(tuning_data):,}[/cyan] rows")

    # Prepare eval_data (for offline leaderboard only, NOT passed to fit)
    eval_data = None
    if eval_df is not None:
        eval_data = eval_df.drop(columns=[c for c in drop_cols if c in eval_df.columns], errors="ignore")

        if target_column not in eval_data.columns:
            console.print(f"[yellow]⚠[/yellow] [bold]Eval:[/bold] target '{target_column}' not in eval_df, [red]ignoring[/red]")
            eval_data = None
        else:
            console.print(f"[green]✓[/green] [bold]Eval:[/bold] Using eval_data with [cyan]{len(eval_data):,}[/cyan] rows for leaderboard")

    # Handle sample weights (if provided) and (optionally) extend for external rows.
    # Preprocessing weights are typically a single-column DataFrame (column name may vary).
    sample_weight_param = None  # Value passed to TabularPredictor(sample_weight=...)

    # Check for config override (sample_weight_strategy: 'auto_weight', 'balance_weight', or custom column)
    sample_weight_strategy = config.dataset.sample_weight_strategy

    if sample_weight_strategy:
        # User explicitly configured sample_weight_strategy in template
        if sample_weight_strategy in ["auto_weight", "balance_weight"]:
            # Special AutoGluon strategies - pass directly to TabularPredictor
            sample_weight_param = sample_weight_strategy
            console.print(f"[cyan]i[/cyan] [bold]Sample Weights:[/bold] Using strategy [magenta]{sample_weight_strategy}[/magenta]")
        else:
            # Custom column name - assume it's already in train_data
            if sample_weight_strategy in train_data.columns:
                sample_weight_param = sample_weight_strategy
                console.print(f"[green]✓[/green] [bold]Sample Weights:[/bold] Using custom column [yellow]{sample_weight_strategy}[/yellow]")
            else:
                console.print(f"[yellow]⚠[/yellow] [bold]Sample Weights:[/bold] column '{sample_weight_strategy}' [red]not found[/red], ignoring")
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
                console.print(f"[green]✓[/green] [bold]Sample Weights:[/bold] Using weights from artifacts: [yellow]{sample_weight_param}[/yellow]")

                # Apply neutral weight to tuning data
                if tuning_data is not None:
                    tuning_weight = float(weights.mean()) if weights.notna().any() else 1.0
                    tuning_data[sample_weight_param] = tuning_weight
                    console.print(f"[cyan]i[/cyan] [bold]Sample Weights:[/bold] Applied neutral weight ([green]{tuning_weight:.4f}[/green]) to tuning data")
                # Apply neutral weight to eval data (leaderboard)
                if eval_data is not None:
                    eval_weight = float(weights.mean()) if weights.notna().any() else 1.0
                    eval_data[sample_weight_param] = eval_weight
                    console.print(f"[cyan]i[/cyan] [bold]Sample Weights:[/bold] Applied neutral weight ([green]{eval_weight:.4f}[/green]) to eval data")
            else:
                console.print(
                    f"[yellow]⚠[/yellow] [bold]Sample Weights:[/bold] [red]Ignoring weights[/red]: expected {expected_rows:,} rows, got {len(weights):,}"
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
    elif weight_evaluation_param and not sample_weight_param:
        console.print(
            "[yellow]⚠[/yellow] [bold]Sample Weights:[/bold] "
            "weight_evaluation=True but no weights found; disabling."
        )
        weight_evaluation_param = False

    if sample_weight_param:
        console.print(f"[cyan]i[/cyan] [bold]Sample Weights:[/bold] weight_evaluation=[magenta]{weight_evaluation_param}[/magenta]")
        if weight_evaluation_param and sample_weight_param in ["auto_weight", "balance_weight"]:
            console.print(f"[yellow]⚠[/yellow] [bold]Sample Weights:[/bold] [red]WARNING[/red]: weight_evaluation=True with {sample_weight_param} is not recommended")

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

    if tuning_data is not None:
        fit_kwargs["tuning_data"] = tuning_data

        # Set use_bag_holdout if bagging enabled
        num_bag_folds = getattr(config.hyperparameters, "num_bag_folds", 0)
        num_bag_sets = getattr(config.hyperparameters, "num_bag_sets", None)
        bagging_enabled = (num_bag_folds and num_bag_folds > 0) or (num_bag_sets and num_bag_sets > 0)

        if bagging_enabled:
            use_bag_holdout = True  # Default
            # Allow template override via config.model dict
            if hasattr(config, "model") and isinstance(config.model, dict):
                use_bag_holdout = config.model.get("use_bag_holdout", True)

            fit_kwargs["use_bag_holdout"] = use_bag_holdout
            console.print(f"[cyan]i[/cyan] [bold]Tuning:[/bold] Bagging enabled with use_bag_holdout=[magenta]{use_bag_holdout}[/magenta]")

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
        console.print(f"[cyan]i[/cyan] [bold]HPO:[/bold] Enabled with [yellow]{hpo_tune_kwargs['num_trials']}[/yellow] trials")
        console.print(f"[cyan]i[/cyan] [bold]HPO:[/bold] Scheduler: [magenta]{hpo_tune_kwargs['scheduler']}[/magenta], Searcher: [magenta]{hpo_tune_kwargs['searcher']}[/magenta]")

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

        console.print(f"[cyan]i[/cyan] [bold]HPO:[/bold] Search spaces defined for: [yellow]{list(converted_space.keys())}[/yellow]")
        for model_type in converted_space:
            console.print(f"    • [cyan]{model_type:10s}[/cyan] | [yellow]{len(converted_space[model_type])}[/yellow] parameters")

    # Forward any model-specific hyperparameters (e.g., NN_TORCH, FASTAI) to AutoGluon.
    known_keys = {
        "presets",
        "time_limit",
        "use_gpu",
        "excluded_models",
        "included_model_types",
        "preset",
        "verbosity",
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

    # Get leaderboard (use eval_data if provided for offline evaluation)
    eval_score = None
    if eval_data is not None:
        leaderboard = predictor.leaderboard(data=eval_data, silent=True)
        # Extract best model score on eval data
        best_model_row = leaderboard[leaderboard['model'] == best_model_name]
        eval_score = best_model_row['score_val'].values[0] if not best_model_row.empty else None
        console.print(f"[green]✓[/green] [bold]Eval:[/bold] Leaderboard score on eval_data: [yellow]{eval_score:.6f}[/yellow]")
    else:
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
        "tuning_rows": len(tuning_df) if tuning_df is not None else 0,
        "eval_rows": len(eval_df) if eval_df is not None else 0,
        "eval_score": eval_score if eval_score is not None else None,
        "preset": preset,
        "time_limit": time_limit,
        "use_gpu": use_gpu,
    }

    return predictor, training_summary
