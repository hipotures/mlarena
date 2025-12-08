"""
AutoGluon boost models with native categorical feature handling.

Designed for XGBoost, LightGBM, and CatBoost to use their native categorical
mechanisms instead of AutoGluon's default encoding (one-hot, label, target).

Prerequisites:
- Run preprocessing with 'categorical_boost' template to convert columns to dtype='category'
- Preprocessing stores categorical_columns list in state.json

Usage:
    uv run python scripts/mla.py model \
        --project <project-name> \
        --model-template baseline_boost \
        --skip-submit

Configuration:
- Reads categorical_columns from preprocessing state.json
- Configures AutoGluon CategoryFeatureGenerator with minimal processing
- Enables XGBoost native categorical handling (enable_categorical=True)
- LightGBM and CatBoost use categories natively by default

References:
- XGBoost categorical: https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html
- AutoGluon features: https://auto.gluon.ai/stable/api/autogluon.features.html
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from autogluon.tabular import TabularPredictor
from rich.console import Console

from kaggle_tools.config_models import ModelConfig

console = Console()


def get_default_config() -> Dict[str, Any]:
    return {
        "hyperparameters": {
            "presets": "medium",
            "time_limit": 300,
            "use_gpu": False,
        },
        "model": {
            "leaderboard_rows": 10,
        },
    }


def _read_categorical_metadata(config: ModelConfig, preprocess_template: str | None) -> List[str]:
    """
    Read categorical_columns from preprocessing state.json.

    Args:
        config: Model configuration
        preprocess_template: Name of preprocessing template used

    Returns:
        List of categorical column names
    """
    if not preprocess_template:
        return []

    project_root = config.system.project_root
    state_path = project_root / "experiments" / f"pre-{preprocess_template}" / "state.json"

    if not state_path.exists():
        console.print(f"[yellow]Warning: Preprocessing state not found: {state_path}[/yellow]")
        return []

    try:
        with open(state_path) as f:
            state = json.load(f)

        preprocess_payload = state.get("modules", {}).get("preprocess", {}).get("payload", {})
        custom_state = preprocess_payload.get("custom_module_state", {})
        categorical_columns = custom_state.get("categorical_columns", [])

        if categorical_columns:
            console.print(f"[green]✓[/green] Loaded {len(categorical_columns)} categorical columns from preprocessing")
            console.print(f"  Columns: {', '.join(categorical_columns)}")

        return categorical_columns

    except Exception as e:
        console.print(f"[yellow]Warning: Failed to read categorical metadata: {e}[/yellow]")
        return []


def _verify_and_convert_categorical_dtypes(
    df: pd.DataFrame,
    categorical_cols: List[str],
    df_name: str = "data",
) -> pd.DataFrame:
    """
    Verify and convert categorical columns to dtype='category'.

    Args:
        df: DataFrame to check/convert
        categorical_cols: List of expected categorical columns
        df_name: Name for logging

    Returns:
        DataFrame with categorical dtypes
    """
    df = df.copy()
    converted = []

    for col in categorical_cols:
        if col not in df.columns:
            continue

        if df[col].dtype.name != 'category':
            try:
                df[col] = df[col].astype('category')
                converted.append(col)
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to convert {col} to category in {df_name}: {e}[/yellow]")

    if converted:
        console.print(f"[yellow]Note: Converted {len(converted)} columns to category dtype in {df_name}[/yellow]")

    return df


def _configure_feature_generator(categorical_cols: List[str]):
    """
    Configure AutoGluon feature generator for native categorical handling.

    Strategy:
    - Use CategoryFeatureGenerator with minimal processing
    - Preserve original category order and values
    - Don't trim or merge categories
    - Explicit missing value marker

    Args:
        categorical_cols: List of categorical column names

    Returns:
        Feature generator configuration or 'auto'
    """
    if not categorical_cols:
        # No categorical metadata, use AutoGluon defaults
        return 'auto'

    from autogluon.features.generators import (
        PipelineFeatureGenerator,
        CategoryFeatureGenerator,
        IdentityFeatureGenerator,
    )
    from autogluon.common.features.types import R_INT, R_FLOAT

    console.print(f"[cyan]Configuring feature generator for native categorical handling[/cyan]")

    feature_generator = PipelineFeatureGenerator(
        generators=[[
            CategoryFeatureGenerator(
                cat_order='original',       # Preserve original category order
                maximum_num_cat=None,       # Don't trim categories
                minimum_cat_count=1,        # Don't merge rare categories
                fillna='__NaN__',           # Explicit missing marker
            ),
            IdentityFeatureGenerator(
                infer_features_in_args=dict(valid_raw_types=[R_INT, R_FLOAT])
            ),
        ]]
    )

    return feature_generator


def _build_hyperparameters(config: ModelConfig) -> Dict[str, Any]:
    """
    Build model-specific hyperparameters for boost algorithms.

    Key configurations:
    - XGB: enable_categorical=True (CRITICAL for native categorical handling)
    - GBM: Uses categories natively by default
    - CAT: Uses categories natively by default

    Args:
        config: Model configuration

    Returns:
        Hyperparameters dictionary for AutoGluon
    """
    hyperparameters = {
        'GBM': {},  # LightGBM uses categories natively
        'CAT': {},  # CatBoost uses categories natively
        'XGB': {
            'enable_categorical': True,  # CRITICAL: Enable XGBoost native categorical
            'tree_method': 'hist',       # Required for categorical support
            'max_cat_to_onehot': 1,      # Disable one-hot fallback (0 or 1)
        },
    }

    # Merge with user-provided hyperparameters from config
    hyper_dict = config.hyperparameters.model_dump(exclude_none=True)
    known_keys = {
        "presets",
        "time_limit",
        "use_gpu",
        "excluded_models",
        "included_model_types",
        "preset",
    }

    user_hyperparams = {k: v for k, v in hyper_dict.items() if k not in known_keys}

    # Merge user hyperparams (user takes precedence)
    for model, params in user_hyperparams.items():
        if model in hyperparameters:
            hyperparameters[model].update(params)
        else:
            hyperparameters[model] = params

    return hyperparameters


def _drop_ignored(df: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    drop_cols = set(config.dataset.ignored_columns + [config.dataset.id_column])
    drop_cols.discard(config.dataset.target)
    return df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")


def train(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> Tuple[TabularPredictor, Dict[str, Any]]:
    """
    Train AutoGluon boost models with native categorical handling.

    Workflow:
    1. Read categorical_columns from preprocessing state (if available)
    2. Verify/convert columns to dtype='category'
    3. Configure feature generator for minimal processing
    4. Set model-specific categorical parameters (XGB, GBM, CAT)
    5. Train predictor

    Args:
        train_df: Training DataFrame
        val_df: Optional validation DataFrame
        config: Model configuration
        artifacts: Optional artifacts (unused)

    Returns:
        Tuple of (predictor, summary_dict)
    """
    console.print("\n[bold cyan]AutoGluon Boost Categorical Model[/bold cyan]")

    # Determine preprocessing template from invocation
    # Note: preprocess_template is stored in experiment state, not in config
    # We need to read it from the model module's context
    preprocess_template = None
    if hasattr(config, 'preprocess_template'):
        preprocess_template = config.preprocess_template
    else:
        # Try to infer from template config (passed via hyperparameters)
        hyper_dict = config.hyperparameters.model_dump(exclude_none=True)
        preprocess_template = hyper_dict.get('preprocess_template')

    # Read categorical metadata
    categorical_cols = _read_categorical_metadata(config, preprocess_template)

    # Verify and convert categorical dtypes
    if categorical_cols:
        train_df = _verify_and_convert_categorical_dtypes(train_df, categorical_cols, "train")
        if val_df is not None:
            val_df = _verify_and_convert_categorical_dtypes(val_df, categorical_cols, "validation")

    # Prepare training data
    features = _drop_ignored(train_df, config)
    train_data = features.copy()
    train_data[config.dataset.target] = train_df[config.dataset.target]

    tuning_data = None
    if val_df is not None:
        val_features = _drop_ignored(val_df, config)
        tuning_data = val_features.copy()
        tuning_data[config.dataset.target] = val_df[config.dataset.target]

    # Configure feature generator
    feature_generator = _configure_feature_generator(categorical_cols)

    # Build hyperparameters
    hyperparameters = _build_hyperparameters(config)

    # Create predictor
    predictor = TabularPredictor(
        label=config.dataset.target,
        path=str(config.system.model_path),
        problem_type=config.dataset.problem_type,
        eval_metric=config.dataset.metric,
        verbosity=2,
    )

    # Prepare fit kwargs
    fit_kwargs = {
        "presets": config.hyperparameters.presets,
        "time_limit": config.hyperparameters.time_limit,
        "num_gpus": 1 if config.hyperparameters.use_gpu else 0,
        "feature_generator": feature_generator,
        "hyperparameters": hyperparameters,
    }

    # Add excluded/included models
    if config.hyperparameters.excluded_models:
        fit_kwargs["excluded_model_types"] = config.hyperparameters.excluded_models

    included_models = getattr(config.hyperparameters, "included_model_types", None)
    if included_models:
        fit_kwargs["included_model_types"] = included_models

    console.print(f"[cyan]Training with presets={fit_kwargs['presets']}, time_limit={fit_kwargs['time_limit']}s[/cyan]")

    # Train
    predictor.fit(
        train_data,
        tuning_data=tuning_data,
        **fit_kwargs,
    )

    # Extract summary
    leaderboard = predictor.leaderboard(train_data, silent=True)
    local_cv = None
    best_model = None

    if not leaderboard.empty and "score_val" in leaderboard:
        scores = leaderboard["score_val"].dropna()
        if not scores.empty:
            local_cv = float(scores.max())
            best_model = leaderboard.loc[scores.idxmax(), "model"]

    summary = {
        "local_cv": local_cv,
        "best_model": best_model,
        "categorical_columns": categorical_cols,
        "num_categorical": len(categorical_cols),
    }

    console.print(f"\n[green]✓[/green] Training completed")
    if local_cv is not None:
        console.print(f"  Local CV: {local_cv:.6f}")
    if best_model:
        console.print(f"  Best model: {best_model}")

    return predictor, summary


def predict(
    model: TabularPredictor,
    test_df: pd.DataFrame,
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Generate predictions using trained predictor.

    Args:
        model: Trained TabularPredictor
        test_df: Test DataFrame
        config: Model configuration
        artifacts: Optional artifacts (unused)

    Returns:
        Submission DataFrame
    """
    features = _drop_ignored(test_df, config)
    submission = pd.DataFrame()
    submission[config.dataset.id_column] = test_df[config.dataset.id_column]

    if config.dataset.submission_probas:
        preds = model.predict_proba(features, as_multiclass=False)
        if isinstance(preds, pd.DataFrame):
            submission[config.dataset.target] = preds.iloc[:, 1]
        else:
            submission[config.dataset.target] = preds
    else:
        submission[config.dataset.target] = model.predict(features)

    return submission
