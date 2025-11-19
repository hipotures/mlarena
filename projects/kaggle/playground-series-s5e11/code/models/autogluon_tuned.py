"""
AutoGluon model with hyperparameter tuning.

Based on autogluon_baseline.py with added hyperparameter optimization.
Uses Bayesian optimization to find best hyperparameters within time budget.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pandas as pd
from autogluon.tabular import TabularPredictor

from kaggle_tools.config_models import ModelConfig


def get_default_config() -> Dict[str, Any]:
    return {
        "hyperparameters": {
            "presets": "medium_quality",
            "time_limit": 600,  # 10 minutes
            "use_gpu": False,
        },
        "model": {
            "leaderboard_rows": 10,
            "num_trials": 10,  # Number of hyperparameter configurations to try
            "searcher": "auto",  # 'auto', 'random', or 'bayesopt'
        },
    }


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
    features = _drop_ignored(train_df, config)
    train_data = features.copy()
    train_data[config.dataset.target] = train_df[config.dataset.target]
    tuning_data = None
    if val_df is not None:
        val_features = _drop_ignored(val_df, config)
        tuning_data = val_features.copy()
        tuning_data[config.dataset.target] = val_df[config.dataset.target]

    predictor = TabularPredictor(
        label=config.dataset.target,
        path=str(config.system.model_path),
        problem_type=config.dataset.problem_type,
        eval_metric=config.dataset.metric,
        verbosity=2,
    )

    # Hyperparameter tuning configuration
    num_trials = config.model.get("num_trials", 10)
    searcher = config.model.get("searcher", "auto")

    hyperparameter_tune_kwargs = {
        "num_trials": num_trials,
        "searcher": searcher,
        "scheduler": "local",
    }

    # Force GPU usage at model level to avoid splitting across folds
    ag_args = None
    if config.hyperparameters.use_gpu:
        ag_args = {"num_gpus": 1}  # Each model gets 1 GPU

    fit_kwargs = {
        "presets": config.hyperparameters.presets,
        "time_limit": config.hyperparameters.time_limit,
        "num_cpus": 16,  # Total CPUs for predictor (max 16 of 32 to avoid freezing)
        "hyperparameter_tune_kwargs": hyperparameter_tune_kwargs,
        "ag_args_fit": ag_args,  # Apply GPU to each model, not split across folds
    }
    if config.hyperparameters.excluded_models:
        fit_kwargs["excluded_model_types"] = config.hyperparameters.excluded_models

    print(f"[AutoGluon Tuned] Training configuration:")
    print(f"  Preset: {config.hyperparameters.presets}")
    print(f"  Time limit: {config.hyperparameters.time_limit}s ({config.hyperparameters.time_limit/3600:.1f}h)")
    print(f"  Total CPUs: {fit_kwargs['num_cpus']}")
    print(f"  GPU per model: {1 if ag_args else 0}")
    print(f"  HPO trials: {num_trials}")
    print(f"  HPO searcher: {searcher}")

    predictor.fit(
        train_data,
        tuning_data=tuning_data,
        **fit_kwargs,
    )

    leaderboard = predictor.leaderboard(train_data, silent=True)
    local_cv = float(leaderboard.iloc[0]["score_val"]) if not leaderboard.empty else None

    # Get best model info
    best_model = leaderboard.iloc[0]["model"] if not leaderboard.empty else None

    summary = {
        "local_cv": local_cv,
        "best_model": best_model,
        "num_models_trained": len(leaderboard),
    }

    print(f"[AutoGluon Tuned] Training completed:")
    print(f"  - Best CV score: {local_cv}")
    print(f"  - Best model: {best_model}")
    print(f"  - Total models trained: {len(leaderboard)}")

    return predictor, summary


def predict(
    model: TabularPredictor,
    test_df: pd.DataFrame,
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> pd.DataFrame:
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
