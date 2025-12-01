"""
AutoGluon baseline model compatible with the generic ML runner.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pandas as pd
from autogluon.tabular import TabularPredictor

from kaggle_tools.config_models import ModelConfig


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

    fit_kwargs = {
        "presets": config.hyperparameters.presets,
        "time_limit": config.hyperparameters.time_limit,
        "num_gpus": 1 if config.hyperparameters.use_gpu else 0,
    }
    if config.hyperparameters.excluded_models:
        fit_kwargs["excluded_model_types"] = config.hyperparameters.excluded_models
    included_models = getattr(config.hyperparameters, "included_model_types", None)
    if included_models:
        fit_kwargs["included_model_types"] = included_models
    hyper_dict = config.hyperparameters.model_dump(exclude_none=True)
    known_keys = {
        "presets",
        "time_limit",
        "use_gpu",
        "excluded_models",
        "included_model_types",
        "preset",
    }
    model_hparams = {k: v for k, v in hyper_dict.items() if k not in known_keys}
    if model_hparams:
        fit_kwargs["hyperparameters"] = model_hparams

    predictor.fit(
        train_data,
        tuning_data=tuning_data,
        **fit_kwargs,
    )

    leaderboard = predictor.leaderboard(train_data, silent=True)
    local_cv = None
    if not leaderboard.empty and "score_val" in leaderboard:
        scores = leaderboard["score_val"].dropna()
        if not scores.empty:
            local_cv = float(scores.max())
    summary = {"local_cv": local_cv}
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
