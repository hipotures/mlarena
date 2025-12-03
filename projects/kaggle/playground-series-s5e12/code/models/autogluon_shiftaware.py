"""
Shift-aware AutoGluon baseline compatible with the generic ML runner.

Adds:
- direct control over bagging/stacking knobs (auto_stack, num_bag_folds, etc.)
- optional sample_weight column (removed from features, passed to fit)
- optional dropping of high-drift features
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

from kaggle_tools.config_models import ModelConfig


def get_default_config() -> Dict[str, Any]:
    return {
        "hyperparameters": {
            "presets": "good",
            "time_limit": 600,
            "use_gpu": False,
            # Fit-level controls (passed directly to TabularPredictor.fit)
            "auto_stack": False,
            "num_bag_folds": 0,
            "num_bag_sets": 1,
            "num_stack_levels": 0,
            "dynamic_stacking": None,
            # Model inclusion/exclusion
            "excluded_models": None,
            "included_model_types": None,
        },
        "model": {
            "leaderboard_rows": 10,
            # Sample-weight column (removed from features, forwarded to fit)
            "sample_weight_column": None,
            # Optional drift-based drops
            "drop_drift_features": False,
            "drift_feature_names": [
                "physical_activity_minutes_per_week",
                "triglycerides",
                "cholesterol_total",
            ],
        },
    }


def _model_cfg(config: ModelConfig) -> Dict[str, Any]:
    cfg = getattr(config, "model", {}) or {}
    if hasattr(cfg, "model_dump"):
        cfg = cfg.model_dump(exclude_none=True)
    return dict(cfg)


def _effective_drift_features(config: ModelConfig) -> List[str]:
    cfg = _model_cfg(config)
    if not cfg.get("drop_drift_features"):
        return []

    names = cfg.get("drift_feature_names") or []
    if isinstance(names, (list, tuple, set)):
        return [str(n) for n in names]
    return [str(names)]


def _sample_weight_column(config: ModelConfig) -> Optional[str]:
    cfg = _model_cfg(config)
    col = cfg.get("sample_weight_column")
    return str(col) if col else None


def _drop_ignored(df: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    drop_cols = set(config.dataset.ignored_columns + [config.dataset.id_column])
    drop_cols.discard(config.dataset.target)
    return df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")


def _prepare_features(df: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    """Remove ignored/id, sample-weight, and optional drift columns."""
    features = _drop_ignored(df, config)

    weight_col = _sample_weight_column(config)
    if weight_col and weight_col in features.columns:
        features = features.drop(columns=[weight_col], errors="ignore")

    drift_cols = _effective_drift_features(config)
    if drift_cols:
        to_drop = [c for c in drift_cols if c in features.columns]
        if to_drop:
            features = features.drop(columns=to_drop, errors="ignore")

    return features


def _fit_kwargs(config: ModelConfig, sample_weight: Optional[pd.Series]) -> Dict[str, Any]:
    hp_cfg = config.hyperparameters
    kwargs: Dict[str, Any] = {
        "presets": hp_cfg.presets,
        "time_limit": hp_cfg.time_limit,
        "num_gpus": 1 if hp_cfg.use_gpu else 0,
    }

    for attr in ["auto_stack", "num_bag_folds", "num_bag_sets", "num_stack_levels", "dynamic_stacking"]:
        if hasattr(hp_cfg, attr):
            value = getattr(hp_cfg, attr)
            if value is not None:
                kwargs[attr] = value

    if hp_cfg.excluded_models:
        kwargs["excluded_model_types"] = hp_cfg.excluded_models
    included_models = getattr(hp_cfg, "included_model_types", None)
    if included_models:
        kwargs["included_model_types"] = included_models

    if sample_weight is not None:
        kwargs["sample_weight"] = sample_weight

    hyper_dict = hp_cfg.model_dump(exclude_none=True)
    known_keys = {
        "presets",
        "time_limit",
        "use_gpu",
        "excluded_models",
        "included_model_types",
        "preset",
        "auto_stack",
        "num_bag_folds",
        "num_bag_sets",
        "num_stack_levels",
        "dynamic_stacking",
    }
    model_hparams = {k: v for k, v in hyper_dict.items() if k not in known_keys}
    if model_hparams:
        kwargs["hyperparameters"] = model_hparams

    return kwargs


def train(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> Tuple[TabularPredictor, Dict[str, Any]]:
    features = _prepare_features(train_df, config)
    train_data = features.copy()
    train_data[config.dataset.target] = train_df[config.dataset.target].values

    tuning_data = None
    if val_df is not None:
        val_features = _prepare_features(val_df, config)
        tuning_data = val_features.copy()
        tuning_data[config.dataset.target] = val_df[config.dataset.target].values

    weight_col = _sample_weight_column(config)
    sample_weight = train_df[weight_col] if weight_col and weight_col in train_df.columns else None

    predictor = TabularPredictor(
        label=config.dataset.target,
        path=str(config.system.model_path),
        problem_type=config.dataset.problem_type,
        eval_metric=config.dataset.metric,
        verbosity=2,
    )

    predictor.fit(
        train_data,
        tuning_data=tuning_data,
        **_fit_kwargs(config, sample_weight),
    )

    leaderboard = predictor.leaderboard(train_data, silent=True)
    local_cv = None
    if not leaderboard.empty and "score_val" in leaderboard:
        scores = leaderboard["score_val"].dropna()
        if not scores.empty:
            local_cv = float(scores.max())

    summary: Dict[str, Any] = {"local_cv": local_cv}

    model_cfg = _model_cfg(config)
    leader_rows = int(model_cfg.get("leaderboard_rows") or 0)
    if leader_rows > 0 and not leaderboard.empty:
        keep_cols = [col for col in ["model", "score_val", "pred_time_val", "fit_time", "stack_level"] if col in leaderboard.columns]
        summary["leaderboard_head"] = (
            leaderboard.head(leader_rows)[keep_cols].replace({np.nan: None}).to_dict(orient="records")
        )

    drift_cols = _effective_drift_features(config)
    if drift_cols:
        summary["dropped_drift_features"] = [c for c in drift_cols if c in train_df.columns]

    if weight_col:
        summary["sample_weight_column"] = weight_col

    return predictor, summary


def predict(
    model: TabularPredictor,
    test_df: pd.DataFrame,
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> pd.DataFrame:
    features = _prepare_features(test_df, config)

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
