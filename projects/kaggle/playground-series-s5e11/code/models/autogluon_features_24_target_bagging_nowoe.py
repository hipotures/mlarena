"""
AutoGluon variant #24: fe21 features + CV target encoding with bagging/stacking,
but without WOE to reduce categorical overfitting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from autogluon.tabular import TabularPredictor

from kaggle_tools.config_models import ModelConfig

MODEL_DIR = Path(__file__).parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import autogluon_features_21 as base_model  # noqa: E402


VARIANT_NAME = "feature-set-24-target-encoding-no-woe"
DEFAULT_CATEGORICAL_COLS = ["grade_subgrade", "loan_purpose", "employment_status", "home_ownership"]

target_encoder = None
woe_mappings: Dict[str, Dict[Any, float]] = {}


def get_default_config() -> Dict[str, Any]:
    return {
        "hyperparameters": {
            "presets": "best_quality",
            "time_limit": 7200,
            "use_gpu": False,
            "num_bag_folds": 8,
            "num_stack_levels": 2,
            "excluded_models": ["NN_TORCH"],
        },
        "model": {
            "leaderboard_rows": 30,
            "target_encoding_folds": 6,
            "target_encoding_smoothing": 2.0,
            "enable_woe": False,
        },
    }


class TargetEncoder:
    """Target encoder with CV to avoid leakage."""

    def __init__(self, n_splits: int = 5, smoothing: float = 1.0, random_state: int = 42):
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.random_state = random_state
        self.global_mean: Optional[float] = None
        self.category_means: Dict[str, Dict[Any, float]] = {}

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, cols: list) -> pd.DataFrame:
        X = X.copy()
        self.global_mean = float(y.mean())

        for col in cols:
            X[f"{col}_target_encoded"] = self.global_mean

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for train_idx, val_idx in skf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            for col in cols:
                stats = y_train.groupby(X_train[col]).agg(["mean", "count"])
                smoothed = {}
                for cat in stats.index:
                    cat_mean = stats.loc[cat, "mean"]
                    cat_count = stats.loc[cat, "count"]
                    smoothed_mean = (cat_mean * cat_count + self.global_mean * self.smoothing) / (
                        cat_count + self.smoothing
                    )
                    smoothed[cat] = smoothed_mean
                X.loc[val_idx, f"{col}_target_encoded"] = X.loc[val_idx, col].map(smoothed).fillna(self.global_mean)

        for col in cols:
            stats = y.groupby(X[col]).agg(["mean", "count"])
            self.category_means[col] = {}
            for cat in stats.index:
                cat_mean = stats.loc[cat, "mean"]
                cat_count = stats.loc[cat, "count"]
                smoothed_mean = (cat_mean * cat_count + self.global_mean * self.smoothing) / (
                    cat_count + self.smoothing
                )
                self.category_means[col][cat] = smoothed_mean
        return X

    def transform(self, X: pd.DataFrame, cols: list) -> pd.DataFrame:
        X = X.copy()
        for col in cols:
            X[f"{col}_target_encoded"] = X[col].map(self.category_means.get(col, {})).fillna(self.global_mean)
        return X


def _drop_ignored(df: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    drop_cols = set(config.dataset.ignored_columns + [config.dataset.id_column])
    drop_cols.discard(config.dataset.target)
    return df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")


def _engineer_features(
    df: pd.DataFrame,
    config: ModelConfig,
    is_train: bool = True,
    enable_woe: bool = False,
) -> pd.DataFrame:
    global target_encoder, woe_mappings

    enriched = base_model._engineer_features(df)
    target_col = config.dataset.target
    categorical_cols = config.model.get("target_encoding_cols", DEFAULT_CATEGORICAL_COLS)
    cols_to_encode = [col for col in categorical_cols if col in enriched.columns]
    if not cols_to_encode:
        return enriched

    n_splits = int(config.model.get("target_encoding_folds", 6))
    smoothing = float(config.model.get("target_encoding_smoothing", 2.0))
    random_state = config.system.random_seed

    if is_train and target_col in enriched.columns:
        target = enriched[target_col].fillna(0).astype(float)
        target_encoder = TargetEncoder(n_splits=n_splits, smoothing=smoothing, random_state=random_state)
        enriched = target_encoder.fit_transform(enriched, target, cols_to_encode)

        if enable_woe:
            woe_mappings = {}
            total_positive = target.sum()
            total_negative = len(target) - total_positive
            total_positive = max(total_positive, 0.5)
            total_negative = max(total_negative, 0.5)
            for col in cols_to_encode:
                mapping: Dict[Any, float] = {}
                for cat in enriched[col].unique():
                    mask = enriched[col] == cat
                    n_positive = target.loc[mask].sum()
                    n_negative = mask.sum() - n_positive
                    n_positive = max(n_positive, 0.5)
                    n_negative = max(n_negative, 0.5)
                    mapping[cat] = float(np.log((n_positive / total_positive) / (n_negative / total_negative)))
                enriched[f"{col}_woe"] = enriched[col].map(mapping).fillna(0)
                woe_mappings[col] = mapping
    else:
        if target_encoder is not None:
            enriched = target_encoder.transform(enriched, cols_to_encode)
        else:
            for col in cols_to_encode:
                enriched[f"{col}_target_encoded"] = 0.0

        if enable_woe:
            for col in cols_to_encode:
                mapping = woe_mappings.get(col, {})
                enriched[f"{col}_woe"] = enriched[col].map(mapping).fillna(0)

    return enriched


def preprocess(df: pd.DataFrame, config: ModelConfig, is_train: bool = True) -> pd.DataFrame:
    enable_woe = bool(config.model.get("enable_woe", False))
    return _engineer_features(df, config, is_train=is_train, enable_woe=enable_woe)


def _build_fit_kwargs(config: ModelConfig) -> Dict[str, Any]:
    fit_kwargs: Dict[str, Any] = {
        "presets": config.hyperparameters.presets,
        "time_limit": config.hyperparameters.time_limit,
        "num_cpus": config.model.get("num_cpus", 16),
        "num_gpus": 1 if config.hyperparameters.use_gpu else 0,
    }

    if config.hyperparameters.excluded_models:
        fit_kwargs["excluded_model_types"] = config.hyperparameters.excluded_models

    for key in ["num_bag_folds", "num_stack_levels", "ag_args", "ag_args_fit", "ag_args_ensemble"]:
        val = getattr(config.hyperparameters, key, None)
        if val is not None:
            fit_kwargs[key] = val

    if "ag_args_ensemble" not in fit_kwargs:
        fit_kwargs["ag_args_ensemble"] = {"fold_fitting_strategy": "sequential_local"}

    return fit_kwargs


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

    fit_kwargs = _build_fit_kwargs(config)
    print(f"[{VARIANT_NAME}] Preset: {fit_kwargs.get('presets')}")
    print(
        f"[{VARIANT_NAME}] Time limit: {fit_kwargs.get('time_limit')}s | "
        f"Bag folds: {fit_kwargs.get('num_bag_folds')} | "
        f"Stack levels: {fit_kwargs.get('num_stack_levels')}"
    )

    predictor.fit(train_data, tuning_data=tuning_data, **fit_kwargs)

    leaderboard = predictor.leaderboard(train_data, silent=True)
    local_cv = float(leaderboard.iloc[0]["score_val"]) if not leaderboard.empty else None
    summary = {
        "local_cv": local_cv,
        "variant": VARIANT_NAME,
        "bag_folds": fit_kwargs.get("num_bag_folds"),
        "stack_levels": fit_kwargs.get("num_stack_levels"),
    }

    if artifacts is None:
        artifacts = {}
    artifacts["target_encoder"] = target_encoder
    artifacts["woe_mappings"] = woe_mappings

    return predictor, summary


def predict(
    model: TabularPredictor,
    test_df: pd.DataFrame,
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> pd.DataFrame:
    global target_encoder, woe_mappings

    if artifacts:
        target_encoder = artifacts.get("target_encoder", target_encoder)
        woe_mappings = artifacts.get("woe_mappings", woe_mappings)

    enable_woe = bool(config.model.get("enable_woe", False))
    test_df = _engineer_features(test_df, config, is_train=False, enable_woe=enable_woe)

    submission = pd.DataFrame()
    submission[config.dataset.id_column] = test_df[config.dataset.id_column]

    drop_features = _drop_ignored(test_df, config)
    if config.dataset.submission_probas:
        preds = model.predict_proba(drop_features, as_multiclass=False)
        if isinstance(preds, pd.DataFrame):
            submission[config.dataset.target] = preds.iloc[:, 1]
        else:
            submission[config.dataset.target] = preds
    else:
        submission[config.dataset.target] = model.predict(drop_features)
    return submission
