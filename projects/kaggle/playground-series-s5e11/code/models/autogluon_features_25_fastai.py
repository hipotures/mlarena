"""
AutoGluon variant #25: fe24 feature set (TE, no WOE) but restricted to NeuralNetFastAI.
Targets a 1h, GPU-enabled run with lightweight bagging/stacking.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import sys

import pandas as pd
from autogluon.tabular import TabularPredictor
import torch

from kaggle_tools.config_models import ModelConfig

MODEL_DIR = Path(__file__).parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import autogluon_features_24_target_bagging_nowoe as base  # noqa: E402

VARIANT_NAME = "feature-set-25-fastai"


def get_default_config() -> Dict[str, Any]:
    """GPU FastAI-only setup with TE (no WOE), 1h budget."""
    return {
        "hyperparameters": {
            "presets": "best_quality",
            "time_limit": 3600,
            "use_gpu": True,
            "num_bag_folds": 5,
            "num_stack_levels": 1,
            "ag_args_fit": {"num_gpus": 1},
            # No exclusions by default; let template control included/excluded models.
            "excluded_models": None,
        },
        "model": {
            "leaderboard_rows": 30,
            "target_encoding_folds": 6,
            "target_encoding_smoothing": 2.0,
            "enable_woe": False,
        },
    }


def _drop_ignored(df: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    drop_cols = set(config.dataset.ignored_columns + [config.dataset.id_column])
    drop_cols.discard(config.dataset.target)
    return df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")


def preprocess(df: pd.DataFrame, config: ModelConfig, is_train: bool = True) -> pd.DataFrame:
    """Reuse fe24 feature engineering (TE, no WOE)."""
    return base._engineer_features(df, config, is_train=is_train, enable_woe=False)


def _build_fit_kwargs(config: ModelConfig) -> Dict[str, Any]:
    fit_kwargs: Dict[str, Any] = {
        "presets": config.hyperparameters.presets,
        "time_limit": config.hyperparameters.time_limit,
        "num_cpus": config.model.get("num_cpus", 16),
        "num_gpus": 1 if config.hyperparameters.use_gpu else 0,
    }

    if config.hyperparameters.excluded_models:
        fit_kwargs["excluded_model_types"] = config.hyperparameters.excluded_models

    for key in [
        "num_bag_folds",
        "num_stack_levels",
        "ag_args",
        "ag_args_fit",
        "ag_args_ensemble",
        "included_model_types",
        "feature_prune_kwargs",
        "hyperparameter_tune_kwargs",
    ]:
        val = getattr(config.hyperparameters, key, None)
        if val is not None:
            fit_kwargs[key] = val

    # Ensure FastAI sees a GPU when requested
    if config.hyperparameters.use_gpu:
        fit_kwargs.setdefault("ag_args_fit", {}).setdefault("num_gpus", 1)
        fit_kwargs["ag_args_fit"].setdefault("device", "cuda")
        # Keep data loading simple to avoid worker contention in constrained envs
        fit_kwargs["ag_args_fit"].setdefault("num_workers", 0)
        fit_kwargs["ag_args_fit"].setdefault("batch_size", 256)
        fit_kwargs["ag_args_fit"].setdefault("epochs", 15)

    # Sequential folding to avoid Ray/socket issues in constrained environments.
    if "ag_args_ensemble" not in fit_kwargs:
        fit_kwargs["ag_args_ensemble"] = {"fold_fitting_strategy": "sequential_local"}

    return fit_kwargs


def train(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> Tuple[TabularPredictor, Dict[str, Any]]:
    # FastAI uses internal validation; we pass full train and let AG handle folds/bagging.
    features = _drop_ignored(train_df, config)
    train_data = features.copy()
    train_data[config.dataset.target] = train_df[config.dataset.target]

    tuning_data = None
    if val_df is not None:
        val_features = _drop_ignored(val_df, config)
        tuning_data = val_features.copy()
        tuning_data[config.dataset.target] = val_df[config.dataset.target]

    if config.hyperparameters.use_gpu:
        gpu_ok = torch.cuda.is_available()
        print(
            f"[{VARIANT_NAME}] GPU requested. torch.cuda.is_available={gpu_ok}, "
            f"device_count={torch.cuda.device_count() if gpu_ok else 0}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
        )
        if not gpu_ok:
            raise RuntimeError("GPU requested but torch.cuda.is_available() is False. Check drivers/visibility.")

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
        f"Stack levels: {fit_kwargs.get('num_stack_levels')} | "
        "Models: FASTAI only"
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
    artifacts["target_encoder"] = base.target_encoder
    artifacts["woe_mappings"] = base.woe_mappings

    return predictor, summary


def predict(
    model: TabularPredictor,
    test_df: pd.DataFrame,
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> pd.DataFrame:
    # Restore encoders
    if artifacts:
        base.target_encoder = artifacts.get("target_encoder", base.target_encoder)
        base.woe_mappings = artifacts.get("woe_mappings", base.woe_mappings)

    test_df = base._engineer_features(test_df, config, is_train=False, enable_woe=False)

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
