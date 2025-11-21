"""
LightGBM model with enriched feature engineering (fe_rich) for experiment_manager/ml_runner.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pathlib import Path
import sys

MODEL_DIR = Path(__file__).parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

PREP_DIR = Path(__file__).parent.parent / "preprocessing"
if str(PREP_DIR) not in sys.path:
    sys.path.insert(0, str(PREP_DIR))

from fe_rich import add_features  # noqa: E402
from kaggle_tools.config_models import ModelConfig  # noqa: E402


VARIANT_NAME = "lgbm-fe-rich"


def get_default_config() -> Dict[str, Any]:
    return {
        "hyperparameters": {
            "presets": "best_quality",
            "time_limit": 3600,
            "use_gpu": False,
        },
        "model": {
            "leaderboard_rows": 20,
        },
    }


def _drop_ignored(df: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    drop_cols = set(config.dataset.ignored_columns + [config.dataset.id_column])
    drop_cols.discard(config.dataset.target)
    return df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")


def _prep_features(df: pd.DataFrame, config: ModelConfig) -> Tuple[pd.DataFrame, list[str]]:
    feats = _drop_ignored(df, config)
    if config.dataset.target in feats.columns:
        feats = feats.drop(columns=[config.dataset.target])
    cat_cols = [c for c in feats.columns if feats[c].dtype == "object"]
    for col in cat_cols:
        feats[col] = feats[col].astype("category")
    return feats, cat_cols


def preprocess(df: pd.DataFrame, config: ModelConfig, is_train: bool = True) -> pd.DataFrame:
    return add_features(df)


def train(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> Tuple[lgb.Booster, Dict[str, Any]]:
    # Apply feature engineering
    train_fe = add_features(train_df)
    X, cat_cols = _prep_features(train_fe, config)
    y = train_df[config.dataset.target]

    if val_df is not None:
        val_fe = add_features(val_df)
        X_val, _ = _prep_features(val_fe, config)
        y_val = val_df[config.dataset.target]
    else:
        X, X_val, y, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    train_ds = lgb.Dataset(X, label=y, categorical_feature=cat_cols, free_raw_data=False)
    val_ds = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, free_raw_data=False)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 96,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 0.5,
        "min_data_in_leaf": 100,
        "class_weight": "balanced",
        "seed": config.system.random_seed,
        "verbose": -1,
    }

    print(f"[{VARIANT_NAME}] Training LightGBM with fe_rich...")
    booster = lgb.train(
        params,
        train_ds,
        num_boost_round=5000,
        valid_sets=[val_ds],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )

    best_auc = booster.best_score.get("valid", {}).get("auc", None)
    summary = {"local_cv": best_auc}
    return booster, summary


def predict(
    model: lgb.Booster,
    test_df: pd.DataFrame,
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> pd.DataFrame:
    test_fe = add_features(test_df)
    X_test, _ = _prep_features(test_fe, config)
    preds = model.predict(X_test, num_iteration=model.best_iteration or model.num_trees())

    submission = pd.DataFrame()
    submission[config.dataset.id_column] = test_df[config.dataset.id_column]
    submission[config.dataset.target] = preds
    return submission
