"""
XGBoost model with enriched feature engineering (fe_rich) for experiment_manager/ml_runner.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb

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


VARIANT_NAME = "xgb-fe-rich"


class XGBWithMapping:
    """Container to keep booster + categorical mappings together."""

    def __init__(self, booster: xgb.Booster, mappings: Dict[str, Dict[Any, int]]):
        self.booster = booster
        self.mappings = mappings


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


def _build_cat_mappings(df: pd.DataFrame, cat_cols: list[str]) -> Dict[str, Dict[Any, int]]:
    mappings: Dict[str, Dict[Any, int]] = {}
    for col in cat_cols:
        uniques = pd.Series(df[col].astype(str).unique())
        mapping = {val: idx for idx, val in enumerate(uniques)}
        mappings[col] = mapping
    return mappings


def _apply_cat_mappings(df: pd.DataFrame, cat_cols: list[str], mappings: Dict[str, Dict[Any, int]]) -> pd.DataFrame:
    out = df.copy()
    for col in cat_cols:
        mapping = mappings[col]
        out[col] = out[col].astype(str).map(mapping).fillna(-1).astype(int)
    return out


def preprocess(df: pd.DataFrame, config: ModelConfig, is_train: bool = True) -> pd.DataFrame:
    return add_features(df)


def train(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> Tuple[XGBWithMapping, Dict[str, Any]]:
    train_fe = add_features(train_df)
    feats = _drop_ignored(train_fe, config)
    y = train_df[config.dataset.target].values
    if config.dataset.target in feats.columns:
        feats = feats.drop(columns=[config.dataset.target])
    cat_cols = [c for c in feats.columns if feats[c].dtype == "object"]
    mappings = _build_cat_mappings(feats, cat_cols)
    feats_enc = _apply_cat_mappings(feats, cat_cols, mappings)

    if val_df is not None:
        val_fe = add_features(val_df)
        val_feats = _drop_ignored(val_fe, config)
        if config.dataset.target in val_feats.columns:
            val_feats = val_feats.drop(columns=[config.dataset.target])
        val_feats_enc = _apply_cat_mappings(val_feats, cat_cols, mappings)
        y_val = val_df[config.dataset.target].values
    else:
        feats_enc, val_feats_enc, y, y_val = train_test_split(
            feats_enc, y, test_size=0.2, random_state=42, stratify=y
        )

    dtrain = xgb.DMatrix(feats_enc, label=y)
    dvalid = xgb.DMatrix(val_feats_enc, label=y_val)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "eta": 0.05,
        "max_depth": 7,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "lambda": 1.0,
        "alpha": 0.1,
        "tree_method": "hist",
        "scale_pos_weight": float((len(y) - y.sum()) / (y.sum() + 1e-6)),
        "seed": config.system.random_seed,
    }

    print(f"[{VARIANT_NAME}] Training XGBoost with fe_rich...")
    evals = [(dvalid, "valid")]
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=evals,
        early_stopping_rounds=200,
        verbose_eval=False,
    )

    best_auc = booster.best_score
    summary = {"local_cv": best_auc}
    return XGBWithMapping(booster, mappings), summary


def predict(
    model: XGBWithMapping,
    test_df: pd.DataFrame,
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> pd.DataFrame:
    test_fe = add_features(test_df)
    feats = _drop_ignored(test_fe, config)
    if config.dataset.target in feats.columns:
        feats = feats.drop(columns=[config.dataset.target])
    cat_cols = [c for c in feats.columns if feats[c].dtype == "object"]
    mappings = model.mappings or _build_cat_mappings(feats, cat_cols)
    feats_enc = _apply_cat_mappings(feats, cat_cols, mappings)

    dtest = xgb.DMatrix(feats_enc)
    booster = model.booster
    best_it = booster.best_iteration if booster.best_iteration is not None else 0
    preds = booster.predict(dtest, iteration_range=(0, best_it + 1))

    submission = pd.DataFrame()
    submission[config.dataset.id_column] = test_df[config.dataset.id_column]
    submission[config.dataset.target] = preds
    return submission
