"""
TabICL + skrub pipeline integrated with the generic ML runner.

Two templates are expected:
- tabicl-fast: 10% train sample, 3-fold CV, smaller ensemble.
- tabicl-full: 100% train, 5-fold CV, larger ensemble.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from skrub import TableVectorizer
from tabicl import TabICLClassifier

from kaggle_tools.config_models import ModelConfig


def get_default_config() -> Dict[str, Any]:
    return {
        "model": {
            "sample_fraction": 1.0,
            "cv_folds": 3,
            "n_estimators": 16,
            "batch_size": 8,
            "checkpoint_version": "tabicl-classifier-v1.1-0506.ckpt",
            "norm_methods": ["none", "power"],
            "feat_shuffle_method": "latin",
        "class_shift": True,
        "outlier_threshold": 4.0,
        "softmax_temperature": 0.9,
        "average_logits": True,
        "use_hierarchical": True,
            "use_amp": True,
            "allow_auto_download": True,
            "device": "cuda",
        "random_state": 42,
        "n_jobs": None,
        "verbose": False,
        "inference_config": None,
        "table_vectorizer": {},
        "output_labels": False,
        "label_threshold": 0.5,
    }
}


def preprocess(df: Optional[pd.DataFrame], config: ModelConfig, is_train: bool = True) -> Optional[pd.DataFrame]:
    if df is None:
        return None
    df = df.copy()
    if is_train:
        sample_fraction = float(config.model.get("sample_fraction", 1.0))
        target_col = config.dataset.target
        if sample_fraction < 1.0 and target_col in df.columns:
            if config.dataset.problem_type == "binary":
                df, _ = train_test_split(
                    df,
                    test_size=1 - sample_fraction,
                    random_state=config.system.random_seed,
                    stratify=df[target_col],
                )
            else:
                df = df.sample(
                    frac=sample_fraction,
                    random_state=config.system.random_seed,
                )
    return df


def _drop_unused(df: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    drop_cols = set(config.dataset.ignored_columns + [config.dataset.id_column])
    drop_cols.discard(config.dataset.target)
    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")


def _split_features_and_target(df: pd.DataFrame, config: ModelConfig) -> Tuple[pd.DataFrame, pd.Series]:
    features = _drop_unused(df, config)
    target = features.pop(config.dataset.target)
    return features, target


def _build_pipeline(config: ModelConfig):
    cfg = config.model
    table_vectorizer = TableVectorizer(**(cfg.get("table_vectorizer") or {}))
    clf = TabICLClassifier(
        n_estimators=int(cfg.get("n_estimators", 16)),
        norm_methods=cfg.get("norm_methods"),
        feat_shuffle_method=cfg.get("feat_shuffle_method", "latin"),
        class_shift=bool(cfg.get("class_shift", True)),
        outlier_threshold=float(cfg.get("outlier_threshold", 4.0)),
        softmax_temperature=float(cfg.get("softmax_temperature", 0.9)),
        average_logits=bool(cfg.get("average_logits", True)),
        use_hierarchical=bool(cfg.get("use_hierarchical", True)),
        use_amp=bool(cfg.get("use_amp", True)),
        batch_size=cfg.get("batch_size", 8),
        model_path=None,
        allow_auto_download=bool(cfg.get("allow_auto_download", True)),
        checkpoint_version=cfg.get("checkpoint_version", "tabicl-classifier-v1.1-0506.ckpt"),
        device=cfg.get("device"),
        random_state=int(cfg.get("random_state", config.system.random_seed)),
        n_jobs=cfg.get("n_jobs"),
        verbose=bool(cfg.get("verbose", False)),
        inference_config=cfg.get("inference_config"),
    )
    return make_pipeline(table_vectorizer, clf)


def _set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def train(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    config: ModelConfig,
    artifacts: Optional[Any] = None,
):
    _set_seeds(config.system.random_seed)
    X, y = _split_features_and_target(train_df, config)

    cfg = config.model
    cv_folds = int(cfg.get("cv_folds", 3))
    local_cv = None

    if cv_folds > 1 and len(np.unique(y)) > 1 and len(y) > cv_folds:
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.system.random_seed)
        oof = np.zeros(len(y))
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            print(f"[TabICL] CV fold {fold}/{cv_folds}")
            model = _build_pipeline(config)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            proba = model.predict_proba(X.iloc[val_idx])
            if proba.ndim == 2 and proba.shape[1] > 1:
                oof[val_idx] = proba[:, 1]
            else:
                oof[val_idx] = proba.reshape(-1)
        local_cv = float(roc_auc_score(y, oof))
        print(f"[TabICL] OOF ROC-AUC: {local_cv:.5f}")

    print("[TabICL] Fitting final model on full (sampled) training data...")
    final_model = _build_pipeline(config)
    final_model.fit(X, y)

    summary = {
        "local_cv": local_cv,
        "cv_folds": cv_folds,
        "sample_fraction": cfg.get("sample_fraction", 1.0),
        "n_estimators": cfg.get("n_estimators", 16),
    }
    return final_model, summary


def predict(
    model,
    test_df: pd.DataFrame,
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> pd.DataFrame:
    features = _drop_unused(test_df, config)
    submit_probas = bool(config.dataset.submission_probas)
    output_labels = bool(config.model.get("output_labels", False))
    label_threshold = float(config.model.get("label_threshold", 0.5))

    submission = pd.DataFrame()
    submission[config.dataset.id_column] = test_df[config.dataset.id_column]

    if submit_probas and not output_labels:
        proba = model.predict_proba(features)
        if proba.ndim == 2 and proba.shape[1] > 1:
            submission[config.dataset.target] = proba[:, 1]
        else:
            submission[config.dataset.target] = proba.reshape(-1)
    else:
        proba = model.predict_proba(features)
        if proba.ndim == 2 and proba.shape[1] > 1:
            pos = proba[:, 1]
        else:
            pos = proba.reshape(-1)
        labels = (pos >= label_threshold).astype(int)
        submission[config.dataset.target] = labels

    return submission
