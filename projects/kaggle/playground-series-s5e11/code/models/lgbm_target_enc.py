"""
LightGBM with target encoding + enriched features (fe_rich).

Run from repo root:
  uv run python projects/kaggle/playground-series-s5e11/code/models/lgbm_target_enc.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from utils import config  # noqa: E402
from utils.submission import create_submission  # noqa: E402

PREP_DIR = Path(__file__).resolve().parents[1] / "preprocessing"
if str(PREP_DIR) not in sys.path:
    sys.path.insert(0, str(PREP_DIR))

from fe_rich import add_features  # noqa: E402

try:
    import lightgbm as lgb  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit("lightgbm is required for this script") from exc


def target_encode_cv(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    cat_cols: Iterable[str],
    n_splits: int = 5,
    smoothing: float = 20.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-validated target encoding with simple smoothing."""
    train_enc = train_df.copy()
    test_enc = test_df.copy()
    prior = train_df[target_col].mean()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for col in cat_cols:
        enc_col = f"{col}_te"
        train_enc[enc_col] = 0.0

        for tr_idx, va_idx in skf.split(train_df, train_df[target_col]):
            tr_fold = train_df.iloc[tr_idx]
            va_fold = train_df.iloc[va_idx]
            stats = (
                tr_fold.groupby(col)[target_col]
                .agg(["mean", "count"])
                .rename(columns={"mean": "m", "count": "n"})
            )
            stats["enc"] = (stats["m"] * stats["n"] + prior * smoothing) / (stats["n"] + smoothing)
            mapped = va_fold[col].map(stats["enc"])
            train_enc.loc[va_idx, enc_col] = mapped.fillna(prior)

        # Fit on full train for test transform
        full_stats = (
            train_df.groupby(col)[target_col]
            .agg(["mean", "count"])
            .rename(columns={"mean": "m", "count": "n"})
        )
        full_stats["enc"] = (full_stats["m"] * full_stats["n"] + prior * smoothing) / (full_stats["n"] + smoothing)
        test_enc[enc_col] = test_df[col].map(full_stats["enc"]).fillna(prior)

    return train_enc, test_enc


def label_encode(df: pd.DataFrame, cat_cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cat_cols:
        out[col] = out[col].astype("category").cat.codes.replace(-1, np.nan)
    return out


def train_lgbm(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str):
    id_col = getattr(config, "ID_COLUMN", "id")
    y = train_df[target_col].values

    # Target encoding for high-cardinality categories
    cat_cols = [c for c in train_df.columns if train_df[c].dtype == "object"]
    te_train, te_test = target_encode_cv(train_df, test_df, target_col, cat_cols)

    # Drop raw cats; keep numerics + encoded.
    drop_cols = cat_cols + [target_col]
    X = te_train.drop(columns=drop_cols)
    X_test = te_test.drop(columns=cat_cols)

    lgb_train = lgb.Dataset(X, label=y, free_raw_data=False)

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
        "seed": 42,
    }

    # CV to estimate best iteration
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(train_df))
    best_iters = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        dtrain = lgb.Dataset(X.iloc[tr_idx], label=y[tr_idx], free_raw_data=False)
        dvalid = lgb.Dataset(X.iloc[va_idx], label=y[va_idx], free_raw_data=False)
        clf = lgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        oof_preds[va_idx] = clf.predict(X.iloc[va_idx], num_iteration=clf.best_iteration)
        best_iters.append(clf.best_iteration)
        auc = roc_auc_score(y[va_idx], oof_preds[va_idx])
        print(f"[Fold {fold}] AUC: {auc:.5f}, best_iter={clf.best_iteration}")

    cv_auc = roc_auc_score(y, oof_preds)
    best_iter = int(np.median(best_iters))
    print(f"[CV] AUC={cv_auc:.5f}, median_best_iter={best_iter}")

    # Train final model on full data
    final_clf = lgb.train(
        params,
        lgb_train,
        num_boost_round=best_iter,
    )

    test_preds = final_clf.predict(X_test, num_iteration=best_iter)
    submission = pd.DataFrame({id_col: test_df[id_col], config.TARGET_COLUMN: test_preds})
    return submission, cv_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--note", default="lgbm target encoding + fe_rich", help="submission note")
    args = parser.parse_args()

    train_df = pd.read_csv(config.TRAIN_PATH)
    test_df = pd.read_csv(config.TEST_PATH)

    # Feature engineering
    train_fe = add_features(train_df)
    test_fe = add_features(test_df)

    submission, cv_auc = train_lgbm(train_fe, test_fe, config.TARGET_COLUMN)

    artifact = create_submission(
        predictions=submission[config.TARGET_COLUMN],
        test_ids=submission[getattr(config, "ID_COLUMN", "id")],
        model_name="lgbm_target_enc_fe_rich",
        local_cv_score=cv_auc,
        notes=args.note,
        config={"model": "lightgbm", "feature_set": "fe_rich+target_encoding"},
        track=True,
    )
    print(f"✓ Submission created at {artifact.path}")


if __name__ == "__main__":
    main()
