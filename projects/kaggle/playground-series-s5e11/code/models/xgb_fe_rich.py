"""
XGBoost with enriched features (fe_rich) + label encoding for categories.

Run from repo root:
  uv run python projects/kaggle/playground-series-s5e11/code/models/xgb_fe_rich.py
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
    import xgboost as xgb  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit("xgboost is required for this script") from exc


def label_encode(df: pd.DataFrame, cat_cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cat_cols:
        out[col] = out[col].astype("category").cat.codes
    return out


def prepare_data(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    cat_cols = [c for c in train_df.columns if train_df[c].dtype == "object"]
    train_enc = label_encode(train_df, cat_cols)
    test_enc = label_encode(test_df, cat_cols)
    y = train_enc[target_col].values
    X = train_enc.drop(columns=[target_col])
    X_test = test_enc.drop(columns=[target_col], errors="ignore")
    return X, X_test, y


def train_xgb(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str):
    id_col = getattr(config, "ID_COLUMN", "id")
    X, X_test, y = prepare_data(train_df, test_df, target_col)

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
        "nthread": 8,
        "seed": 42,
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(train_df))
    best_iters = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        dtrain = xgb.DMatrix(X.iloc[tr_idx], label=y[tr_idx])
        dvalid = xgb.DMatrix(X.iloc[va_idx], label=y[va_idx])
        evals = [(dvalid, "valid")]
        clf = xgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            evals=evals,
            early_stopping_rounds=100,
            verbose_eval=False,
        )
        oof_preds[va_idx] = clf.predict(dvalid, iteration_range=(0, clf.best_iteration + 1))
        best_iters.append(clf.best_iteration)
        auc = roc_auc_score(y[va_idx], oof_preds[va_idx])
        print(f"[Fold {fold}] AUC: {auc:.5f}, best_iter={clf.best_iteration}")

    cv_auc = roc_auc_score(y, oof_preds)
    best_iter = int(np.median(best_iters))
    print(f"[CV] AUC={cv_auc:.5f}, median_best_iter={best_iter}")

    dtrain_full = xgb.DMatrix(X, label=y)
    dtest = xgb.DMatrix(X_test)
    final = xgb.train(
        params,
        dtrain_full,
        num_boost_round=best_iter,
        verbose_eval=False,
    )
    test_preds = final.predict(dtest, iteration_range=(0, best_iter + 1))
    submission = pd.DataFrame({id_col: test_df[id_col], target_col: test_preds})
    return submission, cv_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--note", default="xgb + fe_rich", help="submission note")
    args = parser.parse_args()

    train_df = pd.read_csv(config.TRAIN_PATH)
    test_df = pd.read_csv(config.TEST_PATH)

    train_fe = add_features(train_df)
    test_fe = add_features(test_df)

    submission, cv_auc = train_xgb(train_fe, test_fe, config.TARGET_COLUMN)
    artifact = create_submission(
        predictions=submission[config.TARGET_COLUMN],
        test_ids=submission[getattr(config, "ID_COLUMN", "id")],
        model_name="xgb_fe_rich",
        local_cv_score=cv_auc,
        notes=args.note,
        config={"model": "xgboost", "feature_set": "fe_rich+label_encoding"},
        track=True,
    )
    print(f"✓ Submission created at {artifact.path}")


if __name__ == "__main__":
    main()
