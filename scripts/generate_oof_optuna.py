"""
Generate OOF predictions and test submissions for base models (xgb/lgbm/cat) using Optuna tuning.

Usage (example):
    uv run python scripts/generate_oof_optuna.py \
        --project playground-series-s5e11 \
        --models xgboost lightgbm catboost \
        --n-trials 10 \
        --cv-folds 3 \
        --time-limit 120

Outputs:
    - OOF CSV per model: projects/kaggle/<project>/oof/oof-<model>.csv
    - Submission per model: projects/kaggle/<project>/submissions/submission-oof-<model>-<ts>.csv
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from kaggle_tools.optuna import CVObjective, xgboost_param_space, lightgbm_param_space, catboost_param_space


def load_project(project: str) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parent.parent
    project_root = repo_root / "projects" / "kaggle" / project
    config_path = project_root / "configs" / "project.yaml"
    with open(config_path) as f:
        project_cfg = yaml.safe_load(f)
    return {
        "root": project_root,
        "data_dir": project_root / "data",
        "optuna_cfg": project_cfg.get("optuna", {}),
        "id_column": project_cfg.get("ID_COLUMN", "id"),
        "target_column": project_cfg.get("TARGET_COLUMN", project_cfg.get("target_column", "target")),
    }


def get_model_class(name: str):
    if name == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier
    if name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier
    if name == "catboost":
        import catboost as cb
        return cb.CatBoostClassifier
    raise ValueError(f"Unknown model {name}")


def get_param_space_fn(name: str):
    if name == "xgboost":
        return xgboost_param_space
    if name == "lightgbm":
        return lightgbm_param_space
    if name == "catboost":
        return catboost_param_space
    raise ValueError(f"Unknown model {name}")


def tune_model(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    param_space_cfg: Dict[str, List],
    n_trials: int,
    cv_folds: int,
    timeout: int,
    random_seed: int = 42,
) -> Dict[str, Any]:
    study = optuna.create_study(direction="maximize")

    def objective(trial: optuna.Trial) -> float:
        params = get_param_space_fn(model_name)(trial, param_space_cfg)
        model_class = get_model_class(model_name)
        # model_kwargs for catboost / xgboost categorical handled in CVObjective
        obj = CVObjective(
            model_class=model_class,
            X=X,
            y=y,
            param_space_fn=lambda t: params,
            metric_fn=roc_auc_score,
            cv_folds=cv_folds,
            early_stopping_rounds=50,
            random_seed=random_seed,
            stratified=True,
        )
        return obj(trial)

    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    return study.best_params


def generate_oof_and_submission(
    model_name: str,
    best_params: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    test_df: pd.DataFrame,
    id_column: str,
    target_column: str,
    cv_folds: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    oof = pd.DataFrame({id_column: test_df[id_column]})  # placeholder will replace
    oof = pd.DataFrame({id_column: X.index})  # using index aligns with train rows
    oof[target_column] = 0.0
    test_preds = np.zeros(len(test_df))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model_class = get_model_class(model_name)
        params = dict(best_params)

        if model_name == "xgboost":
            params.setdefault("enable_categorical", True)
            params.setdefault("tree_method", "hist")
            model = model_class(**params, n_estimators=300, random_state=42)
        elif model_name == "lightgbm":
            params.setdefault("objective", "binary")
            params.setdefault("metric", "auc")
            model = model_class(**params, n_estimators=300, random_state=42)
        else:  # catboost
            params.pop("verbose", None)
            cat_features = [i for i, col in enumerate(X.columns) if str(X[col].dtype) == "category"]
            model = model_class(
                **params,
                iterations=300,
                random_state=42,
                cat_features=cat_features,
                verbose=False,
            )

        model.fit(X_train, y_train)
        if hasattr(model, "predict_proba"):
            val_pred = model.predict_proba(X_val)[:, 1]
            test_pred = model.predict_proba(test_df)[:, 1]
        else:
            val_pred = model.predict(X_val)
            test_pred = model.predict(test_df)

        oof.loc[X_val.index, target_column] = val_pred
        test_preds += test_pred / cv_folds

    submission = pd.DataFrame({id_column: test_df[id_column], target_column: test_preds})
    oof_out = pd.DataFrame({id_column: X.index, target_column: oof[target_column]})
    return oof_out, submission


def main():
    parser = argparse.ArgumentParser(description="Generate OOF + submissions for Optuna-tuned models")
    parser.add_argument("--project", required=True)
    parser.add_argument("--models", nargs="+", default=["xgboost", "lightgbm", "catboost"])
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    ctx = load_project(args.project)
    train_path = ctx["data_dir"] / "train.csv"
    test_path = ctx["data_dir"] / "test.csv"
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    id_col = ctx["id_column"]
    target_col = ctx["target_column"]
    # Drop target and id from features
    feature_drop = [target_col]
    if id_col in train_df.columns:
        feature_drop.append(id_col)
    X = train_df.drop(columns=feature_drop)
    y = train_df[target_col]

    # Cast categorical columns to category
    cat_cols = X.select_dtypes(include=["object"]).columns
    if len(cat_cols) > 0:
        X[cat_cols] = X[cat_cols].astype("category")
        test_df[cat_cols] = test_df[cat_cols].astype("category")

    param_space_cfg = ctx["optuna_cfg"].get("param_space", {})
    oof_dir = ctx["root"] / "oof"
    oof_dir.mkdir(exist_ok=True)
    submissions_dir = ctx["root"] / "submissions"
    submissions_dir.mkdir(exist_ok=True)

    for model_name in args.models:
        if model_name not in param_space_cfg:
            raise ValueError(f"Param space for {model_name} not found in project optuna config")
        print(f"=== {model_name} | tuning {args.n_trials} trials, cv={args.cv_folds} ===")
        best_params = tune_model(
            model_name,
            X,
            y,
            param_space_cfg[model_name],
            n_trials=args.n_trials,
            cv_folds=args.cv_folds,
            timeout=args.timeout,
        )
        print(f"Best params for {model_name}: {best_params}")

        oof_df, sub_df = generate_oof_and_submission(
            model_name,
            best_params,
            X,
            y,
            test_df,
            id_col,
            target_col,
            cv_folds=args.cv_folds,
        )

        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        oof_path = oof_dir / f"oof-{model_name}.csv"
        sub_path = submissions_dir / f"submission-oof-{model_name}-{ts}.csv"
        oof_df.to_csv(oof_path, index=False)
        sub_df.to_csv(sub_path, index=False)
        print(f"Saved OOF: {oof_path}")
        print(f"Saved submission: {sub_path}")


if __name__ == "__main__":
    main()
