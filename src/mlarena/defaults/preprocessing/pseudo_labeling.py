"""
Pseudo-Labeling Sub-Module

Purpose: Train a quick model and add confident test predictions as pseudo-labeled rows.
Libraries: pandas, numpy, sklearn
Parameters:
  - include_cols: list[str] | None
  - exclude_cols: list[str]
  - use_original_features_only: bool
  - model_type: logreg|rf
  - logreg_max_iter: int
  - logreg_c: float
  - rf_n_estimators: int
  - rf_max_depth: int | None
  - confidence_threshold: float
  - max_pseudo_fraction: float | None
  - use_soft_labels: bool
  - weight_by_confidence: bool
  - scale_features: bool
  - missing_strategy: mean|median|zero
  - fit_on_val: bool
  - allow_regression: bool
  - regression_keep_fraction: float
  - random_state: int
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from mlarena.preprocessing.utils import (
    validation,
    artifacts,
    dataframe_utils,
    report,
)


def _impute_values(
    df: pd.DataFrame, cols: List[str], strategy: str
) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for col in cols:
        if strategy == "median":
            values[col] = float(df[col].median()) if col in df.columns else 0.0
        elif strategy == "zero":
            values[col] = 0.0
        else:
            values[col] = float(df[col].mean()) if col in df.columns else 0.0
    return values


def _apply_impute(
    df: pd.DataFrame, cols: List[str], values: Dict[str, float]
) -> pd.DataFrame:
    df_out = df.copy()
    for col in cols:
        if col not in df_out.columns:
            df_out[col] = np.nan
        df_out[col] = df_out[col].fillna(values.get(col, 0.0))
    return df_out


def fit_transform(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
    orig_df: pd.DataFrame | None = None,
    eval_df: pd.DataFrame | None = None,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame | None,
    pd.DataFrame,
    pd.DataFrame | None,
    pd.DataFrame | None,
    Dict[str, Any],
]:
    artifact_dir = Path(config.get("_artifact_dir", "."))
    dataset_config = config.get("_dataset", {})
    id_column = dataset_config.get("id_column", "id")
    target_column = dataset_config.get("target")
    ignored_columns = dataset_config.get("ignored_columns", [])
    problem_type = (dataset_config.get("problem_type") or "binary").lower()

    required_params: List[str] = []
    optional_params: Dict[str, Any] = {
        "include_cols": None,
        "exclude_cols": [],
        "use_original_features_only": True,
        "model_type": "logreg",
        "logreg_max_iter": 200,
        "logreg_c": 1.0,
        "rf_n_estimators": 200,
        "rf_max_depth": None,
        "confidence_threshold": 0.9,
        "max_pseudo_fraction": 0.2,
        "use_soft_labels": False,
        "weight_by_confidence": False,
        "scale_features": True,
        "missing_strategy": "median",
        "fit_on_val": False,
        "allow_regression": False,
        "regression_keep_fraction": 0.2,
        "random_state": 42,
    }
    validation.validate_config(config, required_params, optional_params)
    validation.validate_choice(config["model_type"], ["logreg", "rf"], "model_type")
    validation.validate_choice(
        config["missing_strategy"], ["mean", "median", "zero"], "missing_strategy"
    )

    submodule_dir = artifacts.get_submodule_artifact_dir(
        artifact_dir, "pseudo_labeling"
    )

    train_df_original = dataframe_utils.copy_dataframe(train_df)
    test_df_original = dataframe_utils.copy_dataframe(test_df)

    if not target_column or target_column not in train_df.columns:
        raise ValueError("Target column is required for pseudo-labeling.")
    if train_df[target_column].isnull().any():
        raise ValueError(
            "Target column contains NaNs; pseudo-labeling requires complete targets."
        )

    exclude_cols = [id_column, target_column] + ignored_columns + config["exclude_cols"]
    exclude_cols = [c for c in exclude_cols if c]
    numeric_cols = dataframe_utils.get_numeric_columns(train_df, exclude=exclude_cols)

    use_orig_only = bool(config.get("use_original_features_only"))
    if use_orig_only:
        numeric_cols = dataframe_utils.filter_original_columns(
            numeric_cols, config.get("_original_features")
        )

    if config["include_cols"]:
        numeric_cols = [c for c in config["include_cols"] if c in numeric_cols]

    if not numeric_cols:
        state_dict = {
            "version": "1.0",
            "config": {k: v for k, v in config.items() if not k.startswith("_")},
            "message": "No numeric columns available for pseudo-labeling.",
            "n_added": 0,
        }
        return train_df, val_df, test_df, eval_df, orig_df, state_dict

    if problem_type == "regression" and not config["allow_regression"]:
        state_dict = {
            "version": "1.0",
            "config": {k: v for k, v in config.items() if not k.startswith("_")},
            "message": "Regression pseudo-labeling disabled.",
            "n_added": 0,
        }
        return train_df, val_df, test_df, eval_df, orig_df, state_dict

    fit_df = train_df.copy()
    if config["fit_on_val"] and val_df is not None and target_column in val_df.columns:
        fit_df = pd.concat([fit_df, val_df], axis=0)

    impute_values = _impute_values(fit_df, numeric_cols, config["missing_strategy"])
    fit_df = _apply_impute(fit_df, numeric_cols, impute_values)

    x_train = fit_df[numeric_cols].to_numpy(dtype=float)
    y_train = fit_df[target_column].to_numpy()

    scaler = None
    if config["scale_features"] and config["model_type"] == "logreg":
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)

    if problem_type in {"binary", "multiclass"}:
        if config["model_type"] == "logreg":
            model = LogisticRegression(
                max_iter=int(config["logreg_max_iter"]),
                C=float(config["logreg_c"]),
                random_state=int(config["random_state"]),
                n_jobs=-1,
                multi_class="auto",
            )
        else:
            model = RandomForestClassifier(
                n_estimators=int(config["rf_n_estimators"]),
                max_depth=config["rf_max_depth"],
                random_state=int(config["random_state"]),
                n_jobs=-1,
            )
    else:
        model = RandomForestRegressor(
            n_estimators=int(config["rf_n_estimators"]),
            max_depth=config["rf_max_depth"],
            random_state=int(config["random_state"]),
            n_jobs=-1,
        )

    model.fit(x_train, y_train)

    test_imputed = _apply_impute(test_df, numeric_cols, impute_values)
    x_test = test_imputed[numeric_cols].to_numpy(dtype=float)
    if scaler is not None:
        x_test = scaler.transform(x_test)

    if problem_type in {"binary", "multiclass"}:
        proba = model.predict_proba(x_test)
        confidence = proba.max(axis=1)
        preds = model.classes_[proba.argmax(axis=1)]

        if config["use_soft_labels"]:
            if problem_type == "multiclass":
                warnings.warn(
                    "Soft labels for multiclass not supported; using hard labels."
                )
                config["use_soft_labels"] = False
            elif proba.shape[1] != 2:
                config["use_soft_labels"] = False

        if config["use_soft_labels"]:
            y_pseudo = proba[:, 1]
        else:
            y_pseudo = preds

        mask = confidence >= float(config["confidence_threshold"])
        indices = np.where(mask)[0]
        max_frac = config.get("max_pseudo_fraction")
        if max_frac is not None:
            max_count = int(len(test_df) * float(max_frac))
            if max_count < len(indices):
                top_idx = np.argsort(confidence[indices])[::-1][:max_count]
                indices = indices[top_idx]
    else:
        y_pseudo = model.predict(x_test)
        max_frac = config.get("max_pseudo_fraction")
        if max_frac is None:
            max_frac = config.get("regression_keep_fraction")
        max_count = int(len(test_df) * float(max_frac))
        indices = np.arange(len(test_df))
        if max_count < len(indices):
            rng = np.random.default_rng(config["random_state"])
            indices = rng.choice(indices, size=max_count, replace=False)
        confidence = np.ones(len(test_df))

    if len(indices) == 0:
        state_dict = {
            "version": "1.0",
            "config": {k: v for k, v in config.items() if not k.startswith("_")},
            "message": "No pseudo-labeled rows met the threshold.",
            "n_added": 0,
        }
        return train_df, val_df, test_df, eval_df, orig_df, state_dict

    pseudo_df = test_imputed.iloc[indices].copy()
    pseudo_df[target_column] = y_pseudo[indices]

    if config["weight_by_confidence"]:
        if "sample_weight" not in pseudo_df.columns:
            pseudo_df["sample_weight"] = confidence[indices]
        else:
            pseudo_df["sample_weight"] = (
                pseudo_df["sample_weight"] * confidence[indices]
            )

    train_df = pd.concat([train_df, pseudo_df], axis=0)
    train_df = train_df.sample(
        frac=1.0, random_state=config["random_state"]
    ).reset_index(drop=True)

    artifacts.save_fitted_object(model, submodule_dir, "pseudo_model.pkl")
    if scaler is not None:
        artifacts.save_fitted_object(scaler, submodule_dir, "scaler.pkl")

    transformation_summary = report.create_preprocessing_report(
        train_df_original,
        train_df,
        test_df_original,
        test_df,
        config,
    )
    transformation_summary["pseudo_labeling"] = {
        "n_added": len(indices),
        "confidence_threshold": float(config["confidence_threshold"]),
        "model_type": config["model_type"],
    }
    artifacts.save_report(transformation_summary, submodule_dir, "summary.json")

    state_dict = {
        "version": "1.0",
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
        "n_added": int(len(indices)),
        "model_type": config["model_type"],
        "model_path": str(
            (submodule_dir / "pseudo_model.pkl").relative_to(artifact_dir)
        ),
    }
    if scaler is not None:
        state_dict["scaler_path"] = str(
            (submodule_dir / "scaler.pkl").relative_to(artifact_dir)
        )
    state_dict["impute_values"] = impute_values

    return train_df, val_df, test_df, eval_df, orig_df, state_dict
