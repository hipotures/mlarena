"""
KNN Graph Features Sub-Module

Purpose: Create KNN distance-based features for each row.
Libraries: pandas, numpy, sklearn
Parameters:
  - include_cols: list[str] | None
  - exclude_cols: list[str]
  - use_original_features_only: bool
  - k: int
  - metric: str
  - fit_on: train|train_val|train_test|train_val_test
  - scale: bool
  - missing_strategy: mean|median|zero
  - include_self: bool
  - add_density: bool
  - prefix: str
  - random_state: int
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from mlarena.preprocessing.utils import (
    validation,
    artifacts,
    dataframe_utils,
    report,
)


def _build_fit_frames(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    fit_on: str,
) -> List[pd.DataFrame]:
    frames = [train_df]
    if fit_on in {"train_val", "train_val_test"} and val_df is not None:
        frames.append(val_df)
    if fit_on in {"train_test", "train_val_test"}:
        frames.append(test_df)
    return frames


def _impute_values(df: pd.DataFrame, cols: List[str], strategy: str) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for col in cols:
        if strategy == "median":
            values[col] = float(df[col].median()) if col in df.columns else 0.0
        elif strategy == "zero":
            values[col] = 0.0
        else:
            values[col] = float(df[col].mean()) if col in df.columns else 0.0
    return values


def _apply_impute(df: pd.DataFrame, cols: List[str], values: Dict[str, float]) -> pd.DataFrame:
    df_out = df.copy()
    for col in cols:
        if col not in df_out.columns:
            df_out[col] = np.nan
        df_out[col] = df_out[col].fillna(values.get(col, 0.0))
    return df_out


def _knn_features(
    nn: NearestNeighbors,
    x: np.ndarray,
    k: int,
    drop_self: bool,
    include_self: bool,
    add_density: bool,
) -> np.ndarray:
    if x.size == 0:
        return np.zeros((0, 0))
    n_neighbors = k + 1 if drop_self and not include_self else k
    distances, _ = nn.kneighbors(x, n_neighbors=n_neighbors, return_distance=True)
    if drop_self and not include_self:
        distances = distances[:, 1:]
    if distances.shape[1] == 0:
        return np.zeros((distances.shape[0], 0))

    mean = distances.mean(axis=1)
    std = distances.std(axis=1)
    dmin = distances.min(axis=1)
    dmax = distances.max(axis=1)
    dk = distances[:, -1]

    feats = [mean, std, dmin, dmax, dk]
    if add_density:
        feats.append(1.0 / (mean + 1e-9))

    return np.vstack(feats).T


def fit_transform(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
    orig_df: pd.DataFrame | None = None,
    eval_df: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, Dict[str, Any]]:
    artifact_dir = Path(config.get("_artifact_dir", "."))
    dataset_config = config.get("_dataset", {})
    id_column = dataset_config.get("id_column", "id")
    target_column = dataset_config.get("target")
    ignored_columns = dataset_config.get("ignored_columns", [])

    required_params: List[str] = []
    optional_params: Dict[str, Any] = {
        "include_cols": None,
        "exclude_cols": [],
        "use_original_features_only": True,
        "k": 5,
        "metric": "euclidean",
        "fit_on": "train",
        "scale": True,
        "missing_strategy": "median",
        "include_self": False,
        "add_density": True,
        "prefix": "knn",
        "random_state": 42,
    }
    validation.validate_config(config, required_params, optional_params)
    validation.validate_choice(config["fit_on"], ["train", "train_val", "train_test", "train_val_test"], "fit_on")
    validation.validate_choice(config["missing_strategy"], ["mean", "median", "zero"], "missing_strategy")

    submodule_dir = artifacts.get_submodule_artifact_dir(artifact_dir, "knn_graph_features")

    train_df_original = dataframe_utils.copy_dataframe(train_df)
    test_df_original = dataframe_utils.copy_dataframe(test_df)

    exclude_cols = [id_column, target_column] + ignored_columns + config["exclude_cols"]
    exclude_cols = [c for c in exclude_cols if c]
    numeric_cols = dataframe_utils.get_numeric_columns(train_df, exclude=exclude_cols)

    use_orig_only = bool(config.get("use_original_features_only"))
    if use_orig_only:
        numeric_cols = dataframe_utils.filter_original_columns(numeric_cols, config.get("_original_features"))

    if config["include_cols"]:
        numeric_cols = [c for c in config["include_cols"] if c in numeric_cols]

    if not numeric_cols:
        state_dict = {
            "version": "1.0",
            "config": {k: v for k, v in config.items() if not k.startswith("_")},
            "message": "No numeric columns available for KNN features.",
            "knn_columns": [],
        }
        return train_df, val_df, test_df, eval_df, orig_df, state_dict

    fit_frames = _build_fit_frames(train_df, val_df, test_df, config["fit_on"])
    fit_df = pd.concat(fit_frames, axis=0, ignore_index=True)

    impute_values = _impute_values(fit_df, numeric_cols, config["missing_strategy"])
    fit_df = _apply_impute(fit_df, numeric_cols, impute_values)

    x_fit = fit_df[numeric_cols].to_numpy(dtype=float)
    if x_fit.shape[0] < 2:
        warnings.warn("Not enough rows for KNN features; skipping.")
        state_dict = {
            "version": "1.0",
            "config": {k: v for k, v in config.items() if not k.startswith("_")},
            "message": "Insufficient rows for KNN features.",
            "knn_columns": [],
        }
        return train_df, val_df, test_df, eval_df, orig_df, state_dict

    scaler = None
    if config["scale"]:
        scaler = StandardScaler()
        x_fit = scaler.fit_transform(x_fit)

    k = int(config["k"])
    if k < 1:
        raise ValueError("k must be >= 1")

    nn = NearestNeighbors(metric=config["metric"])
    nn.fit(x_fit)

    if scaler is not None:
        artifacts.save_fitted_object(scaler, submodule_dir, "scaler.pkl")
    artifacts.save_fitted_object(nn, submodule_dir, "knn.pkl")

    prefix = str(config["prefix"]).strip() or "knn"

    drop_self_train = config["fit_on"] in {"train", "train_val", "train_test", "train_val_test"}
    drop_self_val = config["fit_on"] in {"train_val", "train_val_test"} and val_df is not None
    drop_self_test = config["fit_on"] in {"train_test", "train_val_test"}

    def _transform_df(df: pd.DataFrame | None, drop_self: bool) -> pd.DataFrame | None:
        if df is None:
            return None
        df_out = df.copy()
        df_out = _apply_impute(df_out, numeric_cols, impute_values)
        x = df_out[numeric_cols].to_numpy(dtype=float)
        if scaler is not None:
            x = scaler.transform(x)
        k_eff = min(k, max(1, nn.n_samples_fit_ - (1 if drop_self and not config["include_self"] else 0)))
        features = _knn_features(nn, x, k_eff, drop_self, config["include_self"], config["add_density"])
        if features.size == 0:
            return df_out
        col_names = [
            f"{prefix}_dist_mean",
            f"{prefix}_dist_std",
            f"{prefix}_dist_min",
            f"{prefix}_dist_max",
            f"{prefix}_dist_k",
        ]
        if config["add_density"]:
            col_names.append(f"{prefix}_density")
        feat_df = pd.DataFrame(features, columns=col_names, index=df_out.index)
        df_out = pd.concat([df_out, feat_df], axis=1)
        return df_out

    train_df = _transform_df(train_df, drop_self_train)
    val_df = _transform_df(val_df, drop_self_val)
    test_df = _transform_df(test_df, drop_self_test)
    eval_df = _transform_df(eval_df, False)
    orig_df = _transform_df(orig_df, False)

    transformation_summary = report.create_preprocessing_report(
        train_df_original,
        train_df,
        test_df_original,
        test_df,
        config,
    )
    artifacts.save_report(transformation_summary, submodule_dir, "summary.json")

    knn_cols = [c for c in train_df.columns if c.startswith(f"{prefix}_")]
    state_dict = {
        "version": "1.0",
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
        "knn_columns": knn_cols,
        "numeric_source_columns": numeric_cols,
        "model_path": str((submodule_dir / "knn.pkl").relative_to(artifact_dir)),
    }
    if scaler is not None:
        state_dict["scaler_path"] = str((submodule_dir / "scaler.pkl").relative_to(artifact_dir))
    state_dict["impute_values"] = impute_values

    return train_df, val_df, test_df, eval_df, orig_df, state_dict
