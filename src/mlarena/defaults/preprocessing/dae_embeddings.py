"""
DAE Embeddings Sub-Module

Purpose: Train a denoising autoencoder (swap/gaussian noise) and append hidden embeddings.
Libraries: pandas, numpy, sklearn
Parameters:
  - include_cols: list[str] | None
  - exclude_cols: list[str]
  - use_original_features_only: bool
  - embedding_dim: int
  - hidden_layers: list[int] | None
  - activation: relu|tanh|logistic|identity
  - max_iter: int
  - batch_size: int
  - learning_rate_init: float
  - alpha: float (L2)
  - early_stopping: bool
  - validation_fraction: float
  - random_state: int
  - noise_type: swap|gaussian
  - swap_prob: float
  - gaussian_sigma: float
  - gaussian_scale_by_std: bool
  - fit_on: train|train_val|train_test|train_val_test|all
  - max_rows: int | None
  - scale: bool
  - missing_strategy: mean|median|zero
  - add_original: bool
  - drop_original: bool
  - prefix: str
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from mlarena.preprocessing.utils import (
    validation,
    artifacts,
    dataframe_utils,
    report,
)


def _activation_fn(name: str):
    if name == "relu":
        return lambda x: np.maximum(0.0, x)
    if name == "tanh":
        return np.tanh
    if name == "logistic":
        return lambda x: 1.0 / (1.0 + np.exp(-x))
    return lambda x: x


def _apply_swap_noise(
    x: np.ndarray, swap_prob: float, rng: np.random.Generator
) -> np.ndarray:
    noisy = x.copy()
    n_rows, n_cols = noisy.shape
    if n_rows <= 1:
        return noisy
    for j in range(n_cols):
        mask = rng.random(n_rows) < swap_prob
        if not mask.any():
            continue
        swap_idx = rng.integers(0, n_rows, size=int(mask.sum()))
        noisy[mask, j] = noisy[swap_idx, j]
    return noisy


def _apply_gaussian_noise(
    x: np.ndarray, sigma: float, scales: np.ndarray | None, rng: np.random.Generator
) -> np.ndarray:
    if scales is None:
        noise = rng.normal(0.0, sigma, size=x.shape)
    else:
        noise = rng.normal(0.0, sigma, size=x.shape) * scales
    return x + noise


def _build_fit_frames(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    eval_df: pd.DataFrame | None,
    fit_on: str,
) -> List[pd.DataFrame]:
    frames = [train_df]
    if fit_on in {"train_val", "train_val_test", "all"} and val_df is not None:
        frames.append(val_df)
    if fit_on in {"train_test", "train_val_test", "all"}:
        frames.append(test_df)
    if fit_on == "all" and eval_df is not None:
        frames.append(eval_df)
    return frames


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


def _compute_embeddings(mlp: MLPRegressor, x: np.ndarray) -> np.ndarray:
    activ = _activation_fn(mlp.activation)
    hidden = x
    for i in range(len(mlp.coefs_) - 1):
        hidden = hidden @ mlp.coefs_[i] + mlp.intercepts_[i]
        hidden = activ(hidden)
    return hidden


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

    required_params: List[str] = []
    optional_params: Dict[str, Any] = {
        "include_cols": None,
        "exclude_cols": [],
        "use_original_features_only": True,
        "embedding_dim": 16,
        "hidden_layers": None,
        "activation": "relu",
        "max_iter": 200,
        "batch_size": 256,
        "learning_rate_init": 0.001,
        "alpha": 0.0001,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "random_state": 42,
        "noise_type": "swap",
        "swap_prob": 0.15,
        "gaussian_sigma": 0.01,
        "gaussian_scale_by_std": True,
        "fit_on": "train_test",
        "max_rows": None,
        "scale": True,
        "missing_strategy": "mean",
        "add_original": True,
        "drop_original": False,
        "prefix": "dae",
    }
    validation.validate_config(config, required_params, optional_params)
    validation.validate_choice(
        config["activation"], ["relu", "tanh", "logistic", "identity"], "activation"
    )
    validation.validate_choice(config["noise_type"], ["swap", "gaussian"], "noise_type")
    validation.validate_choice(
        config["fit_on"],
        ["train", "train_val", "train_test", "train_val_test", "all"],
        "fit_on",
    )
    validation.validate_choice(
        config["missing_strategy"], ["mean", "median", "zero"], "missing_strategy"
    )

    submodule_dir = artifacts.get_submodule_artifact_dir(artifact_dir, "dae_embeddings")

    train_df_original = dataframe_utils.copy_dataframe(train_df)
    test_df_original = dataframe_utils.copy_dataframe(test_df)

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
            "message": "No numeric columns available for DAE embeddings.",
            "embedded_columns": [],
        }
        return train_df, val_df, test_df, eval_df, orig_df, state_dict

    fit_frames = _build_fit_frames(train_df, val_df, test_df, eval_df, config["fit_on"])
    fit_df = pd.concat(fit_frames, axis=0, ignore_index=True)

    if config["max_rows"] and len(fit_df) > int(config["max_rows"]):
        fit_df = fit_df.sample(
            n=int(config["max_rows"]), random_state=config["random_state"]
        )

    impute_values = _impute_values(fit_df, numeric_cols, config["missing_strategy"])
    fit_df = _apply_impute(fit_df, numeric_cols, impute_values)

    x_fit = fit_df[numeric_cols].to_numpy(dtype=float)

    scaler = None
    if config["scale"]:
        scaler = StandardScaler()
        x_fit = scaler.fit_transform(x_fit)

    rng = np.random.default_rng(config["random_state"])
    if config["noise_type"] == "swap":
        x_noisy = _apply_swap_noise(x_fit, float(config["swap_prob"]), rng)
    else:
        scales = (
            x_fit.std(axis=0, keepdims=True)
            if config["gaussian_scale_by_std"]
            else None
        )
        x_noisy = _apply_gaussian_noise(
            x_fit, float(config["gaussian_sigma"]), scales, rng
        )

    hidden_layers = config.get("hidden_layers")
    if hidden_layers:
        hidden_sizes = tuple(int(x) for x in hidden_layers)
    else:
        hidden_sizes = (int(config["embedding_dim"]),)

    if any(h <= 0 for h in hidden_sizes):
        raise ValueError("hidden_layers/embedding_dim must be positive")

    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_sizes,
        activation=config["activation"],
        max_iter=int(config["max_iter"]),
        batch_size=int(config["batch_size"]),
        learning_rate_init=float(config["learning_rate_init"]),
        alpha=float(config["alpha"]),
        early_stopping=bool(config["early_stopping"]),
        validation_fraction=float(config["validation_fraction"]),
        random_state=int(config["random_state"]),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mlp.fit(x_noisy, x_fit)

    artifacts.save_fitted_object(mlp, submodule_dir, "dae_model.pkl")
    if scaler is not None:
        artifacts.save_fitted_object(scaler, submodule_dir, "scaler.pkl")

    prefix = str(config["prefix"]).strip() or "dae"
    drop_original = bool(config["drop_original"]) or not bool(config["add_original"])

    def _transform_df(df: pd.DataFrame | None) -> pd.DataFrame | None:
        if df is None:
            return None
        df_out = df.copy()
        df_out = _apply_impute(df_out, numeric_cols, impute_values)
        x = df_out[numeric_cols].to_numpy(dtype=float)
        if scaler is not None:
            x = scaler.transform(x)
        emb = _compute_embeddings(mlp, x)
        emb_cols = [f"{prefix}_{i}" for i in range(emb.shape[1])]
        emb_df = pd.DataFrame(emb, columns=emb_cols, index=df_out.index)
        df_out = pd.concat([df_out, emb_df], axis=1)
        if drop_original:
            df_out = dataframe_utils.safe_drop_columns(df_out, numeric_cols)
        return df_out

    train_df = _transform_df(train_df)
    val_df = _transform_df(val_df)
    test_df = _transform_df(test_df)
    eval_df = _transform_df(eval_df)
    orig_df = _transform_df(orig_df)

    transformation_summary = report.create_preprocessing_report(
        train_df_original,
        train_df,
        test_df_original,
        test_df,
        config,
    )
    artifacts.save_report(transformation_summary, submodule_dir, "summary.json")

    state_dict = {
        "version": "1.0",
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
        "embedded_columns": [
            col for col in train_df.columns if col.startswith(f"{prefix}_")
        ],
        "numeric_source_columns": numeric_cols,
        "model_path": str((submodule_dir / "dae_model.pkl").relative_to(artifact_dir)),
    }
    if scaler is not None:
        state_dict["scaler_path"] = str(
            (submodule_dir / "scaler.pkl").relative_to(artifact_dir)
        )
    state_dict["impute_values"] = impute_values

    return train_df, val_df, test_df, eval_df, orig_df, state_dict
