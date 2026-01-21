"""
Noise Injection Sub-Module

Purpose: Augment training data by injecting gaussian or swap noise into numeric features.
Libraries: pandas, numpy
Parameters:
  - include_cols: list[str] | None
  - exclude_cols: list[str]
  - use_original_features_only: bool
  - noise_type: gaussian|swap
  - augment_ratio: float
  - gaussian_sigma: float
  - gaussian_scale_by_std: bool
  - swap_prob: float
  - random_state: int
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from mlarena.preprocessing.utils import (
    validation,
    artifacts,
    dataframe_utils,
    report,
)


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
        "noise_type": "gaussian",
        "augment_ratio": 0.3,
        "gaussian_sigma": 0.01,
        "gaussian_scale_by_std": True,
        "swap_prob": 0.1,
        "random_state": 42,
    }
    validation.validate_config(config, required_params, optional_params)
    validation.validate_choice(config["noise_type"], ["gaussian", "swap"], "noise_type")

    submodule_dir = artifacts.get_submodule_artifact_dir(
        artifact_dir, "noise_injection"
    )

    train_df_original = dataframe_utils.copy_dataframe(train_df)
    test_df_original = dataframe_utils.copy_dataframe(test_df)

    if not target_column or target_column not in train_df.columns:
        raise ValueError("Target column is required for noise augmentation.")

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
            "message": "No numeric columns available for noise injection.",
            "n_added": 0,
        }
        return train_df, val_df, test_df, eval_df, orig_df, state_dict

    augment_ratio = float(config["augment_ratio"])
    if augment_ratio <= 0:
        state_dict = {
            "version": "1.0",
            "config": {k: v for k, v in config.items() if not k.startswith("_")},
            "message": "augment_ratio <= 0; skipping.",
            "n_added": 0,
        }
        return train_df, val_df, test_df, eval_df, orig_df, state_dict

    rng = np.random.default_rng(config["random_state"])
    n_rows = len(train_df)
    n_new = int(n_rows * augment_ratio)
    if n_new <= 0:
        state_dict = {
            "version": "1.0",
            "config": {k: v for k, v in config.items() if not k.startswith("_")},
            "message": "No rows to augment after rounding.",
            "n_added": 0,
        }
        return train_df, val_df, test_df, eval_df, orig_df, state_dict

    sample_idx = rng.integers(0, n_rows, size=n_new)
    aug_df = train_df.iloc[sample_idx].copy()

    x = aug_df[numeric_cols].to_numpy(dtype=float)

    if config["noise_type"] == "gaussian":
        if config["gaussian_scale_by_std"]:
            stds = train_df[numeric_cols].std(axis=0).to_numpy(dtype=float)
            noise = (
                rng.normal(0.0, float(config["gaussian_sigma"]), size=x.shape) * stds
            )
        else:
            noise = rng.normal(0.0, float(config["gaussian_sigma"]), size=x.shape)
        x_noisy = x + noise
    else:
        x_noisy = _apply_swap_noise(x, float(config["swap_prob"]), rng)

    aug_df[numeric_cols] = x_noisy

    train_df = pd.concat([train_df, aug_df], axis=0)
    train_df = train_df.sample(
        frac=1.0, random_state=config["random_state"]
    ).reset_index(drop=True)

    transformation_summary = report.create_preprocessing_report(
        train_df_original,
        train_df,
        test_df_original,
        test_df,
        config,
    )
    transformation_summary["augmentation"] = {
        "n_added": n_new,
        "noise_type": config["noise_type"],
        "numeric_cols": numeric_cols,
    }
    artifacts.save_report(transformation_summary, submodule_dir, "summary.json")

    state_dict = {
        "version": "1.0",
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
        "n_added": n_new,
        "noise_type": config["noise_type"],
        "numeric_cols": numeric_cols,
    }

    return train_df, val_df, test_df, eval_df, orig_df, state_dict
