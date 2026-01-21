"""
MixUp Augmentation Sub-Module

Purpose: Augment training data by mixing pairs of samples.
Libraries: pandas, numpy
Parameters:
  - include_cols: list[str] | None
  - exclude_cols: list[str]
  - use_original_features_only: bool
  - augment_ratio: float
  - alpha: float (Beta distribution)
  - lambda_clip: float | None
  - allow_soft_labels: bool
  - hard_label_threshold: float
  - random_state: int
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd

from mlarena.preprocessing.utils import (
    validation,
    artifacts,
    dataframe_utils,
    report,
)


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
        "augment_ratio": 0.3,
        "alpha": 0.2,
        "lambda_clip": None,
        "allow_soft_labels": False,
        "hard_label_threshold": 0.5,
        "random_state": 42,
    }
    validation.validate_config(config, required_params, optional_params)

    submodule_dir = artifacts.get_submodule_artifact_dir(
        artifact_dir, "mixup_augmentation"
    )

    train_df_original = dataframe_utils.copy_dataframe(train_df)
    test_df_original = dataframe_utils.copy_dataframe(test_df)

    if not target_column or target_column not in train_df.columns:
        raise ValueError("Target column is required for mixup augmentation.")

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
            "message": "No numeric columns available for mixup.",
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

    alpha = float(config["alpha"])
    if alpha <= 0:
        raise ValueError("alpha must be > 0")

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

    idx_a = rng.integers(0, n_rows, size=n_new)
    idx_b = rng.integers(0, n_rows, size=n_new)

    lam = rng.beta(alpha, alpha, size=n_new)
    if config["lambda_clip"] is not None:
        clip_val = float(config["lambda_clip"])
        if clip_val < 0.0 or clip_val > 0.5:
            raise ValueError("lambda_clip must be between 0 and 0.5")
        lam = np.clip(lam, clip_val, 1.0 - clip_val)

    base_df = train_df.iloc[idx_a].copy()
    # Remove identifier from synthetic rows to avoid duplicated IDs
    # (and leakage via joins / group-splitting keyed by id).
    if id_column in base_df.columns:
        base_df[id_column] = pd.NA

    x_a = train_df.iloc[idx_a][numeric_cols].to_numpy(dtype=float)
    x_b = train_df.iloc[idx_b][numeric_cols].to_numpy(dtype=float)
    mixed = lam[:, None] * x_a + (1.0 - lam)[:, None] * x_b
    base_df[numeric_cols] = mixed

    y_a = train_df.iloc[idx_a][target_column].to_numpy()
    y_b = train_df.iloc[idx_b][target_column].to_numpy()

    if problem_type in {"binary", "multiclass"}:
        if config["allow_soft_labels"]:
            if problem_type == "multiclass":
                warnings.warn(
                    "Soft labels for multiclass are not supported; using hard labels."
                )
                config["allow_soft_labels"] = False
        if config["allow_soft_labels"]:
            if not np.issubdtype(y_a.dtype, np.number):
                warnings.warn("Soft labels require numeric targets; using hard labels.")
                config["allow_soft_labels"] = False
        if config["allow_soft_labels"]:
            y_new = lam * y_a + (1.0 - lam) * y_b
        else:
            threshold = float(config["hard_label_threshold"])
            choose_a = lam >= threshold
            y_new = np.where(choose_a, y_a, y_b)
    else:
        y_new = lam * y_a + (1.0 - lam) * y_b

    base_df[target_column] = y_new

    train_df = pd.concat([train_df, base_df], axis=0)
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
        "alpha": alpha,
        "allow_soft_labels": bool(config["allow_soft_labels"]),
        "numeric_cols": numeric_cols,
    }
    artifacts.save_report(transformation_summary, submodule_dir, "summary.json")

    state_dict = {
        "version": "1.0",
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
        "n_added": n_new,
        "alpha": alpha,
        "allow_soft_labels": bool(config["allow_soft_labels"]),
        "numeric_cols": numeric_cols,
    }

    return train_df, val_df, test_df, eval_df, orig_df, state_dict
