"""
Imbalance Handler Sub-Module

Purpose: Handle class imbalance via class weights or simple resampling
Libraries: pandas, numpy
Parameters:
  - imbalance_method: none|class_weight|random_over|random_under|smote|smotenc|adasyn
  - sampling_strategy: auto (currently only auto supported)
  - use_sample_weights: bool (for class_weight)
  - categorical_features: list[str] (for SMOTENC)
  - random_state: int
"""

from pathlib import Path
from typing import Any, Dict, Tuple, List
import warnings

import numpy as np
import pandas as pd

from mlarena.preprocessing.utils import (
    validation,
    artifacts,
    dataframe_utils,
    report,
)


def _compute_class_weights(target: pd.Series) -> Dict[Any, float]:
    """Compute balanced class weights."""
    counts = target.value_counts()
    n_classes = len(counts)
    total = len(target)
    weights = {}
    for cls, cnt in counts.items():
        if cnt == 0:
            weights[cls] = 0.0
        else:
            weights[cls] = total / (n_classes * cnt)
    return weights


def _apply_random_over(train_df: pd.DataFrame, target_col: str, random_state: int) -> pd.DataFrame:
    counts = train_df[target_col].value_counts()
    max_count = counts.max()
    resampled = []
    rng = np.random.default_rng(random_state)
    for cls, cnt in counts.items():
        cls_df = train_df[train_df[target_col] == cls]
        if cnt < max_count:
            extra = cls_df.sample(
                n=max_count - cnt,
                replace=True,
                random_state=random_state
            )
            cls_df = pd.concat([cls_df, extra], axis=0)
        resampled.append(cls_df)
    return pd.concat(resampled, axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)


def _apply_random_under(train_df: pd.DataFrame, target_col: str, random_state: int) -> pd.DataFrame:
    counts = train_df[target_col].value_counts()
    min_count = counts.min()
    resampled = []
    for cls, cnt in counts.items():
        cls_df = train_df[train_df[target_col] == cls]
        if cnt > min_count:
            cls_df = cls_df.sample(
                n=min_count,
                replace=False,
                random_state=random_state
            )
        resampled.append(cls_df)
    return pd.concat(resampled, axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)


def fit_transform(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
    orig_df: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None, Dict[str, Any]]:
    """
    Handle class imbalance for classification tasks.

    Adds sample weights or resamples the training data; leaves test unchanged.

    Args:
        train_df: Training data
        val_df: Validation data (can be None)
        test_df: Test data
        config: Configuration dictionary with keys:
            - _artifact_dir: Path to save artifacts
            - _dataset: {id_column, target, ignored_columns, problem_type}
            - imbalance_method: none|class_weight|random_over|random_under|smote|smotenc|adasyn
            - sampling_strategy: "auto" (only supported value)
            - use_sample_weights: bool (for class_weight)
            - random_state: int
        orig_df: External dataset (can be None) - passed through unchanged

    Returns:
        Tuple of (train_df, val_df, test_df, orig_df, state_dict)
    """
    # 1. Extract config
    artifact_dir = Path(config.get("_artifact_dir", "."))
    dataset_config = config.get("_dataset", {})
    target_column = dataset_config.get("target")
    problem_type = dataset_config.get("problem_type", "binary")

    if not target_column:
        raise ValueError("Target column not specified in dataset config")
    if target_column not in train_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in training data")
    if train_df[target_column].isnull().any():
        raise ValueError("Target column contains NaN. Impute or drop before imbalance handling.")

    # 2. Validate config
    required_params: List[str] = []
    optional_params: Dict[str, Any] = {
        "imbalance_method": "none",
        "sampling_strategy": "auto",
        "use_sample_weights": True,
        "categorical_features": [],
        "random_state": 42,
    }
    validation.validate_config(config, required_params, optional_params)

    validation.validate_choice(
        config["imbalance_method"],
        ["none", "class_weight", "random_over", "random_under", "smote", "smotenc", "adasyn"],
        "imbalance_method",
    )
    if config["sampling_strategy"] != "auto":
        raise ValueError("Only sampling_strategy='auto' is supported in imbalance_handler.")

    if problem_type not in ["binary", "multiclass"]:
        warnings.warn(f"Imbalance handler is intended for classification; detected problem_type={problem_type}. Skipping.")
        return train_df, val_df, test_df, orig_df, {
            "version": "1.0",
            "method": "none",
            "message": f"Skipped (problem_type={problem_type})",
            "config": {k: v for k, v in config.items() if not k.startswith("_")},
        }

    # 3. Create sub-module artifact directory
    submodule_dir = artifacts.get_submodule_artifact_dir(artifact_dir, "imbalance_handler")

    # 4. Save original DataFrames for reporting
    train_df_original = dataframe_utils.copy_dataframe(train_df)
    test_df_original = dataframe_utils.copy_dataframe(test_df)

    class_counts_before = train_df[target_column].value_counts().to_dict()

    method = config["imbalance_method"]
    random_state = config["random_state"]
    class_weights = None
    sample_weight_col = None

    # 5. Apply method
    if method == "none":
        # No changes
        train_df_resampled = train_df
    elif method == "class_weight":
        class_weights = _compute_class_weights(train_df[target_column])
        sample_weight_col = "sample_weight"
        train_df_resampled = train_df.copy()
        train_df_resampled[sample_weight_col] = train_df[target_column].map(class_weights).astype(float)
        if val_df is not None and target_column in val_df.columns and config["use_sample_weights"]:
            val_df = val_df.copy()
            val_df[sample_weight_col] = val_df[target_column].map(class_weights).astype(float)
    elif method == "random_over":
        train_df_resampled = _apply_random_over(train_df, target_column, random_state)
    elif method == "random_under":
        train_df_resampled = _apply_random_under(train_df, target_column, random_state)
    else:
        # smote/smotenc/adasyn
        try:
            from imblearn.over_sampling import SMOTE, SMOTENC, ADASYN
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError(
                f"Imbalance method '{method}' requires imbalanced-learn. "
                "Install it (pip install imbalanced-learn) or use class_weight/random_over/random_under."
            ) from e

        feature_cols = [c for c in train_df.columns if c != target_column]
        X = train_df[feature_cols]
        y = train_df[target_column]

        if method in ["smote", "adasyn"]:
            if not all(pd.api.types.is_numeric_dtype(X[c]) for c in feature_cols):
                raise ValueError(
                    f"{method.upper()} requires numeric features. Run encoders before imbalance_handler "
                    "or switch to class_weight/random sampling."
                )
            sampler = SMOTE(random_state=random_state) if method == "smote" else ADASYN(random_state=random_state)
        else:
            cat_cols = config.get("categorical_features", [])
            if not cat_cols:
                raise ValueError("SMOTENC requires categorical_features to be provided (list of column names).")
            missing = [c for c in cat_cols if c not in feature_cols]
            if missing:
                raise ValueError(f"SMOTENC categorical_features not found: {missing}")
            cat_indices = [feature_cols.index(c) for c in cat_cols]
            sampler = SMOTENC(categorical_features=cat_indices, random_state=random_state)

        X_res, y_res = sampler.fit_resample(X, y)
        train_df_resampled = pd.DataFrame(X_res, columns=feature_cols)
        train_df_resampled[target_column] = y_res

    class_counts_after = train_df_resampled[target_column].value_counts().to_dict()

    # 6. Reports
    imbalance_report = {
        "method": method,
        "class_counts_before": class_counts_before,
        "class_counts_after": class_counts_after,
        "sampling_strategy": config["sampling_strategy"],
        "sample_weight_column": sample_weight_col,
        "class_weights": class_weights,
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
    }
    artifacts.save_report(imbalance_report, submodule_dir, "imbalance_report.json")

    transformation_summary = report.create_preprocessing_report(
        train_before=train_df_original,
        train_after=train_df_resampled,
        test_before=test_df_original,
        test_after=test_df,
        config=config,
    )
    artifacts.save_report(transformation_summary, submodule_dir, "summary.json")

    # 6b. Persist sample weights separately to avoid leaking into features
    weights_path = None
    if sample_weight_col and sample_weight_col in train_df_resampled.columns:
        weights_path = submodule_dir / "sample_weights.csv"
        train_df_resampled[[sample_weight_col]].to_csv(weights_path, index=False, compression='infer')
        train_df_resampled = train_df_resampled.drop(columns=[sample_weight_col])
        if val_df is not None and sample_weight_col in val_df.columns:
            val_df = val_df.drop(columns=[sample_weight_col])

    # 7. State dict
    state_dict = {
        "version": "1.0",
        "method": method,
        "class_counts_before": class_counts_before,
        "class_counts_after": class_counts_after,
        "class_weights": class_weights,
        "sample_weight_column": sample_weight_col,
        "weights_path": str(weights_path) if weights_path else None,
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
    }

    return train_df_resampled, val_df, test_df, orig_df, state_dict
