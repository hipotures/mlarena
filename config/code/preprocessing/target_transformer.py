"""
Target Transformer Sub-Module

Purpose: Apply configurable transformations to the target column for regression tasks
Libraries: numpy, pandas, sklearn.preprocessing.PowerTransformer
Parameters:
  - target_transform: Transformation to apply (none|log1p|boxcox|yeo_johnson)
  - clip_lower_quantile / clip_upper_quantile: Optional quantile clipping before transform
  - shift_before_log: Auto-shift non-positive targets for log/Box-Cox
  - shift_value: Manual shift override (added before log/Box-Cox)
  - standardize: Whether PowerTransformer should standardize output (Box-Cox / Yeo-Johnson)
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer

from mlarena.preprocessing.utils import (
    validation,
    artifacts,
    dataframe_utils,
    report,
)


def _clip_series(series: pd.Series, lower: float | None, upper: float | None) -> pd.Series:
    """Clip a series to provided bounds if given."""
    if lower is not None:
        series = series.clip(lower=lower)
    if upper is not None:
        series = series.clip(upper=upper)
    return series


def fit_transform(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
    orig_df: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None, Dict[str, Any]]:
    """
    Transform target column for regression tasks.

    Args:
        train_df: Training data (must contain target column)
        val_df: Validation data (can be None)
        test_df: Test data (target not required)
        config: Configuration dictionary with keys:
            - _artifact_dir: Path to save artifacts
            - _dataset: {id_column, target, ignored_columns}
            - target_transform: none|log1p|boxcox|yeo_johnson
            - clip_lower_quantile: Optional lower quantile for clipping (0-1)
            - clip_upper_quantile: Optional upper quantile for clipping (0-1)
            - shift_before_log: Auto-shift if data <= 0 (log/Box-Cox)
            - shift_value: Manual shift override (added before log/Box-Cox)
            - standardize: Whether to standardize PowerTransformer output
        orig_df: External dataset (can be None)

    Returns:
        Tuple of (train_df, val_df, test_df, orig_df, state_dict)
    """
    # 1. Extract config
    artifact_dir = Path(config.get("_artifact_dir", "."))
    dataset_config = config.get("_dataset", {})
    target_column = dataset_config.get("target")
    problem_type = dataset_config.get("problem_type", "regression")

    if not target_column:
        raise ValueError("Target column not specified in dataset config")
    if target_column not in train_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in training data")

    # 2. Validate config
    required_params: list[str] = []
    optional_params: Dict[str, Any] = {
        "target_transform": "none",
        "clip_lower_quantile": None,
        "clip_upper_quantile": None,
        "shift_before_log": True,
        "shift_value": None,
        "standardize": True,
    }
    validation.validate_config(config, required_params, optional_params)

    validation.validate_choice(
        config["target_transform"],
        ["none", "log1p", "boxcox", "yeo_johnson"],
        "target_transform",
    )
    for q_key in ["clip_lower_quantile", "clip_upper_quantile"]:
        if config[q_key] is not None:
            validation.validate_numeric_range(
                config[q_key],
                min_value=0.0,
                max_value=1.0,
                param_name=q_key,
            )

    # 3. Create sub-module artifact directory
    submodule_dir = artifacts.get_submodule_artifact_dir(artifact_dir, "target_transformer")

    # 4. Save original DataFrames for reporting
    train_df_original = dataframe_utils.copy_dataframe(train_df)
    test_df_original = dataframe_utils.copy_dataframe(test_df)

    # 5. Prepare target series and clipping
    train_target = train_df[target_column].astype(float)
    val_target = val_df[target_column].astype(float) if val_df is not None and target_column in val_df.columns else None
    orig_target = orig_df[target_column].astype(float) if orig_df is not None and target_column in orig_df.columns else None

    lower_q = config["clip_lower_quantile"]
    upper_q = config["clip_upper_quantile"]
    lower_bound = train_target.quantile(lower_q) if lower_q is not None else None
    upper_bound = train_target.quantile(upper_q) if upper_q is not None else None

    train_target = _clip_series(train_target, lower_bound, upper_bound)
    if val_target is not None:
        val_target = _clip_series(val_target, lower_bound, upper_bound)
    if orig_target is not None:
        orig_target = _clip_series(orig_target, lower_bound, upper_bound)

    # 6. Determine shift for log/Box-Cox
    method = config["target_transform"]
    shift_used = 0.0
    if method in ["log1p", "boxcox"]:
        if config["shift_value"] is not None:
            shift_used = float(config["shift_value"])
        elif config["shift_before_log"]:
            min_val = train_target.min()
            if val_target is not None:
                min_val = min(min_val, val_target.min())
            if orig_target is not None:
                min_val = min(min_val, orig_target.min())
            if min_val <= 0:
                shift_used = abs(min_val) + 1e-6

    # 7. Apply transformation
    transformer_path = None

    if method == "none":
        transformed_train = train_target
        transformed_val = val_target
        transformed_orig = orig_target
    elif method == "log1p":
        transformed_train = np.log1p(train_target + shift_used)
        transformed_val = np.log1p(val_target + shift_used) if val_target is not None else None
        transformed_orig = np.log1p(orig_target + shift_used) if orig_target is not None else None
    elif method in ["boxcox", "yeo_johnson"]:
        adjusted_train = train_target + shift_used if method == "boxcox" else train_target
        adjusted_val = val_target + shift_used if (val_target is not None and method == "boxcox") else val_target
        adjusted_orig = orig_target + shift_used if (orig_target is not None and method == "boxcox") else orig_target

        if method == "boxcox" and (adjusted_train <= 0).any():
            raise ValueError("Box-Cox requires strictly positive data after shift; adjust shift_value/shift_before_log.")

        power_transformer = PowerTransformer(
            method="box-cox" if method == "boxcox" else "yeo-johnson",
            standardize=config["standardize"],
        )

        transformed_train = power_transformer.fit_transform(adjusted_train.values.reshape(-1, 1)).ravel()
        transformed_val = (
            power_transformer.transform(adjusted_val.values.reshape(-1, 1)).ravel()
            if adjusted_val is not None
            else None
        )
        transformed_orig = (
            power_transformer.transform(adjusted_orig.values.reshape(-1, 1)).ravel()
            if adjusted_orig is not None
            else None
        )

        transformer_path = artifacts.save_fitted_object(power_transformer, submodule_dir, "power_transformer.pkl")
    else:
        raise ValueError(f"Unsupported target_transform: {method}")

    # 8. Assign back to DataFrames
    train_df[target_column] = transformed_train
    if val_df is not None and transformed_val is not None:
        val_df[target_column] = transformed_val
    if orig_df is not None and transformed_orig is not None:
        orig_df[target_column] = transformed_orig

    # 9. Reports
    transform_report = {
        "method": method,
        "target_column": target_column,
        "problem_type": problem_type,
        "clip_bounds": {"lower": lower_bound, "upper": upper_bound},
        "shift_used": shift_used if method in ["log1p", "boxcox"] else 0.0,
        "standardize": config["standardize"] if method in ["boxcox", "yeo_johnson"] else None,
        "transformer_path": str(transformer_path.relative_to(artifact_dir)) if transformer_path else None,
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
    }
    artifacts.save_report(transform_report, submodule_dir, "target_transform_report.json")

    transformation_summary = report.create_preprocessing_report(
        train_before=train_df_original,
        train_after=train_df,
        test_before=test_df_original,
        test_after=test_df,
        config=config,
    )
    artifacts.save_report(transformation_summary, submodule_dir, "summary.json")

    # 10. State dict
    state_dict = {
        "version": "1.0",
        "method": method,
        "target_column": target_column,
        "clip_bounds": {"lower": lower_bound, "upper": upper_bound},
        "shift_used": shift_used if method in ["log1p", "boxcox"] else 0.0,
        "standardize": config["standardize"] if method in ["boxcox", "yeo_johnson"] else None,
        "transformer_path": str(transformer_path.relative_to(artifact_dir)) if transformer_path else None,
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
    }

    return train_df, val_df, test_df, orig_df, state_dict
