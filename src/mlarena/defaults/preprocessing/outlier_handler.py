"""
Outlier Handler Sub-Module

Purpose: Detect and handle outliers in numeric features using configurable strategies.
Libraries: numpy, pandas, sklearn (IsolationForest)
Parameters:
  - outlier_method: none|quantile|percentile|iqr|zscore|gaussian|mad|isolation_forest
  - lower_quantile / upper_quantile: bounds for quantile method
  - iqr_factor: multiplier for IQR bounds
  - zscore_threshold: absolute z-score cutoff
  - mad_threshold: absolute MAD cutoff
  - mad_scale: scale factor applied to MAD
  - isoforest_contamination: contamination rate for IsolationForest
  - action: clip|set_na|flag_only|trim
  - include_cols / exclude_cols: column selection
  - random_state: int
  - use_original_features_only: bool
"""

from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from mlarena.preprocessing.utils import (
    validation,
    artifacts,
    dataframe_utils,
    report,
)


def _compute_bounds_quantile(
    series: pd.Series, lower_q: float, upper_q: float
) -> Tuple[float, float]:
    if lower_q is None or upper_q is None:
        raise ValueError(
            "Quantile method requires both lower_quantile and upper_quantile."
        )
    lower = float(series.quantile(lower_q))
    upper = float(series.quantile(upper_q))
    return lower, upper


def _compute_bounds_iqr(series: pd.Series, factor: float) -> Tuple[float, float]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = float(q1 - factor * iqr)
    upper = float(q3 + factor * iqr)
    return lower, upper


def _compute_bounds_zscore(series: pd.Series, threshold: float) -> Tuple[float, float]:
    mean = series.mean()
    std = series.std()
    if std == 0 or np.isnan(std):
        return float(mean), float(mean)
    lower = float(mean - threshold * std)
    upper = float(mean + threshold * std)
    return lower, upper


def _compute_bounds_mad(
    series: pd.Series, threshold: float, scale: float
) -> Tuple[float, float]:
    median = series.median()
    mad = np.median(np.abs(series - median))
    scaled_mad = mad * scale
    if scaled_mad == 0 or np.isnan(scaled_mad):
        return float(median), float(median)
    lower = float(median - threshold * scaled_mad)
    upper = float(median + threshold * scaled_mad)
    return lower, upper


def _apply_bounds_action(
    series: pd.Series,
    lower: float | None,
    upper: float | None,
    action: str,
    flag_col: str | None = None,
    target_df: pd.DataFrame | None = None,
) -> Tuple[pd.Series, pd.Series | None]:
    if lower is None or upper is None:
        mask = pd.Series(False, index=series.index)
    else:
        mask = (series < lower) | (series > upper)

    if action == "flag_only" and target_df is not None and flag_col:
        target_df[flag_col] = mask.astype(int)
        return series, mask

    if action == "trim":
        return series, mask

    if action == "set_na":
        series = series.mask(mask, np.nan)
    elif action == "clip":
        if lower is not None:
            series = series.mask(series < lower, lower)
        if upper is not None:
            series = series.mask(series > upper, upper)
    return series, mask


def _apply_isolation_forest(
    train_series: pd.Series,
    val_series: pd.Series | None,
    test_series: pd.Series,
    contamination: float,
    random_state: int,
    action: str,
    flag_col: str | None,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    orig_series: pd.Series | None = None,
    orig_df: pd.DataFrame | None = None,
) -> Tuple[pd.Series, pd.Series | None, pd.Series, pd.Series | None, Dict[str, Any]]:
    non_null = train_series.dropna()
    if non_null.empty:
        return (
            train_series,
            val_series,
            test_series,
            orig_series,
            {
                "skipped": True,
                "reason": "all_nan",
            },
        )

    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(non_null.values.reshape(-1, 1))

    def predict_mask(series: pd.Series) -> pd.Series:
        mask = pd.Series(False, index=series.index)
        non_null_mask = series.notnull()
        if non_null_mask.any():
            preds = model.predict(series[non_null_mask].values.reshape(-1, 1))
            mask.loc[non_null_mask] = preds == -1
        return mask

    train_mask = predict_mask(train_series)
    val_mask = predict_mask(val_series) if val_series is not None else None
    test_mask = predict_mask(test_series)
    orig_mask = predict_mask(orig_series) if orig_series is not None else None

    if action == "flag_only":
        if flag_col:
            train_df[flag_col] = train_mask.astype(int)
            test_df[flag_col] = test_mask.astype(int)
            if val_df is not None and val_mask is not None:
                val_df[flag_col] = val_mask.astype(int)
            if orig_df is not None and orig_mask is not None:
                orig_df[flag_col] = orig_mask.astype(int)
        return (
            train_series,
            val_series,
            test_series,
            orig_series,
            {
                "skipped": False,
                "bounds": None,
            },
        )

    if action == "set_na":
        train_series = train_series.mask(train_mask, np.nan)
        if val_series is not None and val_mask is not None:
            val_series = val_series.mask(val_mask, np.nan)
        test_series = test_series.mask(test_mask, np.nan)
        if orig_series is not None and orig_mask is not None:
            orig_series = orig_series.mask(orig_mask, np.nan)
    elif action == "clip":
        median_train = train_series[~train_mask].median()
        train_series = train_series.mask(train_mask, median_train)
        if val_series is not None and val_mask is not None:
            val_series = val_series.mask(val_mask, median_train)
        test_series = test_series.mask(test_mask, median_train)
        if orig_series is not None and orig_mask is not None:
            orig_series = orig_series.mask(orig_mask, median_train)

    return (
        train_series,
        val_series,
        test_series,
        orig_series,
        {
            "skipped": False,
            "bounds": None,
        },
    )


def fit_transform(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
    orig_df: pd.DataFrame | None = None,
) -> Tuple[
    pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None, Dict[str, Any]
]:
    """
    Outlier handling preprocessing.

    Args:
        train_df: Training data
        val_df: Validation data (can be None)
        test_df: Test data
        config: Configuration dictionary with keys:
            - _artifact_dir: Path to save artifacts
            - _dataset: {id_column, target, ignored_columns}
            - outlier_method: none|quantile|iqr|zscore|isolation_forest
            - lower_quantile: float (for quantile method)
            - upper_quantile: float (for quantile method)
            - iqr_factor: float (for iqr method)
            - zscore_threshold: float (for zscore method)
            - isoforest_contamination: float (for isolation_forest)
            - action: clip|set_na|flag_only
            - include_cols: List[str] or None
            - exclude_cols: List[str]
            - random_state: int
        orig_df: External dataset (can be None)

    Returns:
        Tuple of (train_df, val_df, test_df, orig_df, state_dict)
    """
    # 1. Extract config
    artifact_dir = Path(config.get("_artifact_dir", "."))
    dataset_config = config.get("_dataset", {})
    id_column = dataset_config.get("id_column", "id")
    target_column = dataset_config.get("target")
    ignored_columns = dataset_config.get("ignored_columns", [])

    # 2. Validate config
    required_params: List[str] = []
    optional_params: Dict[str, Any] = {
        "outlier_method": "iqr",
        "lower_quantile": 0.01,
        "upper_quantile": 0.99,
        "iqr_factor": 1.5,
        "zscore_threshold": 3.0,
        "mad_threshold": 3.5,
        "mad_scale": 1.4826,
        "isoforest_contamination": 0.05,
        "action": "clip",
        "include_cols": None,
        "exclude_cols": [],
        "random_state": 42,
        "use_original_features_only": True,
    }
    validation.validate_config(config, required_params, optional_params)

    validation.validate_choice(
        config["outlier_method"],
        [
            "none",
            "quantile",
            "percentile",
            "iqr",
            "zscore",
            "gaussian",
            "mad",
            "isolation_forest",
        ],
        "outlier_method",
    )
    validation.validate_choice(
        config["action"],
        ["clip", "set_na", "flag_only", "trim"],
        "action",
    )
    if config["lower_quantile"] is not None:
        validation.validate_numeric_range(
            config["lower_quantile"], 0.0, 1.0, "lower_quantile"
        )
    if config["upper_quantile"] is not None:
        validation.validate_numeric_range(
            config["upper_quantile"], 0.0, 1.0, "upper_quantile"
        )
    if config["upper_quantile"] is not None and config["lower_quantile"] is not None:
        if config["upper_quantile"] <= config["lower_quantile"]:
            raise ValueError("upper_quantile must be greater than lower_quantile.")

    validation.validate_numeric_range(
        config["iqr_factor"], min_value=0.0, max_value=None, param_name="iqr_factor"
    )
    validation.validate_numeric_range(
        config["zscore_threshold"],
        min_value=0.0,
        max_value=None,
        param_name="zscore_threshold",
    )
    validation.validate_numeric_range(
        config["mad_threshold"],
        min_value=0.0,
        max_value=None,
        param_name="mad_threshold",
    )
    validation.validate_numeric_range(
        config["mad_scale"], min_value=0.0, max_value=None, param_name="mad_scale"
    )
    validation.validate_numeric_range(
        config["isoforest_contamination"],
        min_value=0.0,
        max_value=0.5,
        param_name="isoforest_contamination",
    )

    # 3. Create sub-module artifact directory
    submodule_dir = artifacts.get_submodule_artifact_dir(
        artifact_dir, "outlier_handler"
    )

    # 4. Save original DataFrames for reporting
    train_df_original = dataframe_utils.copy_dataframe(train_df)
    test_df_original = dataframe_utils.copy_dataframe(test_df)

    # 5. Identify numeric columns
    exclude_cols = [id_column, target_column] + ignored_columns + config["exclude_cols"]
    if config["include_cols"] is not None:
        numeric_cols = [
            col for col in config["include_cols"] if col in train_df.columns
        ]
    else:
        numeric_cols = dataframe_utils.get_numeric_columns(
            train_df, exclude=exclude_cols
        )

    numeric_cols = [col for col in numeric_cols if col in test_df.columns]
    use_orig_only = bool(config.get("use_original_features_only"))
    if use_orig_only:
        orig_features = config.get("_original_features")
        numeric_cols = dataframe_utils.filter_original_columns(
            numeric_cols, orig_features
        )

    if not numeric_cols or config["outlier_method"] == "none":
        transformation_summary = report.create_preprocessing_report(
            train_before=train_df_original,
            train_after=train_df,
            test_before=test_df_original,
            test_after=test_df,
            config=config,
        )
        artifacts.save_report(transformation_summary, submodule_dir, "summary.json")
        return (
            train_df,
            val_df,
            test_df,
            {
                "version": "1.0",
                "method": config["outlier_method"],
                "message": "No numeric columns to process or method=none",
                "config": {k: v for k, v in config.items() if not k.startswith("_")},
            },
        )

    outlier_stats: Dict[str, Dict[str, Any]] = {}
    flag_columns: List[str] = []
    trim_masks = {
        "train": pd.Series(False, index=train_df.index),
        "val": pd.Series(False, index=val_df.index) if val_df is not None else None,
        "test": pd.Series(False, index=test_df.index),
        "orig": pd.Series(False, index=orig_df.index) if orig_df is not None else None,
    }

    # 6. Process columns
    for col in numeric_cols:
        method = config["outlier_method"]
        action = config["action"]
        lower = upper = None
        column_detail: Dict[str, Any] = {
            "method": method,
            "action": action,
        }

        if method in ["quantile", "percentile"]:
            lower, upper = _compute_bounds_quantile(
                train_df[col], config["lower_quantile"], config["upper_quantile"]
            )
        elif method == "iqr":
            lower, upper = _compute_bounds_iqr(train_df[col], config["iqr_factor"])
        elif method in ["zscore", "gaussian"]:
            lower, upper = _compute_bounds_zscore(
                train_df[col], config["zscore_threshold"]
            )
        elif method == "mad":
            lower, upper = _compute_bounds_mad(
                train_df[col], config["mad_threshold"], config["mad_scale"]
            )

        if method in ["quantile", "percentile", "iqr", "zscore", "gaussian", "mad"]:
            flag_col = f"{col}_outlier_flag" if action == "flag_only" else None
            # Train
            train_series, train_mask = _apply_bounds_action(
                train_df[col],
                lower,
                upper,
                action,
                flag_col,
                train_df if flag_col else None,
            )
            train_df[col] = train_series
            if action == "trim" and train_mask is not None:
                trim_masks["train"] = trim_masks["train"] | train_mask
            # Val
            if val_df is not None:
                val_series, val_mask = _apply_bounds_action(
                    val_df[col]
                    if col in val_df.columns
                    else pd.Series([], dtype=float),
                    lower,
                    upper,
                    action,
                    flag_col,
                    val_df if flag_col and col in val_df.columns else None,
                )
                if col in val_df.columns:
                    val_df[col] = val_series
                if (
                    action == "trim"
                    and val_mask is not None
                    and trim_masks["val"] is not None
                    and col in val_df.columns
                ):
                    trim_masks["val"] = trim_masks["val"] | val_mask
            else:
                val_mask = None
            # Test
            test_series, test_mask = _apply_bounds_action(
                test_df[col],
                lower,
                upper,
                action,
                flag_col,
                test_df if flag_col else None,
            )
            test_df[col] = test_series
            if action == "trim" and test_mask is not None:
                trim_masks["test"] = trim_masks["test"] | test_mask
            # Orig
            if orig_df is not None:
                orig_series, orig_mask = _apply_bounds_action(
                    orig_df[col]
                    if col in orig_df.columns
                    else pd.Series([], dtype=float),
                    lower,
                    upper,
                    action,
                    flag_col,
                    orig_df if flag_col and col in orig_df.columns else None,
                )
                if col in orig_df.columns:
                    orig_df[col] = orig_series
                if (
                    action == "trim"
                    and orig_mask is not None
                    and trim_masks["orig"] is not None
                    and col in orig_df.columns
                ):
                    trim_masks["orig"] = trim_masks["orig"] | orig_mask
            else:
                orig_mask = None

            column_detail.update(
                {
                    "bounds": {"lower": lower, "upper": upper},
                    "train_outliers": int(train_mask.sum())
                    if train_mask is not None
                    else 0,
                    "val_outliers": int(val_mask.sum()) if val_mask is not None else 0,
                    "test_outliers": int(test_mask.sum())
                    if test_mask is not None
                    else 0,
                    "orig_outliers": int(orig_mask.sum())
                    if orig_mask is not None
                    else 0,
                }
            )
            if flag_col:
                flag_columns.append(flag_col)

        elif method == "isolation_forest":
            if action == "trim":
                raise ValueError("action='trim' is not supported for isolation_forest")
            flag_col = f"{col}_outlier_flag" if action == "flag_only" else None
            train_series, val_series, test_series, orig_series, detail = (
                _apply_isolation_forest(
                    train_df[col],
                    val_df[col]
                    if (val_df is not None and col in val_df.columns)
                    else None,
                    test_df[col],
                    contamination=config["isoforest_contamination"],
                    random_state=config["random_state"],
                    action=action,
                    flag_col=flag_col,
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                    orig_series=orig_df[col]
                    if (orig_df is not None and col in orig_df.columns)
                    else None,
                    orig_df=orig_df,
                )
            )
            train_df[col] = train_series
            if val_df is not None and val_series is not None and col in val_df.columns:
                val_df[col] = val_series
            test_df[col] = test_series
            if (
                orig_df is not None
                and orig_series is not None
                and col in orig_df.columns
            ):
                orig_df[col] = orig_series

            column_detail.update(
                {
                    "bounds": detail.get("bounds"),
                    "train_outliers": int(train_df[col].isna().sum())
                    if action == "set_na"
                    else None,
                    "note": "IsolationForest applied",
                }
            )
            if flag_col:
                flag_columns.append(flag_col)

        outlier_stats[col] = column_detail

    # 7. Reports
    trimmed_rows = {}
    if config["action"] == "trim":
        trimmed_rows["train"] = int(trim_masks["train"].sum())
        train_df = train_df.loc[~trim_masks["train"]].copy()
        if val_df is not None and trim_masks["val"] is not None:
            trimmed_rows["val"] = int(trim_masks["val"].sum())
            val_df = val_df.loc[~trim_masks["val"]].copy()
        if orig_df is not None and trim_masks["orig"] is not None:
            trimmed_rows["orig"] = int(trim_masks["orig"].sum())
            orig_df = orig_df.loc[~trim_masks["orig"]].copy()
        trimmed_rows["test"] = int(trim_masks["test"].sum())

    outlier_report = {
        "method": config["outlier_method"],
        "action": config["action"],
        "columns_processed": numeric_cols,
        "flag_columns": flag_columns,
        "stats": outlier_stats,
        "trimmed_rows": trimmed_rows,
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
    }
    artifacts.save_report(outlier_report, submodule_dir, "outlier_report.json")

    transformation_summary = report.create_preprocessing_report(
        train_before=train_df_original,
        train_after=train_df,
        test_before=test_df_original,
        test_after=test_df,
        config=config,
    )
    artifacts.save_report(transformation_summary, submodule_dir, "summary.json")

    # 8. State dict
    state_dict = {
        "version": "1.0",
        "method": config["outlier_method"],
        "action": config["action"],
        "columns_processed": numeric_cols,
        "flag_columns": flag_columns,
        "stats": outlier_stats,
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
    }

    return train_df, val_df, test_df, orig_df, state_dict
