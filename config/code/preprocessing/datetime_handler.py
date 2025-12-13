"""
Datetime Handler Sub-Module

Purpose: Parse datetime columns, generate derived time features, optional cyclical encodings,
and compute time differences between timestamp columns.

Libraries: pandas, numpy
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, List
import warnings
import math

import numpy as np
import pandas as pd

from mlarena.preprocessing.utils import (
    validation,
    artifacts,
    dataframe_utils,
    report,
)


def _parse_datetime_column(
    df: pd.DataFrame,
    column: str,
    fmt: str | None,
) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype="datetime64[ns]")
    return pd.to_datetime(df[column], format=fmt, errors="coerce")


def _extract_feature(series: pd.Series, feature: str) -> pd.Series:
    if feature == "year":
        return series.dt.year
    if feature == "quarter":
        return series.dt.quarter
    if feature == "month":
        return series.dt.month
    if feature == "weekofyear":
        return series.dt.isocalendar().week.astype(int)
    if feature == "day":
        return series.dt.day
    if feature == "dayofweek":
        return series.dt.dayofweek
    if feature == "dayofyear":
        return series.dt.dayofyear
    if feature == "hour":
        return series.dt.hour
    if feature == "minute":
        return series.dt.minute
    if feature == "second":
        return series.dt.second
    if feature == "is_month_start":
        return series.dt.is_month_start.astype(int)
    if feature == "is_month_end":
        return series.dt.is_month_end.astype(int)
    if feature == "is_quarter_start":
        return series.dt.is_quarter_start.astype(int)
    if feature == "is_quarter_end":
        return series.dt.is_quarter_end.astype(int)
    if feature == "is_year_start":
        return series.dt.is_year_start.astype(int)
    if feature == "is_year_end":
        return series.dt.is_year_end.astype(int)
    raise ValueError(f"Unsupported datetime feature: {feature}")


def _cyclical_encode(series: pd.Series, period: int) -> Tuple[pd.Series, pd.Series]:
    radians = 2 * math.pi * series / period
    return np.sin(radians), np.cos(radians)


def _compute_time_diff(
    start: pd.Series,
    end: pd.Series,
    unit: str,
) -> pd.Series:
    delta = end - start
    seconds = delta.dt.total_seconds()
    if unit == "seconds":
        return seconds
    if unit == "minutes":
        return seconds / 60.0
    if unit == "hours":
        return seconds / 3600.0
    if unit == "days":
        return seconds / 86400.0
    raise ValueError(f"Unsupported time_diff unit: {unit}")


def fit_transform(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
    orig_df: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None, Dict[str, Any]]:
    """
    Datetime preprocessing: parse datetime columns, expand features, add cyclical encodings, compute time diffs.
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
        "datetime_cols": [],
        "datetime_formats": {},
        "expand_datetime_cols": None,
        "time_features_set": "basic",  # basic|extended|none|custom
        "custom_features": [],
        "cyclical_features": [],
        "time_diff_pairs": [],
        "time_diff_default_unit": "days",
        "drop_original_datetime": False,
    }
    validation.validate_config(config, required_params, optional_params)

    validation.validate_choice(
        config["time_features_set"],
        ["basic", "extended", "none", "custom"],
        "time_features_set",
    )
    for feature in config["cyclical_features"]:
        validation.validate_choice(
            feature,
            ["hour", "dayofweek", "month", "weekofyear"],
            "cyclical_features",
        )
    validation.validate_choice(
        config["time_diff_default_unit"],
        ["seconds", "minutes", "hours", "days"],
        "time_diff_default_unit",
    )

    # 3. Create sub-module artifact directory
    submodule_dir = artifacts.get_submodule_artifact_dir(artifact_dir, "datetime_handler")

    # 4. Save originals
    train_df_original = dataframe_utils.copy_dataframe(train_df)
    test_df_original = dataframe_utils.copy_dataframe(test_df)

    # 5. Determine columns to process
    datetime_cols = config["datetime_cols"] or []
    datetime_formats = config.get("datetime_formats", {}) or {}
    expand_cols = config["expand_datetime_cols"] if config["expand_datetime_cols"] is not None else datetime_cols

    # Exclude id/target/ignored
    exclude_cols = [id_column, target_column] + ignored_columns
    datetime_cols = [col for col in datetime_cols if col not in exclude_cols and col in train_df.columns]
    expand_cols = [col for col in expand_cols if col in datetime_cols]

    if not datetime_cols:
        transformation_summary = report.create_preprocessing_report(
            train_before=train_df_original,
            train_after=train_df,
            test_before=test_df_original,
            test_after=test_df,
            config=config,
        )
        artifacts.save_report(transformation_summary, submodule_dir, "summary.json")
        return train_df, val_df, test_df, {
            "version": "1.0",
            "parsed_columns": [],
            "derived_columns": [],
            "cyclical_columns": [],
            "time_diff_columns": [],
            "message": "No datetime columns to process",
            "config": {k: v for k, v in config.items() if not k.startswith("_")},
        }

    # 6. Parse datetime columns
    parsed_columns: List[str] = []
    for col in datetime_cols:
        fmt = datetime_formats.get(col)
        train_df[col] = _parse_datetime_column(train_df, col, fmt)
        if val_df is not None and col in val_df.columns:
            val_df[col] = _parse_datetime_column(val_df, col, fmt)
        if col in test_df.columns:
            test_df[col] = _parse_datetime_column(test_df, col, fmt)
        if orig_df is not None and col in orig_df.columns:
            orig_df[col] = _parse_datetime_column(orig_df, col, fmt)
        parsed_columns.append(col)

    # 7. Determine feature set
    basic_features = ["year", "month", "day", "dayofweek"]
    extended_features = basic_features + [
        "quarter",
        "weekofyear",
        "dayofyear",
        "is_month_start",
        "is_month_end",
        "is_quarter_start",
        "is_quarter_end",
        "is_year_start",
        "is_year_end",
        "hour",
        "minute",
    ]
    if config["time_features_set"] == "basic":
        feature_list = basic_features
    elif config["time_features_set"] == "extended":
        feature_list = extended_features
    elif config["time_features_set"] == "none":
        feature_list = []
    else:  # custom
        feature_list = config["custom_features"] or []

    derived_columns: List[str] = []
    cyclical_columns: List[str] = []

    # 8. Expand datetime features
    for col in expand_cols:
        if not pd.api.types.is_datetime64_any_dtype(train_df[col]):
            warnings.warn(f"Column {col} is not datetime after parsing; skipping feature expansion.")
            continue
        for feat in feature_list:
            new_col = f"{col}_{feat}"
            train_df[new_col] = _extract_feature(train_df[col], feat)
            if val_df is not None and col in val_df.columns:
                val_df[new_col] = _extract_feature(val_df[col], feat)
            if col in test_df.columns:
                test_df[new_col] = _extract_feature(test_df[col], feat)
            if orig_df is not None and col in orig_df.columns:
                orig_df[new_col] = _extract_feature(orig_df[col], feat)
            derived_columns.append(new_col)

        # Cyclical encodings
        for cyc_feat in config["cyclical_features"]:
            source_col = f"{col}_{cyc_feat}"
            if source_col not in train_df.columns:
                continue
            period_map = {
                "hour": 24,
                "dayofweek": 7,
                "month": 12,
                "weekofyear": 52,
            }
            period = period_map[cyc_feat]
            sin_col = f"{source_col}_sin"
            cos_col = f"{source_col}_cos"
            sin_train, cos_train = _cyclical_encode(train_df[source_col], period)
            train_df[sin_col] = sin_train
            train_df[cos_col] = cos_train
            if val_df is not None and source_col in val_df.columns:
                sin_val, cos_val = _cyclical_encode(val_df[source_col], period)
                val_df[sin_col] = sin_val
                val_df[cos_col] = cos_val
            if source_col in test_df.columns:
                sin_test, cos_test = _cyclical_encode(test_df[source_col], period)
                test_df[sin_col] = sin_test
                test_df[cos_col] = cos_test
            if orig_df is not None and source_col in orig_df.columns:
                sin_orig, cos_orig = _cyclical_encode(orig_df[source_col], period)
                orig_df[sin_col] = sin_orig
                orig_df[cos_col] = cos_orig
            cyclical_columns.extend([sin_col, cos_col])

    # 9. Time differences
    time_diff_columns: List[str] = []
    default_unit = config["time_diff_default_unit"]
    for entry in config["time_diff_pairs"]:
        if isinstance(entry, dict):
            start_col = entry.get("start")
            end_col = entry.get("end")
            new_name = entry.get("name")
            unit = entry.get("unit", default_unit)
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            start_col = entry[0]
            end_col = entry[1]
            new_name = entry[2] if len(entry) >= 3 else None
            unit = entry[3] if len(entry) >= 4 else default_unit
        else:
            warnings.warn(f"Invalid time_diff_pairs entry: {entry}")
            continue

        if not start_col or not end_col:
            warnings.warn(f"Skipping time diff pair with missing columns: {entry}")
            continue

        if start_col not in train_df.columns or end_col not in train_df.columns:
            warnings.warn(f"Skipping time diff pair {entry} because columns are missing in train.")
            continue
        new_name = new_name or f"{end_col}_minus_{start_col}_{unit}"

        train_df[new_name] = _compute_time_diff(train_df[start_col], train_df[end_col], unit)
        if val_df is not None and start_col in val_df.columns and end_col in val_df.columns:
            val_df[new_name] = _compute_time_diff(val_df[start_col], val_df[end_col], unit)
        if start_col in test_df.columns and end_col in test_df.columns:
            test_df[new_name] = _compute_time_diff(test_df[start_col], test_df[end_col], unit)
        if orig_df is not None and start_col in orig_df.columns and end_col in orig_df.columns:
            orig_df[new_name] = _compute_time_diff(orig_df[start_col], orig_df[end_col], unit)
        time_diff_columns.append(new_name)

    # 10. Optionally drop original datetime columns
    if config["drop_original_datetime"]:
        train_df = dataframe_utils.safe_drop_columns(train_df, datetime_cols)
        test_df = dataframe_utils.safe_drop_columns(test_df, datetime_cols)
        if val_df is not None:
            val_df = dataframe_utils.safe_drop_columns(val_df, datetime_cols)
        if orig_df is not None:
            orig_df = dataframe_utils.safe_drop_columns(orig_df, datetime_cols)

    # 11. Reports
    datetime_report = {
        "parsed_columns": parsed_columns,
        "derived_columns": derived_columns,
        "cyclical_columns": cyclical_columns,
        "time_diff_columns": time_diff_columns,
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
    }
    artifacts.save_report(datetime_report, submodule_dir, "datetime_report.json")

    transformation_summary = report.create_preprocessing_report(
        train_before=train_df_original,
        train_after=train_df,
        test_before=test_df_original,
        test_after=test_df,
        config=config,
    )
    artifacts.save_report(transformation_summary, submodule_dir, "summary.json")

    # 12. State dict
    state_dict = {
        "version": "1.0",
        "parsed_columns": parsed_columns,
        "derived_columns": derived_columns,
        "cyclical_columns": cyclical_columns,
        "time_diff_columns": time_diff_columns,
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
    }

    return train_df, val_df, test_df, orig_df, state_dict
