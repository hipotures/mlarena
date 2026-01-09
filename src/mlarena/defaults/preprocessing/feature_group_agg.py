"""
Feature Group Aggregations Sub-Module

Purpose: Create group-based aggregations (groupby + agg + merge).
Libraries: pandas
Parameters:
  - group_keys: Columns to group by for aggregations
  - group_value_cols: Value columns to aggregate
  - aggs: Aggregations to compute (e.g., mean, std, min, max, count, nunique)
  - max_generated_features: Guardrail for total new features created
"""

from typing import Any, Dict, List, Tuple
import warnings
from pathlib import Path

import pandas as pd

from mlarena.preprocessing.utils import (
    validation,
    artifacts,
    dataframe_utils,
    report,
)


def _unique_name(base: str, existing: set) -> str:
    """Generate a unique column name if base already exists."""
    if base not in existing:
        return base
    idx = 1
    candidate = f"{base}__{idx}"
    while candidate in existing:
        idx += 1
        candidate = f"{base}__{idx}"
    return candidate


def _apply_group_aggregations(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    orig_df: pd.DataFrame | None,
    group_keys: List[str],
    value_cols: List[str],
    aggs: List[str],
    remaining_slots: int,
    existing_cols: set,
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None, List[str], Dict[str, Any]]:
    if not group_keys or not value_cols or not aggs:
        return train_df, val_df, test_df, orig_df, [], {}

    missing_keys = [col for col in group_keys if col not in train_df.columns]
    missing_values = [col for col in value_cols if col not in train_df.columns]
    if missing_keys:
        warnings.warn(f"Group keys missing in train: {missing_keys} - skipping aggregations")
        return train_df, val_df, test_df, orig_df, [], {}
    if missing_values:
        warnings.warn(f"Value columns missing in train: {missing_values} - skipping aggregations")
        return train_df, val_df, test_df, orig_df, [], {}

    if any(col not in test_df.columns for col in group_keys):
        warnings.warn("Group keys missing in test data - skipping aggregations")
        return train_df, val_df, test_df, orig_df, [], {}

    agg_df = train_df[group_keys + value_cols].groupby(group_keys).agg(aggs)
    agg_df.columns = [
        f"{'__'.join(group_keys)}__{val_col}__{agg}"
        for val_col, agg in agg_df.columns
    ]
    agg_df = agg_df.reset_index()

    new_columns = [col for col in agg_df.columns if col not in group_keys]
    if remaining_slots is not None and remaining_slots < len(new_columns):
        warnings.warn(
            f"Truncating group aggregation features to {remaining_slots} due to max_generated_features limit"
        )
        new_columns = new_columns[:remaining_slots]
        agg_df = agg_df[group_keys + new_columns]

    # Apply to all datasets
    train_df = train_df.merge(agg_df, on=group_keys, how="left")
    test_df = test_df.merge(agg_df, on=group_keys, how="left")
    if val_df is not None:
        val_df = val_df.merge(agg_df, on=group_keys, how="left")
    if orig_df is not None and all(col in orig_df.columns for col in group_keys):
        orig_df = orig_df.merge(agg_df, on=group_keys, how="left")

    # Ensure unique names against existing columns
    renamed_columns = []
    for col in new_columns:
        unique_name = _unique_name(col, existing_cols.union(renamed_columns))
        if unique_name != col:
            train_df.rename(columns={col: unique_name}, inplace=True)
            test_df.rename(columns={col: unique_name}, inplace=True)
            if val_df is not None:
                val_df.rename(columns={col: unique_name}, inplace=True)
            if orig_df is not None and col in orig_df.columns:
                orig_df.rename(columns={col: unique_name}, inplace=True)
        renamed_columns.append(unique_name)

    details = {
        "type": "group_aggregation",
        "group_keys": group_keys,
        "value_columns": value_cols,
        "aggs": aggs,
        "generated_columns": renamed_columns,
    }

    return train_df, val_df, test_df, orig_df, renamed_columns, details


def fit_transform(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
    orig_df: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None, Dict[str, Any]]:
    """
    Feature group aggregations preprocessing sub-module.
    """
    # 1. Extract config
    artifact_dir = Path(config.get("_artifact_dir", "."))
    dataset_config = config.get("_dataset", {})
    target_column = dataset_config.get("target")

    # 2. Validate config
    required_params: List[str] = []
    optional_params: Dict[str, Any] = {
        "group_keys": [],
        "group_value_cols": [],
        "aggs": [],
        "max_generated_features": 200,
        "use_original_features_only": True,
    }
    validation.validate_config(config, required_params, optional_params)

    validation.validate_numeric_range(
        config["max_generated_features"],
        min_value=1,
        max_value=5000,
        param_name="max_generated_features",
    )

    # 3. Create sub-module artifact directory
    submodule_dir = artifacts.get_submodule_artifact_dir(artifact_dir, "feature_group_agg")

    # 4. Save original DataFrames for reporting
    train_df_original = dataframe_utils.copy_dataframe(train_df)
    test_df_original = dataframe_utils.copy_dataframe(test_df)

    existing_cols = set(train_df.columns)
    new_columns: List[str] = []
    group_details: Dict[str, Any] = {}

    max_new = config["max_generated_features"]

    # 5. Group aggregations
    # Warn if target is used as value column to avoid leakage
    if target_column and target_column in config["group_value_cols"]:
        warnings.warn("Target column included in group_value_cols - this may cause leakage.")

    group_keys = config["group_keys"]
    group_value_cols = config["group_value_cols"]
    if config.get("use_original_features_only"):
        orig_features = config.get("_original_features")
        if orig_features:
            orig_set = set(orig_features)
            group_keys = [c for c in group_keys if c in orig_set]
            group_value_cols = [c for c in group_value_cols if c in orig_set]

    if group_keys and group_value_cols and config["aggs"]:
        train_df, val_df, test_df, orig_df, agg_cols_added, group_details = _apply_group_aggregations(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            orig_df=orig_df,
            group_keys=group_keys,
            value_cols=group_value_cols,
            aggs=config["aggs"],
            remaining_slots=max_new,
            existing_cols=existing_cols,
        )
        new_columns.extend(agg_cols_added)

    # 6. Reports
    feature_report = {
        "new_columns": new_columns,
        "group_aggregations": group_details,
        "total_new_features": len(new_columns),
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
    }
    artifacts.save_report(feature_report, submodule_dir, "feature_group_agg_report.json")

    transformation_summary = report.create_preprocessing_report(
        train_before=train_df_original,
        train_after=train_df,
        test_before=test_df_original,
        test_after=test_df,
        config=config,
    )
    artifacts.save_report(transformation_summary, submodule_dir, "summary.json")

    # 7. State dict
    state_dict = {
        "version": "1.0",
        "new_columns": new_columns,
        "group_aggregations": group_details,
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
    }

    return train_df, val_df, test_df, orig_df, state_dict
