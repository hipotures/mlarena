"""
Row Aggregates Sub-Module

Purpose: Create row-wise aggregated statistics (mean, sum, std, etc.) for numeric columns.
Libraries: pandas, numpy, scipy.stats
Parameters:
  - include_cols: List[str] (default: all numeric)
  - exclude_cols: List[str]
  - stats: List[str] (sum, mean, std, min, max, range, mad, skew, kurt)
  - prefix: str (default: "row_")
  - nan_policy: "omit" | "fill_zero" (how to handle NaNs in computation)
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import warnings

import pandas as pd
import numpy as np
from scipy import stats as sp_stats

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
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None, Dict[str, Any]]:
    
    # 1. Extract & Validate
    artifact_dir = Path(config.get("_artifact_dir", "."))
    dataset_config = config.get("_dataset", {})
    id_column = dataset_config.get("id_column", "id")
    target_column = dataset_config.get("target")
    ignored_columns = dataset_config.get("ignored_columns", [])

    required_params = []
    optional_params = {
        "include_cols": None,
        "exclude_cols": [],
        "stats": ["mean", "std", "sum"],
        "prefix": "row_",
        "nan_policy": "omit", # omit = ignore NaNs (pandas default), fill_zero = fill 0 before calc
        "use_original_features_only": True,
    }
    validation.validate_config(config, required_params, optional_params)
    
    allowed_stats = ["sum", "mean", "std", "min", "max", "range", "mad", "skew", "kurt"]
    for s in config["stats"]:
        validation.validate_choice(s, allowed_stats, "stats")

    # 2. Submodule dir
    submodule_dir = artifacts.get_submodule_artifact_dir(artifact_dir, "row_aggregates")
    train_df_original = dataframe_utils.copy_dataframe(train_df)
    test_df_original = dataframe_utils.copy_dataframe(test_df)

    # 3. Determine columns
    exclude_cols = [id_column, target_column] + ignored_columns + config["exclude_cols"]
    exclude_cols = [c for c in exclude_cols if c]
    
    all_numeric = dataframe_utils.get_numeric_columns(train_df, exclude=exclude_cols)
    use_orig_only = bool(config.get("use_original_features_only"))
    orig_features = config.get("_original_features") if use_orig_only else None
    
    if config["include_cols"]:
        numeric_cols = [c for c in config["include_cols"] if c in all_numeric]
    else:
        numeric_cols = all_numeric

    if use_orig_only:
        numeric_cols = dataframe_utils.filter_original_columns(numeric_cols, orig_features)

    if not numeric_cols:
        warnings.warn("No numeric columns found for row aggregates.")
        state_dict = {"new_features": [], "config": config}
        return train_df, val_df, test_df, orig_df, state_dict

    new_features = []
    prefix = config["prefix"]
    nan_policy = config["nan_policy"]

    def process_df(df, is_train=False):
        if df is None: return None
        
        subset = df[numeric_cols]
        if nan_policy == "fill_zero":
            subset = subset.fillna(0)
            
        res_df = pd.DataFrame(index=df.index)
        
        # Calculate stats
        requested = set(config["stats"])
        
        if "sum" in requested:
            res_df[f"{prefix}sum"] = subset.sum(axis=1)
        if "mean" in requested:
            res_df[f"{prefix}mean"] = subset.mean(axis=1)
        if "std" in requested:
            res_df[f"{prefix}std"] = subset.std(axis=1)
        if "min" in requested:
            res_df[f"{prefix}min"] = subset.min(axis=1)
        if "max" in requested:
            res_df[f"{prefix}max"] = subset.max(axis=1)
        if "range" in requested:
            res_df[f"{prefix}range"] = subset.max(axis=1) - subset.min(axis=1)
        if "mad" in requested:
            # Median Absolute Deviation. Pandas doesn't have row-wise mad easily since deprecation?
            # Approximation: mean(abs(x - x.mean()))
            mean_val = subset.mean(axis=1)
            mad = subset.sub(mean_val, axis=0).abs().mean(axis=1)
            res_df[f"{prefix}mad"] = mad
        if "skew" in requested:
            res_df[f"{prefix}skew"] = subset.skew(axis=1)
        if "kurt" in requested:
            res_df[f"{prefix}kurt"] = subset.kurt(axis=1)
            
        if is_train:
            new_features.extend(res_df.columns.tolist())
            
        return pd.concat([df, res_df], axis=1)

    train_df = process_df(train_df, is_train=True)
    test_df = process_df(test_df)
    val_df = process_df(val_df)
    orig_df = process_df(orig_df)

    # 4. Reports
    report_data = {
        "input_cols_count": len(numeric_cols),
        "stats_computed": config["stats"],
        "new_features": new_features,
        "config": {k: v for k, v in config.items() if not k.startswith("_")}
    }
    artifacts.save_report(report_data, submodule_dir, "row_aggregates_report.json")
    
    summary = report.create_preprocessing_report(
        train_before=train_df_original,
        train_after=train_df,
        test_before=test_df_original,
        test_after=test_df,
        config=config,
    )
    artifacts.save_report(summary, submodule_dir, "summary.json")

    state_dict = {
        "version": "1.0",
        "new_features": new_features,
        "config": report_data["config"]
    }

    return train_df, val_df, test_df, orig_df, state_dict
