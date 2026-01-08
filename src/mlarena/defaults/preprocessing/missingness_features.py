"""
Missingness Features Sub-Module

Purpose: Create features capturing missing value patterns (row stats, per-column indicators).
Libraries: pandas, numpy
Parameters:
  - include_cols: List of columns to check (default: all)
  - exclude_cols: List of columns to skip
  - add_per_column_indicators: Bool, add {col}_na flag for columns with missing values
  - add_row_missing_count: Bool, add sum of missing values per row
  - add_row_missing_ratio: Bool, add ratio of missing values per row
  - cap_row_missing_count: Int, cap the row count feature (outlier protection)
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import warnings

import pandas as pd
import numpy as np

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
        "add_per_column_indicators": True,
        "add_row_missing_count": True,
        "add_row_missing_ratio": False,
        "cap_row_missing_count": None,
    }
    validation.validate_config(config, required_params, optional_params)

    # 2. Submodule dir
    submodule_dir = artifacts.get_submodule_artifact_dir(artifact_dir, "missingness_features")
    train_df_original = dataframe_utils.copy_dataframe(train_df)
    test_df_original = dataframe_utils.copy_dataframe(test_df)

    # 3. Determine columns
    exclude_cols = [id_column, target_column] + ignored_columns + config["exclude_cols"]
    exclude_cols = [c for c in exclude_cols if c]
    
    if config["include_cols"]:
        cols_to_check = [c for c in config["include_cols"] if c in train_df.columns and c not in exclude_cols]
    else:
        cols_to_check = [c for c in train_df.columns if c not in exclude_cols]

    if not cols_to_check:
        warnings.warn("No columns selected for missingness features.")
        state_dict = {
            "version": "1.0",
            "new_features": [],
            "config": {k: v for k, v in config.items() if not k.startswith("_")},
            "message": "No columns to check"
        }
        return train_df, val_df, test_df, orig_df, state_dict

    new_cols = []
    
    # Helper to apply transformations
    def process_df(df, is_train=False):
        if df is None: return None
        df_out = df.copy() # We are adding columns
        
        # Row stats
        if config["add_row_missing_count"] or config["add_row_missing_ratio"]:
            # Compute missing count row-wise for selected columns
            missing_count = df[cols_to_check].isnull().sum(axis=1)
            
            if config["cap_row_missing_count"]:
                missing_count = missing_count.clip(upper=config["cap_row_missing_count"])
            
            if config["add_row_missing_count"]:
                name = "row_missing_count"
                df_out[name] = missing_count
                if is_train: new_cols.append(name)
                
            if config["add_row_missing_ratio"]:
                name = "row_missing_ratio"
                df_out[name] = missing_count / len(cols_to_check)
                if is_train: new_cols.append(name)

        # Column indicators
        if config["add_per_column_indicators"]:
            # Only add indicators for columns that actually have missing values in TRAIN
            # (or should we check check dataset-specific? Usually consistent with train)
            # Let's check which columns have missing in the specific DF to be safe/accurate per row
            # But feature set consistency is key. We should decide based on Train missingness.
            
            # Better approach: Iterate columns, check if missing exists in TRAIN. If so, add indicator to ALL.
            pass 
            
        return df_out

    # Determine which columns have missing values in TRAIN
    cols_with_missing_train = []
    if config["add_per_column_indicators"]:
        missing_series = train_df[cols_to_check].isnull().any()
        cols_with_missing_train = missing_series[missing_series].index.tolist()

    def apply_indicators(df, is_train=False):
        if df is None: return None
        for col in cols_with_missing_train:
            name = f"{col}_na"
            df[name] = df[col].isnull().astype(int)
            if is_train:
                if name not in new_cols: new_cols.append(name)
        return df

    # Apply row stats
    train_df = process_df(train_df, is_train=True)
    test_df = process_df(test_df)
    val_df = process_df(val_df)
    orig_df = process_df(orig_df)

    # Apply column indicators
    if config["add_per_column_indicators"] and cols_with_missing_train:
        train_df = apply_indicators(train_df, is_train=True)
        test_df = apply_indicators(test_df)
        val_df = apply_indicators(val_df)
        orig_df = apply_indicators(orig_df)

    # 4. Reports
    report_data = {
        "cols_checked": len(cols_to_check),
        "cols_with_missing_train": len(cols_with_missing_train),
        "new_features": new_cols,
        "config": {k: v for k, v in config.items() if not k.startswith("_")}
    }
    artifacts.save_report(report_data, submodule_dir, "missingness_report.json")
    
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
        "new_features": new_cols,
        "config": report_data["config"]
    }

    return train_df, val_df, test_df, orig_df, state_dict
