"""
Sanity Check - Basic data cleaning and type enforcement

Purpose: Ujednolicenie typów i szybkie odfiltrowanie oczywistych problemów zanim ruszy reszta pipeline'u
Libraries: pandas, numpy
Parameters: column_types_override, min_unique_fraction, max_missing_fraction, drop_duplicates, ignore_columns
"""

from pathlib import Path
from typing import Any, Dict, Tuple, List
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
    """
    Sanity check preprocessing - basic cleaning and type enforcement.

    Args:
        train_df: Training data
        val_df: Validation data (can be None)
        test_df: Test data
        config: Configuration dictionary with keys:
            - _artifact_dir: Path to save artifacts
            - _dataset: {id_column, target, ignored_columns}
            - column_types_override: Dict mapping column names to types (e.g., {'col': 'int64'})
            - min_unique_fraction: Minimum fraction of unique values to keep column (default: 0.01)
            - max_missing_fraction: Maximum fraction of missing values to keep column (default: 0.95)
            - drop_duplicates: Whether to drop duplicate rows (default: True)
            - ignore_columns: List of columns to never drop (default: [])
        orig_df: External dataset (can be None)

    Returns:
        Tuple of (train_df, val_df, test_df, orig_df, state_dict)

        state_dict contains:
        - version: str
        - config: Dict
        - issues_found: Dict with detected problems
        - columns_dropped: List[str]
        - duplicates_removed: int
        - types_changed: Dict[str, str]
    """
    # 1. Extract config
    artifact_dir = Path(config.get("_artifact_dir", "."))
    dataset_config = config.get("_dataset", {})
    id_column = dataset_config.get("id_column", "id")
    target_column = dataset_config.get("target")
    ignored_columns = dataset_config.get("ignored_columns", [])

    # 2. Validate config
    required_params = []
    optional_params = {
        "column_types_override": {},
        "min_unique_fraction": 0.01,
        "max_missing_fraction": 0.95,
        "drop_duplicates": True,
        "ignore_columns": [],
    }
    validation.validate_config(config, required_params, optional_params)

    # Validate numeric parameters
    validation.validate_numeric_range(
        config["min_unique_fraction"],
        min_value=0.0,
        max_value=1.0,
        param_name="min_unique_fraction"
    )
    validation.validate_numeric_range(
        config["max_missing_fraction"],
        min_value=0.0,
        max_value=1.0,
        param_name="max_missing_fraction"
    )

    # 3. Create sub-module artifact directory
    submodule_dir = artifacts.get_submodule_artifact_dir(artifact_dir, "sanity_check")

    # 4. Save original DataFrames for reporting
    train_df_original = dataframe_utils.copy_dataframe(train_df)
    test_df_original = dataframe_utils.copy_dataframe(test_df)

    # 5. Protected columns (never drop these)
    protected_cols = [id_column] + ([target_column] if target_column else []) + \
                    ignored_columns + config["ignore_columns"]
    protected_cols = [col for col in protected_cols if col]  # Remove None/empty

    # 6. Perform sanity checks and transformations
    issues_found = {
        "constant_columns": [],
        "high_missing_columns": [],
        "infinite_values": {},
        "duplicate_rows_train": 0,
        "duplicate_rows_test": 0,
        "type_mismatches": [],
    }

    # 6.1. Check for reserved column names (sample weights should be in artifacts, not DataFrame)
    reserved_columns = ["__sample_weight__", "sample_weight"]
    datasets_to_check = [("train", train_df), ("test", test_df)]
    if val_df is not None:
        datasets_to_check.append(("val", val_df))
    if orig_df is not None:
        datasets_to_check.append(("orig", orig_df))

    found_reserved = []
    for df_name, df in datasets_to_check:
        for reserved_col in reserved_columns:
            if reserved_col in df.columns:
                found_reserved.append(f"{df_name}.{reserved_col}")

    if found_reserved:
        raise RuntimeError(
            f"[Sanity Check] Reserved column names found in DataFrames: {', '.join(found_reserved)}\n"
            f"\n"
            f"Sample weights should be returned via artifacts (custom_module_state['weights_path']),\n"
            f"NOT as columns in the DataFrame. This ensures AutoGluon uses them correctly.\n"
            f"\n"
            f"How to fix:\n"
            f"1. Save weights to a separate CSV file in artifacts/preprocess/\n"
            f"2. Return weights_path in state_dict['custom_module_state']\n"
            f"3. Remove the weight column from train_df/test_df\n"
            f"\n"
            f"See: src/mlarena/defaults/preprocessing/adversarial_validation.py for example"
        )

    # 6.2. Check for infinite values
    for df_name, df in datasets_to_check:
        numeric_cols = dataframe_utils.get_numeric_columns(df)
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                issues_found["infinite_values"][f"{df_name}_{col}"] = int(inf_count)
                # Replace inf with NaN
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    # 6.3. Detect constant/nearly-constant columns (in train only)
    constant_cols = []
    for col in train_df.columns:
        if col in protected_cols:
            continue

        unique_fraction = train_df[col].nunique() / len(train_df)
        if unique_fraction < config["min_unique_fraction"]:
            constant_cols.append(col)
            issues_found["constant_columns"].append({
                "column": col,
                "unique_count": int(train_df[col].nunique()),
                "unique_fraction": float(unique_fraction),
            })

    # 6.4. Detect high missing columns (in train only)
    high_missing_cols = []
    for col in train_df.columns:
        if col in protected_cols:
            continue

        missing_fraction = train_df[col].isnull().mean()
        if missing_fraction > config["max_missing_fraction"]:
            high_missing_cols.append(col)
            issues_found["high_missing_columns"].append({
                "column": col,
                "missing_count": int(train_df[col].isnull().sum()),
                "missing_fraction": float(missing_fraction),
            })

    # 6.5. Combine columns to drop
    columns_to_drop = list(set(constant_cols + high_missing_cols))

    # Drop from all DataFrames
    train_df = dataframe_utils.safe_drop_columns(train_df, columns_to_drop)
    test_df = dataframe_utils.safe_drop_columns(test_df, columns_to_drop)
    if val_df is not None:
        val_df = dataframe_utils.safe_drop_columns(val_df, columns_to_drop)
    if orig_df is not None:
        orig_df = dataframe_utils.safe_drop_columns(orig_df, columns_to_drop)

    # 6.6. Drop duplicate rows
    duplicates_removed_train = 0
    duplicates_removed_test = 0
    duplicates_removed_orig = 0

    if config["drop_duplicates"]:
        # For train, keep first occurrence
        duplicates_removed_train = train_df.duplicated().sum()
        if duplicates_removed_train > 0:
            train_df = train_df.drop_duplicates(keep='first')
            issues_found["duplicate_rows_train"] = int(duplicates_removed_train)

        # For test, keep first occurrence
        duplicates_removed_test = test_df.duplicated().sum()
        if duplicates_removed_test > 0:
            test_df = test_df.drop_duplicates(keep='first')
            issues_found["duplicate_rows_test"] = int(duplicates_removed_test)

        # For val, keep first occurrence
        if val_df is not None:
            duplicates_removed_val = val_df.duplicated().sum()
            if duplicates_removed_val > 0:
                val_df = val_df.drop_duplicates(keep='first')

        # For orig, keep first occurrence
        if orig_df is not None:
            duplicates_removed_orig = orig_df.duplicated().sum()
            if duplicates_removed_orig > 0:
                orig_df = orig_df.drop_duplicates(keep='first')
                issues_found["duplicate_rows_orig"] = int(duplicates_removed_orig)

    # 6.7. Enforce column types (if specified)
    types_changed = {}
    for col, dtype in config["column_types_override"].items():
        if col not in train_df.columns:
            continue

        try:
            old_dtype = str(train_df[col].dtype)
            train_df[col] = train_df[col].astype(dtype)
            if col in test_df.columns:
                test_df[col] = test_df[col].astype(dtype)
            if val_df is not None and col in val_df.columns:
                val_df[col] = val_df[col].astype(dtype)
            if orig_df is not None and col in orig_df.columns:
                orig_df[col] = orig_df[col].astype(dtype)

            types_changed[col] = {"from": old_dtype, "to": dtype}
        except Exception as e:
            warnings.warn(f"Could not convert column '{col}' to {dtype}: {e}")
            issues_found["type_mismatches"].append({
                "column": col,
                "target_type": dtype,
                "error": str(e)
            })

    # 7. Reset indices after dropping duplicates
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    if val_df is not None:
        val_df = val_df.reset_index(drop=True)
    if orig_df is not None:
        orig_df = orig_df.reset_index(drop=True)

    # 8. Generate and save detailed report
    sanity_report = {
        "issues_found": issues_found,
        "columns_dropped": columns_to_drop,
        "columns_dropped_count": len(columns_to_drop),
        "duplicates_removed": {
            "train": duplicates_removed_train,
            "test": duplicates_removed_test,
        },
        "types_changed": types_changed,
        "protected_columns": protected_cols,
        "final_columns": list(train_df.columns),
        "final_column_count": len(train_df.columns),
    }

    artifacts.save_report(sanity_report, submodule_dir, "sanity_report.json")

    # 9. Generate transformation summary
    transformation_summary = report.create_preprocessing_report(
        train_before=train_df_original,
        train_after=train_df,
        test_before=test_df_original,
        test_after=test_df,
        config=config,
    )
    artifacts.save_report(transformation_summary, submodule_dir, "summary.json")

    # 10. Create state dict
    state_dict = {
        "version": "1.0",
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
        "issues_found": issues_found,
        "columns_dropped": columns_to_drop,
        "columns_dropped_count": len(columns_to_drop),
        "duplicates_removed_train": int(duplicates_removed_train),
        "duplicates_removed_test": int(duplicates_removed_test),
        "duplicates_removed_orig": int(duplicates_removed_orig) if orig_df is not None else 0,
        "types_changed": types_changed,
    }

    return train_df, val_df, test_df, orig_df, state_dict
