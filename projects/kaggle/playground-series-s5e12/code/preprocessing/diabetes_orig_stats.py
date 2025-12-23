"""
Diabetes Original Dataset Statistics Preprocessor

Maps aggregated statistics from the original (external) dataset to train/test/val.
This creates powerful features that capture target distribution patterns from the
larger external dataset.

Features created:
- orig_mean_{col}: Mean target value for each unique value of column
- orig_count_{col}: Count of samples for each unique value of column

Key characteristics:
- Requires orig_df to be loaded (use external_dataset preprocessor first)
- Works on both numeric and categorical columns
- Handles missing values gracefully (fills with global mean/0)
- No data leakage (statistics computed from external dataset only)

Based on: S5E12 Single CatBoost notebook strategy
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from mlarena.preprocessing.utils import artifacts


def fit_transform(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    test_df: pd.DataFrame,
    config: Dict[str, Any],
    orig_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame, Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Map original dataset statistics to train/test/val.

    Config keys:
        stat_columns (list): Columns to compute stats for (default: all numeric)
        include_binned (bool): Include bin_* columns (default: True)
        include_categorical (bool): Include categorical columns (default: False)
        stat_types (list): Stats to compute (default: ["mean", "count"])

    Returns:
        Tuple of (train_df, val_df, test_df, orig_df, state_dict)

    Raises:
        ValueError: If orig_df is not provided or target column missing
    """
    # 1. Validate orig_df
    if orig_df is None or len(orig_df) == 0:
        raise ValueError(
            "diabetes_orig_stats requires orig_df to be loaded. "
            "Add 'external_dataset' preprocessor before this step in the chain."
        )

    # 2. Extract config
    artifact_dir = Path(config.get("_artifact_dir", "."))
    dataset_meta = config.get("_dataset", {})
    target_col = dataset_meta.get("target")
    id_col = dataset_meta.get("id_column", "id")

    if not target_col or target_col not in orig_df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in orig_df. "
            "Ensure target is present in external dataset."
        )

    # 3. Get configuration
    stat_columns = config.get("stat_columns")
    include_binned = config.get("include_binned", True)
    include_categorical = config.get("include_categorical", False)
    stat_types = config.get("stat_types", ["mean", "count"])

    # Auto-detect columns if not specified
    if stat_columns is None:
        exclude_cols = {id_col, target_col}

        # Start with numeric columns
        stat_columns = [
            col for col in train_df.select_dtypes(include=[np.number]).columns
            if col not in exclude_cols
        ]

        # Add binned features if enabled
        if include_binned:
            binned_cols = [col for col in train_df.columns if col.startswith("bin_")]
            stat_columns.extend(binned_cols)

        # Add categorical if enabled
        if include_categorical:
            cat_cols = [
                col for col in train_df.select_dtypes(include=['object', 'category']).columns
                if col not in exclude_cols and not col.startswith("bin_")
            ]
            stat_columns.extend(cat_cols)

    # 4. Initialize state tracking
    state_dict = {
        "version": "1.0",
        "module": "diabetes_orig_stats",
        "stat_columns": stat_columns,
        "stat_types": stat_types,
        "new_columns": [],
        "global_mean": float(orig_df[target_col].mean()),
        "orig_rows": len(orig_df),
    }

    # Make copies
    train_df = train_df.copy()
    test_df = test_df.copy()
    if val_df is not None:
        val_df = val_df.copy()

    # 5. Compute and map statistics
    global_mean = orig_df[target_col].mean()

    for col in stat_columns:
        if col not in train_df.columns:
            print(f"⚠️  Column '{col}' not found in train - skipping")
            continue
        if col not in orig_df.columns:
            print(f"⚠️  Column '{col}' not found in orig - skipping")
            continue

        try:
            # Compute mean statistic
            if "mean" in stat_types:
                mean_col_name = f"orig_mean_{col}"
                mean_map = orig_df.groupby(col)[target_col].mean()

                # Map to all datasets
                train_df[mean_col_name] = train_df[col].map(mean_map).fillna(global_mean)
                test_df[mean_col_name] = test_df[col].map(mean_map).fillna(global_mean)
                if val_df is not None:
                    val_df[mean_col_name] = val_df[col].map(mean_map).fillna(global_mean)

                state_dict["new_columns"].append(mean_col_name)

            # Compute count statistic
            if "count" in stat_types:
                count_col_name = f"orig_count_{col}"
                count_map = orig_df.groupby(col).size()

                # Map to all datasets
                train_df[count_col_name] = train_df[col].map(count_map).fillna(0).astype(int)
                test_df[count_col_name] = test_df[col].map(count_map).fillna(0).astype(int)
                if val_df is not None:
                    val_df[count_col_name] = val_df[col].map(count_map).fillna(0).astype(int)

                state_dict["new_columns"].append(count_col_name)

        except Exception as e:
            print(f"⚠️  Failed to compute stats for '{col}': {e}")

    # 6. Save report
    report = {
        "stat_columns": stat_columns,
        "new_columns_created": len(state_dict["new_columns"]),
        "global_mean": state_dict["global_mean"],
        "orig_rows": state_dict["orig_rows"],
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
    }
    artifacts.save_report(report, artifact_dir, "orig_stats_report.json")

    print(f"✅ Orig stats complete: {len(state_dict['new_columns'])} new columns created")

    return train_df, val_df, test_df, orig_df, state_dict
