"""
Missing values imputation using sklearn SimpleImputer.

Supports:
- Numeric columns: mean, median, most_frequent, constant
- Categorical columns: most_frequent, constant

This module implements the preprocessing interface expected by MLArena:
- fit_transform(train_df, val_df, test_df, config) -> (train_df, val_df, test_df, state_dict)
- transform(df, state_dict, config) -> df  # Optional, for inference
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


def fit_transform(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
    orig_df: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None, Dict[str, Any]]:
    """
    Impute missing values in numeric and categorical columns.

    Config parameters:
        numeric_strategy: "mean" | "median" | "most_frequent" | "constant" (default: "median")
        categorical_strategy: "most_frequent" | "constant" (default: "most_frequent")
        fill_value: Value for "constant" strategy (default: 0)

    Args:
        train_df: Training dataframe
        val_df: Validation dataframe (optional)
        test_df: Test dataframe
        config: Configuration dictionary
        orig_df: External dataset (optional)

    Returns:
        Tuple of (train_df, val_df, test_df, orig_df, state_dict)
    """
    # Extract config
    numeric_strategy = config.get("numeric_strategy", "median")
    categorical_strategy = config.get("categorical_strategy", "most_frequent")
    fill_value = config.get("fill_value", 0)

    # Make copies to avoid modifying originals
    train_df = train_df.copy()
    test_df = test_df.copy()
    if val_df is not None:
        val_df = val_df.copy()
    if orig_df is not None:
        orig_df = orig_df.copy()

    # Separate numeric and categorical columns
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = train_df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Remove target column from processing if present
    dataset_info = config.get("_dataset", {})
    target_column = dataset_info.get("target")
    if target_column and target_column in numeric_cols:
        numeric_cols.remove(target_column)
    elif target_column and target_column in categorical_cols:
        categorical_cols.remove(target_column)

    state = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "numeric_strategy": numeric_strategy,
        "categorical_strategy": categorical_strategy,
        "missing_counts": {
            "train": {
                "numeric": int(train_df[numeric_cols].isnull().sum().sum()) if numeric_cols else 0,
                "categorical": int(train_df[categorical_cols].isnull().sum().sum()) if categorical_cols else 0,
            },
            "test": {
                "numeric": int(test_df[numeric_cols].isnull().sum().sum()) if numeric_cols else 0,
                "categorical": int(test_df[categorical_cols].isnull().sum().sum()) if categorical_cols else 0,
            },
        },
    }

    # Impute numeric columns
    if numeric_cols:
        num_imputer = SimpleImputer(strategy=numeric_strategy, fill_value=fill_value)
        train_df[numeric_cols] = num_imputer.fit_transform(train_df[numeric_cols])
        test_df[numeric_cols] = num_imputer.transform(test_df[numeric_cols])
        if val_df is not None:
            val_df[numeric_cols] = num_imputer.transform(val_df[numeric_cols])
        if orig_df is not None:
            orig_df[numeric_cols] = num_imputer.transform(orig_df[numeric_cols])

        state["numeric_imputer_params"] = {
            "statistics": num_imputer.statistics_.tolist() if hasattr(num_imputer, 'statistics_') else [],
        }

    # Impute categorical columns
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy=categorical_strategy, fill_value="missing")
        train_df[categorical_cols] = cat_imputer.fit_transform(train_df[categorical_cols])
        test_df[categorical_cols] = cat_imputer.transform(test_df[categorical_cols])
        if val_df is not None:
            val_df[categorical_cols] = cat_imputer.transform(val_df[categorical_cols])
        if orig_df is not None:
            orig_df[categorical_cols] = cat_imputer.transform(orig_df[categorical_cols])

        state["categorical_imputer_params"] = {
            "statistics": [str(s) for s in cat_imputer.statistics_] if hasattr(cat_imputer, 'statistics_') else [],
        }

    return train_df, val_df, test_df, orig_df, state


def transform(df: pd.DataFrame, state_dict: Dict[str, Any], config: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply fitted imputers to new data.

    Note: This function is not currently used in the preprocessing pipeline
    but is provided for potential future inference use cases.

    Args:
        df: Dataframe to transform
        state_dict: State dictionary from fit_transform
        config: Configuration dictionary

    Returns:
        Transformed dataframe
    """
    # Not needed for current workflow (fit_transform handles everything)
    return df.copy()
