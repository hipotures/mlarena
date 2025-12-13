"""
Rare Category Handler - Sub-module 3

Purpose: Reduce high cardinality and rare category values before encoding
Libraries: pandas, numpy
Parameters: min_freq, min_freq_ratio, top_k, rare_label, detect_id_like_columns,
           id_unique_fraction_threshold, protected_categorical_columns
"""

from pathlib import Path
from typing import Any, Dict, Tuple, List

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
    Handle rare and high-cardinality categorical variables.

    Groups rare categories into a special label, limits to top-K categories,
    and detects potential ID columns with high uniqueness.

    Args:
        train_df: Training data
        val_df: Validation data (can be None)
        test_df: Test data
        config: Configuration dictionary with keys:
            - _artifact_dir: Path to save artifacts
            - _dataset: {id_column, target, ignored_columns}
            - min_freq: Minimum absolute frequency for category (default: 10)
            - min_freq_ratio: Minimum relative frequency (0-1) (default: 0.01)
            - top_k: Keep only top K most frequent categories (default: null)
            - rare_label: Label for rare categories (default: "__RARE__")
            - detect_id_like_columns: Detect ID-like columns (default: true)
            - id_unique_fraction_threshold: Uniqueness threshold for ID detection (default: 0.95)
            - protected_categorical_columns: List of columns to exclude (default: [])

    Returns:
        Tuple of (train_df, val_df, test_df, state_dict)

        state_dict contains:
        - version: str - Version of this sub-module
        - config: Dict - Configuration used
        - category_mappings: Dict[str, Dict] - Mapping old → new categories per column
        - detected_id_columns: List[str] - Columns detected as ID-like
        - stats: Dict - Statistics per column
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
        "min_freq": 10,
        "min_freq_ratio": 0.01,
        "top_k": None,
        "rare_label": "__RARE__",
        "detect_id_like_columns": True,
        "id_unique_fraction_threshold": 0.95,
        "protected_categorical_columns": [],
    }
    validation.validate_config(config, required_params, optional_params)

    # Validate numeric parameters
    validation.validate_numeric_range(
        config["min_freq"],
        min_value=1,
        param_name="min_freq"
    )
    validation.validate_numeric_range(
        config["min_freq_ratio"],
        min_value=0.0,
        max_value=1.0,
        param_name="min_freq_ratio"
    )
    if config["top_k"] is not None:
        validation.validate_numeric_range(
            config["top_k"],
            min_value=1,
            param_name="top_k"
        )
    validation.validate_numeric_range(
        config["id_unique_fraction_threshold"],
        min_value=0.0,
        max_value=1.0,
        param_name="id_unique_fraction_threshold"
    )

    # 3. Create sub-module artifact directory
    submodule_dir = artifacts.get_submodule_artifact_dir(artifact_dir, "rare_category_handler")

    # 4. Save original DataFrames for reporting
    train_df_original = dataframe_utils.copy_dataframe(train_df)
    test_df_original = dataframe_utils.copy_dataframe(test_df)

    # 5. Get categorical columns
    exclude_cols = [id_column, target_column] + ignored_columns + config["protected_categorical_columns"]
    categorical_cols = dataframe_utils.get_categorical_columns(train_df, exclude=exclude_cols)

    # 6. Detect ID-like columns
    detected_id_columns = []
    if config["detect_id_like_columns"]:
        for col in categorical_cols:
            unique_fraction = train_df[col].nunique() / len(train_df)
            if unique_fraction >= config["id_unique_fraction_threshold"]:
                detected_id_columns.append(col)

    # Remove ID-like columns from processing
    categorical_cols = [col for col in categorical_cols if col not in detected_id_columns]

    # 7. Build category mappings
    category_mappings = {}
    column_stats = {}

    for col in categorical_cols:
        # Count frequencies in train
        value_counts = train_df[col].value_counts()
        total_count = len(train_df)

        # Determine which categories to keep
        if config["top_k"] is not None:
            # Keep only top K
            keep_categories = set(value_counts.head(config["top_k"]).index)
        else:
            # Keep categories above threshold
            freq_mask = value_counts >= config["min_freq"]
            ratio_mask = (value_counts / total_count) >= config["min_freq_ratio"]
            keep_categories = set(value_counts[freq_mask & ratio_mask].index)

        # Build mapping: rare categories → rare_label
        mapping = {}
        for category in value_counts.index:
            if category not in keep_categories:
                mapping[category] = config["rare_label"]
            else:
                mapping[category] = category

        category_mappings[col] = mapping

        # Collect stats
        n_unique_before = train_df[col].nunique()
        n_unique_after = len(keep_categories) + (1 if len(mapping) > len(keep_categories) else 0)
        n_rare_categories = len([v for v in mapping.values() if v == config["rare_label"]])

        column_stats[col] = {
            "unique_before": int(n_unique_before),
            "unique_after": int(n_unique_after),
            "n_rare_categories": int(n_rare_categories),
            "n_kept_categories": len(keep_categories),
            "reduction_ratio": float(n_unique_after / n_unique_before) if n_unique_before > 0 else 1.0,
        }

    # 8. Apply mappings to all DataFrames
    for col, mapping in category_mappings.items():
        # Apply mapping with fillna for unseen categories in test
        train_df[col] = train_df[col].map(mapping).fillna(config["rare_label"])
        test_df[col] = test_df[col].map(mapping).fillna(config["rare_label"])
        if val_df is not None:
            val_df[col] = val_df[col].map(mapping).fillna(config["rare_label"])
        if orig_df is not None and col in orig_df.columns:
            orig_df[col] = orig_df[col].map(mapping).fillna(config["rare_label"])

    # 9. Save artifacts
    artifacts.save_report(
        {
            "category_mappings": {
                col: {str(k): str(v) for k, v in mapping.items()}
                for col, mapping in category_mappings.items()
            },
            "column_stats": column_stats,
            "detected_id_columns": detected_id_columns,
        },
        submodule_dir,
        "category_mappings.json"
    )

    # 10. Generate and save report
    transformation_summary = report.create_preprocessing_report(
        train_before=train_df_original,
        train_after=train_df,
        test_before=test_df_original,
        test_after=test_df,
        config=config,
    )
    artifacts.save_report(transformation_summary, submodule_dir, "summary.json")

    # 11. Create state dict
    state_dict = {
        "version": "1.0",
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
        "category_mappings": category_mappings,
        "detected_id_columns": detected_id_columns,
        "column_stats": column_stats,
        "n_categorical_processed": len(categorical_cols),
        "n_id_detected": len(detected_id_columns),
    }

    return train_df, val_df, test_df, orig_df, state_dict


def transform(
    df: pd.DataFrame,
    state_dict: Dict[str, Any],
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Apply rare category handling to new data (inference time).

    Args:
        df: New data to transform
        state_dict: State returned by fit_transform
        config: Original config

    Returns:
        Transformed DataFrame
    """
    df = dataframe_utils.copy_dataframe(df)

    rare_label = config.get("rare_label", "__RARE__")
    category_mappings = state_dict["category_mappings"]

    # Apply mappings
    for col, mapping in category_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(rare_label)

    return df
