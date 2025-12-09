"""
{SUB_MODULE_NAME} - {DESCRIPTION}

Purpose: {WHAT_IT_DOES}
Libraries: {USED_LIBRARIES}
Parameters: {CONFIG_PARAMS}
"""

from pathlib import Path
from typing import Any, Dict, Tuple

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
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, Dict[str, Any]]:
    """
    {SUB_MODULE_NAME} preprocessing.

    Args:
        train_df: Training data
        val_df: Validation data (can be None)
        test_df: Test data
        config: Configuration dictionary with keys:
            - _artifact_dir: Path to save artifacts
            - _dataset: {id_column, target, ignored_columns}
            - {PARAM1}: {DESCRIPTION}
            - {PARAM2}: {DESCRIPTION}

    Returns:
        Tuple of (train_df, val_df, test_df, state_dict)

        state_dict contains:
        - version: str - Version of this sub-module
        - config: Dict - Configuration used
        - {STATE_KEY1}: {DESCRIPTION}
        - {STATE_KEY2}: {DESCRIPTION}
    """
    # 1. Extract config
    artifact_dir = Path(config.get("_artifact_dir", "."))
    dataset_config = config.get("_dataset", {})
    id_column = dataset_config.get("id_column", "id")
    target_column = dataset_config.get("target")
    ignored_columns = dataset_config.get("ignored_columns", [])

    # 2. Validate config
    required_params = []  # Add required parameters here
    optional_params = {
        "param1": "default1",
        "param2": "default2",
    }
    validation.validate_config(config, required_params, optional_params)

    # 3. Create sub-module artifact directory
    submodule_dir = artifacts.get_submodule_artifact_dir(artifact_dir, "sub_module_name")

    # 4. Save original DataFrames for reporting
    train_df_original = dataframe_utils.copy_dataframe(train_df)
    test_df_original = dataframe_utils.copy_dataframe(test_df)

    # 5. Perform transformation
    # ========================================
    # TODO: Implement transformation logic here
    # ========================================

    # Example: Get numeric columns
    # numeric_cols = dataframe_utils.get_numeric_columns(
    #     train_df,
    #     exclude=[id_column, target_column] + ignored_columns
    # )

    # Example: Get categorical columns
    # categorical_cols = dataframe_utils.get_categorical_columns(
    #     train_df,
    #     exclude=[id_column, target_column] + ignored_columns
    # )

    # Example: Validate columns exist
    # validation.validate_column_exists(
    #     train_df,
    #     [col1, col2],
    #     context="Training data"
    # )

    # Example: Fit and transform
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # scaler.fit(train_df[numeric_cols])
    # train_df[numeric_cols] = scaler.transform(train_df[numeric_cols])
    # test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])
    # if val_df is not None:
    #     val_df[numeric_cols] = scaler.transform(val_df[numeric_cols])

    # 6. Save artifacts
    # artifacts.save_fitted_object(scaler, submodule_dir, "scaler.pkl")

    # 7. Generate and save report
    transformation_summary = report.create_preprocessing_report(
        train_before=train_df_original,
        train_after=train_df,
        test_before=test_df_original,
        test_after=test_df,
        config=config,
    )
    artifacts.save_report(transformation_summary, submodule_dir, "summary.json")

    # 8. Create state dict
    state_dict = {
        "version": "1.0",
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
        # Add sub-module specific state here
        # "scaler_path": str((submodule_dir / "scaler.pkl").relative_to(artifact_dir)),
    }

    return train_df, val_df, test_df, state_dict


def transform(
    df: pd.DataFrame,
    state_dict: Dict[str, Any],
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Apply transformation to new data (inference time).

    Optional: Implement only if sub-module needs inference-time support.

    Args:
        df: New data to transform
        state_dict: State returned by fit_transform
        config: Original config

    Returns:
        Transformed DataFrame
    """
    # 1. Extract config
    artifact_dir = Path(config.get("_artifact_dir", "."))
    submodule_dir = artifacts.get_submodule_artifact_dir(artifact_dir, "sub_module_name")

    # 2. Load artifacts
    # scaler = artifacts.load_fitted_object(submodule_dir, "scaler.pkl")

    # 3. Apply transformation
    # df = df.copy()
    # numeric_cols = ... # Get from config or state
    # df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df
