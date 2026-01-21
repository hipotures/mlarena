"""
Numerical Scaling and Transformation Sub-Module

Purpose: Standardization and distribution transformations for models sensitive to scale/distribution
Libraries: sklearn.preprocessing (StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer), numpy
Parameters:
  - scaling_method: Method for scaling (none|standard|minmax|robust|quantile_normal|quantile_uniform|power_yeo_johnson)
  - numeric_include: List of specific numeric columns to scale (None = all numeric)
  - numeric_exclude: List of numeric columns to exclude from scaling
  - log_transform: List of columns for log1p transformation (applied before scaling)
  - clip_lower_quantile: Lower quantile for winsorization (e.g., 0.01)
  - clip_upper_quantile: Upper quantile for winsorization (e.g., 0.99)
  - n_quantiles: Number of quantiles for QuantileTransformer (default: 1000)
  - random_state: Random state for QuantileTransformer
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    QuantileTransformer,
    PowerTransformer,
)

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
    eval_df: pd.DataFrame | None = None,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame | None,
    pd.DataFrame,
    pd.DataFrame | None,
    pd.DataFrame | None,
    Dict[str, Any],
]:
    """
    Scale and transform numerical features.

    Args:
        train_df: Training data
        val_df: Validation data (can be None)
        test_df: Test data
        config: Configuration dictionary with keys:
            - _artifact_dir: Path to save artifacts
            - _dataset: {id_column, target, ignored_columns}
            - scaling_method: Scaling method (none|standard|minmax|robust|quantile_normal|quantile_uniform|power_yeo_johnson)
            - numeric_include: Specific columns to scale (None = all numeric)
            - numeric_exclude: Columns to exclude from scaling
            - log_transform: Columns for log1p transformation
            - clip_lower_quantile: Lower quantile for clipping (0.0-1.0)
            - clip_upper_quantile: Upper quantile for clipping (0.0-1.0)
            - n_quantiles: Number of quantiles for QuantileTransformer
            - random_state: Random state for QuantileTransformer
            - power_standardize: Standardize after power transform (default: True)
        orig_df: External dataset (can be None)
        eval_df: Evaluation dataset (can be None)

    Returns:
        Tuple of (train_df, val_df, test_df, eval_df, orig_df, state_dict)

        state_dict contains:
        - version: str
        - config: Dict
        - scaling_method: str
        - scaled_columns: List[str]
        - log_transformed_columns: List[str]
        - clipped_columns: List[str]
        - clip_bounds: Dict[str, Dict] - Lower/upper bounds per column
        - scaler_path: str (if scaling applied)
        - column_stats: Dict - Statistics before/after transformation
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
        "scaling_method": "none",
        "numeric_include": None,
        "numeric_exclude": [],
        "log_transform": [],
        "clip_lower_quantile": None,
        "clip_upper_quantile": None,
        "n_quantiles": 1000,
        "random_state": 42,
        "power_standardize": True,
    }
    validation.validate_config(config, required_params, optional_params)

    # Validate scaling method
    valid_methods = [
        "none",
        "standard",
        "minmax",
        "robust",
        "quantile_normal",
        "quantile_uniform",
        "rank_gauss",
        "power_yeo_johnson",
    ]
    validation.validate_choice(
        config["scaling_method"], valid_methods, "scaling_method"
    )

    # Validate quantile ranges
    if config["clip_lower_quantile"] is not None:
        validation.validate_numeric_range(
            config["clip_lower_quantile"],
            min_value=0.0,
            max_value=1.0,
            param_name="clip_lower_quantile",
        )
    if config["clip_upper_quantile"] is not None:
        validation.validate_numeric_range(
            config["clip_upper_quantile"],
            min_value=0.0,
            max_value=1.0,
            param_name="clip_upper_quantile",
        )

    # 3. Create sub-module artifact directory
    submodule_dir = artifacts.get_submodule_artifact_dir(artifact_dir, "scaler")

    # 4. Save copies for reporting
    train_df_original = dataframe_utils.copy_dataframe(train_df)
    test_df_original = dataframe_utils.copy_dataframe(test_df)

    # 5. Identify columns to process
    exclude_cols = (
        [id_column, target_column] + ignored_columns + config["numeric_exclude"]
    )

    if config["numeric_include"] is not None:
        # Use explicitly specified columns
        numeric_cols = [
            col
            for col in config["numeric_include"]
            if col in train_df.columns and col not in exclude_cols
        ]
    else:
        # Auto-detect numeric columns
        numeric_cols = dataframe_utils.get_numeric_columns(
            train_df, exclude=exclude_cols
        )

    # Ensure columns exist in both train and test
    numeric_cols = [
        col
        for col in numeric_cols
        if col in train_df.columns and col in test_df.columns
    ]

    if not numeric_cols:
        # No numeric columns to process
        state_dict = {
            "version": "1.0",
            "config": {k: v for k, v in config.items() if not k.startswith("_")},
            "scaling_method": config["scaling_method"],
            "scaled_columns": [],
            "log_transformed_columns": [],
            "clipped_columns": [],
            "message": "No numeric columns to process",
        }
        return train_df, val_df, test_df, eval_df, orig_df, state_dict

    # 6. Apply log transformation (if specified)
    log_cols = [col for col in config["log_transform"] if col in numeric_cols]
    log_transformed_cols = []

    if log_cols:
        for col in log_cols:
            # Create new column with log transformation
            new_col = f"{col}_log"

            # Apply log1p (handles zeros) - shift negative values if needed
            train_min = train_df[col].min()
            test_min = test_df[col].min()
            shift = 0

            if train_min < 0 or test_min < 0:
                shift = abs(min(train_min, test_min)) + 1

            train_df[new_col] = np.log1p(train_df[col] + shift)
            test_df[new_col] = np.log1p(test_df[col] + shift)
            if val_df is not None:
                val_df[new_col] = np.log1p(val_df[col] + shift)
            if orig_df is not None and col in orig_df.columns:
                orig_df[new_col] = np.log1p(orig_df[col] + shift)
            if eval_df is not None:
                eval_df[new_col] = np.log1p(eval_df[col] + shift)

            log_transformed_cols.append(new_col)
            # Add to numeric_cols for potential scaling
            if new_col not in numeric_cols:
                numeric_cols.append(new_col)

    # 7. Apply winsorization/clipping (if specified)
    clip_bounds = {}
    clipped_cols = []

    if (
        config["clip_lower_quantile"] is not None
        or config["clip_upper_quantile"] is not None
    ):
        for col in numeric_cols:
            lower_q = config["clip_lower_quantile"]
            upper_q = config["clip_upper_quantile"]

            # Calculate quantiles from train data only
            if lower_q is not None:
                lower_bound = train_df[col].quantile(lower_q)
            else:
                lower_bound = None

            if upper_q is not None:
                upper_bound = train_df[col].quantile(upper_q)
            else:
                upper_bound = None

            # Apply clipping to all datasets
            if lower_bound is not None or upper_bound is not None:
                train_df[col] = train_df[col].clip(lower=lower_bound, upper=upper_bound)
                test_df[col] = test_df[col].clip(lower=lower_bound, upper=upper_bound)
                if val_df is not None:
                    val_df[col] = val_df[col].clip(lower=lower_bound, upper=upper_bound)
                if orig_df is not None and col in orig_df.columns:
                    orig_df[col] = orig_df[col].clip(
                        lower=lower_bound, upper=upper_bound
                    )
                if eval_df is not None:
                    eval_df[col] = eval_df[col].clip(
                        lower=lower_bound, upper=upper_bound
                    )

                clip_bounds[col] = {
                    "lower": float(lower_bound) if lower_bound is not None else None,
                    "upper": float(upper_bound) if upper_bound is not None else None,
                }
                clipped_cols.append(col)

    # 8. Apply scaling (if method != 'none')
    scaled_cols = []
    scaler_path = None

    if config["scaling_method"] != "none":
        # Initialize scaler
        if config["scaling_method"] == "standard":
            scaler = StandardScaler()
        elif config["scaling_method"] == "minmax":
            scaler = MinMaxScaler()
        elif config["scaling_method"] == "robust":
            scaler = RobustScaler()
        elif config["scaling_method"] in ["quantile_normal", "rank_gauss"]:
            scaler = QuantileTransformer(
                n_quantiles=min(config["n_quantiles"], len(train_df)),
                output_distribution="normal",
                random_state=config["random_state"],
            )
        elif config["scaling_method"] == "quantile_uniform":
            scaler = QuantileTransformer(
                n_quantiles=min(config["n_quantiles"], len(train_df)),
                output_distribution="uniform",
                random_state=config["random_state"],
            )
        elif config["scaling_method"] == "power_boxcox":
            if (train_df[numeric_cols] <= 0).any().any():
                raise ValueError(
                    "power_boxcox requires strictly positive values. Use power_yeo_johnson or shift data."
                )
            scaler = PowerTransformer(
                method="box-cox", standardize=config["power_standardize"]
            )
        elif config["scaling_method"] == "power_yeo_johnson":
            scaler = PowerTransformer(
                method="yeo-johnson", standardize=config["power_standardize"]
            )
        else:
            raise ValueError(f"Unknown scaling_method: {config['scaling_method']}")

        # Fit on train data
        scaler.fit(train_df[numeric_cols])

        # Transform all datasets
        train_df[numeric_cols] = scaler.transform(train_df[numeric_cols])
        test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])
        if val_df is not None:
            val_df[numeric_cols] = scaler.transform(val_df[numeric_cols])
        if orig_df is not None:
            # Only scale columns that exist in orig_df
            orig_numeric_cols = [col for col in numeric_cols if col in orig_df.columns]
            if orig_numeric_cols:
                orig_df[orig_numeric_cols] = scaler.transform(
                    orig_df[orig_numeric_cols]
                )
        if eval_df is not None:
            eval_df[numeric_cols] = scaler.transform(eval_df[numeric_cols])

        # Save scaler
        scaler_path = artifacts.save_fitted_object(scaler, submodule_dir, "scaler.pkl")
        scaled_cols = numeric_cols

    # 9. Collect statistics
    column_stats = {}
    for col in numeric_cols:
        if col in train_df_original.columns:
            column_stats[col] = {
                "before": {
                    "mean": float(train_df_original[col].mean()),
                    "std": float(train_df_original[col].std()),
                    "min": float(train_df_original[col].min()),
                    "max": float(train_df_original[col].max()),
                },
                "after": {
                    "mean": float(train_df[col].mean()),
                    "std": float(train_df[col].std()),
                    "min": float(train_df[col].min()),
                    "max": float(train_df[col].max()),
                },
            }

    # 10. Generate and save report
    transformation_summary = report.create_preprocessing_report(
        train_before=train_df_original,
        train_after=train_df,
        test_before=test_df_original,
        test_after=test_df,
        config=config,
    )

    # Add scaling-specific info
    transformation_summary["scaling"] = {
        "method": config["scaling_method"],
        "scaled_columns": scaled_cols,
        "log_transformed_columns": log_transformed_cols,
        "clipped_columns": clipped_cols,
        "clip_bounds": clip_bounds,
        "column_stats": column_stats,
    }

    artifacts.save_report(transformation_summary, submodule_dir, "summary.json")

    # 11. Create state dict
    state_dict = {
        "version": "1.0",
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
        "scaling_method": config["scaling_method"],
        "scaled_columns": scaled_cols,
        "log_transformed_columns": log_transformed_cols,
        "clipped_columns": clipped_cols,
        "clip_bounds": clip_bounds,
        "column_stats": column_stats,
    }

    if scaler_path:
        state_dict["scaler_path"] = str(scaler_path.relative_to(artifact_dir))

    return train_df, val_df, test_df, eval_df, orig_df, state_dict


def transform(
    df: pd.DataFrame, state_dict: Dict[str, Any], config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Apply scaling transformation to new data (inference time).

    Args:
        df: New data to transform
        state_dict: State returned by fit_transform
        config: Original config

    Returns:
        Transformed DataFrame
    """
    artifact_dir = Path(config.get("_artifact_dir", "."))
    submodule_dir = artifacts.get_submodule_artifact_dir(artifact_dir, "scaler")

    df = dataframe_utils.copy_dataframe(df)

    # Apply log transformations
    for col in state_dict.get("log_transformed_columns", []):
        if col.endswith("_log"):
            original_col = col[:-4]
            if original_col in df.columns:
                # Reconstruct shift from clip_bounds or assume 0
                shift = 0
                df[col] = np.log1p(df[original_col] + shift)

    # Apply clipping
    for col, bounds in state_dict.get("clip_bounds", {}).items():
        if col in df.columns:
            df[col] = df[col].clip(lower=bounds["lower"], upper=bounds["upper"])

    # Apply scaling
    if "scaler_path" in state_dict:
        scaler = artifacts.load_fitted_object(submodule_dir, "scaler.pkl")
        scaled_cols = state_dict["scaled_columns"]
        existing_cols = [col for col in scaled_cols if col in df.columns]
        if existing_cols:
            df[existing_cols] = scaler.transform(df[existing_cols])

    return df
