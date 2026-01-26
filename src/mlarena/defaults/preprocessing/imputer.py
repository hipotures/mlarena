"""
Missing Value Imputation Sub-Module

Purpose: Universal imputation of missing values with strategy selection per column type
Libraries: sklearn.impute (SimpleImputer, KNNImputer, IterativeImputer), pandas
Parameters:
  - numeric_strategy: Global strategy for numeric columns (mean|median|most_frequent|constant|knn|iterative)
  - categorical_strategy: Global strategy for categorical columns (most_frequent|constant)
  - column_strategies: Dict mapping column names to specific strategies
  - fill_value: Value for 'constant' strategy
  - knn_n_neighbors: Number of neighbors for KNN imputation
  - iterative_estimator: Estimator for IterativeImputer (default: BayesianRidge)
  - iterative_max_iter: Max iterations for IterativeImputer
  - treat_outliers_as_na: Whether to treat outliers as missing before imputation
  - outlier_method: Method for outlier detection (iqr|zscore)
  - outlier_threshold: Threshold for outlier detection
"""

from pathlib import Path
from typing import Any, Dict, Tuple
import logging

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import BayesianRidge

logger = logging.getLogger(__name__)

# IterativeImputer is experimental
try:
    from sklearn.experimental import enable_iterative_imputer  # noqa
    from sklearn.impute import IterativeImputer

    ITERATIVE_IMPUTER_AVAILABLE = True
except ImportError:
    ITERATIVE_IMPUTER_AVAILABLE = False

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
    Impute missing values with configurable strategies.

    Args:
        train_df: Training data
        val_df: Validation data (can be None)
        test_df: Test data
        config: Configuration dictionary with keys:
            - _artifact_dir: Path to save artifacts
            - _dataset: {id_column, target, ignored_columns}
            - numeric_strategy: Strategy for numeric columns
            - categorical_strategy: Strategy for categorical columns
            - column_strategies: Dict of column-specific strategies
            - fill_value: Value for constant strategy (default: 0)
            - knn_n_neighbors: Neighbors for KNN (default: 5)
            - iterative_estimator: Estimator for IterativeImputer
            - iterative_max_iter: Max iterations (default: 10)
            - treat_outliers_as_na: Convert outliers to NA first (default: False)
            - outlier_method: iqr|zscore (default: iqr)
            - outlier_threshold: Threshold for outliers (default: 3.0 for zscore, 1.5 for IQR)
        orig_df: External dataset (can be None)
        eval_df: Evaluation dataset (can be None)

    Returns:
        Tuple of (train_df, val_df, test_df, eval_df, orig_df, state_dict)
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
        "numeric_strategy": "mean",
        "categorical_strategy": "most_frequent",
        "column_strategies": {},
        "fill_value": 0,
        "knn_n_neighbors": 5,
        "iterative_estimator": "bayesian_ridge",
        "iterative_max_iter": 10,
        "treat_outliers_as_na": False,
        "outlier_method": "iqr",
        "outlier_threshold": None,  # Auto-select based on method
    }
    validation.validate_config(config, required_params, optional_params)

    # Validate strategy choices
    valid_numeric_strategies = [
        "mean",
        "median",
        "most_frequent",
        "constant",
        "knn",
        "iterative",
    ]
    valid_categorical_strategies = ["most_frequent", "constant"]

    validation.validate_choice(
        config["numeric_strategy"], valid_numeric_strategies, "numeric_strategy"
    )
    validation.validate_choice(
        config["categorical_strategy"],
        valid_categorical_strategies,
        "categorical_strategy",
    )

    # 3. Create sub-module artifact directory
    submodule_dir = artifacts.get_submodule_artifact_dir(artifact_dir, "imputer")

    # 4. Save original DataFrames for reporting
    train_df_original = dataframe_utils.copy_dataframe(train_df)
    test_df_original = dataframe_utils.copy_dataframe(test_df)

    # 5. Get column lists
    exclude_cols = [id_column, target_column] + ignored_columns
    exclude_cols = [col for col in exclude_cols if col is not None]

    numeric_cols = dataframe_utils.get_numeric_columns(train_df, exclude=exclude_cols)
    categorical_cols = dataframe_utils.get_categorical_columns(
        train_df, exclude=exclude_cols
    )

    # 6. Optional: Treat outliers as NA
    if config["treat_outliers_as_na"]:
        train_df, test_df, val_df, orig_df, eval_df, outlier_stats = (
            _treat_outliers_as_na(
                train_df,
                test_df,
                val_df,
                orig_df,
                eval_df,
                numeric_cols,
                method=config["outlier_method"],
                threshold=config["outlier_threshold"],
            )
        )
    else:
        outlier_stats = None

    # 7. Track missing values before imputation
    missing_before = _create_missing_report(train_df, numeric_cols + categorical_cols)
    missing_before_total = sum(missing_before.values())

    # DEBUG: Log missing value status
    columns_with_missing = [col for col, count in missing_before.items() if count > 0]
    logger.debug(
        "Imputer invoked: %d total missing values across %d columns (out of %d total columns)",
        missing_before_total,
        len(columns_with_missing),
        len(numeric_cols) + len(categorical_cols)
    )

    # Early return if no missing values
    if missing_before_total == 0:
        logger.debug("Imputer: No missing values detected - skipping imputation (no-op)")
        state_dict = {
            "version": "1.0",
            "config": {k: v for k, v in config.items() if not k.startswith("_")},
            "imputed_columns": 0,
            "empty_columns_no_observed": [],
            "missing_before_total": 0,
            "missing_after_total": 0,
            "column_strategies": {},
            "was_noop": True,
        }
        return train_df, val_df, test_df, eval_df, orig_df, state_dict

    # 8. Perform imputation
    imputers = {}
    column_strategies_used = {}
    empty_columns_no_observed = []

    def _coerce_numeric(col: str, to_float: bool) -> None:
        if not to_float:
            return
        for df in (train_df, test_df, val_df, orig_df, eval_df):
            if df is not None and col in df.columns:
                df[col] = df[col].astype("float64")

    # Impute numeric columns
    for col in numeric_cols:
        # Check for column-specific strategy
        strategy = config["column_strategies"].get(col, config["numeric_strategy"])
        column_strategies_used[col] = strategy

        # Align dtypes across datasets to avoid sklearn casting errors
        dtype_kinds = []
        for df in (train_df, test_df, val_df, orig_df, eval_df):
            if df is not None and col in df.columns:
                dtype_kinds.append(df[col].dtype.kind)
        dtype_mismatch = len(set(dtype_kinds)) > 1
        requires_float = strategy in {"mean", "median", "knn", "iterative"}
        _coerce_numeric(col, dtype_mismatch or requires_float)

        if train_df[col].isna().all():
            empty_columns_no_observed.append(col)
            fill_value = config["fill_value"]
            train_df[col] = fill_value
            test_df[col] = fill_value
            if val_df is not None:
                val_df[col] = fill_value
            if orig_df is not None and col in orig_df.columns:
                orig_df[col] = orig_df[col].fillna(fill_value)
            if eval_df is not None:
                eval_df[col] = eval_df[col].fillna(fill_value)
            continue

        # Fit and transform
        imputer = _create_imputer(
            strategy,
            fill_value=config["fill_value"],
            knn_n_neighbors=config["knn_n_neighbors"],
            iterative_estimator=config["iterative_estimator"],
            iterative_max_iter=config["iterative_max_iter"],
        )

        # Fit on train
        imputer.fit(train_df[[col]])

        # Transform all datasets
        train_df[col] = imputer.transform(train_df[[col]]).ravel()
        test_df[col] = imputer.transform(test_df[[col]]).ravel()
        if val_df is not None:
            val_df[col] = imputer.transform(val_df[[col]]).ravel()
        if orig_df is not None and col in orig_df.columns:
            orig_df[col] = imputer.transform(orig_df[[col]]).ravel()
        if eval_df is not None:
            eval_df[col] = imputer.transform(eval_df[[col]]).ravel()

        imputers[col] = imputer

    # Impute categorical columns
    for col in categorical_cols:
        # Check for column-specific strategy
        strategy = config["column_strategies"].get(col, config["categorical_strategy"])
        column_strategies_used[col] = strategy

        if train_df[col].isna().all():
            empty_columns_no_observed.append(col)
            fill_value = "__MISSING__"
            train_df[col] = fill_value
            test_df[col] = fill_value
            if val_df is not None:
                val_df[col] = fill_value
            if orig_df is not None and col in orig_df.columns:
                orig_df[col] = orig_df[col].fillna(fill_value)
            if eval_df is not None:
                eval_df[col] = eval_df[col].fillna(fill_value)
            continue

        # Fit and transform
        imputer = _create_imputer(
            strategy,
            fill_value=config.get("fill_value", "__MISSING__"),
            is_categorical=True,
        )

        # Fit on train
        imputer.fit(train_df[[col]])

        # Transform all datasets
        train_df[col] = imputer.transform(train_df[[col]]).ravel()
        test_df[col] = imputer.transform(test_df[[col]]).ravel()
        if val_df is not None:
            val_df[col] = imputer.transform(val_df[[col]]).ravel()
        if orig_df is not None and col in orig_df.columns:
            orig_df[col] = imputer.transform(orig_df[[col]]).ravel()
        if eval_df is not None:
            eval_df[col] = imputer.transform(eval_df[[col]]).ravel()

        imputers[col] = imputer

    # 9. Track missing values after imputation
    missing_after = _create_missing_report(train_df, numeric_cols + categorical_cols)

    # 10. Save imputers
    for col, imputer in imputers.items():
        artifacts.save_fitted_object(imputer, submodule_dir, f"imputer_{col}.pkl")

    # 11. Create imputation report
    imputation_report = {
        "missing_before": missing_before,
        "missing_after": missing_after,
        "column_strategies": column_strategies_used,
        "empty_columns_no_observed": empty_columns_no_observed,
        "outlier_treatment": outlier_stats,
        "imputed_columns": {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
        },
    }
    artifacts.save_report(imputation_report, submodule_dir, "imputation_report.json")

    # 12. Generate and save standard report
    transformation_summary = report.create_preprocessing_report(
        train_before=train_df_original,
        train_after=train_df,
        test_before=test_df_original,
        test_after=test_df,
        config=config,
    )
    artifacts.save_report(transformation_summary, submodule_dir, "summary.json")

    # 13. Print summary table of imputed values
    from rich.console import Console
    from rich.table import Table

    imputed_stats = []
    for col, count_before in missing_before.items():
        if count_before > 0:
            count_after = missing_after.get(col, 0)
            imputed_count = count_before - count_after
            strategy = column_strategies_used.get(col, "unknown")
            imputed_stats.append(
                (col, count_before, count_after, imputed_count, strategy)
            )

    if imputed_stats:
        console = Console(force_terminal=True)
        table = Table(
            title="Imputation Summary (Train)",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Feature", style="cyan")
        table.add_column("Missing (Before)", justify="right")
        table.add_column("Missing (After)", justify="right")
        table.add_column("Imputed", justify="right", style="green")
        table.add_column("Strategy", style="magenta")

        for col, before, after, imputed, strategy in sorted(
            imputed_stats, key=lambda x: x[1], reverse=True
        ):
            table.add_row(col, str(before), str(after), str(imputed), strategy)

        console.print(table)
        console.print(
            f"[dim]Total missing values filled: {sum(x[3] for x in imputed_stats)}[/dim]\n"
        )

    # 14. Create state dict
    missing_after_total = sum(missing_after.values())
    total_imputed = missing_before_total - missing_after_total

    state_dict = {
        "version": "1.0",
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
        "imputed_columns": len(numeric_cols) + len(categorical_cols),
        "empty_columns_no_observed": empty_columns_no_observed,
        "missing_before_total": missing_before_total,
        "missing_after_total": missing_after_total,
        "column_strategies": column_strategies_used,
        "was_noop": False,
    }

    # DEBUG: Log imputation summary
    logger.debug(
        "Imputer completed: %d values imputed, %d columns processed, %d values still missing",
        total_imputed,
        len(column_strategies_used),
        missing_after_total
    )
    if empty_columns_no_observed:
        logger.debug(
            "Imputer: %d columns had no observed values (filled with defaults): %s",
            len(empty_columns_no_observed),
            empty_columns_no_observed
        )

    return train_df, val_df, test_df, eval_df, orig_df, state_dict


def _create_imputer(
    strategy: str,
    fill_value: Any = None,
    knn_n_neighbors: int = 5,
    iterative_estimator: str = "bayesian_ridge",
    iterative_max_iter: int = 10,
    is_categorical: bool = False,
):
    """Create imputer based on strategy."""
    if strategy == "knn" and not is_categorical:
        return KNNImputer(n_neighbors=knn_n_neighbors)

    elif strategy == "iterative" and not is_categorical:
        if not ITERATIVE_IMPUTER_AVAILABLE:
            raise ImportError(
                "IterativeImputer not available. Install sklearn with: "
                "pip install scikit-learn>=0.21"
            )

        if iterative_estimator == "bayesian_ridge":
            estimator = BayesianRidge()
        else:
            estimator = None  # Use default

        return IterativeImputer(
            estimator=estimator, max_iter=iterative_max_iter, random_state=42
        )

    elif strategy == "constant":
        return SimpleImputer(
            strategy="constant", fill_value=fill_value, keep_empty_features=True
        )

    else:
        # mean, median, most_frequent
        return SimpleImputer(strategy=strategy, keep_empty_features=True)


def _treat_outliers_as_na(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    orig_df: pd.DataFrame | None,
    eval_df: pd.DataFrame | None,
    numeric_cols: list,
    method: str = "iqr",
    threshold: float = None,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    Dict,
]:
    """Convert outliers to NaN before imputation."""
    outlier_stats = {}

    for col in numeric_cols:
        if col not in train_df.columns:
            continue

        # Detect outliers on train only
        if method == "iqr":
            q1 = train_df[col].quantile(0.25)
            q3 = train_df[col].quantile(0.75)
            iqr = q3 - q1
            iqr_threshold = threshold if threshold is not None else 1.5
            lower_bound = q1 - iqr_threshold * iqr
            upper_bound = q3 + iqr_threshold * iqr
            outlier_mask_train = (train_df[col] < lower_bound) | (
                train_df[col] > upper_bound
            )
            outlier_mask_test = (test_df[col] < lower_bound) | (
                test_df[col] > upper_bound
            )

        elif method == "zscore":
            mean = train_df[col].mean()
            std = train_df[col].std()
            zscore_threshold = threshold if threshold is not None else 3.0
            outlier_mask_train = np.abs((train_df[col] - mean) / std) > zscore_threshold
            outlier_mask_test = np.abs((test_df[col] - mean) / std) > zscore_threshold

        else:
            continue

        # Count outliers
        outlier_stats[col] = {
            "method": method,
            "threshold": iqr_threshold if method == "iqr" else zscore_threshold,
            "train_outliers": int(outlier_mask_train.sum()),
            "test_outliers": int(outlier_mask_test.sum()),
        }

        # Convert to NaN
        train_df.loc[outlier_mask_train, col] = np.nan
        test_df.loc[outlier_mask_test, col] = np.nan
        if val_df is not None and col in val_df.columns:
            if method == "iqr":
                outlier_mask_val = (val_df[col] < lower_bound) | (
                    val_df[col] > upper_bound
                )
            else:
                outlier_mask_val = np.abs((val_df[col] - mean) / std) > zscore_threshold
            outlier_stats[col]["val_outliers"] = int(outlier_mask_val.sum())
            val_df.loc[outlier_mask_val, col] = np.nan
        if orig_df is not None and col in orig_df.columns:
            if method == "iqr":
                outlier_mask_orig = (orig_df[col] < lower_bound) | (
                    orig_df[col] > upper_bound
                )
            else:
                outlier_mask_orig = (
                    np.abs((orig_df[col] - mean) / std) > zscore_threshold
                )
            outlier_stats[col]["orig_outliers"] = int(outlier_mask_orig.sum())
            orig_df.loc[outlier_mask_orig, col] = np.nan
        if eval_df is not None and col in eval_df.columns:
            if method == "iqr":
                outlier_mask_eval = (eval_df[col] < lower_bound) | (
                    eval_df[col] > upper_bound
                )
            else:
                outlier_mask_eval = (
                    np.abs((eval_df[col] - mean) / std) > zscore_threshold
                )
            outlier_stats[col]["eval_outliers"] = int(outlier_mask_eval.sum())
            eval_df.loc[outlier_mask_eval, col] = np.nan

    return train_df, test_df, val_df, orig_df, eval_df, outlier_stats


def _create_missing_report(df: pd.DataFrame, columns: list) -> Dict[str, int]:
    """Create report of missing values per column."""
    missing_counts = {}
    for col in columns:
        if col in df.columns:
            missing_counts[col] = int(df[col].isnull().sum())
    return missing_counts
