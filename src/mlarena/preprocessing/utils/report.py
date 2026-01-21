"""Logging and reporting utilities for preprocessing sub-modules."""

from typing import Dict, List
import pandas as pd


def log_transformation_summary(
    before_df: pd.DataFrame, after_df: pd.DataFrame, submodule_name: str
) -> Dict:
    """
    Generate summary of transformation (shape changes, new columns, etc.).

    Args:
        before_df: DataFrame before transformation
        after_df: DataFrame after transformation
        submodule_name: Name of the sub-module

    Returns:
        Dictionary with transformation summary
    """
    before_cols = set(before_df.columns)
    after_cols = set(after_df.columns)

    added_cols = list(after_cols - before_cols)
    removed_cols = list(before_cols - after_cols)
    kept_cols = list(before_cols & after_cols)

    summary = {
        "submodule": submodule_name,
        "shape_before": list(before_df.shape),
        "shape_after": list(after_df.shape),
        "rows_before": before_df.shape[0],
        "rows_after": after_df.shape[0],
        "cols_before": before_df.shape[1],
        "cols_after": after_df.shape[1],
        "added_columns": added_cols,
        "removed_columns": removed_cols,
        "kept_columns_count": len(kept_cols),
        "rows_changed": after_df.shape[0] - before_df.shape[0],
        "cols_changed": after_df.shape[1] - before_df.shape[1],
    }

    return summary


def create_preprocessing_report(
    train_before: pd.DataFrame,
    train_after: pd.DataFrame,
    test_before: pd.DataFrame,
    test_after: pd.DataFrame,
    config: Dict,
) -> Dict:
    """
    Standard report structure for all sub-modules.

    Args:
        train_before: Training DataFrame before transformation
        train_after: Training DataFrame after transformation
        test_before: Test DataFrame before transformation
        test_after: Test DataFrame after transformation
        config: Configuration dictionary

    Returns:
        Dictionary with preprocessing report
    """
    train_summary = log_transformation_summary(
        train_before, train_after, "preprocessing"
    )
    test_summary = log_transformation_summary(test_before, test_after, "preprocessing")

    report = {
        "version": "1.0",
        "train": train_summary,
        "test": test_summary,
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
        "metadata": {
            "train_test_cols_match": set(train_after.columns)
            == set(test_after.columns),
            "train_test_rows_ratio": test_after.shape[0] / train_after.shape[0]
            if train_after.shape[0] > 0
            else None,
        },
    }

    return report


def create_column_stats_report(
    df: pd.DataFrame, columns: List[str] = None
) -> Dict[str, Dict]:
    """
    Generate statistics report for specified columns.

    Args:
        df: DataFrame to analyze
        columns: List of columns to include (None for all)

    Returns:
        Dictionary mapping column names to their statistics
    """
    columns = columns or df.columns.tolist()

    stats = {}
    for col in columns:
        if col not in df.columns:
            continue

        col_stats = {
            "dtype": str(df[col].dtype),
            "missing_count": int(df[col].isnull().sum()),
            "missing_rate": float(df[col].isnull().mean()),
            "unique_count": int(df[col].nunique()),
        }

        # Add numeric-specific stats
        if pd.api.types.is_numeric_dtype(df[col]):
            col_stats.update(
                {
                    "mean": float(df[col].mean())
                    if not df[col].isnull().all()
                    else None,
                    "std": float(df[col].std()) if not df[col].isnull().all() else None,
                    "min": float(df[col].min()) if not df[col].isnull().all() else None,
                    "max": float(df[col].max()) if not df[col].isnull().all() else None,
                    "median": float(df[col].median())
                    if not df[col].isnull().all()
                    else None,
                }
            )

        stats[col] = col_stats

    return stats


def create_missing_values_report(df: pd.DataFrame) -> Dict:
    """
    Generate missing values report.

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary with missing values statistics
    """
    missing_counts = df.isnull().sum()
    missing_rates = df.isnull().mean()

    cols_with_missing = missing_counts[missing_counts > 0].index.tolist()

    report = {
        "total_missing": int(df.isnull().sum().sum()),
        "missing_rate_overall": float(df.isnull().mean().mean()),
        "columns_with_missing": cols_with_missing,
        "columns_with_missing_count": len(cols_with_missing),
        "columns_by_missing_rate": {
            col: {
                "count": int(missing_counts[col]),
                "rate": float(missing_rates[col]),
            }
            for col in cols_with_missing
        },
    }

    return report


def create_data_quality_report(df: pd.DataFrame) -> Dict:
    """
    Generate comprehensive data quality report.

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary with data quality metrics
    """
    report = {
        "shape": list(df.shape),
        "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_rows_rate": float(df.duplicated().mean()),
        "column_types": {
            "numeric": len(df.select_dtypes(include=["number"]).columns),
            "categorical": len(
                df.select_dtypes(include=["object", "category"]).columns
            ),
            "datetime": len(df.select_dtypes(include=["datetime"]).columns),
            "boolean": len(df.select_dtypes(include=["bool"]).columns),
        },
        "missing_values": create_missing_values_report(df),
    }

    return report
