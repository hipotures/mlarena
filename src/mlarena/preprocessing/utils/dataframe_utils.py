"""Common DataFrame operations for preprocessing sub-modules."""

from typing import List, Tuple, Optional
import pandas as pd
import numpy as np


def get_numeric_columns(df: pd.DataFrame, exclude: List[str] = None) -> List[str]:
    """
    Return list of numeric column names.

    Args:
        df: DataFrame to analyze
        exclude: List of column names to exclude

    Returns:
        List of numeric column names
    """
    exclude = exclude or []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if col not in exclude]


def get_categorical_columns(df: pd.DataFrame, exclude: List[str] = None) -> List[str]:
    """
    Return list of categorical column names.

    Args:
        df: DataFrame to analyze
        exclude: List of column names to exclude

    Returns:
        List of categorical/object column names
    """
    exclude = exclude or []
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return [col for col in categorical_cols if col not in exclude]


def get_datetime_columns(df: pd.DataFrame, exclude: List[str] = None) -> List[str]:
    """
    Return list of datetime column names.

    Args:
        df: DataFrame to analyze
        exclude: List of column names to exclude

    Returns:
        List of datetime column names
    """
    exclude = exclude or []
    datetime_cols = df.select_dtypes(include=["datetime", "datetime64"]).columns.tolist()
    return [col for col in datetime_cols if col not in exclude]


def get_boolean_columns(df: pd.DataFrame, exclude: List[str] = None) -> List[str]:
    """
    Return list of boolean column names.

    Args:
        df: DataFrame to analyze
        exclude: List of column names to exclude

    Returns:
        List of boolean column names
    """
    exclude = exclude or []
    boolean_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    return [col for col in boolean_cols if col not in exclude]


def filter_original_columns(columns: List[str], original_features: Optional[List[str]] = None) -> List[str]:
    """
    Filter a column list to only those present in the original feature set.

    Args:
        columns: Candidate column list
        original_features: Original feature list (or None to skip filtering)

    Returns:
        Filtered list limited to original_features when provided.
    """
    if not original_features:
        return columns
    orig_set = set(original_features)
    return [col for col in columns if col in orig_set]


def safe_drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Drop columns that exist in DataFrame.

    Args:
        df: DataFrame to modify
        columns: List of column names to drop

    Returns:
        DataFrame with specified columns removed
    """
    existing_cols = [col for col in columns if col in df.columns]
    if existing_cols:
        return df.drop(columns=existing_cols)
    return df


def align_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fill_value: any = np.nan
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ensure train and test have same columns (add missing with fill_value).

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        fill_value: Value to use for missing columns (default: np.nan)

    Returns:
        Tuple of (train_df, test_df) with aligned columns
    """
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    # Add missing columns to test
    missing_in_test = train_cols - test_cols
    for col in missing_in_test:
        test_df[col] = fill_value

    # Add missing columns to train
    missing_in_train = test_cols - train_cols
    for col in missing_in_train:
        train_df[col] = fill_value

    # Reorder columns to match
    test_df = test_df[train_df.columns]

    return train_df, test_df


def get_columns_by_dtype(df: pd.DataFrame, dtypes: List[str]) -> List[str]:
    """
    Get columns matching any of the specified dtypes.

    Args:
        df: DataFrame to analyze
        dtypes: List of dtype strings (e.g., ['int64', 'float64'])

    Returns:
        List of column names matching the dtypes
    """
    return df.select_dtypes(include=dtypes).columns.tolist()


def get_constant_columns(df: pd.DataFrame, threshold: float = 1.0) -> List[str]:
    """
    Get columns with constant values (nunique <= threshold).

    Args:
        df: DataFrame to analyze
        threshold: Maximum number of unique values to consider constant

    Returns:
        List of constant column names
    """
    return [col for col in df.columns if df[col].nunique() <= threshold]


def get_high_missing_columns(df: pd.DataFrame, threshold: float = 0.5) -> List[str]:
    """
    Get columns with high missing value rate.

    Args:
        df: DataFrame to analyze
        threshold: Minimum missing rate to flag (0.0 to 1.0)

    Returns:
        List of column names with high missing rates
    """
    missing_rates = df.isnull().mean()
    return missing_rates[missing_rates >= threshold].index.tolist()


def copy_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a deep copy of DataFrame.

    Args:
        df: DataFrame to copy

    Returns:
        Deep copy of DataFrame
    """
    return df.copy()
