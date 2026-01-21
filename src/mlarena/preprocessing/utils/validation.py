"""Config validation utilities for preprocessing sub-modules."""

from typing import Any, Dict, List
import pandas as pd


def validate_config(
    config: Dict[str, Any], required: List[str], optional: Dict[str, Any]
) -> None:
    """
    Validate preprocessing config has required params, set defaults for optional.

    Args:
        config: Configuration dictionary to validate
        required: List of required parameter names
        optional: Dictionary of optional parameters with default values

    Raises:
        ValueError: If required parameters are missing
    """
    # Check required parameters
    missing = [param for param in required if param not in config]
    if missing:
        raise ValueError(f"Missing required configuration parameters: {missing}")

    # Set defaults for optional parameters
    for param, default_value in optional.items():
        if param not in config:
            config[param] = default_value


def validate_column_exists(df: pd.DataFrame, columns: List[str], context: str) -> None:
    """
    Raise error if columns missing from DataFrame.

    Args:
        df: DataFrame to check
        columns: List of required column names
        context: Description of where this validation is happening

    Raises:
        ValueError: If any columns are missing
    """
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"{context}: Missing columns in DataFrame: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def infer_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Return column types as dict of lists.

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary with keys:
        - 'numeric': List of numeric column names
        - 'categorical': List of categorical/object column names
        - 'datetime': List of datetime column names
        - 'boolean': List of boolean column names
    """
    column_types = {
        "numeric": [],
        "categorical": [],
        "datetime": [],
        "boolean": [],
    }

    for col in df.columns:
        dtype = df[col].dtype

        if pd.api.types.is_numeric_dtype(dtype):
            column_types["numeric"].append(col)
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            column_types["datetime"].append(col)
        elif pd.api.types.is_bool_dtype(dtype):
            column_types["boolean"].append(col)
        else:
            column_types["categorical"].append(col)

    return column_types


def validate_choice(value: str, choices: List[str], param_name: str) -> None:
    """
    Validate that a value is one of allowed choices.

    Args:
        value: Value to validate
        choices: List of allowed values
        param_name: Name of the parameter (for error message)

    Raises:
        ValueError: If value not in choices
    """
    if value not in choices:
        raise ValueError(
            f"Invalid value for '{param_name}': {value}. Must be one of: {choices}"
        )


def validate_numeric_range(
    value: float,
    min_value: float = None,
    max_value: float = None,
    param_name: str = "parameter",
) -> None:
    """
    Validate that a numeric value is within allowed range.

    Args:
        value: Value to validate
        min_value: Minimum allowed value (inclusive), None for no minimum
        max_value: Maximum allowed value (inclusive), None for no maximum
        param_name: Name of the parameter (for error message)

    Raises:
        ValueError: If value outside allowed range
    """
    if min_value is not None and value < min_value:
        raise ValueError(f"'{param_name}' must be >= {min_value}, got {value}")
    if max_value is not None and value > max_value:
        raise ValueError(f"'{param_name}' must be <= {max_value}, got {value}")
