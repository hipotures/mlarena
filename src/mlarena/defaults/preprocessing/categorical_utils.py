"""
Shared utilities for handling categorical columns in preprocessing pipelines.

Helper functions to read and apply categorical metadata from previous preprocessing steps.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from rich.console import Console

console = Console()


def read_categorical_columns_from_previous_step(
    config: Dict,
    input_source: Optional[str] = None,
) -> List[str]:
    """
    Read categorical_columns from previous preprocessing step's state.json.

    Args:
        config: Config dict with _system.project_root and optional input_source
        input_source: Name of previous preprocessing template (e.g., "categorical_boost")

    Returns:
        List of categorical column names, or empty list if not found

    Example:
        # In av_weights.py:
        categorical_cols = read_categorical_columns_from_previous_step(
            config, input_source="categorical_boost"
        )
        train_df = apply_categorical_dtypes(train_df, categorical_cols)
    """
    # Try to get input_source from config if not provided
    if input_source is None:
        input_source = config.get("input_source")

    if not input_source:
        return []

    # Get project root
    system_config = config.get("_system", {})
    project_root = Path(system_config.get("project_root", "."))

    # Read state.json from previous step
    state_path = project_root / "experiments" / f"pre-{input_source}" / "state.json"

    if not state_path.exists():
        return []

    try:
        with open(state_path) as f:
            state = json.load(f)

        # Navigate to custom_module_state
        preprocess_payload = (
            state.get("modules", {}).get("preprocess", {}).get("payload", {})
        )
        custom_state = preprocess_payload.get("custom_module_state", {})
        categorical_columns = custom_state.get("categorical_columns", [])

        if categorical_columns:
            console.print(
                f"[cyan]✓[/cyan] Loaded {len(categorical_columns)} categorical columns from {input_source}"
            )
            console.print(
                f"  Columns: {', '.join(categorical_columns[:5])}"
                + (
                    f" ... (+{len(categorical_columns) - 5} more)"
                    if len(categorical_columns) > 5
                    else ""
                )
            )

        return categorical_columns

    except Exception as e:
        console.print(
            f"[yellow]Warning: Failed to read categorical metadata from {input_source}: {e}[/yellow]"
        )
        return []


def apply_categorical_dtypes(
    df: pd.DataFrame,
    categorical_cols: List[str],
    df_name: str = "data",
) -> pd.DataFrame:
    """
    Convert specified columns to dtype='category'.

    Args:
        df: DataFrame to convert
        categorical_cols: List of column names to convert
        df_name: Name for logging (e.g., "train", "test")

    Returns:
        DataFrame with categorical dtypes (copy)

    Example:
        train_df = apply_categorical_dtypes(train_df, ["gender", "education"], "train")
    """
    if not categorical_cols:
        return df

    df = df.copy()
    converted = []
    skipped = []

    for col in categorical_cols:
        if col not in df.columns:
            skipped.append(col)
            continue

        if df[col].dtype.name == "category":
            # Already category, skip
            continue

        try:
            df[col] = df[col].astype("category")
            converted.append(col)
        except Exception as e:
            console.print(
                f"[yellow]Warning: Failed to convert '{col}' to category in {df_name}: {e}[/yellow]"
            )

    if converted:
        console.print(
            f"[cyan]✓[/cyan] Converted {len(converted)} columns to category dtype in {df_name}"
        )

    if skipped:
        console.print(
            f"[yellow]Note: {len(skipped)} columns not found in {df_name}: {', '.join(skipped[:3])}"
            + (f" ... (+{len(skipped) - 3} more)" if len(skipped) > 3 else "")
            + "[/yellow]"
        )

    return df


def restore_categorical_types_from_chain(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function: read categorical columns from previous step and apply to train/test.

    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        config: Config dict with input_source and project_root

    Returns:
        Tuple of (train_df, test_df) with categorical dtypes applied

    Example:
        # In av_weights.py fit_transform():
        train_df, test_df = restore_categorical_types_from_chain(train_df, test_df, config)
        # Now train_df and test_df have correct categorical dtypes
    """
    categorical_cols = read_categorical_columns_from_previous_step(config)

    if categorical_cols:
        train_df = apply_categorical_dtypes(train_df, categorical_cols, "train")
        test_df = apply_categorical_dtypes(test_df, categorical_cols, "test")

    return train_df, test_df
