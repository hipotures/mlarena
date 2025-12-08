"""
Categorical feature encoder using EDA metadata.

Converts columns detected as categorical by EDA (ydata-profiling) to dtype='category'
for native categorical handling in boost algorithms (XGBoost, LightGBM, CatBoost).

This module implements the preprocessing interface expected by MLArena:
- fit_transform(train_df, val_df, test_df, config) -> (train_df, val_df, test_df, state_dict)
- transform(df, state_dict, config) -> df  # Optional, for inference
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from rich.console import Console

console = Console()


def _read_eda_metadata(project_root: Path) -> Dict[str, Any]:
    """
    Read EDA metadata from experiments/eda/artifacts/eda/eda_summary.json.

    Args:
        project_root: Project root directory

    Returns:
        Dictionary with EDA metadata

    Raises:
        FileNotFoundError: If EDA summary not found
    """
    eda_summary_path = project_root / "experiments" / "eda" / "artifacts" / "eda" / "eda_summary.json"

    if not eda_summary_path.exists():
        raise FileNotFoundError(
            f"EDA summary not found: {eda_summary_path}\n"
            f"Run: uv run python scripts/mla.py eda --project <project-name> --force"
        )

    with open(eda_summary_path) as f:
        return json.load(f)


def _extract_categorical_columns(
    eda_data: Dict[str, Any],
    max_cardinality: int,
    exclude_text_type: bool,
    include_numeric_categories: bool,
    target_column: str | None,
) -> Tuple[List[str], Dict[str, Dict]]:
    """
    Extract categorical columns from EDA metadata.

    Args:
        eda_data: EDA summary dictionary
        max_cardinality: Maximum number of distinct values for categorical
        exclude_text_type: Whether to exclude "Text" type columns
        include_numeric_categories: Whether to include numeric low-cardinality columns
        target_column: Target column to exclude

    Returns:
        Tuple of (categorical_columns, eda_metadata)
    """
    variables = eda_data.get("train", {}).get("variables", {})

    categorical_cols = []
    eda_metadata = {}

    for col, meta in variables.items():
        # Skip target column
        if target_column and col == target_column:
            continue

        col_type = meta.get("type", "")
        n_distinct = meta.get("n_distinct", 0)

        # Filter by cardinality
        if n_distinct > max_cardinality:
            continue

        # Categorical type (e.g., Sex, Pclass, Embarked)
        if col_type == "Categorical":
            categorical_cols.append(col)
            eda_metadata[col] = {
                "type": col_type,
                "n_distinct": n_distinct,
                "n_missing": meta.get("n_missing", 0),
            }

        # Text type (e.g., Name, Ticket, Cabin) - include if low cardinality
        elif col_type == "Text" and not exclude_text_type:
            categorical_cols.append(col)
            eda_metadata[col] = {
                "type": col_type,
                "n_distinct": n_distinct,
                "n_missing": meta.get("n_missing", 0),
            }

        # Numeric type with low cardinality (e.g., Pclass: 1,2,3)
        elif col_type == "Numeric" and include_numeric_categories and n_distinct <= 10:
            categorical_cols.append(col)
            eda_metadata[col] = {
                "type": f"{col_type} (treated as categorical)",
                "n_distinct": n_distinct,
                "n_missing": meta.get("n_missing", 0),
            }

    return categorical_cols, eda_metadata


def _convert_to_category(
    df: pd.DataFrame,
    categorical_cols: List[str],
    df_name: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convert specified columns to dtype='category'.

    Args:
        df: DataFrame to convert
        categorical_cols: List of column names to convert
        df_name: Name for logging (e.g., "train", "test")

    Returns:
        Tuple of (converted_df, actually_converted_cols)
    """
    df = df.copy()
    converted = []

    for col in categorical_cols:
        if col not in df.columns:
            console.print(f"[yellow]Warning: Column '{col}' not found in {df_name} data, skipping[/yellow]")
            continue

        try:
            df[col] = df[col].astype('category')
            converted.append(col)
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to convert '{col}' to category in {df_name}: {e}[/yellow]")

    return df, converted


def fit_transform(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, Dict[str, Any]]:
    """
    Convert categorical columns to dtype='category' based on EDA metadata.

    Workflow:
    1. Read EDA summary from experiments/eda/artifacts/eda/eda_summary.json
    2. Extract categorical columns (Categorical + Text with cardinality filter)
    3. Convert columns to dtype='category' in train/val/test
    4. Store metadata in state for model to access

    Config parameters:
        max_cardinality: int (default: 50) - Maximum distinct values for categorical
        exclude_text_type: bool (default: False) - Skip "Text" type columns from EDA
        include_numeric_categories: bool (default: True) - Include numeric low-cardinality columns

    Args:
        train_df: Training dataframe
        val_df: Validation dataframe (optional)
        test_df: Test dataframe
        config: Configuration dictionary

    Returns:
        Tuple of (train_df, val_df, test_df, state_dict)

    Raises:
        FileNotFoundError: If EDA summary not found
    """
    # Extract config
    max_cardinality = config.get("max_cardinality", 50)
    exclude_text_type = config.get("exclude_text_type", False)
    include_numeric_categories = config.get("include_numeric_categories", True)

    # Get project root and target column
    system_config = config.get("_system", {})
    project_root = Path(system_config.get("project_root", "."))

    dataset_config = config.get("_dataset", {})
    target_column = dataset_config.get("target")

    console.print(f"\n[bold cyan]Categorical Encoder:[/bold cyan]")
    console.print(f"  Max cardinality: {max_cardinality}")
    console.print(f"  Exclude text type: {exclude_text_type}")
    console.print(f"  Include numeric categories: {include_numeric_categories}")

    # Read EDA metadata
    try:
        eda_data = _read_eda_metadata(project_root)
        console.print(f"  [green]✓[/green] EDA metadata loaded")
    except FileNotFoundError as e:
        console.print(f"  [yellow]⚠[/yellow] {e}")
        console.print(f"  [yellow]Falling back to pandas dtype detection[/yellow]")

        # Fallback: detect categorical columns using pandas
        categorical_cols = train_df.select_dtypes(exclude=["number"]).columns.tolist()
        if target_column and target_column in categorical_cols:
            categorical_cols.remove(target_column)

        eda_metadata = {col: {"type": "object (fallback)", "n_distinct": train_df[col].nunique()} for col in categorical_cols}
    else:
        # Extract categorical columns from EDA
        categorical_cols, eda_metadata = _extract_categorical_columns(
            eda_data,
            max_cardinality,
            exclude_text_type,
            include_numeric_categories,
            target_column,
        )

    console.print(f"  [green]✓[/green] Found {len(categorical_cols)} categorical columns")

    # Convert to dtype='category'
    train_df, train_converted = _convert_to_category(train_df, categorical_cols, "train")
    test_df, test_converted = _convert_to_category(test_df, categorical_cols, "test")

    if val_df is not None:
        val_df, val_converted = _convert_to_category(val_df, categorical_cols, "validation")

    console.print(f"  [green]✓[/green] Converted {len(train_converted)} columns to category dtype")

    # Print summary table
    if categorical_cols:
        from rich.table import Table

        table = Table(title="Categorical Columns", show_header=True)
        table.add_column("Column", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Distinct", style="yellow", justify="right")
        table.add_column("Missing", style="red", justify="right")

        for col in categorical_cols:
            meta = eda_metadata.get(col, {})
            table.add_row(
                col,
                meta.get("type", "unknown"),
                str(meta.get("n_distinct", "?")),
                str(meta.get("n_missing", 0)),
            )

        console.print(table)

    # Build state dictionary
    state = {
        "categorical_columns": categorical_cols,
        "eda_metadata": eda_metadata,
        "conversion_summary": {
            "train_converted": len(train_converted),
            "test_converted": len(test_converted),
            "val_converted": len(val_converted) if val_df is not None else 0,
            "target_excluded": target_column,
        },
        "config": {
            "max_cardinality": max_cardinality,
            "exclude_text_type": exclude_text_type,
            "include_numeric_categories": include_numeric_categories,
        },
    }

    return train_df, val_df, test_df, state


def transform(df: pd.DataFrame, state_dict: Dict[str, Any], config: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply categorical conversion to new data.

    Args:
        df: Dataframe to transform
        state_dict: State dictionary from fit_transform
        config: Configuration dictionary

    Returns:
        Transformed dataframe
    """
    categorical_cols = state_dict.get("categorical_columns", [])

    if not categorical_cols:
        return df.copy()

    df = df.copy()

    for col in categorical_cols:
        if col in df.columns:
            try:
                df[col] = df[col].astype('category')
            except Exception:
                pass  # Silently skip conversion errors in inference

    return df
