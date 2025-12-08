"""
Categorical feature encoder using EDA metadata and auto-detection.

Converts columns detected as categorical by:
1. EDA (ydata-profiling): Categorical, Text types
2. Auto-detection: Integer/float-encoded categorical columns (e.g., 0/1 flags, ordinal scales)

For native categorical handling in boost algorithms (XGBoost, LightGBM, CatBoost).

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
from rich.table import Table

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


def _auto_detect_categorical(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    threshold: int,
    target_column: str | None,
) -> Tuple[List[str], Dict[str, Dict]]:
    """
    Auto-detect integer/float-encoded categorical columns.

    Analyzes train + test together to find columns with:
    - Low cardinality (< threshold unique values)
    - Numeric dtype (int64 or float64)

    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        threshold: Maximum unique values for categorical
        target_column: Target column to exclude

    Returns:
        Tuple of (categorical_columns, metadata)
    """
    exclude_cols = ["id"]
    if target_column:
        exclude_cols.append(target_column)

    total_rows = len(train_df) + len(test_df)
    categorical_cols = []
    metadata = {}

    # Find numeric columns present in both datasets
    numeric_cols = [
        col
        for col in train_df.columns
        if col not in exclude_cols
        and train_df[col].dtype in ["int64", "float64"]
        and col in test_df.columns
    ]

    for col in numeric_cols:
        # Combine values from train + test
        combined_values = pd.concat([train_df[col].dropna(), test_df[col].dropna()])

        n_unique = combined_values.nunique()
        unique_ratio = n_unique / total_rows

        # Criteria: max unique <= threshold AND ratio < 1%
        is_categorical = n_unique <= threshold and unique_ratio < 0.01

        if is_categorical:
            unique_vals = sorted(combined_values.unique())

            # Detect type
            is_binary = n_unique == 2
            is_sequential = False

            if train_df[col].dtype == "int64" and n_unique > 2:
                min_val = int(min(unique_vals))
                max_val = int(max(unique_vals))
                expected_range = list(range(min_val, max_val + 1))
                is_sequential = unique_vals == expected_range

            # Classify type
            if is_binary:
                col_type = "Binary (auto-detected)"
            elif is_sequential:
                col_type = "Ordinal (auto-detected)"
            else:
                col_type = "Nominal (auto-detected)"

            categorical_cols.append(col)
            metadata[col] = {
                "type": col_type,
                "n_distinct": n_unique,
                "n_missing": train_df[col].isna().sum(),
                "unique_values": unique_vals,
                "is_binary": is_binary,
                "is_sequential": is_sequential,
                "train_unique": train_df[col].nunique(),
                "test_unique": test_df[col].nunique(),
            }

    return categorical_cols, metadata


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


def _create_feature_summary_table(
    train_df: pd.DataFrame,
    categorical_metadata: Dict[str, Dict],
    target_column: str | None,
) -> None:
    """
    Create and display comprehensive feature type summary footer.

    Shows all columns with their detected types, cardinality, and dtype.

    Args:
        train_df: Training dataframe (after conversion)
        categorical_metadata: Metadata for all categorical columns
        target_column: Target column name (if any)
    """
    console.print("\n" + "=" * 80)
    console.print("[bold cyan]ALL FEATURES TYPE SUMMARY[/bold cyan]")
    console.print("=" * 80 + "\n")

    table = Table(show_header=True, box=None)
    table.add_column("Column", style="cyan", width=30)
    table.add_column("Type", style="green", width=28)
    table.add_column("Distinct", style="yellow", justify="right", width=8)
    table.add_column("Dtype", style="magenta", width=12)

    # Categorize all columns
    categorical_cols = set(categorical_metadata.keys())
    all_cols = [col for col in train_df.columns if col not in ["id", target_column]]

    # Sort: categorical first (by type), then numeric
    def sort_key(col):
        if col in categorical_cols:
            meta = categorical_metadata[col]
            type_str = meta.get("type", "")
            # Order: Binary → Ordinal → Nominal → Categorical → Text → Numeric
            if "Binary" in type_str:
                return (0, meta.get("n_distinct", 0), col)
            elif "Ordinal" in type_str:
                return (1, meta.get("n_distinct", 0), col)
            elif "Nominal" in type_str:
                return (2, meta.get("n_distinct", 0), col)
            elif "Categorical" in type_str:
                return (3, meta.get("n_distinct", 0), col)
            elif "Text" in type_str:
                return (4, meta.get("n_distinct", 0), col)
            else:
                return (5, meta.get("n_distinct", 0), col)
        else:
            return (10, train_df[col].nunique(), col)  # Numeric last

    sorted_cols = sorted(all_cols, key=sort_key)

    # Build table rows
    cat_count = 0
    num_count = 0

    for col in sorted_cols:
        n_distinct = train_df[col].nunique()
        dtype_str = str(train_df[col].dtype)
        original_dtype_str = dtype_str  # Track original dtype before conversion

        if col in categorical_cols:
            meta = categorical_metadata[col]
            col_type = meta.get("type", "Categorical")
            cat_count += 1

            # Determine original dtype for conversion indicator
            if dtype_str == "category":
                # Try to infer original dtype from unique values
                if "unique_values" in meta and meta["unique_values"]:
                    import numpy as np
                    sample_val = meta["unique_values"][0]
                    # Check actual Python type
                    if isinstance(sample_val, (int, np.integer)):
                        original_dtype = "int64"
                    elif isinstance(sample_val, (float, np.floating)):
                        original_dtype = "float64"
                    else:
                        original_dtype = "object"
                else:
                    # Fallback: assume int64 for auto-detected, object for EDA
                    original_dtype = "int64" if "auto-detected" in col_type else "object"

                dtype_display = f"{original_dtype}→cat"
            else:
                dtype_display = dtype_str
        else:
            # Not in categorical_cols - check if object/string type
            if dtype_str in ["object", "string"]:
                col_type = "Categorical (not converted)"
                cat_count += 1  # Count as categorical but note not converted
                dtype_display = dtype_str
            else:
                col_type = "Numeric"
                dtype_display = dtype_str
                num_count += 1

        table.add_row(col, col_type, str(n_distinct), dtype_display)

    console.print(table)

    # Summary stats
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Categorical features: [cyan]{cat_count}[/cyan]")
    console.print(f"  Numeric features:     [yellow]{num_count}[/yellow]")
    console.print(f"  Total features:       [green]{cat_count + num_count}[/green]")
    if target_column:
        console.print(f"  Target (excluded):    [red]{target_column}[/red]")
    console.print()


def fit_transform(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, Dict[str, Any]]:
    """
    Convert categorical columns to dtype='category' using EDA metadata + auto-detection.

    Workflow:
    1. Read EDA summary from experiments/eda/artifacts/eda/eda_summary.json (optional)
    2. Extract categorical columns from EDA (Categorical + Text + low-cardinality Numeric)
    3. Auto-detect integer/float-encoded categorical columns (e.g., 0/1 flags, ordinal scales)
    4. Merge results (unique columns only)
    5. Convert columns to dtype='category' in train/val/test
    6. Display comprehensive feature type summary
    7. Store metadata in state for model to access

    Config parameters:
        max_cardinality: int (default: 50) - Maximum distinct values for categorical (EDA)
        exclude_text_type: bool (default: False) - Skip "Text" type columns from EDA
        include_numeric_categories: bool (default: True) - Include numeric low-cardinality columns (EDA)
        enable_auto_detect: bool (default: True) - Enable auto-detection of numeric categorical columns
        auto_detect_threshold: int (default: 25) - Max unique values for auto-detection

    Args:
        train_df: Training dataframe
        val_df: Validation dataframe (optional)
        test_df: Test dataframe
        config: Configuration dictionary

    Returns:
        Tuple of (train_df, val_df, test_df, state_dict)

    Raises:
        FileNotFoundError: If EDA summary not found (and fallback enabled)
    """
    # Extract config
    max_cardinality = config.get("max_cardinality", 50)
    exclude_text_type = config.get("exclude_text_type", False)
    include_numeric_categories = config.get("include_numeric_categories", True)
    enable_auto_detect = config.get("enable_auto_detect", True)
    auto_detect_threshold = config.get("auto_detect_threshold", 25)

    # Get project root and target column
    system_config = config.get("_system", {})
    project_root = Path(system_config.get("project_root", "."))

    dataset_config = config.get("_dataset", {})
    target_column = dataset_config.get("target")

    console.print(f"\n[bold cyan]Categorical Encoder:[/bold cyan]")
    console.print(f"  Max cardinality (EDA): {max_cardinality}")
    console.print(f"  Exclude text type: {exclude_text_type}")
    console.print(f"  Include numeric categories (EDA): {include_numeric_categories}")
    console.print(f"  Auto-detect enabled: {enable_auto_detect}")
    if enable_auto_detect:
        console.print(f"  Auto-detect threshold: {auto_detect_threshold}")

    # Step 1: Read EDA metadata (optional)
    eda_metadata = {}
    eda_cols = []

    try:
        eda_data = _read_eda_metadata(project_root)
        console.print(f"  [green]✓[/green] EDA metadata loaded")

        # Extract categorical columns from EDA
        eda_cols, eda_metadata = _extract_categorical_columns(
            eda_data,
            max_cardinality,
            exclude_text_type,
            include_numeric_categories,
            target_column,
        )
        console.print(f"  [green]✓[/green] Found {len(eda_cols)} categorical columns from EDA")

    except FileNotFoundError as e:
        console.print(f"  [yellow]⚠[/yellow] EDA metadata not found")
        console.print(f"  [yellow]Skipping EDA-based detection[/yellow]")

        # Fallback: detect categorical columns using pandas (only object dtypes)
        eda_cols = train_df.select_dtypes(include=["object"]).columns.tolist()
        if target_column and target_column in eda_cols:
            eda_cols.remove(target_column)

        eda_metadata = {
            col: {
                "type": "object (fallback)",
                "n_distinct": train_df[col].nunique(),
                "n_missing": train_df[col].isna().sum(),
            }
            for col in eda_cols
        }
        if eda_cols:
            console.print(f"  [green]✓[/green] Fallback: detected {len(eda_cols)} object columns")

    # Step 2: Auto-detect integer/float-encoded categorical columns
    auto_detect_metadata = {}
    auto_detect_cols = []

    if enable_auto_detect:
        console.print(f"\n[bold cyan]Auto-detecting numeric categorical columns...[/bold cyan]")
        auto_detect_cols, auto_detect_metadata = _auto_detect_categorical(
            train_df, test_df, auto_detect_threshold, target_column
        )
        console.print(f"  [green]✓[/green] Found {len(auto_detect_cols)} numeric categorical columns")

        # Show detected columns
        if auto_detect_cols:
            for col in auto_detect_cols:
                meta = auto_detect_metadata[col]
                type_str = meta.get("type", "")
                n_distinct = meta.get("n_distinct", "?")
                console.print(f"    • {col:30s} | {type_str:28s} | {n_distinct} distinct")

    # Step 3: Merge results (unique columns only)
    all_categorical_metadata = {**eda_metadata, **auto_detect_metadata}
    categorical_cols = list(dict.fromkeys(eda_cols + auto_detect_cols))  # Preserve order, remove duplicates

    console.print(f"\n[bold]Total categorical columns:[/bold] {len(categorical_cols)}")
    console.print(f"  From EDA: {len(eda_cols)}")
    console.print(f"  From auto-detect: {len(auto_detect_cols)}")
    console.print(f"  Overlap: {len(set(eda_cols) & set(auto_detect_cols))}")

    # Step 4: Convert to dtype='category'
    train_df, train_converted = _convert_to_category(train_df, categorical_cols, "train")
    test_df, test_converted = _convert_to_category(test_df, categorical_cols, "test")

    if val_df is not None:
        val_df, val_converted = _convert_to_category(val_df, categorical_cols, "validation")
    else:
        val_converted = []

    console.print(f"  [green]✓[/green] Converted {len(train_converted)} columns to category dtype")

    # Step 5: Display comprehensive feature type summary (footer)
    _create_feature_summary_table(train_df, all_categorical_metadata, target_column)

    # Build state dictionary
    state = {
        "categorical_columns": categorical_cols,
        "eda_metadata": eda_metadata,
        "auto_detect_metadata": auto_detect_metadata,
        "all_categorical_metadata": all_categorical_metadata,
        "conversion_summary": {
            "train_converted": len(train_converted),
            "test_converted": len(test_converted),
            "val_converted": len(val_converted),
            "target_excluded": target_column,
            "eda_count": len(eda_cols),
            "auto_detect_count": len(auto_detect_cols),
            "overlap_count": len(set(eda_cols) & set(auto_detect_cols)),
        },
        "config": {
            "max_cardinality": max_cardinality,
            "exclude_text_type": exclude_text_type,
            "include_numeric_categories": include_numeric_categories,
            "enable_auto_detect": enable_auto_detect,
            "auto_detect_threshold": auto_detect_threshold,
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
