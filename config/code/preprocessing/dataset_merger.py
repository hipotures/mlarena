"""
Dataset merger preprocessing module.

Merges external/original dataset with Kaggle competition training data.
Supports column alignment, name mapping, and optional source tracking.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from rich.console import Console
from rich.table import Table


def fit_transform(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    test_df: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame, Dict[str, Any]]:
    """
    Merge external dataset with Kaggle train data.

    Config keys:
        orig_path (str): Path to original dataset CSV (required)
        mode (str): 'align' (intersection) or 'union' (all columns), default 'align'
        source_flag (str|null): Column name for source tracking, default null
        column_mapping (dict): Map original column names to Kaggle names, default {}

    Returns:
        - train_merged: Kaggle train + original dataset
        - val_df: Unchanged (pass-through)
        - test_df: With source_flag if enabled (for feature consistency)
        - state_dict: Column alignment report and merge statistics
    """
    console = Console(force_terminal=True)

    # Get artifact directory and project root
    artifact_dir = Path(config.get("_artifact_dir", ".")).resolve()
    # For preprocessing chains: artifacts/preprocess → artifacts → step → chain → experiments → project_root
    # artifact_dir: {project}/experiments/{chain}/{step}/artifacts/preprocess/
    project_root = artifact_dir.parent.parent.parent.parent.parent

    # 1. Load original dataset
    orig_path_str = config.get("orig_path")
    if not orig_path_str:
        raise ValueError("Config key 'orig_path' is required")

    orig_path = Path(orig_path_str)
    if not orig_path.is_absolute():
        orig_path = project_root / orig_path

    if not orig_path.exists():
        raise FileNotFoundError(
            f"Original dataset not found: {orig_path}\n"
            f"Project root: {project_root}"
        )

    console.print(f"[cyan]Loading original dataset:[/cyan] {orig_path.relative_to(project_root)}")
    orig_df = pd.read_csv(orig_path)
    console.print(f"  Original rows: {len(orig_df):,}, columns: {len(orig_df.columns)}")

    # 1b. Apply column mapping (if provided)
    column_mapping = config.get("column_mapping", {})

    if column_mapping:
        # Validate: all mapping keys must exist in original
        missing_cols = set(column_mapping.keys()) - set(orig_df.columns)
        if missing_cols:
            raise ValueError(
                f"Column mapping references non-existent columns in original dataset: "
                f"{sorted(missing_cols)}\n"
                f"Available columns: {sorted(orig_df.columns)}"
            )

        # Rename columns: original_name -> kaggle_name
        orig_df = orig_df.rename(columns=column_mapping)

        # Report applied mappings
        console.print(f"\n[cyan]Applied column mappings ({len(column_mapping)}):[/cyan]")
        for orig_col, kaggle_col in sorted(column_mapping.items()):
            console.print(f"  {orig_col} → {kaggle_col}")

    # 2. Analyze column alignment
    kaggle_cols = set(train_df.columns)
    orig_cols = set(orig_df.columns)

    common_cols = kaggle_cols & orig_cols
    kaggle_only = kaggle_cols - orig_cols  # Missing in original
    orig_only = orig_cols - kaggle_cols    # Extra in original

    # Store for reporting
    alignment_stats = {
        "kaggle_columns": len(kaggle_cols),
        "original_columns": len(orig_cols),
        "matched_columns": len(common_cols),
        "kaggle_only_columns": sorted(list(kaggle_only)),
        "original_only_columns": sorted(list(orig_only)),
    }

    # 3. Column alignment based on mode
    mode = config.get("mode", "align")

    # Get dataset metadata (target column might not be in test)
    dataset_meta = config.get("_dataset", {})
    target_col = dataset_meta.get("target")

    if mode == "align":
        # Keep only common columns (intersection)
        if not common_cols:
            raise ValueError(
                "No common columns found between Kaggle and original datasets. "
                "Use 'column_mapping' to align column names or mode='union' to keep all columns."
            )

        final_columns_train = sorted(common_cols)
        # Test might not have target column
        final_columns_test = sorted(common_cols - {target_col} if target_col else common_cols)

        orig_aligned = orig_df[final_columns_train].copy()
        train_aligned = train_df[final_columns_train].copy()
        test_aligned = test_df[final_columns_test].copy()

    elif mode == "union":
        # Keep all columns (union), fill missing with NA
        # Train columns (includes target if present)
        train_cols_all = kaggle_cols | orig_cols
        # Test columns (excludes target)
        test_cols_all = (kaggle_cols | orig_cols) - {target_col} if target_col else (kaggle_cols | orig_cols)

        # Add missing columns to original
        for col in kaggle_only:
            orig_df[col] = pd.NA

        # Add missing columns to kaggle train
        for col in orig_only:
            train_df[col] = pd.NA

        # Add missing columns to test (but not target)
        for col in orig_only:
            if col != target_col:
                test_df[col] = pd.NA

        final_columns_train = sorted(train_cols_all)
        final_columns_test = sorted(test_cols_all)

        orig_aligned = orig_df[final_columns_train].copy()
        train_aligned = train_df[final_columns_train].copy()
        test_aligned = test_df[final_columns_test].copy()

    else:
        raise ValueError(
            f"Invalid mode: '{mode}'. Must be 'align' (intersection) or 'union' (all columns)"
        )

    # 4. Add source tracking flag (if enabled)
    source_flag = config.get("source_flag")

    if source_flag:
        # Add to train (kaggle=0, original=1)
        train_aligned[source_flag] = 0
        orig_aligned[source_flag] = 1

        # CRITICAL: Add to test too (all test is kaggle=0)
        test_aligned[source_flag] = 0

        console.print(f"\n[cyan]Source tracking enabled:[/cyan] '{source_flag}' (0=Kaggle, 1=Original)")

    # 5. Merge datasets
    train_merged = pd.concat([train_aligned, orig_aligned], ignore_index=True)

    # 6. Generate detailed report
    console.print("\n")
    table = Table(title="Dataset Merger Report", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Value", style="white")

    table.add_row("Kaggle columns", str(len(kaggle_cols)))
    table.add_row("Original columns", str(len(orig_cols)))
    table.add_row("Matched columns", f"{len(common_cols)}/{len(kaggle_cols)}")
    table.add_row("Mode", mode)
    table.add_row("Source flag", source_flag or "None")
    table.add_row("Column mapping", "Yes" if column_mapping else "No")
    table.add_row("", "")  # Spacer
    table.add_row("Kaggle rows", f"{len(train_df):,}")
    table.add_row("Original rows", f"{len(orig_df):,}")
    table.add_row("Merged rows", f"{len(train_merged):,}")
    table.add_row("Final columns", str(len(final_columns_train)))

    console.print(table)

    # Warn about mismatches
    if kaggle_only:
        console.print(f"\n[yellow]⚠ Columns in Kaggle but NOT in original ({len(kaggle_only)}):[/yellow]")
        for col in sorted(kaggle_only)[:10]:  # Show first 10
            console.print(f"  - {col}")
        if len(kaggle_only) > 10:
            console.print(f"  ... and {len(kaggle_only)-10} more")

    if orig_only:
        console.print(f"\n[yellow]⚠ Columns in original but NOT in Kaggle ({len(orig_only)}):[/yellow]")
        for col in sorted(orig_only)[:10]:
            console.print(f"  - {col}")
        if len(orig_only) > 10:
            console.print(f"  ... and {len(orig_only)-10} more")

    # Mode-specific notes
    if mode == "align" and (kaggle_only or orig_only):
        console.print(f"\n[dim]Mode 'align': Using only {len(common_cols)} matched columns[/dim]")
    elif mode == "union" and (kaggle_only or orig_only):
        console.print(f"\n[dim]Mode 'union': Missing columns filled with NA[/dim]")

    # Warn if low match rate
    match_rate = len(common_cols) / len(kaggle_cols) if kaggle_cols else 0
    if match_rate < 0.5:
        console.print(
            f"\n[bold yellow]⚠ Low column match rate ({match_rate:.1%}). "
            f"Consider using 'column_mapping' to align column names.[/bold yellow]"
        )

    # 7. Return state
    state_dict = {
        "version": "1.0",
        "module": "dataset_merger",
        "config": {
            "orig_path": str(orig_path),
            "mode": mode,
            "source_flag": source_flag,
            "column_mapping": column_mapping,
        },
        "alignment": alignment_stats,
        "rows": {
            "kaggle": len(train_df),
            "original": len(orig_df),
            "merged": len(train_merged),
        },
        "column_mapping_applied": len(column_mapping) > 0,
        "match_rate": match_rate,
    }

    return train_merged, val_df, test_aligned, state_dict


def transform(df: pd.DataFrame, state_dict: Dict[str, Any], config: Dict[str, Any]) -> pd.DataFrame:
    """
    Transform function (not used for merger, but required for interface compatibility).

    Dataset merger is a training-time operation only.
    """
    return df.copy()
