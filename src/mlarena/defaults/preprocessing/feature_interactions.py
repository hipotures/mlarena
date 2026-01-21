"""
Feature Interactions Sub-Module

Purpose: Create simple arithmetic interaction features (add, sub, mul, div) between numeric pairs.
Libraries: pandas, numpy
Parameters:
  - interaction_types: List of operations to create between numeric pairs (add|sub|mul|div)
  - numeric_pairs: Explicit list of column pairs for interactions
  - auto_pair_numeric: Automatically generate numeric pairs (limited by max_auto_pairs)
  - max_auto_pairs: Limit for auto-generated pairs
  - max_generated_features: Guardrail for total new features created
"""

from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd

from mlarena.preprocessing.utils import (
    validation,
    artifacts,
    dataframe_utils,
    report,
)


def _unique_name(base: str, existing: set) -> str:
    """Generate a unique column name if base already exists."""
    if base not in existing:
        return base
    idx = 1
    candidate = f"{base}__{idx}"
    while candidate in existing:
        idx += 1
        candidate = f"{base}__{idx}"
    return candidate


def _append_new_columns(df: pd.DataFrame, new_cols: Dict[str, Any]) -> pd.DataFrame:
    if not new_cols:
        return df
    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)


def _prepare_interaction_pairs(
    numeric_pairs: List[List[str]],
    numeric_cols: List[str],
    auto_pair_numeric: bool,
    max_auto_pairs: int,
) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []

    # Add explicit pairs
    for pair in numeric_pairs or []:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            warnings.warn(f"Skipping invalid pair definition: {pair}")
            continue
        col_a, col_b = pair
        if col_a == col_b:
            continue
        pairs.append((col_a, col_b))

    # Auto-generate pairs if requested
    if auto_pair_numeric:
        auto_pairs = list(combinations(numeric_cols, 2))[: max(0, int(max_auto_pairs))]
        pairs.extend(auto_pairs)

    # Deduplicate while preserving order
    seen = set()
    deduped_pairs = []
    for a, b in pairs:
        if (a, b) not in seen and (b, a) not in seen:
            deduped_pairs.append((a, b))
            seen.add((a, b))
    return deduped_pairs


def _apply_interactions(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    orig_df: pd.DataFrame | None,
    pairs: List[Tuple[str, str]],
    interaction_types: List[str],
    max_new_features: int,
    existing_cols: set,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame | None,
    pd.DataFrame,
    pd.DataFrame | None,
    List[str],
    List[Dict[str, Any]],
]:
    new_cols: List[str] = []
    details: List[Dict[str, Any]] = []
    allowed_ops = {"add", "sub", "mul", "div"}
    train_new_cols: Dict[str, Any] = {}
    test_new_cols: Dict[str, Any] = {}
    val_new_cols: Dict[str, Any] = {}
    orig_new_cols: Dict[str, Any] = {}
    limit_reached = False

    seen_names = existing_cols.copy()
    for col_a, col_b in pairs:
        if col_a not in train_df.columns or col_b not in train_df.columns:
            warnings.warn(f"Skipping pair ({col_a}, {col_b}) - column missing in train")
            continue
        if col_a not in test_df.columns or col_b not in test_df.columns:
            warnings.warn(f"Skipping pair ({col_a}, {col_b}) - column missing in test")
            continue
        if not (
            pd.api.types.is_numeric_dtype(train_df[col_a])
            and pd.api.types.is_numeric_dtype(train_df[col_b])
        ):
            warnings.warn(f"Skipping pair ({col_a}, {col_b}) - non-numeric dtype")
            continue

        for op in interaction_types:
            if op not in allowed_ops:
                warnings.warn(f"Unsupported interaction '{op}' - skipping")
                continue
            if len(new_cols) >= max_new_features:
                limit_reached = True
                break

            base_name = f"{col_a}_{op}_{col_b}"
            new_name = _unique_name(base_name, seen_names)
            seen_names.add(new_name)

            if op == "add":
                train_series = train_df[col_a] + train_df[col_b]
                test_series = test_df[col_a] + test_df[col_b]
                val_series = (
                    val_df[col_a] + val_df[col_b] if val_df is not None else None
                )
            elif op == "sub":
                train_series = train_df[col_a] - train_df[col_b]
                test_series = test_df[col_a] - test_df[col_b]
                val_series = (
                    val_df[col_a] - val_df[col_b] if val_df is not None else None
                )
            elif op == "mul":
                train_series = train_df[col_a] * train_df[col_b]
                test_series = test_df[col_a] * test_df[col_b]
                val_series = (
                    val_df[col_a] * val_df[col_b] if val_df is not None else None
                )
            elif op == "div":
                with np.errstate(divide="ignore", invalid="ignore"):
                    train_series = np.where(
                        train_df[col_b] != 0, train_df[col_a] / train_df[col_b], np.nan
                    )
                    test_series = np.where(
                        test_df[col_b] != 0, test_df[col_a] / test_df[col_b], np.nan
                    )
                    val_series = (
                        np.where(
                            val_df[col_b] != 0, val_df[col_a] / val_df[col_b], np.nan
                        )
                        if val_df is not None
                        else None
                    )

            train_new_cols[new_name] = train_series
            test_new_cols[new_name] = test_series
            if val_df is not None and val_series is not None:
                val_new_cols[new_name] = val_series
            if (
                orig_df is not None
                and col_a in orig_df.columns
                and col_b in orig_df.columns
            ):
                if op == "add":
                    orig_new_cols[new_name] = orig_df[col_a] + orig_df[col_b]
                elif op == "sub":
                    orig_new_cols[new_name] = orig_df[col_a] - orig_df[col_b]
                elif op == "mul":
                    orig_new_cols[new_name] = orig_df[col_a] * orig_df[col_b]
                elif op == "div":
                    with np.errstate(divide="ignore", invalid="ignore"):
                        orig_new_cols[new_name] = np.where(
                            orig_df[col_b] != 0,
                            orig_df[col_a] / orig_df[col_b],
                            np.nan,
                        )

            new_cols.append(new_name)
            details.append(
                {
                    "type": "interaction",
                    "operation": op,
                    "columns": [col_a, col_b],
                    "new_column": new_name,
                }
            )

            if len(new_cols) >= max_new_features:
                limit_reached = True
                break
        if limit_reached:
            break

    train_df = _append_new_columns(train_df, train_new_cols)
    test_df = _append_new_columns(test_df, test_new_cols)
    if val_df is not None:
        val_df = _append_new_columns(val_df, val_new_cols)
    if orig_df is not None:
        orig_df = _append_new_columns(orig_df, orig_new_cols)

    return train_df, val_df, test_df, orig_df, new_cols, details


def fit_transform(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
    orig_df: pd.DataFrame | None = None,
) -> Tuple[
    pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None, Dict[str, Any]
]:
    """
    Feature interactions preprocessing sub-module.
    """
    # 1. Extract config
    artifact_dir = Path(config.get("_artifact_dir", "."))
    dataset_config = config.get("_dataset", {})
    id_column = dataset_config.get("id_column", "id")
    target_column = dataset_config.get("target")
    ignored_columns = dataset_config.get("ignored_columns", [])

    # 2. Validate config
    required_params: List[str] = []
    optional_params: Dict[str, Any] = {
        "interaction_types": [],
        "numeric_pairs": [],
        "auto_pair_numeric": False,
        "max_auto_pairs": 30,
        "max_generated_features": 200,
        "use_original_features_only": True,
    }
    validation.validate_config(config, required_params, optional_params)

    # Validate choices
    for op in config["interaction_types"]:
        validation.validate_choice(
            op, ["add", "sub", "mul", "div"], "interaction_types"
        )

    validation.validate_numeric_range(
        config["max_generated_features"],
        min_value=1,
        max_value=5000,
        param_name="max_generated_features",
    )

    # 3. Create sub-module artifact directory
    submodule_dir = artifacts.get_submodule_artifact_dir(
        artifact_dir, "feature_interactions"
    )

    # 4. Save original DataFrames for reporting
    train_df_original = dataframe_utils.copy_dataframe(train_df)
    test_df_original = dataframe_utils.copy_dataframe(test_df)

    # 5. Prepare column sets
    exclude_cols = [id_column, target_column] + ignored_columns
    exclude_cols = [col for col in exclude_cols if col]
    numeric_cols = dataframe_utils.get_numeric_columns(train_df, exclude=exclude_cols)
    use_orig_only = bool(config.get("use_original_features_only"))
    orig_features = config.get("_original_features") if use_orig_only else None
    if use_orig_only:
        numeric_cols = dataframe_utils.filter_original_columns(
            numeric_cols, orig_features
        )

    if not numeric_cols and config["interaction_types"]:
        warnings.warn("No numeric columns available for interactions")

    existing_cols = set(train_df.columns)
    new_columns: List[str] = []
    interaction_details: List[Dict[str, Any]] = []

    max_new = config["max_generated_features"]

    # 6. Interaction features
    numeric_pairs = config["numeric_pairs"]
    if use_orig_only and orig_features:
        orig_set = set(orig_features)
        numeric_pairs = [
            pair
            for pair in numeric_pairs
            if isinstance(pair, (list, tuple))
            and len(pair) == 2
            and pair[0] in orig_set
            and pair[1] in orig_set
        ]

    pairs = _prepare_interaction_pairs(
        numeric_pairs=numeric_pairs,
        numeric_cols=numeric_cols,
        auto_pair_numeric=config["auto_pair_numeric"],
        max_auto_pairs=config["max_auto_pairs"],
    )

    if pairs and config["interaction_types"]:
        train_df, val_df, test_df, orig_df, inter_cols, inter_details = (
            _apply_interactions(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                orig_df=orig_df,
                pairs=pairs,
                interaction_types=config["interaction_types"],
                max_new_features=max_new,
                existing_cols=existing_cols,
            )
        )
        new_columns.extend(inter_cols)
        interaction_details.extend(inter_details)

    # 7. Reports
    feature_report = {
        "new_columns": new_columns,
        "interactions": interaction_details,
        "total_new_features": len(new_columns),
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
    }
    artifacts.save_report(
        feature_report, submodule_dir, "feature_interactions_report.json"
    )

    transformation_summary = report.create_preprocessing_report(
        train_before=train_df_original,
        train_after=train_df,
        test_before=test_df_original,
        test_after=test_df,
        config=config,
    )
    artifacts.save_report(transformation_summary, submodule_dir, "summary.json")

    # 8. State dict
    state_dict = {
        "version": "1.0",
        "new_columns": new_columns,
        "interactions": interaction_details,
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
    }

    return train_df, val_df, test_df, orig_df, state_dict
