"""
Feature Engineering Sub-Module

Purpose: Create interaction features, polynomial expansions, and group-based aggregations in a configurable, reusable way.
Libraries: pandas, numpy, sklearn.preprocessing.PolynomialFeatures
Parameters:
  - interaction_types: List of operations to create between numeric pairs (add|sub|mul|div)
  - numeric_pairs: Explicit list of column pairs for interactions
  - auto_pair_numeric: Automatically generate numeric pairs (limited by max_auto_pairs)
  - poly_degree: Degree for polynomial features (None disables)
  - poly_columns: Columns to include in polynomial expansion (None = all numeric)
  - poly_include_bias: Whether to include bias term in polynomial features
  - poly_interaction_only: If True, use interaction-only polynomials
  - group_keys: Columns to group by for aggregations
  - group_value_cols: Value columns to aggregate
  - aggs: Aggregations to compute (e.g., mean, std, min, max, count, nunique)
  - max_generated_features: Guardrail for total new features created
"""

from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

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
        auto_pairs = list(combinations(numeric_cols, 2))[:max(0, int(max_auto_pairs))]
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
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None, List[str], List[Dict[str, Any]]]:
    new_cols: List[str] = []
    details: List[Dict[str, Any]] = []
    allowed_ops = {"add", "sub", "mul", "div"}
    train_new_cols: Dict[str, Any] = {}
    test_new_cols: Dict[str, Any] = {}
    val_new_cols: Dict[str, Any] = {}
    orig_new_cols: Dict[str, Any] = {}
    limit_reached = False

    for col_a, col_b in pairs:
        if col_a not in train_df.columns or col_b not in train_df.columns:
            warnings.warn(f"Skipping pair ({col_a}, {col_b}) - column missing in train")
            continue
        if col_a not in test_df.columns or col_b not in test_df.columns:
            warnings.warn(f"Skipping pair ({col_a}, {col_b}) - column missing in test")
            continue
        if not (pd.api.types.is_numeric_dtype(train_df[col_a]) and pd.api.types.is_numeric_dtype(train_df[col_b])):
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
            new_name = _unique_name(base_name, existing_cols.union(new_cols))

            if op == "add":
                train_series = train_df[col_a] + train_df[col_b]
                test_series = test_df[col_a] + test_df[col_b]
                val_series = val_df[col_a] + val_df[col_b] if val_df is not None else None
            elif op == "sub":
                train_series = train_df[col_a] - train_df[col_b]
                test_series = test_df[col_a] - test_df[col_b]
                val_series = val_df[col_a] - val_df[col_b] if val_df is not None else None
            elif op == "mul":
                train_series = train_df[col_a] * train_df[col_b]
                test_series = test_df[col_a] * test_df[col_b]
                val_series = val_df[col_a] * val_df[col_b] if val_df is not None else None
            elif op == "div":
                with np.errstate(divide="ignore", invalid="ignore"):
                    train_series = np.where(train_df[col_b] != 0, train_df[col_a] / train_df[col_b], np.nan)
                    test_series = np.where(test_df[col_b] != 0, test_df[col_a] / test_df[col_b], np.nan)
                    val_series = (
                        np.where(val_df[col_b] != 0, val_df[col_a] / val_df[col_b], np.nan)
                        if val_df is not None else None
                    )

            train_new_cols[new_name] = train_series
            test_new_cols[new_name] = test_series
            if val_df is not None and val_series is not None:
                val_new_cols[new_name] = val_series
            if orig_df is not None and col_a in orig_df.columns and col_b in orig_df.columns:
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
            details.append({
                "type": "interaction",
                "operation": op,
                "columns": [col_a, col_b],
                "new_column": new_name,
            })

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


def _apply_polynomial_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    orig_df: pd.DataFrame | None,
    poly_cols: List[str],
    degree: int,
    include_bias: bool,
    interaction_only: bool,
    remaining_slots: int,
    existing_cols: set,
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None, List[str], Dict[str, Any]]:
    if degree is None or degree <= 1 or not poly_cols:
        return train_df, val_df, test_df, orig_df, [], {}

    # Skip if NaNs present to avoid PolynomialFeatures failure; user should impute earlier.
    if train_df[poly_cols].isnull().any().any():
        warnings.warn(
            "Skipping polynomial features because input contains NaN. "
            "Run an imputer earlier in the chain or set poly_degree: null."
        )
        return train_df, val_df, test_df, orig_df, [], {
            "type": "polynomial",
            "skipped": True,
            "reason": "nan_in_input",
            "input_columns": poly_cols,
        }

    # Safety check: estimate number of features to avoid OOM
    n = len(poly_cols)
    from math import comb
    if interaction_only:
        num_expected = sum(comb(n, i) for i in range(1, degree + 1))
    else:
        num_expected = comb(n + degree, degree) - 1 # excluding bias if not requested, but close enough
    
    if num_expected > 10000:
        warnings.warn(
            f"Polynomial expansion (degree {degree}) for {n} columns would create ~{num_expected} features. "
            f"Skipping to avoid OOM (limit: 10,000)."
        )
        return train_df, val_df, test_df, orig_df, [], {
            "type": "polynomial",
            "skipped": True,
            "reason": "too_many_features",
            "num_expected": num_expected,
            "input_columns": poly_cols,
        }

    poly = PolynomialFeatures(
        degree=degree,
        include_bias=include_bias,
        interaction_only=interaction_only,
    )

    train_poly = poly.fit_transform(train_df[poly_cols])
    test_poly = poly.transform(test_df[poly_cols])
    val_poly = poly.transform(val_df[poly_cols]) if val_df is not None else None
    orig_poly = poly.transform(orig_df[poly_cols]) if orig_df is not None and all(c in orig_df.columns for c in poly_cols) else None

    feature_names = poly.get_feature_names_out(poly_cols)
    poly_train_df = pd.DataFrame(train_poly, columns=feature_names, index=train_df.index)
    poly_test_df = pd.DataFrame(test_poly, columns=feature_names, index=test_df.index)
    poly_val_df = pd.DataFrame(val_poly, columns=feature_names, index=val_df.index) if val_df is not None else None
    poly_orig_df = pd.DataFrame(orig_poly, columns=feature_names, index=orig_df.index) if orig_poly is not None else None

    # Only keep new columns to avoid overwriting originals
    new_feature_names = [name for name in feature_names if name not in train_df.columns]

    if remaining_slots is not None and remaining_slots < len(new_feature_names):
        warnings.warn(
            f"Truncating polynomial features to {remaining_slots} due to max_generated_features limit"
        )
        new_feature_names = new_feature_names[:remaining_slots]

    created_names: List[str] = []
    for name in new_feature_names:
        unique_name = _unique_name(name, existing_cols.union(created_names))
        created_names.append(unique_name)

    if created_names:
        poly_train_sel = poly_train_df[new_feature_names].copy()
        poly_test_sel = poly_test_df[new_feature_names].copy()
        poly_train_sel.columns = created_names
        poly_test_sel.columns = created_names
        train_df = pd.concat([train_df, poly_train_sel], axis=1)
        test_df = pd.concat([test_df, poly_test_sel], axis=1)
        if val_df is not None and poly_val_df is not None:
            poly_val_sel = poly_val_df[new_feature_names].copy()
            poly_val_sel.columns = created_names
            val_df = pd.concat([val_df, poly_val_sel], axis=1)
        if orig_df is not None and poly_orig_df is not None:
            poly_orig_sel = poly_orig_df[new_feature_names].copy()
            poly_orig_sel.columns = created_names
            orig_df = pd.concat([orig_df, poly_orig_sel], axis=1)

    details = {
        "type": "polynomial",
        "degree": degree,
        "include_bias": include_bias,
        "interaction_only": interaction_only,
        "input_columns": poly_cols,
        "generated_columns": created_names,
    }

    return train_df, val_df, test_df, orig_df, created_names, details


def _apply_group_aggregations(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    orig_df: pd.DataFrame | None,
    group_keys: List[str],
    value_cols: List[str],
    aggs: List[str],
    remaining_slots: int,
    existing_cols: set,
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None, List[str], Dict[str, Any]]:
    if not group_keys or not value_cols or not aggs:
        return train_df, val_df, test_df, orig_df, [], {}

    missing_keys = [col for col in group_keys if col not in train_df.columns]
    missing_values = [col for col in value_cols if col not in train_df.columns]
    if missing_keys:
        warnings.warn(f"Group keys missing in train: {missing_keys} - skipping aggregations")
        return train_df, val_df, test_df, orig_df, [], {}
    if missing_values:
        warnings.warn(f"Value columns missing in train: {missing_values} - skipping aggregations")
        return train_df, val_df, test_df, orig_df, [], {}

    if any(col not in test_df.columns for col in group_keys):
        warnings.warn("Group keys missing in test data - skipping aggregations")
        return train_df, val_df, test_df, orig_df, [], {}

    agg_df = train_df[group_keys + value_cols].groupby(group_keys).agg(aggs)
    agg_df.columns = [
        f"{'__'.join(group_keys)}__{val_col}__{agg}"
        for val_col, agg in agg_df.columns
    ]
    agg_df = agg_df.reset_index()

    new_columns = [col for col in agg_df.columns if col not in group_keys]
    if remaining_slots is not None and remaining_slots < len(new_columns):
        warnings.warn(
            f"Truncating group aggregation features to {remaining_slots} due to max_generated_features limit"
        )
        new_columns = new_columns[:remaining_slots]
        agg_df = agg_df[group_keys + new_columns]

    # Apply to all datasets
    train_df = train_df.merge(agg_df, on=group_keys, how="left")
    test_df = test_df.merge(agg_df, on=group_keys, how="left")
    if val_df is not None:
        val_df = val_df.merge(agg_df, on=group_keys, how="left")
    if orig_df is not None and all(col in orig_df.columns for col in group_keys):
        orig_df = orig_df.merge(agg_df, on=group_keys, how="left")

    # Ensure unique names against existing columns
    renamed_columns = []
    for col in new_columns:
        unique_name = _unique_name(col, existing_cols.union(renamed_columns))
        if unique_name != col:
            train_df.rename(columns={col: unique_name}, inplace=True)
            test_df.rename(columns={col: unique_name}, inplace=True)
            if val_df is not None:
                val_df.rename(columns={col: unique_name}, inplace=True)
            if orig_df is not None and col in orig_df.columns:
                orig_df.rename(columns={col: unique_name}, inplace=True)
        renamed_columns.append(unique_name)

    details = {
        "type": "group_aggregation",
        "group_keys": group_keys,
        "value_columns": value_cols,
        "aggs": aggs,
        "generated_columns": renamed_columns,
    }

    return train_df, val_df, test_df, orig_df, renamed_columns, details


def fit_transform(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
    orig_df: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None, Dict[str, Any]]:
    """
    Feature engineering preprocessing sub-module.

    Args:
        train_df: Training data
        val_df: Validation data (can be None)
        test_df: Test data
        config: Configuration dictionary with keys:
            - _artifact_dir: Path to save artifacts
            - _dataset: {id_column, target, ignored_columns}
            - interaction_types: List[str] - Operations to create (add|sub|mul|div)
            - numeric_pairs: List[List[str]] - Explicit pairs for interactions
            - auto_pair_numeric: bool - Auto-generate numeric pairs
            - max_auto_pairs: int - Max auto-generated pairs
            - poly_degree: int|None - Degree for polynomial features (None disables)
            - poly_columns: List[str]|None - Columns to include (None=all numeric)
            - poly_include_bias: bool - Include bias term
            - poly_interaction_only: bool - Only interaction terms
            - group_keys: List[str] - Group-by columns
            - group_value_cols: List[str] - Value columns to aggregate
            - aggs: List[str] - Aggregations to compute (mean, std, min, max, count, nunique, etc.)
            - max_generated_features: int - Safety cap for new features
        orig_df: External dataset (can be None)

    Returns:
        Tuple of (train_df, val_df, test_df, orig_df, state_dict)

        state_dict contains:
        - version: str
        - new_columns: List[str]
        - interactions: List[Dict]
        - polynomial: Dict
        - group_aggregations: Dict
        - config: Dict (sanitized user config)
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
        "poly_degree": None,
        "poly_columns": None,
        "poly_include_bias": False,
        "poly_interaction_only": False,
        "group_keys": [],
        "group_value_cols": [],
        "aggs": [],
        "max_generated_features": 200,
        "use_original_features_only": False,
    }
    validation.validate_config(config, required_params, optional_params)

    # Validate choices
    for op in config["interaction_types"]:
        validation.validate_choice(op, ["add", "sub", "mul", "div"], "interaction_types")

    if config["poly_degree"] is not None:
        validation.validate_numeric_range(
            config["poly_degree"],
            min_value=2,
            max_value=5,
            param_name="poly_degree",
        )

    validation.validate_numeric_range(
        config["max_generated_features"],
        min_value=1,
        max_value=5000,
        param_name="max_generated_features",
    )

    # 3. Create sub-module artifact directory
    submodule_dir = artifacts.get_submodule_artifact_dir(artifact_dir, "feature_engineer")

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
        numeric_cols = dataframe_utils.filter_original_columns(numeric_cols, orig_features)

    if not numeric_cols and (config["interaction_types"] or config["poly_degree"]):
        warnings.warn("No numeric columns available for interactions/polynomial features")

    existing_cols = set(train_df.columns)
    new_columns: List[str] = []
    interaction_details: List[Dict[str, Any]] = []
    polynomial_details: Dict[str, Any] = {}
    group_details: Dict[str, Any] = {}

    max_new = config["max_generated_features"]

    # 6. Interaction features
    numeric_pairs = config["numeric_pairs"]
    if use_orig_only and orig_features:
        orig_set = set(orig_features)
        numeric_pairs = [
            pair for pair in numeric_pairs
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
    train_df, val_df, test_df, orig_df, inter_cols, inter_details = _apply_interactions(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        orig_df=orig_df,
        pairs=pairs,
        interaction_types=config["interaction_types"],
        max_new_features=max_new,
        existing_cols=existing_cols,
    )
    new_columns.extend(inter_cols)
    interaction_details.extend(inter_details)

    remaining_slots = max_new - len(new_columns)

    # 7. Polynomial features
    poly_cols = config["poly_columns"] if config["poly_columns"] is not None else numeric_cols
    poly_cols = [col for col in poly_cols if col in numeric_cols]

    train_df, val_df, test_df, orig_df, poly_cols_added, polynomial_details = _apply_polynomial_features(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        orig_df=orig_df,
        poly_cols=poly_cols,
        degree=config["poly_degree"],
        include_bias=config["poly_include_bias"],
        interaction_only=config["poly_interaction_only"],
        remaining_slots=remaining_slots,
        existing_cols=existing_cols.union(new_columns),
    )
    new_columns.extend(poly_cols_added)
    remaining_slots = max_new - len(new_columns)

    # 8. Group aggregations
    # Warn if target is used as value column to avoid leakage
    if target_column and target_column in config["group_value_cols"]:
        warnings.warn("Target column included in group_value_cols - this may cause leakage.")

    if remaining_slots > 0:
        group_keys = config["group_keys"]
        group_value_cols = config["group_value_cols"]
        if use_orig_only and orig_features:
            orig_set = set(orig_features)
            group_keys = [c for c in group_keys if c in orig_set]
            group_value_cols = [c for c in group_value_cols if c in orig_set]
        train_df, val_df, test_df, orig_df, agg_cols_added, group_details = _apply_group_aggregations(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            orig_df=orig_df,
            group_keys=group_keys,
            value_cols=group_value_cols,
            aggs=config["aggs"],
            remaining_slots=remaining_slots,
            existing_cols=existing_cols.union(new_columns),
        )
    else:
        agg_cols_added = []
        group_details = {}
    new_columns.extend(agg_cols_added)

    # 9. Reports
    feature_report = {
        "new_columns": new_columns,
        "interactions": interaction_details,
        "polynomial": polynomial_details,
        "group_aggregations": group_details,
        "total_new_features": len(new_columns),
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
    }
    artifacts.save_report(feature_report, submodule_dir, "feature_engineering_report.json")

    transformation_summary = report.create_preprocessing_report(
        train_before=train_df_original,
        train_after=train_df,
        test_before=test_df_original,
        test_after=test_df,
        config=config,
    )
    artifacts.save_report(transformation_summary, submodule_dir, "summary.json")

    # 10. State dict
    state_dict = {
        "version": "1.0",
        "new_columns": new_columns,
        "interactions": interaction_details,
        "polynomial": polynomial_details,
        "group_aggregations": group_details,
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
    }

    return train_df, val_df, test_df, orig_df, state_dict
