"""
Feature Polynomial Sub-Module

Purpose: Create polynomial features using sklearn.preprocessing.PolynomialFeatures.
Libraries: pandas, sklearn.preprocessing.PolynomialFeatures
Parameters:
  - poly_degree: Degree for polynomial features (None disables)
  - poly_columns: Columns to include in polynomial expansion (None = all numeric)
  - poly_include_bias: Whether to include bias term in polynomial features
  - poly_interaction_only: If True, use interaction-only polynomials
  - max_generated_features: Guardrail for total new features created
"""

from typing import Any, Dict, List, Tuple
import warnings
from pathlib import Path

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


def fit_transform(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
    orig_df: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None, Dict[str, Any]]:
    """
    Feature polynomial preprocessing sub-module.
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
        "poly_degree": None,
        "poly_columns": None,
        "poly_include_bias": False,
        "poly_interaction_only": False,
        "max_generated_features": 200,
    }
    validation.validate_config(config, required_params, optional_params)

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
    submodule_dir = artifacts.get_submodule_artifact_dir(artifact_dir, "feature_polynomial")

    # 4. Save original DataFrames for reporting
    train_df_original = dataframe_utils.copy_dataframe(train_df)
    test_df_original = dataframe_utils.copy_dataframe(test_df)

    # 5. Prepare column sets
    exclude_cols = [id_column, target_column] + ignored_columns
    exclude_cols = [col for col in exclude_cols if col]
    numeric_cols = dataframe_utils.get_numeric_columns(train_df, exclude=exclude_cols)

    if not numeric_cols and config["poly_degree"]:
        warnings.warn("No numeric columns available for polynomial features")

    existing_cols = set(train_df.columns)
    new_columns: List[str] = []
    polynomial_details: Dict[str, Any] = {}

    max_new = config["max_generated_features"]

    # 6. Polynomial features
    if config["poly_degree"]:
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
            remaining_slots=max_new,
            existing_cols=existing_cols,
        )
        new_columns.extend(poly_cols_added)

    # 7. Reports
    feature_report = {
        "new_columns": new_columns,
        "polynomial": polynomial_details,
        "total_new_features": len(new_columns),
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
    }
    artifacts.save_report(feature_report, submodule_dir, "feature_polynomial_report.json")

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
        "polynomial": polynomial_details,
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
    }

    return train_df, val_df, test_df, orig_df, state_dict
