"""
Feature selection using variance threshold or univariate statistical tests.

Removes low-variance features or selects K best features based on statistical tests.

This module implements the preprocessing interface expected by MLArena:
- fit_transform(train_df, val_df, test_df, config) -> (train_df, val_df, test_df, state_dict)
- transform(df, state_dict, config) -> df  # Optional, for inference
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, f_regression


def fit_transform(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
    orig_df: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None, Dict[str, Any]]:
    """
    Select features based on variance or statistical tests.

    Config parameters:
        method: "variance_threshold" | "select_k_best" (default: "variance_threshold")
        threshold: Variance threshold (default: 0.01)
        k: Number of top features to select (for select_k_best, default: 10)
        score_func: "f_classif" | "f_regression" (for select_k_best, default: "f_classif")

    Args:
        train_df: Training dataframe
        val_df: Validation dataframe (optional)
        test_df: Test dataframe
        config: Configuration dictionary
        orig_df: External dataset (optional)

    Returns:
        Tuple of (train_df, val_df, test_df, orig_df, state_dict)
    """
    # Extract config
    method = config.get("method", "variance_threshold")
    threshold = config.get("threshold", 0.01)
    k = config.get("k", 10)
    score_func_name = config.get("score_func", "f_classif")

    # Make copies to avoid modifying originals
    train_df = train_df.copy()
    test_df = test_df.copy()
    if val_df is not None:
        val_df = val_df.copy()
    if orig_df is not None:
        orig_df = orig_df.copy()

    # Get numeric columns only (feature selection works on numeric data)
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()

    # Get target column and remove from features
    dataset_info = config.get("_dataset", {})
    target_column = dataset_info.get("target")
    if target_column and target_column in numeric_cols:
        numeric_cols.remove(target_column)

    if not numeric_cols:
        # No numeric columns to select from
        return train_df, val_df, test_df, orig_df, {"skipped": True, "reason": "no numeric columns"}

    X_train = train_df[numeric_cols]
    X_test = test_df[numeric_cols]
    X_val = val_df[numeric_cols] if val_df is not None else None
    X_orig = orig_df[numeric_cols] if orig_df is not None else None

    state = {
        "method": method,
        "original_features": numeric_cols,
        "original_feature_count": len(numeric_cols),
    }

    if method == "variance_threshold":
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X_train)

        # Get selected feature names
        selected_mask = selector.get_support()
        selected_features = [f for f, selected in zip(numeric_cols, selected_mask) if selected]

        # Transform
        X_train_selected = selector.transform(X_train)
        X_test_selected = selector.transform(X_test)
        X_val_selected = selector.transform(X_val) if X_val is not None else None
        X_orig_selected = selector.transform(X_orig) if X_orig is not None else None

        # Update dataframes
        # Drop all original numeric columns
        train_df = train_df.drop(columns=numeric_cols)
        test_df = test_df.drop(columns=numeric_cols)
        if val_df is not None:
            val_df = val_df.drop(columns=numeric_cols)
        if orig_df is not None:
            orig_df = orig_df.drop(columns=numeric_cols)

        # Add selected features back
        train_df[selected_features] = X_train_selected
        test_df[selected_features] = X_test_selected
        if val_df is not None:
            val_df[selected_features] = X_val_selected
        if orig_df is not None:
            orig_df[selected_features] = X_orig_selected

        # State tracking
        dropped_features = [f for f in numeric_cols if f not in selected_features]

        state["threshold"] = threshold
        state["selected_features"] = selected_features
        state["selected_feature_count"] = len(selected_features)
        state["dropped_features"] = dropped_features
        state["dropped_feature_count"] = len(dropped_features)
        state["variances"] = selector.variances_.tolist() if hasattr(selector, 'variances_') else []

    elif method == "select_k_best":
        # Need target for SelectKBest
        if not target_column or target_column not in train_df.columns:
            raise ValueError("select_k_best requires target column in training data")

        y_train = train_df[target_column]

        # Choose score function
        score_func = f_classif if score_func_name == "f_classif" else f_regression

        # Select k features (or all if less than k)
        k_actual = min(k, len(numeric_cols))
        selector = SelectKBest(score_func=score_func, k=k_actual)
        selector.fit(X_train, y_train)

        # Get selected feature names
        selected_mask = selector.get_support()
        selected_features = [f for f, selected in zip(numeric_cols, selected_mask) if selected]

        # Transform
        X_train_selected = selector.transform(X_train)
        X_test_selected = selector.transform(X_test)
        X_val_selected = selector.transform(X_val) if X_val is not None else None
        X_orig_selected = selector.transform(X_orig) if X_orig is not None else None

        # Update dataframes
        # Drop all original numeric columns
        train_df = train_df.drop(columns=numeric_cols)
        test_df = test_df.drop(columns=numeric_cols)
        if val_df is not None:
            val_df = val_df.drop(columns=numeric_cols)
        if orig_df is not None:
            orig_df = orig_df.drop(columns=numeric_cols)

        # Add selected features back
        train_df[selected_features] = X_train_selected
        test_df[selected_features] = X_test_selected
        if val_df is not None:
            val_df[selected_features] = X_val_selected
        if orig_df is not None:
            orig_df[selected_features] = X_orig_selected

        # State tracking
        dropped_features = [f for f in numeric_cols if f not in selected_features]

        state["k"] = k_actual
        state["score_func"] = score_func_name
        state["selected_features"] = selected_features
        state["selected_feature_count"] = len(selected_features)
        state["dropped_features"] = dropped_features
        state["dropped_feature_count"] = len(dropped_features)
        state["scores"] = selector.scores_.tolist() if hasattr(selector, 'scores_') else []

    else:
        raise ValueError(f"Unknown feature selection method: {method}")

    return train_df, val_df, test_df, orig_df, state


def transform(df: pd.DataFrame, state_dict: Dict[str, Any], config: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply fitted selector to new data.

    Note: This function is not currently used in the preprocessing pipeline
    but is provided for potential future inference use cases.

    Args:
        df: Dataframe to transform
        state_dict: State dictionary from fit_transform
        config: Configuration dictionary

    Returns:
        Transformed dataframe
    """
    # Not needed for current workflow (fit_transform handles everything)
    return df.copy()
