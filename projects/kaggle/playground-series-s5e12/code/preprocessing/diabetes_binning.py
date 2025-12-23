"""
Diabetes Binning Preprocessor

Implements two types of binning for numerical features:
1. Statistical Binning: Quantile-based binning (qcut) fitted on train
2. AI Binning: Decision tree threshold-based binning fitted on train

Key features:
- Strict train-fit paradigm (no data leakage)
- Handles NaN values gracefully
- Outputs categorical string features for CatBoost
- Configurable columns and parameters

Based on: S5E12 Single CatBoost notebook strategy
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from mlarena.preprocessing.utils import artifacts


def _fit_stat_binning(series: pd.Series, q: int = 10) -> np.ndarray:
    """
    Learn quantile bin boundaries from training set.

    Args:
        series: Training series to learn from
        q: Number of quantiles

    Returns:
        Array of bin edges
    """
    # Use rank to handle ties, drop NaN before binning
    _, bins = pd.qcut(
        series.dropna().rank(method='first'),
        q=q,
        retbins=True,
        duplicates='drop'
    )
    return bins


def _apply_stat_binning(series: pd.Series, bins: np.ndarray) -> pd.Series:
    """
    Apply learned quantile boundaries to any dataset.

    Args:
        series: Series to bin
        bins: Learned bin edges

    Returns:
        Binned series as strings (categorical)
    """
    return pd.cut(
        series,
        bins=bins,
        labels=False,
        include_lowest=True
    ).astype(str)


def _fit_ai_binning(X: pd.DataFrame, y: pd.Series, col: str, max_depth: int = 3, random_state: int = 42) -> List[float]:
    """
    Learn decision tree thresholds from training set.

    Args:
        X: Training features
        y: Training target
        col: Column name to bin
        max_depth: Tree depth
        random_state: Random seed

    Returns:
        List of threshold values
    """
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)

    # Fill NaN with mean for DT training
    X_filled = X[[col]].fillna(X[col].mean())
    dt.fit(X_filled, y)

    # Extract thresholds (exclude -2 which marks leaf nodes)
    thresholds = sorted(list(set([t for t in dt.tree_.threshold if t != -2])))
    return thresholds


def _apply_ai_binning(series: pd.Series, thresholds: List[float]) -> pd.Series:
    """
    Apply learned AI thresholds to any dataset.

    Args:
        series: Series to bin
        thresholds: Learned thresholds

    Returns:
        Binned series as strings (categorical)
    """
    bins = [-np.inf] + thresholds + [np.inf]
    return pd.cut(series, bins=bins, labels=False).astype(str)


def fit_transform(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    test_df: pd.DataFrame,
    config: Dict[str, Any],
    orig_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame, Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Apply statistical and AI binning to numerical features.

    Config keys:
        binning_columns (list): Columns to bin (default: auto-detect numeric)
        stat_quantiles (int): Number of quantiles for statistical binning (default: 10)
        ai_max_depth (int): Max depth for decision tree binning (default: 3)
        random_state (int): Random seed for AI binning (default: 42)
        enable_stat_binning (bool): Enable statistical binning (default: True)
        enable_ai_binning (bool): Enable AI binning (default: True)

    Returns:
        Tuple of (train_df, val_df, test_df, orig_df, state_dict)

    New columns added:
        bin_{col}_stat: Statistical binning (if enabled)
        bin_{col}_ai: AI binning (if enabled)
    """
    # 1. Extract config
    artifact_dir = Path(config.get("_artifact_dir", "."))
    dataset_meta = config.get("_dataset", {})
    target_col = dataset_meta.get("target")
    id_col = dataset_meta.get("id_column", "id")

    # 2. Get configuration
    binning_columns = config.get("binning_columns")
    stat_quantiles = config.get("stat_quantiles", 10)
    ai_max_depth = config.get("ai_max_depth", 3)
    random_state = config.get("random_state", 42)
    enable_stat = config.get("enable_stat_binning", True)
    enable_ai = config.get("enable_ai_binning", True)

    # Auto-detect numeric columns if not specified
    if binning_columns is None:
        exclude_cols = {id_col, target_col}
        binning_columns = [
            col for col in train_df.select_dtypes(include=[np.number]).columns
            if col not in exclude_cols
        ]

    # Validate target column exists for AI binning
    if enable_ai and (not target_col or target_col not in train_df.columns):
        raise ValueError(
            "AI binning requires target column in training data. "
            "Set enable_ai_binning: false or ensure target is present."
        )

    # 3. Initialize state tracking
    state_dict = {
        "version": "1.0",
        "module": "diabetes_binning",
        "binned_columns": binning_columns,
        "stat_binning_enabled": enable_stat,
        "ai_binning_enabled": enable_ai,
        "stat_bins": {},
        "ai_thresholds": {},
        "new_columns": [],
    }

    # Make copies to avoid modifying originals
    train_df = train_df.copy()
    test_df = test_df.copy()
    if val_df is not None:
        val_df = val_df.copy()
    if orig_df is not None:
        orig_df = orig_df.copy()

    # 4. Process each column
    for col in binning_columns:
        if col not in train_df.columns:
            print(f"⚠️  Column '{col}' not found in train - skipping")
            continue
        if col not in test_df.columns:
            print(f"⚠️  Column '{col}' not found in test - skipping")
            continue

        # Statistical binning
        if enable_stat:
            stat_col_name = f"bin_{col}_stat"
            try:
                # Fit on train
                stat_bins = _fit_stat_binning(train_df[col], q=stat_quantiles)
                state_dict["stat_bins"][col] = stat_bins.tolist()

                # Apply to all datasets
                train_df[stat_col_name] = _apply_stat_binning(train_df[col], stat_bins)
                test_df[stat_col_name] = _apply_stat_binning(test_df[col], stat_bins)
                if val_df is not None:
                    val_df[stat_col_name] = _apply_stat_binning(val_df[col], stat_bins)
                if orig_df is not None and col in orig_df.columns:
                    orig_df[stat_col_name] = _apply_stat_binning(orig_df[col], stat_bins)

                state_dict["new_columns"].append(stat_col_name)
            except Exception as e:
                print(f"⚠️  Statistical binning failed for '{col}': {e}")

        # AI binning
        if enable_ai:
            ai_col_name = f"bin_{col}_ai"
            try:
                # Fit on train
                ai_thresholds = _fit_ai_binning(
                    train_df,
                    train_df[target_col],
                    col,
                    max_depth=ai_max_depth,
                    random_state=random_state
                )
                state_dict["ai_thresholds"][col] = ai_thresholds

                # Apply to all datasets
                train_df[ai_col_name] = _apply_ai_binning(train_df[col], ai_thresholds)
                test_df[ai_col_name] = _apply_ai_binning(test_df[col], ai_thresholds)
                if val_df is not None:
                    val_df[ai_col_name] = _apply_ai_binning(val_df[col], ai_thresholds)
                if orig_df is not None and col in orig_df.columns:
                    orig_df[ai_col_name] = _apply_ai_binning(orig_df[col], ai_thresholds)

                state_dict["new_columns"].append(ai_col_name)
            except Exception as e:
                print(f"⚠️  AI binning failed for '{col}': {e}")

    # 5. Save report
    report = {
        "binned_columns": binning_columns,
        "new_columns_created": len(state_dict["new_columns"]),
        "stat_binning_enabled": enable_stat,
        "ai_binning_enabled": enable_ai,
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
    }
    artifacts.save_report(report, artifact_dir, "binning_report.json")

    print(f"✅ Binning complete: {len(state_dict['new_columns'])} new columns created")

    return train_df, val_df, test_df, orig_df, state_dict
