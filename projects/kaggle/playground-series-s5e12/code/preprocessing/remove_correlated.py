"""
Remove Highly Correlated Features (Inter-Feature Correlation)

Removes redundant features that are highly correlated with each other.
For each pair of highly correlated features (|corr| > threshold), keeps one.

Based on: Pairwise correlation matrix analysis
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from mlarena.preprocessing.utils import artifacts


def fit_transform(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    test_df: pd.DataFrame,
    config: Dict[str, Any],
    orig_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame, Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Remove highly correlated features based on pairwise correlation matrix.

    Config keys:
        correlation_threshold (float): Drop if |correlation| > threshold (default: 0.95)
        method (str): 'pearson' | 'spearman' (default: 'pearson')

    Returns:
        Tuple of (train_df, val_df, test_df, orig_df, state_dict)
    """
    # 1. Extract config
    artifact_dir = Path(config.get("_artifact_dir", "."))
    dataset_meta = config.get("_dataset", {})
    target_col = dataset_meta.get("target")
    id_col = dataset_meta.get("id_column", "id")

    correlation_threshold = config.get("correlation_threshold", 0.95)
    method = config.get("method", "pearson")

    # 2. Get numeric features (exclude target and ID)
    exclude_cols = {target_col, id_col}
    numeric_cols = [
        col for col in train_df.select_dtypes(include=[np.number]).columns
        if col not in exclude_cols
    ]

    if len(numeric_cols) == 0:
        print("⚠️  No numeric features found, skipping correlation removal")
        return train_df, val_df, test_df, orig_df, {
            "version": "1.0",
            "module": "remove_correlated",
            "features_dropped": [],
            "features_kept": list(train_df.columns),
        }

    # 3. Calculate correlation matrix
    print(f"📊 Analyzing {len(numeric_cols)} numeric features for correlation...")
    corr_matrix = train_df[numeric_cols].corr(method=method).abs()

    # 4. Find highly correlated pairs (upper triangle only to avoid duplicates)
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # 5. Identify features to drop
    to_drop = set()
    for column in upper_tri.columns:
        # Find features correlated above threshold with this column
        correlated_features = upper_tri.index[upper_tri[column] > correlation_threshold].tolist()
        if correlated_features:
            # Keep first feature, drop the rest
            to_drop.update(correlated_features)

    to_drop = sorted(list(to_drop))
    to_keep = [col for col in train_df.columns if col not in to_drop]

    print(f"✅ Correlation analysis complete:")
    print(f"   Threshold: {correlation_threshold}")
    print(f"   Features dropped: {len(to_drop)}")
    print(f"   Features kept: {len(to_keep)}")

    if to_drop:
        print(f"   Dropped: {', '.join(to_drop[:10])}{' ...' if len(to_drop) > 10 else ''}")

    # 6. Apply to all datasets
    train_df = train_df[to_keep].copy()
    test_df = test_df[[col for col in to_keep if col in test_df.columns]].copy()
    if val_df is not None:
        val_df = val_df[[col for col in to_keep if col in val_df.columns]].copy()
    if orig_df is not None:
        orig_df = orig_df[[col for col in to_keep if col in orig_df.columns]].copy()

    # 7. Save report
    state_dict = {
        "version": "1.0",
        "module": "remove_correlated",
        "correlation_threshold": correlation_threshold,
        "method": method,
        "features_analyzed": len(numeric_cols),
        "features_dropped": to_drop,
        "features_kept": to_keep,
        "drop_count": len(to_drop),
    }

    report = {
        "correlation_threshold": correlation_threshold,
        "method": method,
        "numeric_features_analyzed": len(numeric_cols),
        "features_dropped_count": len(to_drop),
        "features_kept_count": len(to_keep),
        "dropped_features": to_drop[:50],  # Limit for JSON size
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
    }
    artifacts.save_report(report, artifact_dir, "correlation_removal_report.json")

    return train_df, val_df, test_df, orig_df, state_dict
