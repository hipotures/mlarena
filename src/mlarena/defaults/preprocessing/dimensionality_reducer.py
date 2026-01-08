"""
Dimensionality Reducer Sub-Module

Purpose: Reduce dimensionality using PCA or SVD and add components as features.
Libraries: sklearn.decomposition
Parameters:
  - method: "pca" | "svd" (TruncatedSVD)
  - n_components: int
  - include_sparse: bool (for SVD)
  - whiten: bool (for PCA)
  - random_state: int
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import warnings

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD

from mlarena.preprocessing.utils import (
    validation,
    artifacts,
    dataframe_utils,
    report,
)


def fit_transform(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
    orig_df: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None, Dict[str, Any]]:
    
    # 1. Extract & Validate
    artifact_dir = Path(config.get("_artifact_dir", "."))
    dataset_config = config.get("_dataset", {})
    id_column = dataset_config.get("id_column", "id")
    target_column = dataset_config.get("target")
    ignored_columns = dataset_config.get("ignored_columns", [])

    required_params = []
    optional_params = {
        "method": "pca",
        "n_components": 10,
        "include_sparse": False,
        "whiten": False,
        "random_state": 42,
    }
    validation.validate_config(config, required_params, optional_params)
    validation.validate_choice(config["method"], ["pca", "svd"], "method")

    submodule_dir = artifacts.get_submodule_artifact_dir(artifact_dir, "dimensionality_reducer")
    train_df_original = dataframe_utils.copy_dataframe(train_df)
    test_df_original = dataframe_utils.copy_dataframe(test_df)

    # 3. Determine columns (Numeric only usually, unless SVD on sparse)
    exclude_cols = [id_column, target_column] + ignored_columns
    exclude_cols = [c for c in exclude_cols if c]
    
    # If SVD and include_sparse=True, we might want all cols (if onehot)?
    # Standard practice: PCA on numeric.
    numeric_cols = dataframe_utils.get_numeric_columns(train_df, exclude=exclude_cols)
    
    if not numeric_cols:
        warnings.warn("No numeric columns found for dimensionality reduction.")
        return train_df, val_df, test_df, orig_df, {"message": "No numeric cols"}

    if len(numeric_cols) < config["n_components"]:
        warnings.warn(f"n_components ({config['n_components']}) > n_features ({len(numeric_cols)}). Reducing.")
        config["n_components"] = len(numeric_cols)

    if config["n_components"] <= 0:
        warnings.warn("n_components <= 0 after adjustment. Skipping dimensionality reduction.")
        return train_df, val_df, test_df, orig_df, {"message": "Invalid n_components"}

    # 4. Fit
    if config["method"] == "pca":
        model = PCA(
            n_components=config["n_components"],
            whiten=config["whiten"],
            random_state=config["random_state"]
        )
    else:
        model = TruncatedSVD(
            n_components=config["n_components"],
            random_state=config["random_state"]
        )
    
    # Handle NaNs: PCA requires no NaNs. SVD too usually.
    if train_df[numeric_cols].isnull().any().any():
        warnings.warn("Input contains NaNs. Skipping dimensionality reduction (impute first).")
        return train_df, val_df, test_df, orig_df, {"skipped": "NaNs present"}

    model.fit(train_df[numeric_cols])
    
    new_features = []
    prefix = config["method"]

    def process_df(df):
        if df is None: return None
        comps = model.transform(df[numeric_cols])
        
        cols = [f"{prefix}_{i}" for i in range(config["n_components"])]
        df_new = pd.DataFrame(comps, columns=cols, index=df.index)
        
        # Track new features
        for c in cols:
            if c not in new_features: new_features.append(c)
            
        return pd.concat([df, df_new], axis=1)

    train_df = process_df(train_df)
    test_df = process_df(test_df)
    val_df = process_df(val_df)
    orig_df = process_df(orig_df)

    # Save fitted
    artifacts.save_fitted_object(model, submodule_dir, "reducer.pkl")

    # 5. Reports
    report_data = {
        "explained_variance_ratio": model.explained_variance_ratio_.tolist() if hasattr(model, "explained_variance_ratio_") else [],
        "total_explained_variance": float(sum(model.explained_variance_ratio_)) if hasattr(model, "explained_variance_ratio_") else 0.0,
        "new_features": new_features,
        "config": {k: v for k, v in config.items() if not k.startswith("_")}
    }
    artifacts.save_report(report_data, submodule_dir, "reducer_report.json")
    
    summary = report.create_preprocessing_report(
        train_before=train_df_original,
        train_after=train_df,
        test_before=test_df_original,
        test_after=test_df,
        config=config,
    )
    artifacts.save_report(summary, submodule_dir, "summary.json")

    state_dict = {
        "version": "1.0",
        "new_features": new_features,
        "config": report_data["config"]
    }

    return train_df, val_df, test_df, orig_df, state_dict
