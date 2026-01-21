"""
Clustering Features Sub-Module

Purpose: Generate cluster IDs and distances using KMeans.
Libraries: sklearn.cluster
Parameters:
  - numeric_include: List[str]
  - numeric_exclude: List[str]
  - n_clusters: int
  - add_cluster_id: bool
  - add_distances: bool
  - random_state: int
"""

from pathlib import Path
from typing import Any, Dict, Tuple
import warnings

import pandas as pd
from sklearn.cluster import KMeans

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
) -> Tuple[
    pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None, Dict[str, Any]
]:
    # 1. Extract & Validate
    artifact_dir = Path(config.get("_artifact_dir", "."))
    dataset_config = config.get("_dataset", {})
    id_column = dataset_config.get("id_column", "id")
    target_column = dataset_config.get("target")
    ignored_columns = dataset_config.get("ignored_columns", [])

    required_params = []
    optional_params = {
        "numeric_include": None,
        "numeric_exclude": [],
        "n_clusters": 10,
        "add_cluster_id": True,
        "add_distances": False,
        "random_state": 42,
        "n_init": 10,
        "algorithm": "kmeans",  # Reserved for future GMM etc.
        "use_original_features_only": True,
    }
    validation.validate_config(config, required_params, optional_params)

    submodule_dir = artifacts.get_submodule_artifact_dir(
        artifact_dir, "clustering_features"
    )
    train_df_original = dataframe_utils.copy_dataframe(train_df)
    test_df_original = dataframe_utils.copy_dataframe(test_df)

    # 3. Determine columns
    exclude_cols = (
        [id_column, target_column] + ignored_columns + config["numeric_exclude"]
    )
    exclude_cols = [c for c in exclude_cols if c]
    all_numeric = dataframe_utils.get_numeric_columns(train_df, exclude=exclude_cols)
    use_orig_only = bool(config.get("use_original_features_only"))
    orig_features = config.get("_original_features") if use_orig_only else None

    if config["numeric_include"]:
        numeric_cols = [c for c in config["numeric_include"] if c in all_numeric]
    else:
        numeric_cols = all_numeric

    if use_orig_only:
        numeric_cols = dataframe_utils.filter_original_columns(
            numeric_cols, orig_features
        )

    if not numeric_cols:
        return train_df, val_df, test_df, orig_df, {"message": "No numeric cols"}

    # Handle NaNs: KMeans requires no NaNs.
    if train_df[numeric_cols].isnull().any().any():
        warnings.warn("Input contains NaNs. Skipping clustering (impute first).")
        return train_df, val_df, test_df, orig_df, {"skipped": "NaNs present"}

    # 4. Fit
    model = KMeans(
        n_clusters=config["n_clusters"],
        random_state=config["random_state"],
        n_init=config["n_init"],
    )
    model.fit(train_df[numeric_cols])

    new_features = []

    def process_df(df):
        if df is None:
            return None
        df_out = df.copy()

        # Predict clusters
        clusters = model.predict(df[numeric_cols])

        if config["add_cluster_id"]:
            name = "cluster_id"
            df_out[name] = clusters
            if name not in new_features:
                new_features.append(name)

        if config["add_distances"]:
            # Distances to all centers
            dists = model.transform(df[numeric_cols])
            cols = [f"dist_cluster_{i}" for i in range(config["n_clusters"])]
            df_dists = pd.DataFrame(dists, columns=cols, index=df.index)
            df_out = pd.concat([df_out, df_dists], axis=1)
            for c in cols:
                if c not in new_features:
                    new_features.append(c)

        return df_out

    train_df = process_df(train_df)
    test_df = process_df(test_df)
    val_df = process_df(val_df)
    orig_df = process_df(orig_df)

    # Save fitted
    artifacts.save_fitted_object(model, submodule_dir, "kmeans.pkl")

    # 5. Reports
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
        "inertia": float(model.inertia_),
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
    }

    return train_df, val_df, test_df, orig_df, state_dict
