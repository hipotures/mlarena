"""
Groupwise Normalizer Sub-Module

Purpose: Normalize numeric features relative to groups (e.g., price relative to category average).
Libraries: pandas
Parameters:
  - group_keys: List[str]
  - value_cols: List[str]
  - add_group_mean: Bool (add the mean itself as feature)
  - add_centered: Bool (value - mean)
  - add_zscore: Bool ((value - mean) / std)
  - add_ratio: Bool (value / mean)
  - eps: Float (epsilon for division)
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import warnings

import pandas as pd
import numpy as np

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

    required_params = ["group_keys", "value_cols"]
    optional_params = {
        "add_group_mean": True,
        "add_centered": True,
        "add_zscore": True,
        "add_ratio": False,
        "eps": 1e-6,
    }
    validation.validate_config(config, required_params, optional_params)

    submodule_dir = artifacts.get_submodule_artifact_dir(artifact_dir, "groupwise_normalizer")
    train_df_original = dataframe_utils.copy_dataframe(train_df)
    test_df_original = dataframe_utils.copy_dataframe(test_df)

    # 3. Compute Group Stats on Train
    keys = config["group_keys"]
    values = config["value_cols"]
    if isinstance(keys, str):
        keys = [keys]
    if isinstance(values, str):
        values = [values]
    eps = config["eps"]

    if not keys or not values:
        warnings.warn("groupwise_normalizer requires non-empty group_keys and value_cols. Skipping.")
        state_dict = {
            "version": "1.0",
            "new_features": [],
            "config": {k: v for k, v in config.items() if not k.startswith("_")},
            "message": "group_keys/value_cols empty"
        }
        return train_df, val_df, test_df, orig_df, state_dict
    
    # Validate columns
    missing = [c for c in keys + values if c not in train_df.columns]
    if missing:
        return train_df, val_df, test_df, orig_df, {"error": f"Missing cols: {missing}"}

    # Groupby
    stats = train_df.groupby(keys)[values].agg(['mean', 'std'])
    
    # Flatten columns: value_col -> (mean, std)
    # stats.columns is MultiIndex
    
    new_features = []

    # Flatten stats for easier merge
    stats.columns = [f"{v}_{stat}" for v, stat in stats.columns]
    stats = stats.reset_index()
    
    def process_df(df):
        if df is None: return None
        df_out = df.copy()
        
        # Merge stats
        temp = df[keys].merge(stats, on=keys, how='left')
        # temp has same index as df IF we preserve it? 
        # merge resets index if not careful or if relations are 1:N?
        # keys are not unique in df, but unique in stats. M:1 merge.
        # merge preserves order of left key? Not guaranteed?
        # Safer: set index, merge, restore.
        
        # Actually map is safer and faster for single key. For multi-key, merge is needed.
        # Let's use left join on index.
        temp.index = df.index
        
        for v in values:
            mean_col = f"{v}_mean"
            std_col = f"{v}_std"
            
            # If stats missing (unseen group), fillna?
            # Global fallback?
            if temp[mean_col].isnull().any():
                # Fill with global mean
                g_mean = train_df[v].mean()
                g_std = train_df[v].std()
                temp[mean_col] = temp[mean_col].fillna(g_mean)
                temp[std_col] = temp[std_col].fillna(g_std)
            
            if config["add_group_mean"]:
                name = f"{v}_grp_mean"
                df_out[name] = temp[mean_col]
                if name not in new_features: new_features.append(name)
                
            if config["add_centered"]:
                name = f"{v}_centered"
                df_out[name] = df[v] - temp[mean_col]
                if name not in new_features: new_features.append(name)
                
            if config["add_zscore"]:
                name = f"{v}_grp_zscore"
                # Avoid div by zero
                sigma = temp[std_col].fillna(0)
                sigma = sigma.replace(0, eps) # if std is 0 (constant group)
                df_out[name] = (df[v] - temp[mean_col]) / sigma
                if name not in new_features: new_features.append(name)
                
            if config["add_ratio"]:
                name = f"{v}_grp_ratio"
                mu = temp[mean_col].replace(0, eps)
                df_out[name] = df[v] / mu
                if name not in new_features: new_features.append(name)
                
        return df_out

    # Apply
    # Note: process_df does the merge inside.
    
    train_df = process_df(train_df)
    test_df = process_df(test_df)
    val_df = process_df(val_df)
    orig_df = process_df(orig_df)

    # 4. Reports
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
        "config": {k: v for k, v in config.items() if not k.startswith("_")}
    }

    return train_df, val_df, test_df, orig_df, state_dict
