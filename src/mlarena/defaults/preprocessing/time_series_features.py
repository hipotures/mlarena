"""
Time Series Features Sub-Module

Purpose: Generate lags, rolling windows, and time-based aggregations.
Libraries: pandas
Parameters:
  - entity_id_col: Column(s) defining the entity (group by). If None, treat as single series.
  - timestamp_col: Column for sorting.
  - value_cols: Columns to lag/roll.
  - lags: List of integers (e.g. [1, 7, 28]).
  - windows: List of integers for rolling windows.
  - rolling_aggs: List of aggregations (mean, std, min, max).
  - fill_method: "ffill", "bfill", "zero", "none".
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
    
    required_params = ["value_cols"]
    optional_params = {
        "entity_id_col": None, # str or list
        "timestamp_col": None,
        "sort_ascending": True,
        "lags": [],
        "windows": [],
        "rolling_aggs": ["mean"],
        "fill_method": "none", # none = leave NaN
        "drop_original_value_cols": False,
    }
    validation.validate_config(config, required_params, optional_params)

    submodule_dir = artifacts.get_submodule_artifact_dir(artifact_dir, "time_series_features")
    train_df_original = dataframe_utils.copy_dataframe(train_df)
    test_df_original = dataframe_utils.copy_dataframe(test_df)

    # Normalize config values
    value_cols = config["value_cols"]
    if isinstance(value_cols, str):
        value_cols = [value_cols]

    if not value_cols:
        warnings.warn("No value_cols provided for time_series_features. Skipping.")
        state_dict = {
            "version": "1.0",
            "new_features": [],
            "config": {k: v for k, v in config.items() if not k.startswith("_")},
            "message": "value_cols empty"
        }
        return train_df, val_df, test_df, orig_df, state_dict

    missing_values = [c for c in value_cols if c not in train_df.columns]
    if missing_values:
        return train_df, val_df, test_df, orig_df, {"error": f"Missing value_cols: {missing_values}"}

    grp_cols: List[str] = []
    if config["entity_id_col"]:
        if isinstance(config["entity_id_col"], str):
            grp_cols = [config["entity_id_col"]]
        else:
            grp_cols = list(config["entity_id_col"])
        missing_groups = [c for c in grp_cols if c not in train_df.columns]
        if missing_groups:
            return train_df, val_df, test_df, orig_df, {"error": f"Missing entity_id_col: {missing_groups}"}

    ts_col = config.get("timestamp_col")
    if ts_col and ts_col not in train_df.columns:
        return train_df, val_df, test_df, orig_df, {"error": f"Missing timestamp_col: {ts_col}"}

    # 3. Processing
    # This is tricky: TS features usually require the FULL history to be available for test.
    # If we process independently, test lags will be NaN at the beginning of test.
    # Correct approach: Concatenate Train+Val+Test (ordered), generate features, then split back.
    # BUT splitting back requires keeping track of indices or IDs.
    # MLArena architecture assumes fit on train, transform on test. 
    # For TS, "transform" on test implies test rows follow train rows in time.
    # Simple independent transform is usually wrong for lags unless test is very long and we discard beginning.
    
    # Implementation:
    # 1. Concatenate ALL available data (Train + Val + Test + Orig?)
    #    Warning: Orig might be external distribution, maybe not temporal? Skip Orig for TS usually.
    # 2. Sort by (Entity, Time)
    # 3. Compute features
    # 4. Slice back using index (assuming indices are unique/preserved).
    
    # Concatenate
    # Add a marker to split later
    train_df = train_df.copy()
    test_df = test_df.copy()
    val_df = val_df.copy() if val_df is not None else None

    train_df['_ts_split'] = 'train'
    test_df['_ts_split'] = 'test'
    if val_df is not None:
        val_df['_ts_split'] = 'val'

    frames = [train_df]
    if val_df is not None:
        frames.append(val_df)
    frames.append(test_df)
    combined = pd.concat(frames, axis=0)  # Index might duplicate?
    # Reset index to be safe, but keep original index?
    # MLArena relies on index? Usually no, it returns DFs.
    # But if we return DFs with different index, it might break.
    # Let's save original index.
    
    combined['_orig_index'] = combined.index
    combined = combined.reset_index(drop=True)
    
    # Sort
    sort_cols: List[str] = []
    if grp_cols:
        sort_cols.extend(grp_cols)

    if ts_col:
        sort_cols.append(ts_col)
        
    if sort_cols:
        combined = combined.sort_values(sort_cols, ascending=config["sort_ascending"])
        
    # Generate features
    new_features = []
    grouped = combined.groupby(grp_cols) if grp_cols else combined
    
    for v in value_cols:
        # Lags
        for lag in config["lags"]:
            name = f"{v}_lag_{lag}"
            if grp_cols:
                combined[name] = grouped[v].shift(lag)
            else:
                combined[name] = combined[v].shift(lag)
            new_features.append(name)
            
        # Rolling
        for window in config["windows"]:
            for agg in config["rolling_aggs"]:
                name = f"{v}_roll_{window}_{agg}"
                # Rolling usually requires sorted time.
                if grp_cols:
                    # grouped rolling is tricky in pandas < 1.3? No, available.
                    # We need to ensure closed='left' usually to avoid leakage (current row included?)
                    # Shift(1) then rolling is safer for predictive models (don't use current value).
                    # Let's assume user wants predictive features: shift(1) then roll.
                    shifted = grouped[v].shift(1)
                    # Now we need to group the shifted series again? 
                    # Shift within group preserves structure.
                    # But rolling on grouped object?
                    # grouped.rolling() returns a weird index.
                    
                    # Safer: transform.
                    # x.shift(1).rolling(window).agg(agg)
                    # But groupby().transform(lambda x: ...) is slow.
                    
                    # Optimized way: sort, then operations.
                    # We are already sorted.
                    # Just use groupby().shift(1).rolling()...
                    
                    # Note: rolling() on groupby returns MultiIndex (keys + original index).
                    r = combined.groupby(grp_cols)[v].shift(1).rolling(window)
                    if agg == 'mean': res = r.mean()
                    elif agg == 'std': res = r.std()
                    elif agg == 'sum': res = r.sum()
                    elif agg == 'min': res = r.min()
                    elif agg == 'max': res = r.max()
                    else: continue
                    
                    # res has MultiIndex. We need to align with combined.
                    # If we used reset_index() on rolling, it might be easier.
                    # Or since we sorted 'combined' by group, the order should match?
                    # Actually groupby().rolling() keeps order of groups.
                    # Resetting index of result usually aligns if sorted.
                    
                    combined[name] = res.reset_index(level=list(range(len(grp_cols))), drop=True)
                    
                else:
                    # No groups
                    shifted = combined[v].shift(1)
                    r = shifted.rolling(window)
                    if agg == 'mean': combined[name] = r.mean()
                    elif agg == 'std': combined[name] = r.std()
                    elif agg == 'sum': combined[name] = r.sum()
                    elif agg == 'min': combined[name] = r.min()
                    elif agg == 'max': combined[name] = r.max()
                    
                new_features.append(name)

    # Fill NaNs
    fill = config["fill_method"]
    if new_features:
        if fill == "zero":
            combined[new_features] = combined[new_features].fillna(0)
        elif fill == "ffill":
            if grp_cols:
                combined[new_features] = combined.groupby(grp_cols)[new_features].ffill()
            else:
                combined[new_features] = combined[new_features].ffill()
        elif fill == "bfill":
            if grp_cols:
                combined[new_features] = combined.groupby(grp_cols)[new_features].bfill()
            else:
                combined[new_features] = combined[new_features].bfill()
    
    # Restore original structure
    # Split by _ts_split
    # Restore index via _orig_index
    
    combined = combined.set_index('_orig_index')
    
    train_out = combined[combined['_ts_split'] == 'train'].drop(columns=['_ts_split'], errors="ignore")
    test_out = combined[combined['_ts_split'] == 'test'].drop(columns=['_ts_split'], errors="ignore")
    
    val_out = None
    if val_df is not None:
        val_out = combined[combined['_ts_split'] == 'val'].drop(columns=['_ts_split'], errors="ignore")
        
    # Reorder to match input index order?
    # combined.set_index('_orig_index') restores the values to the index label.
    # But the rows are sorted.
    # train_df.loc[idx] will align? Yes.
    
    train_df = train_out.loc[train_df.index]
    test_df = test_out.loc[test_df.index]
    if val_df is not None:
        val_df = val_out.loc[val_df.index]
        
    # orig_df was ignored (standard for TS usually)
    
    if config["drop_original_value_cols"]:
        train_df = train_df.drop(columns=value_cols)
        test_df = test_df.drop(columns=value_cols)
        if val_df is not None: val_df = val_df.drop(columns=value_cols)

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
