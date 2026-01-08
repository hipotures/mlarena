"""
Rank Features Sub-Module

Purpose: Transform numeric features into ranks or percentiles.
Libraries: pandas, scipy.stats
Parameters:
  - numeric_include: List[str]
  - numeric_exclude: List[str]
  - group_keys: List[str] (optional, for grouped ranking)
  - mode: "global" | "by_group"
  - method: "rank" | "percentile" | "gauss_rank" (future?)
  - tie_method: "average" | "min" | "max" | "first" | "dense"
  - add_original: Bool (keep original cols)
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

    required_params = []
    optional_params = {
        "numeric_include": None,
        "numeric_exclude": [],
        "group_keys": [],
        "mode": "global",
        "method": "percentile",
        "tie_method": "average",
        "add_original": True,
    }
    validation.validate_config(config, required_params, optional_params)
    validation.validate_choice(config["mode"], ["global", "by_group"], "mode")
    validation.validate_choice(config["method"], ["rank", "percentile"], "method")

    # 2. Submodule dir
    submodule_dir = artifacts.get_submodule_artifact_dir(artifact_dir, "rank_features")
    train_df_original = dataframe_utils.copy_dataframe(train_df)
    test_df_original = dataframe_utils.copy_dataframe(test_df)

    # 3. Determine columns
    exclude_cols = [id_column, target_column] + ignored_columns + config["numeric_exclude"]
    exclude_cols = [c for c in exclude_cols if c]
    all_numeric = dataframe_utils.get_numeric_columns(train_df, exclude=exclude_cols)
    
    if config["numeric_include"]:
        numeric_cols = [c for c in config["numeric_include"] if c in all_numeric]
    else:
        numeric_cols = all_numeric

    # 4. Processing
    new_features = []
    
    # Global Ranking Strategy:
    # rank/percentile is distribution dependent.
    # Correct way: Compute rank mapping on TRAIN, apply to TEST?
    # Or rank independently? Usually rank features are applied per dataset batch if it's about order relative to current batch.
    # BUT for consistent ML, value X should map to same Rank Y.
    # This implies we should treat ranks like a transformation:
    # fit: learn ECDF on train. transform: apply ECDF.
    
    # However, simpler implementation often ranks the whole concatenated set or per-set. 
    # Ranking per-set changes the meaning of '0.5' if distributions shift.
    # Best practice: Fit on Train (scipy.stats.percentileofscore or similar), apply to Test.
    # OR simpler: pd.Series.rank(pct=True). 
    
    # For this implementation, we will use pandas rank per-dataset (independent).
    # Why? Because ranks are often used to normalize distribution regardless of shift.
    # If test has higher values, we want them to be 0.99, not 1.5.
    
    def process_df(df):
        if df is None: return None
        df_out = df.copy()
        
        # If by_group, we need group keys
        if config["mode"] == "by_group":
            if not config["group_keys"]:
                warnings.warn("mode='by_group' but no group_keys provided. Falling back to global.")
                mode = "global"
            else:
                mode = "by_group"
        else:
            mode = "global"

        cols_to_rank = numeric_cols
        
        for col in cols_to_rank:
            new_col_name = f"{col}_rank" if config["method"] == "rank" else f"{col}_pct"
            
            if mode == "global":
                if config["method"] == "rank":
                    df_out[new_col_name] = df[col].rank(method=config["tie_method"])
                else:
                    df_out[new_col_name] = df[col].rank(pct=True, method=config["tie_method"])
            else:
                # Grouped
                keys = config["group_keys"]
                if config["method"] == "rank":
                    df_out[new_col_name] = df.groupby(keys)[col].rank(method=config["tie_method"])
                else:
                    df_out[new_col_name] = df.groupby(keys)[col].rank(pct=True, method=config["tie_method"])
            
            if not config["add_original"]:
                df_out = df_out.drop(columns=[col])
                
        return df_out

    # NOTE: Independent ranking per dataset!
    train_df = process_df(train_df)
    test_df = process_df(test_df)
    val_df = process_df(val_df)
    orig_df = process_df(orig_df)
    
    # Identify new columns
    current_cols = set(train_df.columns)
    orig_cols = set(train_df_original.columns)
    new_features = list(current_cols - orig_cols)

    # 5. Reports
    report_data = {
        "input_cols": len(numeric_cols),
        "new_features": new_features,
        "config": {k: v for k, v in config.items() if not k.startswith("_")}
    }
    artifacts.save_report(report_data, submodule_dir, "rank_features_report.json")
    
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
