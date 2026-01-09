"""
Categorical Cross Sub-Module

Purpose: Create cross-product features from pairs of categorical columns (e.g., A x B).
Libraries: pandas, sklearn, category_encoders
Parameters:
  - cross_pairs: List[List[str]] (if empty, auto-generate?)
  - max_pair_cardinality: int (limit for auto-generated pairs)
  - output: "hashed" | "onehot" | "target_mean_oof"
  - hash_dim: int (for hashed)
  - separator: str (default "__")
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
from itertools import combinations
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
        "cross_pairs": [], # List of [col1, col2]
        "max_pair_cardinality": 5000,
        "separator": "__",
        "output": "hashed",
        "hash_dim": 12,
        # Params for target encoding
        "oof_folds": 5,
        "oof_random_state": 42,
    }
    validation.validate_config(config, required_params, optional_params)
    validation.validate_choice(config["output"], ["hashed", "onehot", "target_mean_oof"], "output")

    submodule_dir = artifacts.get_submodule_artifact_dir(artifact_dir, "categorical_cross")
    train_df_original = dataframe_utils.copy_dataframe(train_df)
    test_df_original = dataframe_utils.copy_dataframe(test_df)

    exclude_cols = [id_column, target_column] + ignored_columns
    categorical_cols = dataframe_utils.get_categorical_columns(train_df, exclude=exclude_cols)

    # 2. Determine pairs
    pairs = config["cross_pairs"]
    
    # If no pairs provided, should we auto-generate?
    # Let's say yes, but with cardinality guard.
    if not pairs:
        # Auto generate pairs
        # But only if product cardinality is reasonable?
        # We can just pick pairs. Let's limit to top K pairs?
        # For now, simplistic approach: combination of all categoricals
        # This can be huge. Let's rely on manual config mostly, or very strict limits.
        # Let's create pairs but filter by max_pair_cardinality
        
        candidates = list(combinations(categorical_cols, 2))
        valid_pairs = []
        for c1, c2 in candidates:
            card1 = train_df[c1].nunique()
            card2 = train_df[c2].nunique()
            if card1 * card2 <= config["max_pair_cardinality"]:
                valid_pairs.append([c1, c2])
        pairs = valid_pairs

    new_features = []
    
    # Helper to create crossed series
    def create_cross_series(df, c1, c2):
        return df[c1].astype(str) + config["separator"] + df[c2].astype(str)

    # 3. Processing
    sep = config["separator"]
    output_method = config["output"]
    
    # Suppress fragmentation warning for wide creation
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    
    # We collect all crossed columns first
    temp_train = pd.DataFrame(index=train_df.index)
    temp_test = pd.DataFrame(index=test_df.index)
    temp_val = pd.DataFrame(index=val_df.index) if val_df is not None else None
    temp_orig = pd.DataFrame(index=orig_df.index) if orig_df is not None else None
    
    cross_col_names = []
    
    for c1, c2 in pairs:
        if c1 not in train_df.columns or c2 not in train_df.columns:
            continue
            
        name = f"{c1}{sep}{c2}"
        cross_col_names.append(name)
        
        temp_train[name] = create_cross_series(train_df, c1, c2)
        temp_test[name] = create_cross_series(test_df, c1, c2)
        if temp_val is not None:
            temp_val[name] = create_cross_series(val_df, c1, c2)
        if temp_orig is not None:
            temp_orig[name] = create_cross_series(orig_df, c1, c2)

    # Now encode these temp columns
    if not cross_col_names:
        return train_df, val_df, test_df, orig_df, {"message": "No pairs generated"}

    if output_method == "hashed":
        from sklearn.feature_extraction import FeatureHasher
        hash_dim = config["hash_dim"]
        
        hasher = FeatureHasher(n_features=hash_dim, input_type='string')
        
        # Hashing usually works on a list of features or single?
        # FeatureHasher expects iterables of strings.
        # We can hash each cross column separately (better for interpretability/separation)
        # Or hash all together (feature interaction implicit in hashing collision? No).
        # Standard: Hash each crossed feature separately into N dims? Or one big hash space?
        # Usually hashing trick is used on the whole row. 
        # But here we want specific cross features.
        # Let's do: hash each cross-pair into `hash_dim` columns.
        
        for col in cross_col_names:
            cols_out = [f"{col}_h{i}" for i in range(hash_dim)]
            
            # Transform
            def apply_hash(series):
                return hasher.transform([[x] for x in series]).toarray()
            
            # This is slow row-by-row. Sklearn expects iterable of iterables.
            # Faster:
            tr_h = hasher.transform(temp_train[col].map(lambda x: [x])).toarray()
            te_h = hasher.transform(temp_test[col].map(lambda x: [x])).toarray()
            
            train_df[cols_out] = tr_h
            test_df[cols_out] = te_h
            
            if val_df is not None:
                va_h = hasher.transform(temp_val[col].map(lambda x: [x])).toarray()
                val_df[cols_out] = va_h
            if orig_df is not None:
                or_h = hasher.transform(temp_orig[col].map(lambda x: [x])).toarray()
                orig_df[cols_out] = or_h
                
            new_features.extend(cols_out)

    elif output_method == "onehot":
        # OneHot encode the crossed columns
        # Caution: High cardinality!
        from sklearn.preprocessing import OneHotEncoder
        enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        enc.fit(temp_train[cross_col_names])
        
        new_names = enc.get_feature_names_out(cross_col_names)
        
        tr_o = enc.transform(temp_train[cross_col_names])
        te_o = enc.transform(temp_test[cross_col_names])
        
        train_df[new_names] = tr_o
        test_df[new_names] = te_o
        
        if val_df is not None:
            va_o = enc.transform(temp_val[cross_col_names])
            val_df[new_names] = va_o
        if orig_df is not None:
            or_o = enc.transform(temp_orig[cross_col_names])
            orig_df[new_names] = or_o
            
        new_features.extend(new_names)
        artifacts.save_fitted_object(enc, submodule_dir, "cross_onehot.pkl")

    elif output_method == "target_mean_oof":
        # Reuse logic from encoder.py via copy or import?
        # Better to implement minimal logic here.
        if target_column is None:
            raise ValueError("target_mean_oof requires target column")
        from sklearn.model_selection import KFold
        
        folds = config["oof_folds"]
        kf = KFold(n_splits=folds, shuffle=True, random_state=config["oof_random_state"])
        target = train_df[target_column]
        global_mean = target.mean()
        
        for col in cross_col_names:
            out_col = f"{col}_te"
            new_features.append(out_col)
            
            # OOF Train
            oof_vals = np.full(len(train_df), global_mean)
            for tr_idx, va_idx in kf.split(train_df):
                tr_x, tr_y = temp_train.iloc[tr_idx][col], target.iloc[tr_idx]
                va_x = temp_train.iloc[va_idx][col]
                
                # Simple mean
                means = tr_y.groupby(tr_x).mean()
                oof_vals[va_idx] = va_x.map(means).fillna(global_mean).values
            
            train_df[out_col] = oof_vals
            
            # Full Train for Test
            full_means = target.groupby(temp_train[col]).mean()
            test_df[out_col] = temp_test[col].map(full_means).fillna(global_mean)
            
            if val_df is not None:
                val_df[out_col] = temp_val[col].map(full_means).fillna(global_mean)
            if orig_df is not None:
                orig_df[out_col] = temp_orig[col].map(full_means).fillna(global_mean)

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
        "generated_pairs": pairs,
        "new_features": new_features,
        "config": {k: v for k, v in config.items() if not k.startswith("_")}
    }

    return train_df, val_df, test_df, orig_df, state_dict
