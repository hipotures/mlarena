"""
Target encoding + label encoding + standard scaling using external dataset_merger module.

Steps:
- Expects external dataset to be provided via preprocessing chain (e.g. external_dataset)
- (Optional) drop specified columns from merged train (drop_orig_columns config)
- Target-mean encoding per categorical column using KFold (train-only), added as mean_<col>.
- Label-encode categorical columns (fit on train, unseen -> -1).
- Standard-scale numeric columns (including mean_* features) using train stats.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from adversarial_validation import compute_adversarial_weights

RANDOM_SEED = 42


def _dataset_meta(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get("_dataset") or {}


def _infer_columns(train_df: pd.DataFrame, meta: Dict[str, Any], cat_override: Optional[List[str]] = None) -> Tuple[List[str], List[str]]:
    target = meta.get("target")
    id_col = meta.get("id_column")
    ignored = set(meta.get("ignored_columns") or [])
    if target:
        ignored.add(target)
    if id_col:
        ignored.add(id_col)

    cat_cols = set(cat_override or [])
    if not cat_cols:
        cat_cols = set(train_df.select_dtypes(include=["object", "category"]).columns)
    cat_cols -= ignored

    num_cols = set(train_df.columns) - cat_cols - ignored
    return sorted(cat_cols), sorted(num_cols)


def _target_encode(train: pd.DataFrame, test: pd.DataFrame, target: str, cat_cols: List[str], folds: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """KFold target mean encoding with fallback to global mean."""
    kf = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_SEED)
    global_mean = train[target].mean()

    train_out = train.copy()
    test_out = test.copy()

    for col in cat_cols:
        col_codes = train[col].astype("category").cat.codes
        fold_means = np.zeros(len(train_out), dtype=np.float32)

        for tr_idx, val_idx in kf.split(train_out):
            tr_codes = col_codes.iloc[tr_idx]
            tr_target = train_out[target].iloc[tr_idx]
            means = tr_target.groupby(tr_codes).mean()
            val_codes = col_codes.iloc[val_idx]
            val_mean = val_codes.map(means)
            val_mean = val_mean.fillna(global_mean)
            fold_means[val_idx] = val_mean.values

        train_out[f"mean_{col}"] = fold_means

        # Test encoding using global means from full train
        full_means = train_out.groupby(col)[target].mean()
        test_mean = test_out[col].map(full_means).fillna(global_mean)
        train_out[f"mean_{col}"] = train_out[f"mean_{col}"].astype(np.float32)
        test_out[f"mean_{col}"] = test_mean.astype(np.float32)

    return train_out, test_out


def _label_encode(train: pd.DataFrame, test: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_out = train.copy()
    test_out = test.copy()

    for col in cols:
        uniq = pd.Index(train_out[col].astype(str).unique())
        mapping = {v: i for i, v in enumerate(uniq)}
        train_out[col] = train_out[col].astype(str).map(mapping).fillna(-1).astype(int)
        test_out[col] = test_out[col].astype(str).map(mapping).fillna(-1).astype(int)

    return train_out, test_out


def fit_transform(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    test_df: pd.DataFrame,
    config: Dict[str, Any],
    orig_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame, Optional[pd.DataFrame], Dict[str, Any]]:
    artifact_dir = Path(config.get("_artifact_dir", ".")).resolve()
    # For preprocessing chains: artifacts/preprocess → artifacts → step → chain → experiments → project_root
    project_root = artifact_dir.parent.parent.parent.parent.parent
    meta = _dataset_meta(config)
    id_col = meta.get("id_column") or "id"
    target = meta.get("target") or config.get("target_column")
    if not target:
        raise ValueError("target column not provided in dataset metadata")

    # Drop specified columns if provided (e.g., columns from original dataset that we don't want)
    drop_cols = config.get("drop_orig_columns") or []
    train_core = train_df.copy()
    test = test_df.copy()
    orig = orig_df.copy() if orig_df is not None else None

    # Fill missing IDs (from external dataset merger) with sequential values
    # Use safe offset: max(train_max, test_max) + 1 to avoid collision with test IDs
    if orig is not None:
        # When external_dataset is used in union mode, orig often has id_col all-NA.
        # Assign synthetic ids to keep downstream AV utilities stable.
        if id_col in orig.columns and orig[id_col].isna().any():
            train_max = train_core[id_col].max() if id_col in train_core.columns else 0
            test_max = test[id_col].max() if id_col in test.columns else 0

            if pd.isna(train_max):
                train_max = 0
            if pd.isna(test_max):
                test_max = 0

            start_id = int(max(train_max, test_max)) + 1
            na_mask = orig[id_col].isna()
            orig.loc[na_mask, id_col] = range(start_id, start_id + na_mask.sum())

    if id_col in train_core.columns and train_core[id_col].isna().any():
        train_max = train_core[id_col].max()
        test_max = test[id_col].max() if id_col in test.columns else 0

        if pd.isna(train_max):
            train_max = 0
        if pd.isna(test_max):
            test_max = 0

        start_id = int(max(train_max, test_max)) + 1
        na_mask = train_core[id_col].isna()
        train_core.loc[na_mask, id_col] = range(start_id, start_id + na_mask.sum())

    if drop_cols:
        train_core = train_core.drop(columns=[c for c in drop_cols if c in train_core.columns], errors="ignore")
        test = test.drop(columns=[c for c in drop_cols if c in test.columns], errors="ignore")
        if orig is not None:
            orig = orig.drop(columns=[c for c in drop_cols if c in orig.columns], errors="ignore")

    cat_override = config.get("categorical_columns") or None
    cat_cols, _ = _infer_columns(train_core, meta, cat_override)

    n_train = len(train_core)
    combined = pd.concat([train_core, orig], ignore_index=True) if orig is not None else train_core

    # Target encoding
    combined_te, test_te = _target_encode(combined, test, target, cat_cols, folds=int(config.get("te_folds", 5)))

    # Label encode categorical columns
    combined_le, test_le = _label_encode(combined_te, test_te, cat_cols)

    # Standard scale numeric columns (including mean_* features) but exclude target
    num_cols_all = [c for c in combined_le.columns if c not in {id_col, target} and c not in cat_cols]
    scaler = StandardScaler()
    combined_le[num_cols_all] = scaler.fit_transform(combined_le[num_cols_all])
    test_le[num_cols_all] = scaler.transform(test_le[num_cols_all])

    # Split back (NO row-merge output in preprocessing)
    train_le = combined_le.iloc[:n_train].reset_index(drop=True)
    orig_le = combined_le.iloc[n_train:].reset_index(drop=True) if orig is not None else None

    state = {
        "categorical_cols": cat_cols,
        "numeric_cols": num_cols_all,
        "te_folds": int(config.get("te_folds", 5)),
        "random_seed": RANDOM_SEED,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }

    # Adversarial validation weights (train vs test) on processed features
    av_presets = config.get("av_presets")
    av_time_limit = int(config.get("av_time_limit", 0) or 0)
    av_included = config.get("av_included_model_types")
    if av_presets and av_time_limit > 0:
        weights_output = Path(config.get("weights_output", "data/train_av_weights.csv"))
        if not weights_output.is_absolute():
            weights_output = project_root / weights_output
        artifact_weights = artifact_dir / "train_av_weights.csv"
        av_model_dir = artifact_dir / "av_model"

        # Drop label and target-derived TE features to avoid trivial AV (AUC=1)
        drop_for_av = [id_col, target] + [c for c in train_le.columns if c.startswith("mean_")]

        # Compute AV weights only for Kaggle train rows (external/orig stays separate in preprocessing).
        av_df = compute_adversarial_weights(
            train_df=train_le,
            test_df=test_le,
            id_column=id_col,
            target_column=target,
            drop_columns=drop_for_av,
            presets=av_presets,
            time_limit=av_time_limit,
            included_model_types=av_included,
            output_dir=av_model_dir,
        )

        artifact_weights.parent.mkdir(parents=True, exist_ok=True)
        av_df.to_csv(artifact_weights, index=False)
        weights_output.parent.mkdir(parents=True, exist_ok=True)
        av_df.to_csv(weights_output, index=False)

        state.update(
            {
                "av_weights_path": str(weights_output),
                "av_weights_artifact": str(artifact_weights),
                "weights_path": str(artifact_weights),
                "av_rows": len(av_df),
                "av_presets": av_presets,
                "av_time_limit": av_time_limit,
                "av_included_model_types": av_included,
            }
        )

    return train_le, None if val_df is None else val_df, test_le, orig_le, state


def transform(df: pd.DataFrame, state_dict: Dict[str, Any], config: Dict[str, Any]) -> pd.DataFrame:
    # For simplicity, reuse fit logic; this module is intended for training-time use
    return df.copy()
