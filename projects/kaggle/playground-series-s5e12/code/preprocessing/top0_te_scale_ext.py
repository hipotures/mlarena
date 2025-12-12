"""
Target encoding + label encoding + standard scaling with optional merge of external dataset.

Steps:
- (Optional) merge original dataset (e.g., data/diabetes_dataset.csv) into train:
  * drop specified columns from orig
  * add id column if missing (sequential after max train id)
  * align columns to competition train (missing -> NA)
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


def _merge_original(train_df: pd.DataFrame, test_df: pd.DataFrame, config: Dict[str, Any], id_col: str, target: Optional[str]) -> pd.DataFrame:
    """Merge external dataset into train, aligning columns."""
    artifact_dir = Path(config["_artifact_dir"]).resolve()
    project_root = artifact_dir.parent.parent.parent.parent

    orig_path = Path(config.get("orig_path", "data/diabetes_dataset.csv"))
    if not orig_path.is_absolute():
        orig_path = project_root / orig_path
    if not orig_path.exists():
        # No merge if file missing
        return train_df

    drop_cols = config.get("drop_orig_columns") or []

    orig_df = pd.read_csv(orig_path)
    # Drop specified columns
    orig_df = orig_df.drop(columns=[c for c in drop_cols if c in orig_df.columns], errors="ignore")

    # Ensure target column exists
    if target and target not in orig_df.columns:
        orig_df[target] = pd.NA

    # Ensure id column exists
    if id_col not in orig_df.columns:
        start_id = train_df[id_col].max() + 1 if id_col in train_df.columns else 0
        orig_df[id_col] = range(start_id, start_id + len(orig_df))

    # Align to train columns
    cols = list(train_df.columns)
    for c in cols:
        if c not in orig_df.columns:
            orig_df[c] = pd.NA
    orig_df = orig_df[cols]

    merged = pd.concat([train_df, orig_df], ignore_index=True)
    return merged


def fit_transform(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    test_df: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame, Dict[str, Any]]:
    meta = _dataset_meta(config)
    id_col = meta.get("id_column") or "id"
    target = meta.get("target") or config.get("target_column")
    if not target:
        raise ValueError("target column not provided in dataset metadata")

    # Merge external dataset if available
    train_merged = _merge_original(train_df.copy(), test_df, config, id_col=id_col, target=target)
    test = test_df.copy()

    cat_override = config.get("categorical_columns") or None
    cat_cols, num_cols = _infer_columns(train_merged, meta, cat_override)

    # Target encoding
    train_te, test_te = _target_encode(train_merged, test, target, cat_cols, folds=int(config.get("te_folds", 5)))

    # Label encode categorical columns
    train_le, test_le = _label_encode(train_te, test_te, cat_cols)

    # Standard scale numeric columns (including mean_* features) but exclude target
    num_cols_all = [c for c in train_le.columns if c not in {id_col, target} and c not in cat_cols]
    scaler = StandardScaler()
    train_le[num_cols_all] = scaler.fit_transform(train_le[num_cols_all])
    test_le[num_cols_all] = scaler.transform(test_le[num_cols_all])

    # Optional: compute adversarial weights on processed features to match model
    artifact_dir = Path(config.get("_artifact_dir", Path(".")))
    project_root = artifact_dir.parent.parent.parent.parent
    weights_output = Path(config.get("weights_output", project_root / "data" / "train_av_weights.csv"))
    av_model_dir = artifact_dir / "av_model_te_ext"
    av_model_dir.mkdir(parents=True, exist_ok=True)

    drop_cols = [id_col]
    if target:
        drop_cols.append(target)

    av_df = compute_adversarial_weights(
        train_df=train_le,
        test_df=test_le,
        id_column=id_col,
        target_column=None,
        drop_columns=drop_cols,
        presets=config.get("av_presets", "best_quality"),
        time_limit=int(config.get("av_time_limit", 1200)),
        included_model_types=config.get("av_included_model_types", ["GBM"]),
        output_dir=av_model_dir,
        hyperparameter_tune_kwargs=config.get("av_hpo_kwargs"),
        random_seed=RANDOM_SEED,
    )

    # Persist weights relative to project root (data/) and in artifacts
    weights_output = weights_output if weights_output.is_absolute() else project_root / weights_output
    weights_output.parent.mkdir(parents=True, exist_ok=True)
    av_df.to_csv(weights_output, index=False)

    # Save under the preprocess artifacts dir (artifact_dir already points to experiments/pre-*/artifacts/preprocess)
    artifact_weights = artifact_dir / "train_av_weights.csv"
    artifact_weights.parent.mkdir(parents=True, exist_ok=True)
    av_df.to_csv(artifact_weights, index=False)

    state = {
        "categorical_cols": cat_cols,
        "numeric_cols": num_cols_all,
        "te_folds": int(config.get("te_folds", 5)),
        "random_seed": RANDOM_SEED,
        "merged_rows": len(train_merged) - len(train_df),
        "orig_path": str(config.get("orig_path", "data/diabetes_dataset.csv")),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "weights_path": str(weights_output),
    }

    return train_le, None if val_df is None else val_df, test_le, state


def transform(df: pd.DataFrame, state_dict: Dict[str, Any], config: Dict[str, Any]) -> pd.DataFrame:
    # For simplicity, reuse fit logic; this module is intended for training-time use
    return df.copy()
