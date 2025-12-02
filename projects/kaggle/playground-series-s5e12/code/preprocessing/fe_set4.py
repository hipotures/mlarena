"""
Optimized Feature-engineering + target-encoding module using pure Polars logic.
Includes GPU-accelerated Feature Selection using XGBoost (inference on GPU).

Highlights:
- Polars-native binning (`cut`/`qcut`), rounding, digit features, pairwise cat combos
- Target encoding fully in Polars (fold-wise), maps stored for inference
- Feature Selection Stage 1: Drop constant + Redundant (Joint Cardinality) + Correlated
- Feature Selection Stage 2: Permutation Importance on GPU via XGBoost
  (Solves the CPU bottleneck by moving inference to GPU)
- Outputs pandas DataFrames for downstream model compatibility
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
from sklearn.metrics import roc_auc_score, mean_squared_error
from pathlib import Path
import os

# Try importing XGBoost (Preferred for GPU Inference)
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# Try importing cuDF (RAPIDS) for full GPU pipeline
try:
    import cudf
    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False

MISSING_TOKEN = "<MISSING>"

# Polars type buckets
INT_TYPES = {
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
}
FLOAT_TYPES = {pl.Float32, pl.Float64}


def _dataset_meta(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get("_dataset") or {}


def _drop_ignored(df: pl.DataFrame, meta: Dict[str, Any]) -> pl.DataFrame:
    ignored = set(meta.get("ignored_columns") or [])
    target = meta.get("target")
    id_col = meta.get("id_column")

    drops = set(ignored)
    if id_col:
        drops.add(id_col)
    if target in drops:
        drops.remove(target)

    if drops:
        existing = [c for c in drops if c in df.columns]
        if existing:
            return df.drop(existing)
    return df


def _infer_cols(df: pl.DataFrame, meta: Dict[str, Any], cat_over: set[str], num_over: set[str]) -> Tuple[List[str], List[str]]:
    ignore = set(meta.get("ignored_columns") or [])
    target = meta.get("target")
    id_col = meta.get("id_column")
    if target:
        ignore.add(target)
    if id_col:
        ignore.add(id_col)

    cat_cols: List[str] = []
    num_cols: List[str] = []
    schema = df.schema

    for col in df.columns:
        if col in ignore:
            continue
        dtype = schema[col]
        if col in cat_over:
            cat_cols.append(col)
            continue
        if col in num_over:
            num_cols.append(col)
            continue
        if dtype in (pl.Utf8, pl.Categorical, pl.Boolean):
            cat_cols.append(col)
        elif dtype in INT_TYPES:
            try:
                nunique = df[col].n_unique()
            except Exception:
                nunique = 0
            if nunique <= 30:
                cat_cols.append(col)
            else:
                num_cols.append(col)
        elif dtype in FLOAT_TYPES:
            num_cols.append(col)

    return sorted(set(cat_cols)), sorted(set(num_cols))


def _add_binning_features(df: pl.DataFrame, num_cols: list[str], cfg: Dict[str, Any]) -> pl.DataFrame:
    quantile_bins = cfg.get("quantile_bins", []) or []
    uniform_bins = cfg.get("uniform_bins", []) or []
    log1p_bins = cfg.get("log1p_bins", []) or []
    if not (quantile_bins or uniform_bins or log1p_bins):
        return df

    exprs: List[pl.Expr] = []
    for col in num_cols:
        for q in quantile_bins:
            exprs.append(
                pl.col(col)
                .qcut(int(q), labels=None, allow_duplicates=True)
                .cast(pl.Utf8)
                .fill_null("")
                .alias(f"{col}_q{q}")
            )
        if uniform_bins:
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min is None or c_max is None or c_min == c_max:
                for u in uniform_bins:
                    exprs.append(pl.lit("").alias(f"{col}_u{u}"))
            else:
                for u in uniform_bins:
                    breaks = np.linspace(c_min, c_max, int(u) + 1)[1:-1]
                    exprs.append(
                        pl.col(col)
                        .cut(breaks, labels=None)
                        .cast(pl.Utf8)
                        .fill_null("")
                        .alias(f"{col}_u{u}")
                    )
        if log1p_bins:
            log_col = pl.col(col).log1p()
            for lb in log1p_bins:
                exprs.append(
                    log_col
                    .qcut(int(lb), labels=None, allow_duplicates=True)
                    .cast(pl.Utf8)
                    .fill_null("")
                    .alias(f"{col}_logq{lb}")
                )
    if exprs:
        df = df.with_columns(exprs)
    return df


def _add_rounding_features(df: pl.DataFrame, num_cols: list[str], multipliers: list[int]) -> pl.DataFrame:
    if not multipliers:
        return df
    exprs: List[pl.Expr] = []
    for col in num_cols:
        col_expr = pl.col(col)
        for m in multipliers:
            if m <= 0:
                continue
            exprs.append(((col_expr / m).round(0) * m).alias(f"{col}_round_{m}"))
            exprs.append(((col_expr / m).floor() * m).alias(f"{col}_floor_{m}"))
    if exprs:
        df = df.with_columns(exprs)
    return df


def _add_digit_features(df: pl.DataFrame, num_cols: list[str], digit_mods: list[int]) -> pl.DataFrame:
    if not digit_mods:
        return df
    exprs: List[pl.Expr] = []
    for col in num_cols:
        c = pl.col(col)
        int_part = c.floor().abs().cast(pl.Int64)
        frac_part = (c - c.floor()).abs()
        exprs.append((int_part % 10).alias(f"{col}_int_last_digit"))
        exprs.append(((int_part // 10) % 10).alias(f"{col}_int_first_digit"))
        exprs.append((frac_part * 10).floor().cast(pl.Int64).alias(f"{col}_frac_first_digit"))
        for mod in digit_mods:
            if mod <= 0:
                continue
            exprs.append((int_part % mod).alias(f"{col}_mod_{mod}"))
    if exprs:
        df = df.with_columns(exprs)
    return df


def _add_pairwise_cats(df: pl.DataFrame, cat_cols: List[str], limit: int) -> pl.DataFrame:
    if limit < 1 or len(cat_cols) < 2:
        return df
    exprs: List[pl.Expr] = []
    top = cat_cols[:limit]
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            c1, c2 = top[i], top[j]
            exprs.append(pl.concat_str([pl.col(c1).cast(pl.Utf8), pl.col(c2).cast(pl.Utf8)], separator="|").alias(f"{c1}__{c2}"))
    if exprs:
        df = df.with_columns(exprs)
    return df


def _calc_te_map(df: pl.DataFrame, col: str, target: str, smoothing: float, min_leaf: int, global_mean: float) -> pl.DataFrame:
    agg = df.group_by(col).agg(
        [
            pl.count(target).alias("count"),
            pl.mean(target).alias("mean"),
        ]
    )
    agg = agg.with_columns(
        pl.when(pl.col("count") >= min_leaf)
        .then((pl.col("count") * pl.col("mean") + smoothing * global_mean) / (pl.col("count") + smoothing))
        .otherwise(pl.lit(global_mean))
        .alias("encoding")
    )
    return agg.select([col, "encoding"])


def _fit_target_encoding_polars(
    df: pl.DataFrame,
    target_col: str,
    cat_cols: list[str],
    folds: int,
    smoothing: float,
    min_leaf: int,
    seed: int,
) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    global_mean = df.select(pl.mean(target_col)).item()
    df_folds = df.with_columns((pl.int_range(0, pl.len()).shuffle(seed=seed) % folds).alias("__fold__"))

    inference_maps: Dict[str, Dict[Any, float]] = {}
    for col in cat_cols:
        full_map = _calc_te_map(df, col, target_col, smoothing, min_leaf, global_mean)
        inference_maps[col] = {k: v for k, v in full_map.iter_rows()}

    final_parts = []
    for fold_idx in range(folds):
        train_fold = df_folds.filter(pl.col("__fold__") != fold_idx)
        val_fold = df_folds.filter(pl.col("__fold__") == fold_idx)
        val_enriched = val_fold
        for col in cat_cols:
            fmap = _calc_te_map(train_fold, col, target_col, smoothing, min_leaf, global_mean).rename({"encoding": f"{col}_te"})
            val_enriched = val_enriched.join(fmap, on=col, how="left")
            val_enriched = val_enriched.with_columns(pl.col(f"{col}_te").fill_null(global_mean))
        final_parts.append(val_enriched)

    out_df = pl.concat(final_parts).drop("__fold__")
    te_state = {"global_mean": global_mean, "maps": inference_maps, "cols": cat_cols}
    return out_df, te_state


def _apply_te_inference_polars(df: pl.DataFrame, te_state: Dict[str, Any]) -> pl.DataFrame:
    global_mean = te_state["global_mean"]
    cols = te_state["cols"]
    maps = te_state["maps"]
    for col in cols:
        mapping = maps.get(col, {})
        if not mapping:
            df = df.with_columns(pl.lit(global_mean).alias(f"{col}_te"))
            continue
        map_df = pl.DataFrame({col: list(mapping.keys()), f"{col}_te": list(mapping.values())})
        if df.schema.get(col) and df.schema[col] != map_df.schema[col]:
            map_df = map_df.with_columns(pl.col(col).cast(df.schema[col]))
        df = df.join(map_df, on=col, how="left")
        df = df.with_columns(pl.col(f"{col}_te").fill_null(global_mean))
    return df


def _identify_drop_cols(df: pl.DataFrame, cfg: Dict[str, Any], meta: Dict[str, Any]) -> List[str]:
    """
    Stage 1: Drop Constant + Redundant (Joint Cardinality) + Correlated
    """
    target = meta.get("target")
    before = len(df.columns)
    candidates = [c for c in df.columns if c != target]
    if not candidates:
        return []

    n_unique_map = df.select([pl.col(c).n_unique().alias(c) for c in candidates]).row(0, named=True)
    drop_cols = set()

    # Constant
    for col, nu in n_unique_map.items():
        if nu <= 1: drop_cols.add(col)

    # Redundant Categorical
    candidates_by_nu: Dict[int, List[str]] = {}
    for col, nu in n_unique_map.items():
        if col in drop_cols: continue
        if df.schema[col] in FLOAT_TYPES: continue
        candidates_by_nu.setdefault(nu, []).append(col)

    for nu, cols_group in candidates_by_nu.items():
        if len(cols_group) < 2: continue
        cols_group.sort(key=lambda x: (len(x), x))
        for i in range(len(cols_group)):
            c1 = cols_group[i]
            if c1 in drop_cols: continue
            for j in range(i + 1, len(cols_group)):
                c2 = cols_group[j]
                if c2 in drop_cols: continue
                pair_unique = df.select(pl.struct([c1, c2]).n_unique()).item()
                if pair_unique == nu: drop_cols.add(c2)

    # Correlated Numeric
    corr_drop = set()
    corr_threshold = float(cfg.get("correlation_threshold", 0.99))
    if corr_threshold < 1.0:
        num_candidates = [c for c in df.columns if c not in drop_cols and c != target and df.schema[c] in (INT_TYPES | FLOAT_TYPES)]
        if len(num_candidates) > 1:
            try:
                # Use cuDF if available for correlation
                if CUDF_AVAILABLE:
                    pdf = cudf.from_arrow(df.select(num_candidates).to_arrow())
                else:
                    pdf = df.select(num_candidates).to_pandas() # fallback
                
                # Check backend
                if hasattr(pdf, 'to_pandas'): # It's cudf
                    corr_matrix = pdf.corr().to_pandas()
                else:
                    corr_matrix = pdf.corr()
                    
                cols = corr_matrix.columns.tolist()
                mat = corr_matrix.to_numpy()
            except Exception:
                # Fallback to Polars CPU
                corr_pl = df.select(num_candidates).corr()
                cols = corr_pl.columns
                mat = corr_pl.to_numpy()

            for i in range(len(cols)):
                c1 = cols[i]
                if c1 in drop_cols: continue
                for j in range(i + 1, len(cols)):
                    c2 = cols[j]
                    if c2 in drop_cols: continue
                    if abs(mat[i, j]) > corr_threshold:
                        corr_drop.add(c2)

    drop_cols.update(corr_drop)
    cfg["_drop_summary_stage1"] = {"before": before, "after": before - len(drop_cols), "dropped_count": len(drop_cols)}
    return list(drop_cols)


def _load_eda_types(cfg: Dict[str, Any]) -> Dict[str, str]:
    """
    Load column type hints from EDA profile (ydata_profile_min.json).
    Returns mapping: col -> "categorical" | "numeric"
    """
    # Resolve project root from module path (code/preprocessing/..)
    project_root = Path(__file__).resolve().parents[2]

    path_cfg = cfg.get("eda_profile_path")
    if path_cfg:
        eda_path = Path(path_cfg)
        if not eda_path.is_absolute():
            eda_path = project_root / eda_path
    else:
        eda_path = project_root / "experiments" / "init" / "eda" / "ydata_profile_min.json"

    if not eda_path.exists():
        return {}

    try:
        import json
        data = json.loads(eda_path.read_text())
    except Exception:
        return {}

    out: Dict[str, str] = {}
    for col, payload in (data.get("variables") or {}).items():
        t = payload.get("type")
        if t in ("Categorical", "Category", "Boolean"):
            out[col] = "categorical"
        elif t in ("Numeric", "Real", "Number"):
            out[col] = "numeric"
    return out


def _run_permutation_selection_gpu(
    df: pl.DataFrame, target: str, cfg: Dict[str, Any]
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Stage 2: GPU-Accelerated Permutation Importance using XGBoost.
    Uses 'gpu_predictor' to force inference on GPU, solving CPU bottlenecks.
    """
    if not XGB_AVAILABLE:
        print("XGBoost not installed. Skipping selection.")
        return df.columns, {}

    features = [c for c in df.columns if c != target]
    if not features:
        return df.columns, {}

    # 1. Prepare Data
    # Convert Polars -> Pandas -> (Optional) cuDF
    df_pd = df.to_pandas()
    df_pd = df_pd.sample(frac=1.0, random_state=42).reset_index(drop=True)
    split_idx = int(len(df_pd) * 0.8)
    
    train_pd = df_pd.iloc[:split_idx].copy()
    val_pd = df_pd.iloc[split_idx:].copy()

    # Apply EDA-based dtype hints if available
    eda_types = _load_eda_types(cfg)
    if eda_types:
        for df_ in (train_pd, val_pd):
            for col, kind in eda_types.items():
                if col not in df_.columns or col == target:
                    continue
                if kind == "numeric":
                    df_.loc[:, col] = pd.to_numeric(df_[col], errors="coerce")
                elif kind == "categorical":
                    df_.loc[:, col] = df_[col].astype("category")

    # Force any remaining non-numeric columns to category to avoid object dtypes
    for df_ in (train_pd, val_pd):
        for c in df_.columns:
            if c == target:
                continue
            if not pd.api.types.is_numeric_dtype(df_[c]):
                df_.loc[:, c] = df_[c].astype("category")

    # Normalize any remaining object dtypes to category before metric setup
    for df_ in (train_pd, val_pd):
        obj_cols = df_.select_dtypes(include=["object"]).columns.tolist()
        if obj_cols:
            df_.loc[:, obj_cols] = df_.loc[:, obj_cols].astype("category")

    # Determine Metric & Objective
    n_unique_target = df[target].n_unique()
    if n_unique_target <= 2:
        objective = "binary:logistic"
        eval_metric = "auc"
        maximize = True
        def scorer(y_true, y_pred):
            return roc_auc_score(y_true, y_pred)
    else:
        objective = "reg:squarederror"
        eval_metric = "rmse"
        maximize = False
        def scorer(y_true, y_pred):
            return mean_squared_error(y_true, y_pred, squared=False)

    # 2. Setup XGBoost with GPU
    xgb_params = {
        "tree_method": "hist",
        "device": "cuda",             # Use GPU for training
        "objective": objective,
        "eval_metric": eval_metric,
        "verbosity": 0,
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 150,
        "predictor": "gpu_predictor", # FORCE GPU INFERENCE
    }

    # Handle dtypes for XGBoost (no object/string allowed)
    eda_types = _load_eda_types(cfg)
    if eda_types:
        for df_ in (train_pd, val_pd):
            for col, kind in eda_types.items():
                if col not in df_.columns or col == target:
                    continue
                if kind == "numeric":
                    df_.loc[:, col] = pd.to_numeric(df_[col], errors="coerce")
                elif kind == "categorical":
                    df_.loc[:, col] = df_[col].astype("category")

    for df_ in (train_pd, val_pd):
        for c in df_.columns:
            if c == target:
                continue
            if pd.api.types.is_object_dtype(df_[c]) or pd.api.types.is_string_dtype(df_[c]):
                coerced = pd.to_numeric(df_[c], errors="coerce")
                if coerced.notna().sum() == len(df_[c]):
                    df_.loc[:, c] = coerced
                else:
                    df_.loc[:, c] = df_[c].astype("category")
        obj_cols = df_.select_dtypes(include=["object"]).columns.tolist()
        if obj_cols:
            df_.loc[:, obj_cols] = df_.loc[:, obj_cols].astype("category")
    cat_cols = train_pd.select_dtypes(include=['category']).columns.tolist()

    # Extra safety: ensure no object dtypes remain before train/predict
    for df_ in (train_pd, val_pd):
        obj_cols = df_.select_dtypes(include=["object"]).columns.tolist()
        if obj_cols:
            df_.loc[:, obj_cols] = df_.loc[:, obj_cols].astype("category")

    # 3. Train Probe Model
    # If cuDF is available, move data to GPU to avoid CPU shuffling later
    if CUDF_AVAILABLE:
        print("Using RAPIDS (cuDF) for full-GPU permutation loop...")
        X_train = cudf.DataFrame.from_pandas(train_pd.drop(columns=[target]))
        y_train = cudf.Series.from_pandas(train_pd[target])
        X_val = cudf.DataFrame.from_pandas(val_pd.drop(columns=[target]))
        y_val_cpu = val_pd[target].values # Keep true labels on CPU for scoring
    else:
        print("Using Pandas + XGBoost GPU Predictor...")
        X_train = train_pd.drop(columns=[target]).copy()
        y_train = train_pd[target].copy()
        X_val = val_pd.drop(columns=[target]).copy()
        obj_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
        if obj_cols:
            X_train.loc[:, obj_cols] = X_train.loc[:, obj_cols].astype("category")
            X_val.loc[:, obj_cols] = X_val.loc[:, obj_cols].astype("category")
        obj_cols_val = X_val.select_dtypes(include=["object"]).columns.tolist()
        if obj_cols_val:
            X_val.loc[:, obj_cols_val] = X_val.loc[:, obj_cols_val].astype("category")
        y_val_cpu = val_pd[target].values

    # Final guard: encode all non-numeric columns as category codes (avoids object)
    def _encode_df(df: pd.DataFrame) -> pd.DataFrame:
        def encode_col(col: pd.Series):
            if pd.api.types.is_numeric_dtype(col):
                return col
            return col.astype("category").cat.codes
        return df.apply(encode_col)

    X_train = _encode_df(X_train)
    X_val = _encode_df(X_val)

    # Train
    # enable_categorical=True is needed for pandas categorical
    model = xgb.XGBModel(**xgb_params, enable_categorical=True)
    model.fit(X_train, y_train)

    # 4. Baseline Score
    baseline_preds = model.predict(X_val)
    # Move preds to CPU if they are on GPU (cupy/cudf)
    if hasattr(baseline_preds, 'get'): 
        baseline_preds = baseline_preds.get()
    
    baseline_score = scorer(y_val_cpu, baseline_preds)
    print(f"Baseline Score ({eval_metric}): {baseline_score:.5f} | objective={objective} | features={len(features)}")

    # 5. Permutation Loop
    keep_features = []
    drop_stats = []

    # If cuDF, columns are on GPU. Shuffling is fast.
    # If Pandas, columns are on CPU. Shuffling is slow, but predict is GPU.
    
    col_list = features
    
    for col in col_list:
        # Backup column
        if CUDF_AVAILABLE:
            save_col = X_val[col].copy()
            # GPU Shuffle
            X_val[col] = X_val[col].sample(frac=1.0).values
        else:
            save_col = X_val[col].values.copy()
            # CPU Shuffle
            X_val[col] = np.random.permutation(X_val[col].values)
        
        # Predict (GPU)
        shuff_preds = model.predict(X_val)
        if hasattr(shuff_preds, 'get'): shuff_preds = shuff_preds.get()
        
        shuff_score = scorer(y_val_cpu, shuff_preds)
        
        # Restore column
        if CUDF_AVAILABLE:
            X_val[col] = save_col
        else:
            X_val[col] = save_col

        # Calc Importance
        if maximize: # AUC
            imp = baseline_score - shuff_score # Positive = Useful
        else: # RMSE
            imp = shuff_score - baseline_score # Positive = Useful (error increased)

        if imp > 0.00001: 
            keep_features.append(col)
        else:
            drop_stats.append((col, imp))

    print(f"Permutation Selection: Dropped {len(drop_stats)} features. Kept {len(keep_features)}.")
    if drop_stats:
        worst = sorted(drop_stats, key=lambda x: x[1])[:5]
        print("Worst (harmful/neutral) features:", worst)
    
    final_cols = keep_features + [target]
    stats = {
        "baseline": baseline_score,
        "dropped": len(drop_stats),
        "worst_harmful": sorted(drop_stats, key=lambda x: x[1])[:10]
    }
    return final_cols, stats


def _process_pipeline(
    df: pl.DataFrame,
    meta: Dict[str, Any],
    cfg: Dict[str, Any],
    te_state: Optional[Dict[str, Any]],
    is_train: bool,
) -> Tuple[pl.DataFrame, Optional[Dict[str, Any]], set[str], list[str]]:
    cat_overrides = set(cfg.get("categorical_overrides") or [])
    num_overrides = set(cfg.get("numeric_overrides") or [])

    te_features_enabled = bool(cfg.get("target_encoding_features", True))
    cat_cols, num_cols = _infer_cols(df, meta, cat_overrides, num_overrides)
    fill_missing = bool(cfg.get("fill_missing", True))
    string_clean = bool(cfg.get("string_clean", True))

    exprs: List[pl.Expr] = []
    for col in cat_cols:
        e = pl.col(col).cast(pl.Utf8)
        if fill_missing:
            e = e.fill_null(MISSING_TOKEN)
        if string_clean:
            e = e.str.strip_chars().str.to_lowercase()
        exprs.append(e.alias(col))

    if fill_missing and num_cols:
        medians_df = df.select([pl.col(c).median() for c in num_cols])
        medians = medians_df.row(0)
        for col, median_val in zip(num_cols, medians):
            exprs.append(pl.col(col).fill_null(median_val).alias(col))

    if exprs:
        df = df.with_columns(exprs)

    df = _add_binning_features(df, num_cols, cfg)
    df = _add_rounding_features(df, num_cols, cfg.get("round_multipliers", []))
    df = _add_digit_features(df, num_cols, cfg.get("digit_mods", []))

    cat_cols_current = set(cat_cols)
    for c in df.columns:
        if c in cat_cols_current or c in num_cols or c in meta.get("ignored_columns", []):
            continue
        if any(tok in c for tok in ("_q", "_u", "_logq")):
            cat_cols.append(c)

    pair_limit = int(cfg.get("pairwise_cat_limit", 0) or 0)
    if pair_limit > 1:
        df = _add_pairwise_cats(df, cat_cols, pair_limit)
        combo_cols = [c for c in df.columns if "__" in c]
        cat_cols.extend(combo_cols)

    te_cfg = cfg.get("target_encoding", {}) or {}
    te_enabled = te_features_enabled and bool(te_cfg.get("enabled", True))
    target = meta.get("target")

    if te_enabled and target and (target in df.columns or te_state):
        if is_train:
            df, te_state = _fit_target_encoding_polars(
                df,
                target,
                cat_cols,
                folds=int(te_cfg.get("folds", 5)),
                smoothing=float(te_cfg.get("smoothing", 10.0)),
                min_leaf=int(te_cfg.get("min_samples_leaf", 20)),
                seed=int(te_cfg.get("seed", 42)),
            )
        elif te_state:
            df = _apply_te_inference_polars(df, te_state)

    return df, te_state, set(cat_cols), num_cols


def fit_transform(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    test_df: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame, Dict[str, Any]]:
    meta = _dataset_meta(config)
    cfg = {k: v for k, v in config.items() if k != "_dataset"}
    id_col = meta.get("id_column")
    sample_frac = float(cfg.get("sample_frac", 1.0) or 1.0)
    sample_seed = int(cfg.get("sample_seed", 42))

    if sample_frac < 1.0:
        print(f"[fe_set4] Sampling train data: frac={sample_frac}")
        train_df = train_df.sample(frac=sample_frac, random_state=sample_seed).reset_index(drop=True)

    # Avoid multiprocessing semaphores issues in polars conversions
    os.environ.setdefault("POLARS_NO_THREADING", "1")

    train_ids = train_df[id_col].copy() if id_col and id_col in train_df else None
    val_ids = val_df[id_col].copy() if val_df is not None and id_col and id_col in val_df else None
    test_ids = test_df[id_col].copy() if id_col and id_col in test_df else None

    pl_train = _drop_ignored(pl.from_arrow(pa.Table.from_pandas(train_df, preserve_index=False)), meta)
    pl_val = _drop_ignored(pl.from_arrow(pa.Table.from_pandas(val_df, preserve_index=False)), meta) if val_df is not None else None
    pl_test = _drop_ignored(pl.from_arrow(pa.Table.from_pandas(test_df, preserve_index=False)), meta)

    # 1. Generate Features
    pl_train, te_state, cat_cols, num_cols = _process_pipeline(pl_train, meta, cfg, None, True)

    # 2. Stage 1: Drop Duplicates/Corr
    to_drop = _identify_drop_cols(pl_train, cfg, meta)
    final_cols = [c for c in pl_train.columns if c not in to_drop]
    pl_train = pl_train.select(final_cols)
    
    # 3. Stage 2: Permutation Selection (GPU XGBoost)
    perm_stats = {}
    if cfg.get("selection_enabled", False):
        print(f"Starting Permutation Selection on GPU (XGBoost)... [features={len(pl_train.columns)-1}]")
        final_cols, perm_stats = _run_permutation_selection_gpu(pl_train, meta["target"], cfg)
        pl_train = pl_train.select(final_cols)

    # 4. Apply to Val/Test
    pl_val_out = None
    if pl_val is not None:
        pl_val, _, _, _ = _process_pipeline(pl_val, meta, cfg, te_state, False)
        valid_val_cols = [c for c in final_cols if c in pl_val.columns]
        pl_val_out = pl_val.select(valid_val_cols).to_pandas()

    pl_test, _, _, _ = _process_pipeline(pl_test, meta, cfg, te_state, False)
    valid_test_cols = [c for c in final_cols if c in pl_test.columns]
    pl_test_out = pl_test.select(valid_test_cols).to_pandas()

    # Reattach ID
    if id_col and train_ids is not None:
        pl_train_out = pl_train.to_pandas()
        pl_train_out[id_col] = train_ids.reset_index(drop=True)
    else:
        pl_train_out = pl_train.to_pandas()

    if pl_val_out is not None and id_col and val_ids is not None:
        pl_val_out[id_col] = val_ids.reset_index(drop=True)

    if id_col and test_ids is not None:
        pl_test_out[id_col] = test_ids.reset_index(drop=True)

    state = {
        "version": "1.0-gpu-xgb",
        "template": "fe_set1",
        "dataset": meta,
        "config": cfg,
        "cat_cols": sorted(list(cat_cols)),
        "num_cols": num_cols,
        "target_encoding": te_state,
        "final_columns": final_cols,
        "shapes": {
            "train": list(pl_train_out.shape),
            "val": list(pl_val_out.shape) if pl_val_out is not None else None,
            "test": list(pl_test_out.shape),
        },
        "drop_summary_stage1": cfg.get("_drop_summary_stage1", {}),
        "perm_selection_stats": perm_stats
    }

    return pl_train_out, pl_val_out, pl_test_out, state

def transform(df: pd.DataFrame, state_dict: Dict[str, Any], config: Dict[str, Any]) -> pd.DataFrame:
    if df is None: return None
    meta = state_dict.get("dataset") or _dataset_meta(config)
    cfg = state_dict.get("config") or {k: v for k, v in config.items() if k != "_dataset"}
    te_state = state_dict.get("target_encoding")
    final_cols = state_dict.get("final_columns")
    id_col = meta.get("id_column")
    id_series = df[id_col].copy() if id_col and id_col in df else None

    pl_df = _drop_ignored(pl.from_pandas(df), meta)
    processed, _, _, _ = _process_pipeline(pl_df, meta, cfg, te_state, False)

    if final_cols:
        keep = [c for c in final_cols if c in processed.columns]
        processed = processed.select(keep)

    out_df = processed.to_pandas()
    if id_col and id_series is not None:
        out_df[id_col] = id_series.reset_index(drop=True)
    return out_df
