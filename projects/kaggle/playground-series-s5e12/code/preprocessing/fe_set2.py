"""
Optimized Feature-engineering + target-encoding module using pure Polars logic.
Includes GPU-accelerated RFE (Recursive Feature Elimination) with LightGBM.

Highlights:
- Polars-native binning (`cut`/`qcut`), rounding, digit features, pairwise cat combos
- Target encoding fully in Polars (fold-wise), maps stored for inference
- Feature Selection Stage 1: Drop constant + Redundant (Joint Cardinality) + Correlated
- Feature Selection Stage 2: Recursive Feature Elimination (RFE) via LightGBM (GPU)
- Outputs pandas DataFrames for downstream model compatibility
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl

# Try importing LightGBM for RFE
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

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
    Stage 1 Selection: Drop Constant + Redundant (Joint Cardinality) + Correlated
    """
    target = meta.get("target")
    before = len(df.columns)

    candidates = [c for c in df.columns if c != target]
    if not candidates:
        return []

    # Precompute n_unique for all candidates
    n_unique_map = df.select([pl.col(c).n_unique().alias(c) for c in candidates]).row(0, named=True)

    drop_cols = set()

    # Constant columns
    for col, nu in n_unique_map.items():
        if nu <= 1:
            drop_cols.add(col)

    # Redundant categoricals via joint cardinality
    candidates_by_nu: Dict[int, List[str]] = {}
    for col, nu in n_unique_map.items():
        if col in drop_cols:
            continue
        if df.schema[col] in FLOAT_TYPES:
            continue
        candidates_by_nu.setdefault(nu, []).append(col)

    for nu, cols_group in candidates_by_nu.items():
        if len(cols_group) < 2:
            continue
        cols_group.sort(key=lambda x: (len(x), x))
        for i in range(len(cols_group)):
            c1 = cols_group[i]
            if c1 in drop_cols:
                continue
            for j in range(i + 1, len(cols_group)):
                c2 = cols_group[j]
                if c2 in drop_cols:
                    continue
                pair_unique = df.select(pl.struct([c1, c2]).n_unique()).item()
                if pair_unique == nu:
                    drop_cols.add(c2)

    # Correlated numeric columns
    corr_drop = set()
    corr_threshold = float(cfg.get("correlation_threshold", 0.99))
    if corr_threshold < 1.0:
        num_candidates = [
            c for c in df.columns
            if c not in drop_cols and c != target and df.schema[c] in (INT_TYPES | FLOAT_TYPES)
        ]
        if len(num_candidates) > 1:
            try:
                import cudf
                pdf = cudf.from_arrow(df.select(num_candidates).to_arrow())
                corr_matrix_df = pdf.corr()
                cols = corr_matrix_df.columns.to_pandas().tolist()
                mat = corr_matrix_df.to_numpy()
            except Exception:
                corr_pl = df.select(num_candidates).corr()
                cols = corr_pl.columns
                mat = corr_pl.to_numpy()

            for i in range(len(cols)):
                c1 = cols[i]
                if c1 in drop_cols:
                    continue
                for j in range(i + 1, len(cols)):
                    c2 = cols[j]
                    if c2 in drop_cols:
                        continue
                    if abs(mat[i, j]) > corr_threshold:
                        corr_drop.add(c2)

    drop_cols.update(corr_drop)

    cfg["_drop_summary_stage1"] = {
        "before": before,
        "after": before - len(drop_cols),
        "const_dropped": len({c for c, nu in n_unique_map.items() if nu <= 1}),
        "corr_dropped": len(corr_drop),
        "dropped_count": len(drop_cols),
    }
    return list(drop_cols)


def _run_lightgbm_rfe_gpu(
    df: pl.DataFrame, target: str, cfg: Dict[str, Any]
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Stage 2 Selection: Recursive Feature Elimination using LightGBM on GPU.
    """
    if not LGBM_AVAILABLE:
        print("LightGBM not installed, skipping RFE.")
        return df.columns, {}

    # Config params
    n_features_to_select = int(cfg.get("rfe_n_features", 100))
    step = int(cfg.get("rfe_step", 5))  # How many to drop per iteration
    
    current_features = [c for c in df.columns if c != target]
    if len(current_features) <= n_features_to_select:
        return df.columns, {}

    # Check target type to determine objective
    # (Simplified heuristic: few unique values -> binary, else regression)
    n_unique_target = df[target].n_unique()
    if n_unique_target <= 2:
        objective = "binary"
        metric = "auc"
    else:
        objective = "regression"
        metric = "rmse"

    # LightGBM GPU Params
    params = {
        "device": "gpu",  # or 'cuda' depending on LGBM version build
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
        "objective": objective,
        "metric": metric,
        "verbosity": -1,
        "n_jobs": -1,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 100, # Fast estimators for selection
    }
    
    # We must convert to pandas/numpy for LGBM
    # (Pass pandas to handle categories automatically if set as 'category')
    # Note: Target encoding features are numeric, bins are strings/cats.
    # We need to cast categorical strings to 'category' dtype for LGBM
    pdf = df.to_pandas()
    y = pdf[target]
    
    # Prepare X (cast objects to category)
    X = pdf.drop(columns=[target])
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype('category')

    log_history = []
    
    while len(current_features) > n_features_to_select:
        # Train
        train_set = lgb.Dataset(X[current_features], label=y)
        model = lgb.train(params, train_set)
        
        # Get Importance (Gain is usually better for selection than Split)
        importance = model.feature_importance(importance_type='gain')
        feature_name = model.feature_name()
        
        # Create DataFrame to sort
        imp_df = pd.DataFrame({'feature': feature_name, 'importance': importance})
        imp_df = imp_df.sort_values(by='importance', ascending=True) # Ascending -> worst first
        
        # Identify features to drop
        # Ensure we don't drop more than needed to reach target
        num_to_drop = min(step, len(current_features) - n_features_to_select)
        worst_features = imp_df.head(num_to_drop)['feature'].tolist()
        
        # Update list
        current_features = [f for f in current_features if f not in worst_features]
        
        log_history.append({
            "n_features": len(current_features) + num_to_drop,
            "dropped": worst_features,
            "min_importance": imp_df.iloc[0]['importance']
        })
        
        print(f"RFE: {len(current_features) + num_to_drop} -> {len(current_features)} features. Dropped {num_to_drop} worst.")

    # Always keep target in the returned list logic if it was in input
    final_cols = current_features + [target] if target in df.columns else current_features
    
    stats = {
        "rfe_history": log_history,
        "final_count": len(current_features)
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

    # update cat cols after new string features
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
    te_enabled = bool(te_cfg.get("enabled", True))
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
    train_ids = train_df[id_col].copy() if id_col and id_col in train_df else None
    val_ids = val_df[id_col].copy() if val_df is not None and id_col and id_col in val_df else None
    test_ids = test_df[id_col].copy() if id_col and id_col in test_df else None

    pl_train = _drop_ignored(pl.from_pandas(train_df), meta)
    pl_val = _drop_ignored(pl.from_pandas(val_df), meta) if val_df is not None else None
    pl_test = _drop_ignored(pl.from_pandas(test_df), meta)

    # 1. Generate ALL features on Train
    pl_train, te_state, cat_cols, num_cols = _process_pipeline(pl_train, meta, cfg, None, True)

    # 2. Stage 1 Selection: Drop Constant / Redundant / Correlated
    # (Fast, heuristic-based)
    to_drop = _identify_drop_cols(pl_train, cfg, meta)
    final_cols = [c for c in pl_train.columns if c not in to_drop]
    pl_train = pl_train.select(final_cols)
    
    # 3. Stage 2 Selection: Recursive Feature Elimination (RFE) on GPU
    # (Model-based, slower but smarter)
    # Only run if enabled in config
    rfe_stats = {}
    if cfg.get("rfe_enabled", False):
        print(f"Starting RFE on GPU... reducing to {cfg.get('rfe_n_features', 100)} features.")
        final_cols, rfe_stats = _run_lightgbm_rfe_gpu(pl_train, meta["target"], cfg)
        pl_train = pl_train.select(final_cols)

    # 4. Process Val/Test and Apply Selection (Intersection of columns)
    pl_val_out = None
    if pl_val is not None:
        pl_val, _, _, _ = _process_pipeline(pl_val, meta, cfg, te_state, False)
        valid_val_cols = [c for c in final_cols if c in pl_val.columns]
        pl_val_out = pl_val.select(valid_val_cols).to_pandas()

    pl_test, _, _, _ = _process_pipeline(pl_test, meta, cfg, te_state, False)
    valid_test_cols = [c for c in final_cols if c in pl_test.columns]
    pl_test_out = pl_test.select(valid_test_cols).to_pandas()

    # Reattach ID column
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
        "version": "1.0-opt-rfe",
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
        "rfe_stats": rfe_stats
    }

    return pl_train_out, pl_val_out, pl_test_out, state


def transform(df: pd.DataFrame, state_dict: Dict[str, Any], config: Dict[str, Any]) -> pd.DataFrame:
    if df is None:
        return None

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
