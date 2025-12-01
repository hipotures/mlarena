"""
Feature-engineering + target-encoding preprocess module (set1) using Polars for transforms.

Steps:
- Drop ignored/id columns (keeps target when present)
- Missing-value handling and string cleaning
- Numeric binning (quantile/uniform/log1p), rounding/flooring, digit-level features
- Pairwise categorical combos
- CV-safe target encoding (pandas for OOF computation; maps cached in state)

Outputs pandas DataFrames to stay compatible with downstream model code.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import KFold

MISSING_TOKEN = "<MISSING>"
INT_TYPES = {
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
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
    drops.discard(target)
    if drops:
        keep = [c for c in df.columns if c not in drops]
        return df.select(keep)
    return df


def _infer_categorical_columns(df: pl.DataFrame, meta: Dict[str, Any], cat_over: set[str], num_over: set[str]) -> set[str]:
    ignore = set(meta.get("ignored_columns") or [])
    target = meta.get("target")
    id_col = meta.get("id_column")
    ignore.update([c for c in (target, id_col) if c])
    cat_cols: set[str] = set(cat_over)
    for col, dtype in df.schema.items():
        if col in ignore or col in num_over:
            continue
        if dtype in (pl.Utf8, pl.Categorical, pl.Boolean):
            cat_cols.add(col)
            continue
        if dtype in INT_TYPES:
            try:
                nunique = df[col].n_unique()
            except Exception:
                nunique = 0
            if nunique <= 30:
                cat_cols.add(col)
    return cat_cols


def _numeric_columns(df: pl.DataFrame, cat_cols: set[str], meta: Dict[str, Any]) -> list[str]:
    ignore = set(meta.get("ignored_columns") or [])
    target = meta.get("target")
    id_col = meta.get("id_column")
    ignore.update([c for c in (target, id_col) if c])
    cols = []
    for col, dtype in df.schema.items():
        if col in cat_cols or col in ignore:
            continue
        if dtype in INT_TYPES or dtype in FLOAT_TYPES:
            cols.append(col)
    return cols


def _bin_edges_quantile(values: np.ndarray, bins: int) -> Optional[np.ndarray]:
    clean = values[~np.isnan(values)]
    if clean.size < 2 or bins < 2:
        return None
    edges = np.quantile(clean, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if edges.size < 2:
        return None
    return edges


def _bin_edges_uniform(values: np.ndarray, bins: int) -> Optional[np.ndarray]:
    clean = values[~np.isnan(values)]
    if clean.size < 2 or bins < 2:
        return None
    lo, hi = float(clean.min()), float(clean.max())
    if lo == hi:
        return None
    edges = np.linspace(lo, hi, bins + 1)
    edges = np.unique(edges)
    if edges.size < 2:
        return None
    return edges


def _edges_to_labels(edges: np.ndarray) -> List[str]:
    labels = []
    for a, b in zip(edges[:-1], edges[1:]):
        labels.append(f"[{a:.6g},{b:.6g})")
    return labels


def _digitize_to_labels(values: np.ndarray, edges: np.ndarray) -> List[str]:
    labels = _edges_to_labels(edges)
    if len(labels) == 0:
        return ["" for _ in range(len(values))]
    idx = np.digitize(values, edges[1:-1], right=False)
    idx = np.clip(idx, 0, len(labels) - 1)
    return [labels[i] if not np.isnan(v) else "" for i, v in zip(idx, values)]


def _add_binning_features(df: pl.DataFrame, num_cols: list[str], cfg: Dict[str, Any]) -> pl.DataFrame:
    quantile_bins = cfg.get("quantile_bins", []) or []
    uniform_bins = cfg.get("uniform_bins", []) or []
    log1p_bins = cfg.get("log1p_bins", []) or []

    new_cols = []
    for col in num_cols:
        values = df[col].to_numpy()
        for q in quantile_bins:
            edges = _bin_edges_quantile(values, int(q))
            if edges is None:
                continue
            labels = _digitize_to_labels(values, edges)
            new_cols.append(pl.Series(f"{col}_q{q}", labels))
        for u in uniform_bins:
            edges = _bin_edges_uniform(values, int(u))
            if edges is None:
                continue
            labels = _digitize_to_labels(values, edges)
            new_cols.append(pl.Series(f"{col}_u{u}", labels))
        for lb in log1p_bins:
            pos_mask = values > 0
            if not pos_mask.any():
                continue
            log_vals = np.full_like(values, np.nan, dtype=float)
            log_vals[pos_mask] = np.log1p(values[pos_mask])
            edges = _bin_edges_quantile(log_vals, int(lb))
            if edges is None:
                continue
            labels = _digitize_to_labels(log_vals, edges)
            new_cols.append(pl.Series(f"{col}_logq{lb}", labels))

    if new_cols:
        df = df.with_columns(new_cols)
    return df


def _add_rounding_features(df: pl.DataFrame, num_cols: list[str], multipliers: list[int]) -> pl.DataFrame:
    new_cols = []
    for col in num_cols:
        series = df[col]
        for m in multipliers:
            if m <= 0:
                continue
            new_cols.append(((series / m).round(0) * m).alias(f"{col}_round_{m}"))
            new_cols.append(((series / m).floor() * m).alias(f"{col}_floor_{m}"))
    if new_cols:
        df = df.with_columns(new_cols)
    return df


def _add_digit_features(df: pl.DataFrame, num_cols: list[str], digit_mods: list[int]) -> pl.DataFrame:
    new_cols = []
    for col in num_cols:
        s = df[col]
        int_part = s.floor()
        frac_part = (s - int_part).abs()
        new_cols.append((int_part.abs().cast(pl.Int64) % 10).alias(f"{col}_int_last_digit"))
        new_cols.append(((int_part.abs() // 10) % 10).alias(f"{col}_int_first_digit"))
        new_cols.append((frac_part * 10).floor().cast(pl.Int64).alias(f"{col}_frac_first_digit"))
        for mod in digit_mods:
            if mod <= 0:
                continue
            new_cols.append((int_part.abs().cast(pl.Int64) % mod).alias(f"{col}_mod_{mod}"))
    if new_cols:
        df = df.with_columns(new_cols)
    return df


def _add_pairwise_cats(df: pl.DataFrame, cat_cols: List[str], limit: int) -> pl.DataFrame:
    new_cols = []
    top = cat_cols[:limit]
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            c1, c2 = top[i], top[j]
            new_cols.append(pl.concat_str([pl.col(c1).cast(pl.Utf8), pl.lit("|"), pl.col(c2).cast(pl.Utf8)]).alias(f"{c1}__{c2}"))
    if new_cols:
        df = df.with_columns(new_cols)
    return df


def _fit_target_encoding_pandas(
    df: pd.DataFrame,
    target_col: str,
    cat_cols: list[str],
    folds: int,
    smoothing: float,
    min_samples_leaf: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    global_mean = df[target_col].mean()
    oof_encoded = pd.DataFrame(index=df.index)
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    def _build_map(fold_df: pd.DataFrame, col: str) -> Dict[Any, float]:
        stats = fold_df.groupby(col)[target_col].agg(["mean", "count"])
        smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
        if min_samples_leaf > 1:
            smooth = smooth[stats["count"] >= min_samples_leaf]
        return smooth.to_dict()

    for col in cat_cols:
        oof_col = pd.Series(index=df.index, dtype=float)
        for train_idx, val_idx in kf.split(df):
            train_fold = df.iloc[train_idx]
            val_fold = df.iloc[val_idx]
            enc_map = _build_map(train_fold, col)
            oof_col.iloc[val_idx] = val_fold[col].map(enc_map).fillna(global_mean)
        oof_col = oof_col.fillna(global_mean)
        oof_encoded[f"{col}_te"] = oof_col

    inference_maps = {col: _build_map(df, col) for col in cat_cols}
    te_state = {
        "global_mean": global_mean,
        "maps": inference_maps,
        "cols": cat_cols,
    }
    encoded_df = df.copy()
    for col in oof_encoded.columns:
        encoded_df[col] = oof_encoded[col]
    return encoded_df, te_state


def _apply_target_encoding_inference(df: pd.DataFrame, te_state: Dict[str, Any]) -> pd.DataFrame:
    result = df.copy()
    global_mean = te_state["global_mean"]
    for col in te_state["cols"]:
        te_col = f"{col}_te"
        enc_map = te_state["maps"].get(col, {})
        result[te_col] = result[col].map(enc_map).fillna(global_mean)
    return result


def _process_polars(
    df: pl.DataFrame,
    meta: Dict[str, Any],
    cfg: Dict[str, Any],
    te_state: Optional[Dict[str, Any]],
    is_train: bool,
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]], set[str], list[str]]:
    if df is None:
        return None, te_state, set(), []

    cat_overrides = set(cfg.get("categorical_overrides") or [])
    num_overrides = set(cfg.get("numeric_overrides") or [])
    fill_missing = bool(cfg.get("fill_missing", True))
    string_clean = bool(cfg.get("string_clean", True))
    round_multipliers = cfg.get("round_multipliers", []) or []
    digit_mods = cfg.get("digit_mods", []) or []
    pair_limit = int(cfg.get("pairwise_cat_limit", 0) or 0)
    te_cfg = cfg.get("target_encoding", {}) or {}

    processed = df.clone()

    cat_cols = _infer_categorical_columns(processed, meta, cat_overrides, num_overrides)
    num_cols = _numeric_columns(processed, cat_cols, meta)

    if fill_missing or string_clean:
        fill_exprs = []
        for col in cat_cols:
            expr = pl.col(col).cast(pl.Utf8)
            if fill_missing:
                expr = expr.fill_null(MISSING_TOKEN)
            if string_clean:
                expr = expr.str.strip_chars().str.to_lowercase()
            fill_exprs.append(expr.alias(col))
        for col in num_cols:
            expr = pl.col(col)
            if fill_missing:
                median_val = processed[col].median()
                expr = expr.fill_null(median_val)
            fill_exprs.append(expr.alias(col))
        if fill_exprs:
            processed = processed.with_columns(fill_exprs)

    processed = _add_binning_features(processed, num_cols, cfg)
    processed = _add_rounding_features(processed, num_cols, round_multipliers)
    processed = _add_digit_features(processed, num_cols, digit_mods)

    new_cat_cols = _infer_categorical_columns(processed, meta, cat_overrides, num_overrides)
    cat_cols = sorted(list(new_cat_cols))

    if pair_limit > 1 and len(cat_cols) > 1:
        processed = _add_pairwise_cats(processed, cat_cols, pair_limit)
        combo_cols = [c for c in processed.columns if "__" in c]
        cat_cols = sorted(list(set(cat_cols).union(combo_cols)))

    # Convert to pandas for target encoding
    pd_df = processed.to_pandas()

    te_enabled = bool(te_cfg.get("enabled", True)) and meta.get("target") in pd_df.columns
    if te_enabled:
        folds = int(te_cfg.get("folds", 5))
        smoothing = float(te_cfg.get("smoothing", 10.0))
        min_samples_leaf = int(te_cfg.get("min_samples_leaf", 20))
        seed = int(te_cfg.get("seed", 42))

        if is_train:
            pd_df, te_state = _fit_target_encoding_pandas(
                pd_df, meta.get("target"), cat_cols, folds, smoothing, min_samples_leaf, seed
            )
        elif te_state:
            pd_df = _apply_target_encoding_inference(pd_df, te_state)

    return pd_df, te_state, set(cat_cols), num_cols


def fit_transform(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    test_df: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame, Dict[str, Any]]:
    meta = _dataset_meta(config)
    cfg = {k: v for k, v in config.items() if k != "_dataset"}

    pl_train = pl.from_pandas(train_df)
    pl_val = pl.from_pandas(val_df) if val_df is not None else None
    pl_test = pl.from_pandas(test_df)

    pl_train = _drop_ignored(pl_train, meta)
    if pl_val is not None:
        pl_val = _drop_ignored(pl_val, meta)
    pl_test = _drop_ignored(pl_test, meta)

    train_processed, te_state, cat_cols, num_cols = _process_polars(pl_train, meta, cfg, None, True)
    val_processed = None
    if pl_val is not None:
        val_processed, _, _, _ = _process_polars(pl_val, meta, cfg, te_state, False)
    test_processed, _, _, _ = _process_polars(pl_test, meta, cfg, te_state, False)

    state = {
        "version": "1.0",
        "template": "fe_set1",
        "dataset": meta,
        "config": cfg,
        "cat_cols": sorted(list(cat_cols)),
        "num_cols": num_cols,
        "target_encoding": te_state,
    }
    return train_processed, val_processed, test_processed, state


def transform(df: pd.DataFrame, state_dict: Dict[str, Any], config: Dict[str, Any]) -> pd.DataFrame:
    if df is None:
        return df
    meta = state_dict.get("dataset") or _dataset_meta(config)
    cfg = state_dict.get("config") or {k: v for k, v in config.items() if k != "_dataset"}
    te_state = state_dict.get("target_encoding")

    pl_df = pl.from_pandas(df)
    pl_df = _drop_ignored(pl_df, meta)
    processed_pd, _, _, _ = _process_polars(pl_df, meta, cfg, te_state, False)
    return processed_pd
