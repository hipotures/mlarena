"""
AutoGluon baseline with aggressive feature engineering + CV-safe target encoding.

Implements:
- Binning (quantile, uniform, log-binning) for high-cardinality numeric vars
- Rounding/truncation buckets
- Digit-level artifacts (integer/frac parts, last/first digit, modulo)
- Lightweight categorical combos
- CV target encoding with smoothing (leak-safe via out-of-fold encodings)

State needed for inference (encoding maps) is cached per experiment_id so that
preprocess(train) → preprocess(val/test) within ml_runner share encoders.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import KFold

from kaggle_tools.config_models import ModelConfig

MISSING_TOKEN = "<MISSING>"
_TE_STATE: Dict[str, Dict[str, Any]] = {}


def get_default_config() -> Dict[str, Any]:
    return {
        "hyperparameters": {
            "presets": "medium",
            "time_limit": 600,
            "use_gpu": False,
        },
        "model": {
            "leaderboard_rows": 20,
        },
        "preprocess": {
            "enabled": True,
            "fill_missing": True,
            "string_clean": True,
            "quantile_bins": [20, 100],
            "uniform_bins": [20],
            "log1p_bins": [20],
            "round_multipliers": [1, 10, 100],
            "digit_mods": [3, 5, 7, 10],
            "pairwise_cat_limit": 8,  # combine up to N most frequent categorical cols
            "target_encoding": {
                "enabled": True,
                "folds": 5,
                "smoothing": 10.0,
                "min_samples_leaf": 20,
            },
            "categorical_overrides": [],
            "numeric_overrides": [],
        },
    }


def _drop_ignored(df: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    drop_cols = set(config.dataset.ignored_columns + [config.dataset.id_column])
    drop_cols.discard(config.dataset.target)
    return df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")


def _get_preprocess_cfg(config: ModelConfig) -> Dict[str, Any]:
    cfg = getattr(config, "preprocess", {}) or {}
    if hasattr(cfg, "model_dump"):
        cfg = cfg.model_dump(exclude_none=True)
    te_cfg = cfg.get("target_encoding", {}) or {}
    cfg["target_encoding"] = te_cfg
    return cfg


def _infer_categorical_columns(
    df: pd.DataFrame,
    config: ModelConfig,
    cat_overrides: set[str],
    num_overrides: set[str],
) -> set[str]:
    ignore_cols = set(config.dataset.ignored_columns + [config.dataset.id_column, config.dataset.target])
    cat_cols: set[str] = set(cat_overrides)
    for col in df.columns:
        if col in ignore_cols or col in num_overrides:
            continue
        series = df[col]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
            cat_cols.add(col)
            continue
        if pd.api.types.is_bool_dtype(series):
            cat_cols.add(col)
            continue
        if pd.api.types.is_integer_dtype(series) and series.nunique(dropna=True) <= 30:
            cat_cols.add(col)
    return cat_cols


def _numeric_columns(df: pd.DataFrame, cat_cols: set[str], config: ModelConfig) -> list[str]:
    ignore_cols = set(config.dataset.ignored_columns + [config.dataset.id_column, config.dataset.target])
    return [
        col
        for col in df.columns
        if col not in cat_cols
        and col not in ignore_cols
        and pd.api.types.is_numeric_dtype(df[col])
    ]


def _safe_bin(series: pd.Series, strategy: str, bins: int) -> pd.Series:
    try:
        if strategy == "quantile":
            binned = pd.qcut(series, q=min(bins, series.nunique()), duplicates="drop")
        elif strategy == "uniform":
            binned = pd.cut(series, bins=bins)
        else:
            return series
        return binned.astype(str)
    except Exception:
        return series.astype(str)


def _add_binning_features(df: pd.DataFrame, num_cols: list[str], cfg: Dict[str, Any]) -> pd.DataFrame:
    quantile_bins = cfg.get("quantile_bins", []) or []
    uniform_bins = cfg.get("uniform_bins", []) or []
    log1p_bins = cfg.get("log1p_bins", []) or []

    enriched = df.copy()
    for col in num_cols:
        col_series = enriched[col]
        for q in quantile_bins:
            enriched[f"{col}_q{q}"] = _safe_bin(col_series, "quantile", int(q))
        for b in uniform_bins:
            enriched[f"{col}_u{b}"] = _safe_bin(col_series, "uniform", int(b))
        for lb in log1p_bins:
            if (col_series <= 0).all():
                continue
            log_s = np.log1p(col_series.clip(lower=0))
            enriched[f"{col}_logq{lb}"] = _safe_bin(log_s, "quantile", int(lb))
    return enriched


def _add_rounding_features(df: pd.DataFrame, num_cols: list[str], multipliers: list[int]) -> pd.DataFrame:
    enriched = df.copy()
    for col in num_cols:
        series = enriched[col]
        for m in multipliers:
            if m <= 0:
                continue
            enriched[f"{col}_round_{m}"] = np.round(series / m) * m
            enriched[f"{col}_floor_{m}"] = np.floor(series / m) * m
    return enriched


def _add_digit_features(df: pd.DataFrame, num_cols: list[str], digit_mods: list[int]) -> pd.DataFrame:
    enriched = df.copy()
    for col in num_cols:
        series = enriched[col]
        int_part = np.floor(series)
        frac_part = np.abs(series - int_part)
        enriched[f"{col}_int_last_digit"] = (np.abs(int_part).astype(int) % 10).astype(int)
        enriched[f"{col}_int_first_digit"] = (np.abs(int_part) // 10 % 10).astype(int)
        enriched[f"{col}_frac_first_digit"] = np.floor(frac_part * 10).astype(int)
        for mod in digit_mods:
            if mod <= 0:
                continue
            enriched[f"{col}_mod_{mod}"] = (np.abs(int_part).astype(int) % mod).astype(int)
    return enriched


def _add_pairwise_cats(df: pd.DataFrame, cat_cols: list[str], limit: int) -> pd.DataFrame:
    enriched = df.copy()
    top_cats = cat_cols[:limit]
    for i in range(len(top_cats)):
        for j in range(i + 1, len(top_cats)):
            c1, c2 = top_cats[i], top_cats[j]
            enriched[f"{c1}__{c2}"] = enriched[c1].astype(str) + "|" + enriched[c2].astype(str)
    return enriched


def _fit_target_encoding(
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

    # Build full-data maps for inference
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


def preprocess(df: pd.DataFrame, config: ModelConfig, is_train: bool = True) -> pd.DataFrame:
    if df is None:
        return df

    cfg = _get_preprocess_cfg(config)
    if not cfg.get("enabled", True):
        return df

    cat_overrides = set(cfg.get("categorical_overrides") or [])
    num_overrides = set(cfg.get("numeric_overrides") or [])
    fill_missing = bool(cfg.get("fill_missing", True))
    string_clean = bool(cfg.get("string_clean", True))
    round_multipliers = cfg.get("round_multipliers", []) or []
    digit_mods = cfg.get("digit_mods", []) or []
    pair_limit = int(cfg.get("pairwise_cat_limit", 0) or 0)
    te_cfg = cfg.get("target_encoding", {}) or {}

    processed = df.copy()

    # Type inference
    cat_cols = _infer_categorical_columns(processed, config, cat_overrides, num_overrides)
    num_cols = _numeric_columns(processed, cat_cols, config)

    # Fill missing + string clean
    if fill_missing:
        for col in cat_cols:
            if col in processed.columns:
                processed[col] = processed[col].fillna(MISSING_TOKEN)
        for col in num_cols:
            if col in processed.columns:
                processed[col] = processed[col].fillna(processed[col].median())
    if string_clean:
        for col in cat_cols:
            if col in processed.columns and pd.api.types.is_string_dtype(processed[col]):
                processed[col] = processed[col].str.strip().str.lower()

    # Numeric transformations
    processed = _add_binning_features(processed, num_cols, cfg)
    processed = _add_rounding_features(processed, num_cols, round_multipliers)
    processed = _add_digit_features(processed, num_cols, digit_mods)

    # Refresh categorical set after new features
    new_cat_cols = _infer_categorical_columns(processed, config, cat_overrides, num_overrides)
    cat_cols = sorted(list(new_cat_cols))

    if pair_limit > 1 and len(cat_cols) > 1:
        processed = _add_pairwise_cats(processed, cat_cols, pair_limit)
        # Update cat cols with new combos
        combo_cols = [c for c in processed.columns if "__" in c]
        cat_cols = sorted(list(set(cat_cols).union(combo_cols)))

    # Target encoding (CV-safe, cached for inference)
    te_enabled = bool(te_cfg.get("enabled", True)) and config.dataset.target in processed.columns
    if te_enabled:
        folds = int(te_cfg.get("folds", 5))
        smoothing = float(te_cfg.get("smoothing", 10.0))
        min_samples_leaf = int(te_cfg.get("min_samples_leaf", 20))
        seed = getattr(config.system, "random_seed", 42)

        if is_train:
            encoded_df, te_state = _fit_target_encoding(
                processed, config.dataset.target, cat_cols, folds, smoothing, min_samples_leaf, seed
            )
            _TE_STATE[config.system.experiment_id] = te_state
            processed = encoded_df
        else:
            te_state = _TE_STATE.get(config.system.experiment_id)
            if te_state:
                processed = _apply_target_encoding_inference(processed, te_state)

    return processed


def train(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> Tuple[TabularPredictor, Dict[str, Any]]:
    features = _drop_ignored(train_df, config)
    train_data = features.copy()
    train_data[config.dataset.target] = train_df[config.dataset.target]
    tuning_data = None
    if val_df is not None:
        val_features = _drop_ignored(val_df, config)
        tuning_data = val_features.copy()
        tuning_data[config.dataset.target] = val_df[config.dataset.target]

    predictor = TabularPredictor(
        label=config.dataset.target,
        path=str(config.system.model_path),
        problem_type=config.dataset.problem_type,
        eval_metric=config.dataset.metric,
        verbosity=2,
    )

    fit_kwargs = {
        "presets": config.hyperparameters.presets,
        "time_limit": config.hyperparameters.time_limit,
        "num_gpus": 1 if config.hyperparameters.use_gpu else 0,
    }
    if config.hyperparameters.excluded_models:
        fit_kwargs["excluded_model_types"] = config.hyperparameters.excluded_models
    included_models = getattr(config.hyperparameters, "included_model_types", None)
    if included_models:
        fit_kwargs["included_model_types"] = included_models

    predictor.fit(
        train_data,
        tuning_data=tuning_data,
        **fit_kwargs,
    )

    leaderboard = predictor.leaderboard(train_data, silent=True)
    local_cv_score = None
    if not leaderboard.empty and "score_val" in leaderboard:
        scores = leaderboard["score_val"].dropna()
        if not scores.empty:
            local_cv_score = float(scores.max())
    summary = {"local_cv_score": local_cv_score}
    return predictor, summary


def predict(
    model: TabularPredictor,
    test_df: pd.DataFrame,
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> pd.DataFrame:
    features = _drop_ignored(test_df, config)
    submission = pd.DataFrame()
    submission[config.dataset.id_column] = test_df[config.dataset.id_column]

    if config.dataset.submission_probas:
        preds = model.predict_proba(features, as_multiclass=False)
        if isinstance(preds, pd.DataFrame):
            submission[config.dataset.target] = preds.iloc[:, 1]
        else:
            submission[config.dataset.target] = preds
    else:
        submission[config.dataset.target] = model.predict(features)
    return submission
