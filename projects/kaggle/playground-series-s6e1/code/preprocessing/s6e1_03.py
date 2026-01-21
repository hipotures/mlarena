"""
S6E1_03 custom preprocessing pipeline.

Differences vs s6e1_01:
  - target stats computed OOF on CV folds
  - external prediction features removed
  - feature engineering fit on train only (applied to other splits via train mappings)
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from mlarena.preprocessing.utils import (
    validation,
    artifacts,
    dataframe_utils,
    report,
)


def _fill_missing_cats(df: pd.DataFrame | None, cat_cols: List[str]) -> pd.DataFrame | None:
    if df is None or not cat_cols:
        return df
    df = df.copy()
    for col in cat_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(object).fillna("NaN")
    return df


def _factorize_with_mapping(values: pd.Series) -> Tuple[pd.Series, Dict[Any, int]]:
    codes, uniques = pd.factorize(values)
    mapping = {val: int(idx) for idx, val in enumerate(uniques)}
    return pd.Series(codes, index=values.index), mapping


def _apply_mapping(values: pd.Series, mapping: Dict[Any, int]) -> pd.Series:
    return values.map(mapping)


def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series | None:
    if col not in df.columns:
        return None
    series = df[col]
    if pd.api.types.is_categorical_dtype(series):
        codes = series.cat.codes.replace(-1, np.nan)
        return codes.astype(float)
    return pd.to_numeric(series, errors="coerce")


def _build_target_stats(
    stats_df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    max_cardinality: int | None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, float], List[str]]:
    stats: Dict[str, pd.DataFrame] = {}
    skipped: List[str] = []

    for col in feature_cols:
        if col not in stats_df.columns:
            skipped.append(col)
            continue
        if max_cardinality is not None:
            try:
                if int(stats_df[col].nunique()) > int(max_cardinality):
                    skipped.append(col)
                    continue
            except Exception:
                pass
        stats[col] = stats_df.groupby(col)[target_col].agg(["mean", "count"])

    global_stats = {
        "mean": float(stats_df[target_col].mean()),
        "count": 0.0,
    }
    return stats, global_stats, skipped


def _apply_target_stats(
    df: pd.DataFrame | None,
    stats: Dict[str, pd.DataFrame],
    global_stats: Dict[str, float],
) -> pd.DataFrame | None:
    if df is None:
        return None
    df = df.copy()
    for col, agg in stats.items():
        if col not in df.columns:
            continue
        for agg_name in ["mean", "count"]:
            new_col = f"{col}_org_{agg_name}"
            if new_col in df.columns:
                continue
            tmp = agg[agg_name].rename(new_col).reset_index()
            df = df.merge(tmp, on=col, how="left")
            df[new_col] = df[new_col].fillna(global_stats[agg_name])
    return df


def _apply_oof_target_stats(
    X_train: pd.DataFrame,
    train_df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    n_splits: int,
    random_state: int,
    shuffle: bool,
) -> pd.DataFrame:
    out = X_train.copy()
    for col in feature_cols:
        for agg_name in ["mean", "count"]:
            new_col = f"{col}_org_{agg_name}"
            if new_col not in out.columns:
                out[new_col] = np.nan

    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    for train_idx, val_idx in kf.split(train_df):
        fold_df = train_df.iloc[train_idx]
        stats, global_stats, _ = _build_target_stats(
            fold_df,
            target_col,
            feature_cols,
            None,
        )
        for col, agg in stats.items():
            if col not in out.columns:
                continue
            for agg_name in ["mean", "count"]:
                new_col = f"{col}_org_{agg_name}"
                values = out.loc[val_idx, col].map(agg[agg_name])
                values = values.fillna(global_stats[agg_name])
                out.loc[val_idx, new_col] = values.values

    return out


def _feature_engineering(
    df: pd.DataFrame,
    num_features: List[str],
    cat_features: List[str],
    config: Dict[str, Any],
    state: Dict[str, Any] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    df = df.copy()
    bin_count = int(config.get("bin_count", 5))
    highcard_threshold = int(config.get("highcard_threshold", 20))
    ratio_eps = float(config.get("ratio_eps", 1e-6))

    num_features = [c for c in num_features if c in df.columns]
    cat_features = [c for c in cat_features if c in df.columns]

    fit_mode = state is None
    if fit_mode:
        highcard = [c for c in num_features if df[c].nunique() > highcard_threshold]
        state = {
            "highcard_features": highcard,
            "bin_edges": {},
            "category_mappings": {},
            "num_features": num_features,
            "cat_features": cat_features,
        }
    else:
        highcard = state.get("highcard_features", [])

    category_mappings = state.setdefault("category_mappings", {})
    bin_edges = state.setdefault("bin_edges", {})

    bin_features: List[str] = []
    for c in num_features:
        new_col = f"{c}_bin"
        if fit_mode:
            bins, edges = pd.cut(df[c], bins=bin_count, labels=False, retbins=True, duplicates="drop")
            bin_edges[c] = edges
            codes, mapping = _factorize_with_mapping(pd.Series(bins, index=df.index))
            category_mappings[new_col] = mapping
        else:
            edges = bin_edges.get(c)
            if edges is None:
                continue
            bins = pd.cut(df[c], bins=edges, labels=False, include_lowest=True)
            mapping = category_mappings.get(new_col, {})
            codes = _apply_mapping(pd.Series(bins, index=df.index), mapping)
        df[new_col] = pd.Series(codes, index=df.index).astype("category")
        bin_features.append(new_col)

    comb_features: List[str] = []
    if cat_features:
        str_df = df[cat_features].astype("string")
        for c1, c2 in combinations(str_df.columns, 2):
            comb_name = f"{c1}_{c2}_comb"
            raw = str_df[c1] + "_" + str_df[c2]
            if fit_mode:
                codes, mapping = _factorize_with_mapping(raw)
                category_mappings[comb_name] = mapping
            else:
                mapping = category_mappings.get(comb_name, {})
                codes = _apply_mapping(raw, mapping)
            df[comb_name] = pd.Series(codes, index=df.index).astype("category")
            comb_features.append(comb_name)

    numtocat_features: List[str] = []
    for c in num_features:
        new_col = f"{c}_cat"
        raw = df[c]
        if fit_mode:
            codes, mapping = _factorize_with_mapping(raw)
            category_mappings[new_col] = mapping
        else:
            mapping = category_mappings.get(new_col, {})
            codes = _apply_mapping(raw, mapping)
        df[new_col] = pd.Series(codes, index=df.index).astype("category")
        numtocat_features.append(new_col)

    for c in cat_features:
        raw = df[c]
        if fit_mode:
            codes, mapping = _factorize_with_mapping(raw)
            category_mappings[c] = mapping
        else:
            mapping = category_mappings.get(c, {})
            codes = _apply_mapping(raw, mapping)
        df[c] = pd.Series(codes, index=df.index).astype("category")

    for col in highcard:
        new_col_name = f"{col}_round"
        raw = df[col].round()
        if fit_mode:
            codes, mapping = _factorize_with_mapping(raw)
            category_mappings[new_col_name] = mapping
        else:
            mapping = category_mappings.get(new_col_name, {})
            codes = _apply_mapping(raw, mapping)
        df[new_col_name] = pd.Series(codes, index=df.index).astype("category")
        numtocat_features.append(new_col_name)

    for c in highcard:
        df[f"Log_{c}"] = np.log1p(df[c])
        df[f"{c}_sq"] = df[c] ** 2
        df[f"{c}_sqrt"] = df[c] ** 0.5

    for c1, c2 in combinations(num_features, 2):
        df[f"{c1}_{c2}_ratio"] = df[c1] / (df[c2] + ratio_eps)
        df[f"{c1}_*_{c2}"] = df[c1] * df[c2]

    def _has_cols(cols: List[str]) -> bool:
        return all(c in df.columns for c in cols)

    if _has_cols(["study_hours", "class_attendance", "sleep_hours"]):
        study_hours = _numeric_series(df, "study_hours")
        class_att = _numeric_series(df, "class_attendance")
        sleep_hours = _numeric_series(df, "sleep_hours")
        if study_hours is not None and class_att is not None and sleep_hours is not None:
            df["efficiency"] = (study_hours * class_att) / (sleep_hours + 1)
            df["high_att_high_study"] = (
                (class_att >= 90) & (study_hours >= 6)
            ).astype(int)
            df["high_study_flag"] = (study_hours >= 7).astype(int)
            df["ideal_sleep_flag"] = (
                (sleep_hours >= 7) & (sleep_hours <= 9)
            ).astype(int)

    if _has_cols(["facility_rating", "sleep_quality", "exam_difficulty"]):
        facility = _numeric_series(df, "facility_rating")
        sleepq = _numeric_series(df, "sleep_quality")
        difficulty = _numeric_series(df, "exam_difficulty")
        if facility is not None and sleepq is not None:
            df["facility_x_sleepq"] = facility * sleepq
        if difficulty is not None and facility is not None:
            df["difficulty_x_facility"] = difficulty * facility

    if _has_cols(["study_hours", "sleep_quality"]):
        study_hours = _numeric_series(df, "study_hours")
        sleepq = _numeric_series(df, "sleep_quality")
        if study_hours is not None and sleepq is not None:
            df["study_hours_times_sleep_quality"] = study_hours * sleepq

    if _has_cols(["class_attendance", "facility_rating"]):
        class_att = _numeric_series(df, "class_attendance")
        facility = _numeric_series(df, "facility_rating")
        if class_att is not None and facility is not None:
            df["attendance_times_facility"] = class_att * facility

    if _has_cols(["sleep_hours", "exam_difficulty"]):
        sleep_hours = _numeric_series(df, "sleep_hours")
        difficulty = _numeric_series(df, "exam_difficulty")
        if sleep_hours is not None and difficulty is not None:
            df["sleep_hours_times_difficulty"] = sleep_hours * difficulty

    if "study_hours" in df.columns:
        study_hours = _numeric_series(df, "study_hours")
        if study_hours is not None:
            df["study_hours_sin"] = np.sin(2 * np.pi * study_hours / 12).astype("float32")
    if "class_attendance" in df.columns:
        class_att = _numeric_series(df, "class_attendance")
        if class_att is not None:
            df["class_attendance_sin"] = np.sin(2 * np.pi * class_att / 12).astype("float32")

    if cat_features:
        df[cat_features] = df[cat_features].astype("category")

    if fit_mode:
        state["bin_features"] = bin_features
        state["comb_features"] = comb_features
        state["numtocat_features"] = numtocat_features
        state["highcard_features"] = highcard

    details = {
        "bin_features": state.get("bin_features", bin_features),
        "comb_features": state.get("comb_features", comb_features),
        "numtocat_features": state.get("numtocat_features", numtocat_features),
        "highcard_features": state.get("highcard_features", highcard),
        "cat_features": state.get("cat_features", cat_features),
        "num_features": state.get("num_features", num_features),
    }
    return df, details, state


def _fit_frequency_encoding(
    df: pd.DataFrame,
    features: List[str],
    min_count: int,
) -> Dict[str, Dict[Any, float]]:
    mappings: Dict[str, Dict[Any, float]] = {}
    for c in features:
        if c not in df.columns:
            continue
        counts = df[c].value_counts(normalize=True)
        if min_count and min_count > 1:
            raw_counts = df[c].value_counts()
            keep = raw_counts[raw_counts >= min_count].index
            counts = counts[counts.index.isin(keep)]
        mappings[c] = counts.to_dict()
    return mappings


def _apply_frequency_encoding(
    df: pd.DataFrame | None,
    features: List[str],
    mappings: Dict[str, Dict[Any, float]],
) -> pd.DataFrame | None:
    if df is None:
        return None
    df = df.copy()
    for c in features:
        if c not in df.columns:
            continue
        mapping = mappings.get(c, {})
        df[f"{c}_fe"] = df[c].map(mapping).astype(float).fillna(0)
    return df


def fit_transform(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
    orig_df: pd.DataFrame | None = None,
    eval_df: pd.DataFrame | None = None,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame | None,
    pd.DataFrame,
    pd.DataFrame | None,
    pd.DataFrame | None,
    Dict[str, Any],
]:
    """
    Custom preprocessing module implementing s6e1_03 feature engineering.
    """
    artifact_dir = Path(config.get("_artifact_dir", "."))
    dataset_config = config.get("_dataset", {})
    target_col = dataset_config.get("target")

    required_params: List[str] = []
    optional_params: Dict[str, Any] = {
        "missing": False,
        "outliers": False,
        "log_trf": False,
        "oof_n_splits": 5,
        "oof_shuffle": True,
        "oof_random_state": 42,
        "stats_max_cardinality": None,
        "frequency_encoding": True,
        "frequency_min_count": 0,
        "bin_count": 5,
        "highcard_threshold": 20,
        "ratio_eps": 1e-6,
        "apply_outliers_to_val": False,
        "apply_outliers_to_orig": False,
        "log_transform_val": True,
        "log_transform_eval": True,
        "log_transform_orig": True,
        "quiet": False,
        "save_frequency_mappings": False,
        "save_target_stats": False,
    }
    validation.validate_config(config, required_params, optional_params)

    submodule_dir = artifacts.get_submodule_artifact_dir(artifact_dir, "s6e1_03")

    train_df_original = dataframe_utils.copy_dataframe(train_df)
    test_df_original = dataframe_utils.copy_dataframe(test_df)

    train_target = train_df[target_col] if target_col and target_col in train_df.columns else None
    val_target = val_df[target_col] if val_df is not None and target_col in val_df.columns else None
    eval_target = eval_df[target_col] if eval_df is not None and target_col in eval_df.columns else None
    orig_target = orig_df[target_col] if orig_df is not None and target_col in orig_df.columns else None

    X_train = train_df.drop(columns=[target_col]) if train_target is not None else train_df.copy()
    X_val = (
        val_df.drop(columns=[target_col])
        if val_df is not None and val_target is not None
        else val_df.copy()
        if val_df is not None
        else None
    )
    X_test = test_df.copy()
    X_eval = (
        eval_df.drop(columns=[target_col])
        if eval_df is not None and eval_target is not None
        else eval_df.copy()
        if eval_df is not None
        else None
    )
    X_orig = (
        orig_df.drop(columns=[target_col])
        if orig_df is not None and orig_target is not None
        else orig_df.copy()
        if orig_df is not None
        else None
    )

    num_features = X_train.select_dtypes(exclude=["object", "bool", "category"]).columns.tolist()
    cat_features = X_train.select_dtypes(include=["object", "bool", "category"]).columns.tolist()

    if config["missing"]:
        X_train = _fill_missing_cats(X_train, cat_features)
        X_test = _fill_missing_cats(X_test, cat_features)
        X_val = _fill_missing_cats(X_val, cat_features)
        X_eval = _fill_missing_cats(X_eval, cat_features)
        X_orig = _fill_missing_cats(X_orig, cat_features)

    target_stats_used: Dict[str, pd.DataFrame] = {}
    global_stats: Dict[str, float] = {}
    skipped_stats_cols: List[str] = []

    if target_col and target_col in train_df.columns:
        target_stats_used, global_stats, skipped_stats_cols = _build_target_stats(
            train_df,
            target_col,
            num_features + cat_features,
            config["stats_max_cardinality"],
        )
        stats_features = list(target_stats_used.keys())
        n_splits = int(config.get("oof_n_splits", 5))
        shuffle = bool(config.get("oof_shuffle", True))
        random_state = int(config.get("oof_random_state", 42))

        if stats_features and n_splits >= 2 and len(train_df) >= n_splits:
            X_train = _apply_oof_target_stats(
                X_train,
                train_df,
                target_col,
                stats_features,
                n_splits,
                random_state,
                shuffle,
            )
            X_val = _apply_target_stats(X_val, target_stats_used, global_stats)
            X_test = _apply_target_stats(X_test, target_stats_used, global_stats)
            X_eval = _apply_target_stats(X_eval, target_stats_used, global_stats)
            X_orig = _apply_target_stats(X_orig, target_stats_used, global_stats)
        else:
            warnings.warn("Skipping OOF target stats due to insufficient data or folds")
            target_stats_used = {}
            global_stats = {}
            skipped_stats_cols = []
    else:
        warnings.warn("Target column missing; skipping OOF target stats")

    X_train, fe_details, fe_state = _feature_engineering(
        X_train, num_features, cat_features, config, state=None
    )
    if X_val is not None:
        X_val, _, _ = _feature_engineering(X_val, num_features, cat_features, config, state=fe_state)
    if X_test is not None:
        X_test, _, _ = _feature_engineering(X_test, num_features, cat_features, config, state=fe_state)
    if X_eval is not None:
        X_eval, _, _ = _feature_engineering(X_eval, num_features, cat_features, config, state=fe_state)
    if X_orig is not None:
        X_orig, _, _ = _feature_engineering(X_orig, num_features, cat_features, config, state=fe_state)

    freq_features = (
        fe_details["numtocat_features"]
        + fe_details["comb_features"]
        + fe_details["cat_features"]
        + fe_details["bin_features"]
    )
    freq_features = [c for c in freq_features if c in X_train.columns] if freq_features else []

    freq_mappings: Dict[str, Dict[Any, float]] = {}
    if config["frequency_encoding"] and freq_features:
        freq_mappings = _fit_frequency_encoding(
            X_train,
            freq_features,
            int(config.get("frequency_min_count", 0)),
        )
        X_train = _apply_frequency_encoding(X_train, freq_features, freq_mappings)
        X_val = _apply_frequency_encoding(X_val, freq_features, freq_mappings)
        X_test = _apply_frequency_encoding(X_test, freq_features, freq_mappings)
        X_eval = _apply_frequency_encoding(X_eval, freq_features, freq_mappings)
        X_orig = _apply_frequency_encoding(X_orig, freq_features, freq_mappings)

    if config["save_frequency_mappings"] and freq_mappings:
        artifacts.save_report(
            {"frequency_mappings": freq_mappings},
            submodule_dir,
            "frequency_mappings.json",
        )

    if config["save_target_stats"] and target_stats_used:
        stats_payload = {
            "global_stats": global_stats,
            "features": list(target_stats_used.keys()),
            "skipped": skipped_stats_cols,
        }
        artifacts.save_report(stats_payload, submodule_dir, "target_stats_summary.json")

    if config["outliers"] and train_target is not None:
        q1 = train_target.quantile(0.25)
        q3 = train_target.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (train_target >= lower) & (train_target <= upper)
        X_train = X_train[mask].reset_index(drop=True)
        train_target = train_target[mask].reset_index(drop=True)

        if config["apply_outliers_to_val"] and val_target is not None and X_val is not None:
            val_mask = (val_target >= lower) & (val_target <= upper)
            X_val = X_val[val_mask].reset_index(drop=True)
            val_target = val_target[val_mask].reset_index(drop=True)

        if config["apply_outliers_to_orig"] and orig_target is not None and X_orig is not None:
            orig_mask = (orig_target >= lower) & (orig_target <= upper)
            X_orig = X_orig[orig_mask].reset_index(drop=True)
            orig_target = orig_target[orig_mask].reset_index(drop=True)

    if config["log_trf"] and train_target is not None:
        train_target = np.log1p(train_target)
        if config["log_transform_val"] and val_target is not None:
            val_target = np.log1p(val_target)
        if config["log_transform_eval"] and eval_target is not None:
            eval_target = np.log1p(eval_target)
        if config["log_transform_orig"] and orig_target is not None:
            orig_target = np.log1p(orig_target)

    new_train = X_train.copy()
    if train_target is not None:
        new_train[target_col] = train_target.values

    new_val = X_val.copy() if X_val is not None else None
    if new_val is not None and val_target is not None:
        new_val[target_col] = val_target.values

    new_eval = X_eval.copy() if X_eval is not None else None
    if new_eval is not None and eval_target is not None:
        new_eval[target_col] = eval_target.values

    new_orig = X_orig.copy() if X_orig is not None else None
    if new_orig is not None and orig_target is not None:
        new_orig[target_col] = orig_target.values

    summary = report.create_preprocessing_report(
        train_before=train_df_original,
        train_after=new_train,
        test_before=test_df_original,
        test_after=X_test,
        config=config,
    )
    artifacts.save_report(summary, submodule_dir, "summary.json")

    cat_after = X_test.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
    num_after = X_test.select_dtypes(exclude=["object", "bool", "category"]).columns.tolist()

    state_dict = {
        "version": "3.0",
        "module": "s6e1_03",
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
        "num_features_initial": num_features,
        "cat_features_initial": cat_features,
        "num_features_after": num_after,
        "cat_features_after": cat_after,
        "feature_engineering": fe_details,
        "frequency_encoded_features": freq_features,
        "target_stats_features": list(target_stats_used.keys()),
        "target_stats_skipped": skipped_stats_cols,
    }

    return new_train, new_val, X_test, new_eval, new_orig, state_dict
