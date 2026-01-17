"""
S6E1_01 custom preprocessing pipeline.

Implements a Kaggle-style feature engineering flow:
  - optional categorical missing fill
  - target mean/count stats per feature
  - feature engineering (bins, crosses, numeric interactions, flags)
  - frequency encoding for derived categorical features
  - optional target outlier trimming and log1p transform
  - optional external prediction features (e.g., Ridge OOF/Test)
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd

from mlarena.preprocessing.utils import (
    validation,
    artifacts,
    dataframe_utils,
    report,
)


def _resolve_project_root(artifact_dir: Path) -> Path:
    artifact_dir = Path(artifact_dir).resolve()
    try:
        return artifact_dir.parent.parent.parent.parent.parent.parent
    except IndexError:
        return Path.cwd()


def _load_prediction_series(
    path_str: str,
    project_root: Path,
    expected_len: int,
    column: str | int | None = None,
) -> pd.Series:
    path = Path(path_str)
    if not path.is_absolute():
        path = project_root / path
    df = pd.read_csv(path, compression="infer")

    if column is None:
        series = df.iloc[:, 0]
    elif isinstance(column, int):
        series = df.iloc[:, column]
    else:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in {path}")
        series = df[column]

    if len(series) != expected_len:
        raise ValueError(
            f"Prediction length mismatch for {path}: expected {expected_len}, got {len(series)}"
        )
    return series.reset_index(drop=True)


def _fill_missing_cats(df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    if df is None or not cat_cols:
        return df
    df = df.copy()
    for col in cat_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(object).fillna("NaN")
    return df


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


def _combine_frames(
    frames: List[Tuple[str, pd.DataFrame | None]],
) -> Tuple[pd.DataFrame, List[Tuple[str, int]]]:
    parts: List[pd.DataFrame] = []
    sizes: List[Tuple[str, int]] = []
    for name, df in frames:
        if df is None:
            continue
        parts.append(df)
        sizes.append((name, len(df)))
    combined = pd.concat(parts, axis=0).reset_index(drop=True) if parts else pd.DataFrame()
    return combined, sizes


def _split_frames(
    combined: pd.DataFrame,
    sizes: List[Tuple[str, int]],
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    offset = 0
    for name, length in sizes:
        out[name] = combined.iloc[offset:offset + length].copy()
        offset += length
    return out


def _feature_engineering(
    df: pd.DataFrame,
    num_features: List[str],
    cat_features: List[str],
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df.copy()
    bin_count = int(config.get("bin_count", 5))
    highcard_threshold = int(config.get("highcard_threshold", 20))
    ratio_eps = float(config.get("ratio_eps", 1e-6))

    num_features = [c for c in num_features if c in df.columns]
    cat_features = [c for c in cat_features if c in df.columns]

    highcard = [c for c in num_features if df[c].nunique() > highcard_threshold]

    bin_features: List[str] = []
    for c in num_features:
        new_col = f"{c}_bin"
        df[new_col] = pd.cut(df[c], bins=bin_count, labels=False)
        df[new_col], _ = pd.factorize(df[new_col])
        df[new_col] = df[new_col].astype("category")
        bin_features.append(new_col)

    comb_features: List[str] = []
    if cat_features:
        str_df = df[cat_features].astype("string")
        for c1, c2 in combinations(str_df.columns, 2):
            comb_name = f"{c1}_{c2}_comb"
            df[comb_name], _ = pd.factorize(str_df[c1] + "_" + str_df[c2])
            df[comb_name] = df[comb_name].astype("category")
            comb_features.append(comb_name)

    numtocat_features: List[str] = []
    for c in num_features:
        new_col = f"{c}_cat"
        df[new_col], _ = df[c].factorize()
        df[new_col] = df[new_col].astype("category")
        numtocat_features.append(new_col)

    for c in cat_features:
        df[c], _ = df[c].factorize()

    for col in highcard:
        new_col_name = f"{col}_round"
        df[new_col_name] = df[col].round().astype(int).astype("category")
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
        df["efficiency"] = (df["study_hours"] * df["class_attendance"]) / (df["sleep_hours"] + 1)
        df["high_att_high_study"] = (
            (df["class_attendance"] >= 90) & (df["study_hours"] >= 6)
        ).astype(int)
        df["high_study_flag"] = (df["study_hours"] >= 7).astype(int)
        df["ideal_sleep_flag"] = (
            (df["sleep_hours"] >= 7) & (df["sleep_hours"] <= 9)
        ).astype(int)

    if _has_cols(["facility_rating", "sleep_quality", "exam_difficulty"]):
        df["facility_x_sleepq"] = df["facility_rating"] * df["sleep_quality"]
        df["difficulty_x_facility"] = df["exam_difficulty"] * df["facility_rating"]

    if _has_cols(["study_hours", "sleep_quality"]):
        df["study_hours_times_sleep_quality"] = df["study_hours"] * df["sleep_quality"]

    if _has_cols(["class_attendance", "facility_rating"]):
        df["attendance_times_facility"] = df["class_attendance"] * df["facility_rating"]

    if _has_cols(["sleep_hours", "exam_difficulty"]):
        df["sleep_hours_times_difficulty"] = df["sleep_hours"] * df["exam_difficulty"]

    if "study_hours" in df.columns:
        df["study_hours_sin"] = np.sin(2 * np.pi * df["study_hours"] / 12).astype("float32")
    if "class_attendance" in df.columns:
        df["class_attendance_sin"] = np.sin(2 * np.pi * df["class_attendance"] / 12).astype("float32")

    if cat_features:
        df[cat_features] = df[cat_features].astype("category")

    details = {
        "bin_features": bin_features,
        "comb_features": comb_features,
        "numtocat_features": numtocat_features,
        "highcard_features": highcard,
        "cat_features": cat_features,
        "num_features": num_features,
    }
    return df, details


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
    Custom preprocessing module implementing s6e1_01 feature engineering.
    """
    artifact_dir = Path(config.get("_artifact_dir", "."))
    dataset_config = config.get("_dataset", {})
    target_col = dataset_config.get("target")

    required_params: List[str] = []
    optional_params: Dict[str, Any] = {
        "missing": False,
        "outliers": False,
        "log_trf": False,
        "add_target_stats": True,
        "target_stats_source": "train",
        "stats_max_cardinality": None,
        "frequency_encoding": True,
        "frequency_min_count": 0,
        "bin_count": 5,
        "highcard_threshold": 20,
        "ratio_eps": 1e-6,
        "external_predictions": [],
        "ridge_oof_path": None,
        "ridge_test_path": None,
        "ridge_val_path": None,
        "ridge_eval_path": None,
        "ridge_orig_path": None,
        "ridge_feature_name": "Ridge_predict",
        "ridge_column": None,
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
    validation.validate_choice(
        config["target_stats_source"],
        ["train", "orig", "train_orig"],
        "target_stats_source",
    )

    submodule_dir = artifacts.get_submodule_artifact_dir(artifact_dir, "s6e1_01")

    train_df_original = dataframe_utils.copy_dataframe(train_df)
    test_df_original = dataframe_utils.copy_dataframe(test_df)

    train_target = train_df[target_col] if target_col and target_col in train_df.columns else None
    val_target = val_df[target_col] if val_df is not None and target_col in val_df.columns else None
    eval_target = eval_df[target_col] if eval_df is not None and target_col in eval_df.columns else None
    orig_target = orig_df[target_col] if orig_df is not None and target_col in orig_df.columns else None

    X_train = train_df.drop(columns=[target_col]) if train_target is not None else train_df.copy()
    X_val = val_df.drop(columns=[target_col]) if val_df is not None and val_target is not None else val_df.copy() if val_df is not None else None
    X_test = test_df.copy()
    X_eval = eval_df.drop(columns=[target_col]) if eval_df is not None and eval_target is not None else eval_df.copy() if eval_df is not None else None
    X_orig = orig_df.drop(columns=[target_col]) if orig_df is not None and orig_target is not None else orig_df.copy() if orig_df is not None else None

    num_features = X_train.select_dtypes(exclude=["object", "bool", "category"]).columns.tolist()
    cat_features = X_train.select_dtypes(include=["object", "bool", "category"]).columns.tolist()

    if config["missing"]:
        X_train = _fill_missing_cats(X_train, cat_features)
        X_test = _fill_missing_cats(X_test, cat_features)
        X_val = _fill_missing_cats(X_val, cat_features)
        X_eval = _fill_missing_cats(X_eval, cat_features)
        X_orig = _fill_missing_cats(X_orig, cat_features)

    external_predictions = list(config.get("external_predictions") or [])
    if config.get("ridge_oof_path") or config.get("ridge_test_path"):
        external_predictions.append({
            "train_path": config.get("ridge_oof_path"),
            "test_path": config.get("ridge_test_path"),
            "val_path": config.get("ridge_val_path"),
            "eval_path": config.get("ridge_eval_path"),
            "orig_path": config.get("ridge_orig_path"),
            "feature_name": config.get("ridge_feature_name", "Ridge_predict"),
            "column": config.get("ridge_column"),
        })

    if external_predictions:
        project_root = _resolve_project_root(artifact_dir)
        for spec in external_predictions:
            feature_name = spec.get("feature_name")
            train_path = spec.get("train_path")
            test_path = spec.get("test_path")
            if not feature_name or not train_path or not test_path:
                warnings.warn("Skipping external_predictions entry with missing feature_name/train_path/test_path")
                continue
            col = spec.get("column")
            X_train[feature_name] = _load_prediction_series(
                train_path, project_root, len(X_train), column=col
            ).values
            X_test[feature_name] = _load_prediction_series(
                test_path, project_root, len(X_test), column=col
            ).values

            for df_name, df_ref, path_key in [
                ("val", X_val, "val_path"),
                ("eval", X_eval, "eval_path"),
                ("orig", X_orig, "orig_path"),
            ]:
                p = spec.get(path_key)
                if df_ref is None:
                    continue
                if p:
                    df_ref[feature_name] = _load_prediction_series(
                        p, project_root, len(df_ref), column=col
                    ).values
                else:
                    df_ref[feature_name] = np.nan

    target_stats_used: Dict[str, pd.DataFrame] = {}
    global_stats: Dict[str, float] = {}
    skipped_stats_cols: List[str] = []
    if config["add_target_stats"]:
        if not target_col or target_col not in train_df.columns:
            warnings.warn("Target column missing; skipping target stats.")
        else:
            if config["target_stats_source"] == "orig" and orig_df is not None and target_col in orig_df.columns:
                stats_source = orig_df
            elif config["target_stats_source"] == "train_orig" and orig_df is not None and target_col in orig_df.columns:
                stats_source = pd.concat([train_df, orig_df], axis=0)
            else:
                stats_source = train_df
            target_stats_used, global_stats, skipped_stats_cols = _build_target_stats(
                stats_source,
                target_col,
                num_features + cat_features,
                config["stats_max_cardinality"],
            )
            X_train = _apply_target_stats(X_train, target_stats_used, global_stats)
            X_val = _apply_target_stats(X_val, target_stats_used, global_stats)
            X_test = _apply_target_stats(X_test, target_stats_used, global_stats)
            X_eval = _apply_target_stats(X_eval, target_stats_used, global_stats)
            X_orig = _apply_target_stats(X_orig, target_stats_used, global_stats)

    combined, sizes = _combine_frames([
        ("train", X_train),
        ("val", X_val),
        ("test", X_test),
        ("eval", X_eval),
        ("orig", X_orig),
    ])

    if not combined.empty:
        combined, fe_details = _feature_engineering(combined, num_features, cat_features, config)
    else:
        fe_details = {
            "bin_features": [],
            "comb_features": [],
            "numtocat_features": [],
            "highcard_features": [],
            "cat_features": cat_features,
            "num_features": num_features,
        }

    split = _split_frames(combined, sizes)
    X_train = split.get("train", X_train)
    X_val = split.get("val", X_val)
    X_test = split.get("test", X_test)
    X_eval = split.get("eval", X_eval)
    X_orig = split.get("orig", X_orig)

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
        "version": "1.0",
        "module": "s6e1_01",
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
