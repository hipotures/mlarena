"""
Rank Features Sub-Module

Purpose: Transform numeric features into ranks or percentiles.
Libraries: pandas, scipy.stats
Parameters:
  - numeric_include: List[str]
  - numeric_exclude: List[str]
  - group_keys: List[str] (optional, for grouped ranking)
  - mode: "global" | "by_group"
  - method: "rank" | "percentile" | "gauss_rank"
  - tie_method: "average" | "min" | "max" | "first" | "dense"
  - add_original: Bool (keep original cols)
  - fit_on_train: Bool (use train distribution for val/test)
"""

from pathlib import Path
from typing import Any, Dict, Tuple
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
) -> Tuple[
    pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None, Dict[str, Any]
]:
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
        "fit_on_train": False,
        "use_original_features_only": True,
    }
    validation.validate_config(config, required_params, optional_params)
    validation.validate_choice(config["mode"], ["global", "by_group"], "mode")
    validation.validate_choice(
        config["method"], ["rank", "percentile", "gauss_rank"], "method"
    )

    # 2. Submodule dir
    submodule_dir = artifacts.get_submodule_artifact_dir(artifact_dir, "rank_features")
    train_df_original = dataframe_utils.copy_dataframe(train_df)
    test_df_original = dataframe_utils.copy_dataframe(test_df)

    # 3. Determine columns
    exclude_cols = (
        [id_column, target_column] + ignored_columns + config["numeric_exclude"]
    )
    exclude_cols = [c for c in exclude_cols if c]
    all_numeric = dataframe_utils.get_numeric_columns(train_df, exclude=exclude_cols)
    use_orig_only = bool(config.get("use_original_features_only"))
    orig_features = config.get("_original_features") if use_orig_only else None

    if config["numeric_include"]:
        numeric_cols = [c for c in config["numeric_include"] if c in all_numeric]
    else:
        numeric_cols = all_numeric

    if use_orig_only:
        numeric_cols = dataframe_utils.filter_original_columns(
            numeric_cols, orig_features
        )

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

    # Suppress fragmentation warning
    warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

    fit_on_train = bool(config.get("fit_on_train"))
    if fit_on_train and config["mode"] == "by_group":
        warnings.warn(
            "fit_on_train is not supported with mode='by_group'. Falling back to per-dataset ranking."
        )
        fit_on_train = False

    train_sorted = {}
    if fit_on_train:
        for col in numeric_cols:
            non_null = train_df[col].dropna().values
            train_sorted[col] = np.sort(non_null) if non_null.size else np.array([])

    def _rank_from_train(series: pd.Series, sorted_vals: np.ndarray) -> np.ndarray:
        ranks = np.full(len(series), np.nan, dtype=np.float64)
        if sorted_vals.size == 0:
            return ranks
        mask = series.notna()
        if mask.any():
            ranks[mask.values] = (
                np.searchsorted(sorted_vals, series[mask].values, side="left") + 1
            )
        return ranks

    def _percentile_from_train(
        series: pd.Series, sorted_vals: np.ndarray
    ) -> np.ndarray:
        if sorted_vals.size == 0:
            return np.full(len(series), np.nan, dtype=np.float64)
        ranks = _rank_from_train(series, sorted_vals)
        return ranks / float(sorted_vals.size)

    def _gauss_rank_from_percentile(percentiles: np.ndarray) -> np.ndarray:
        try:
            from scipy.special import erfinv
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "gauss_rank requires scipy (scipy.special.erfinv)."
            ) from exc
        clipped = np.clip(percentiles, 1e-6, 1 - 1e-6)
        return np.sqrt(2.0) * erfinv(2.0 * clipped - 1.0)

    def process_df(df):
        if df is None:
            return None
        df_out = df.copy()

        # If by_group, we need group keys
        if config["mode"] == "by_group":
            if not config["group_keys"]:
                warnings.warn(
                    "mode='by_group' but no group_keys provided. Falling back to global."
                )
                mode = "global"
            else:
                mode = "by_group"
        else:
            mode = "global"

        cols_to_rank = numeric_cols

        for col in cols_to_rank:
            if config["method"] == "rank":
                new_col_name = f"{col}_rank"
            elif config["method"] == "percentile":
                new_col_name = f"{col}_pct"
            else:
                new_col_name = f"{col}_gauss"

            if mode == "global":
                if fit_on_train:
                    if config["method"] == "rank":
                        df_out[new_col_name] = _rank_from_train(
                            df[col], train_sorted[col]
                        )
                    elif config["method"] == "percentile":
                        df_out[new_col_name] = _percentile_from_train(
                            df[col], train_sorted[col]
                        )
                    else:
                        pct = _percentile_from_train(df[col], train_sorted[col])
                        df_out[new_col_name] = _gauss_rank_from_percentile(pct)
                else:
                    if config["method"] == "rank":
                        df_out[new_col_name] = df[col].rank(method=config["tie_method"])
                    elif config["method"] == "percentile":
                        df_out[new_col_name] = df[col].rank(
                            pct=True, method=config["tie_method"]
                        )
                    else:
                        pct = (
                            df[col]
                            .rank(pct=True, method=config["tie_method"])
                            .to_numpy()
                        )
                        df_out[new_col_name] = _gauss_rank_from_percentile(pct)
            else:
                # Grouped
                keys = config["group_keys"]
                if config["method"] == "rank":
                    df_out[new_col_name] = df.groupby(keys)[col].rank(
                        method=config["tie_method"]
                    )
                elif config["method"] == "percentile":
                    df_out[new_col_name] = df.groupby(keys)[col].rank(
                        pct=True, method=config["tie_method"]
                    )
                else:
                    pct = (
                        df.groupby(keys)[col]
                        .rank(pct=True, method=config["tie_method"])
                        .to_numpy()
                    )
                    df_out[new_col_name] = _gauss_rank_from_percentile(pct)

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
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
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
        "config": report_data["config"],
    }

    return train_df, val_df, test_df, orig_df, state_dict
