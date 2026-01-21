"""
Groupwise Normalizer Sub-Module

Purpose: Normalize numeric features relative to groups (e.g., price relative to category average).
Libraries: pandas
Parameters:
  - group_keys: List[str]
  - value_cols: List[str]
  - add_group_mean: Bool (add the mean itself as feature)
  - add_centered: Bool (value - mean)
  - add_zscore: Bool ((value - center) / spread)
  - add_ratio: Bool (value / center)
  - reference_stat: mean|median|min|max|quantile
  - quantile_value: float (used when reference_stat=quantile)
  - zscore_method: std|mad
  - mad_scale: float (scale factor for MAD)
  - eps: Float (epsilon for division)
"""

from pathlib import Path
from typing import Any, Dict, Tuple
import warnings

import pandas as pd

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
    dataset_config.get("id_column", "id")
    dataset_config.get("target")
    dataset_config.get("ignored_columns", [])

    required_params = ["group_keys", "value_cols"]
    optional_params = {
        "add_group_mean": True,
        "add_centered": True,
        "add_zscore": True,
        "add_ratio": False,
        "eps": 1e-6,
        "reference_stat": "mean",
        "quantile_value": 0.5,
        "zscore_method": "std",
        "mad_scale": 1.4826,
        "use_original_features_only": True,
    }
    validation.validate_config(config, required_params, optional_params)
    validation.validate_choice(
        config["reference_stat"],
        ["mean", "median", "min", "max", "quantile"],
        "reference_stat",
    )
    validation.validate_choice(
        config["zscore_method"],
        ["std", "mad"],
        "zscore_method",
    )
    if config["reference_stat"] == "quantile":
        validation.validate_numeric_range(
            config["quantile_value"],
            min_value=0.0,
            max_value=1.0,
            param_name="quantile_value",
        )

    submodule_dir = artifacts.get_submodule_artifact_dir(
        artifact_dir, "groupwise_normalizer"
    )
    train_df_original = dataframe_utils.copy_dataframe(train_df)
    test_df_original = dataframe_utils.copy_dataframe(test_df)

    # 3. Compute Group Stats on Train
    keys = config["group_keys"]
    values = config["value_cols"]
    if config.get("use_original_features_only"):
        orig_features = config.get("_original_features")
        if orig_features:
            orig_set = set(orig_features)
            keys = [c for c in keys if c in orig_set]
            values = [c for c in values if c in orig_set]
    if isinstance(keys, str):
        keys = [keys]
    if isinstance(values, str):
        values = [values]
    eps = config["eps"]
    reference_stat = config["reference_stat"]
    quantile_value = config["quantile_value"]
    zscore_method = config["zscore_method"]
    mad_scale = config["mad_scale"]

    if not keys or not values:
        warnings.warn(
            "groupwise_normalizer requires non-empty group_keys and value_cols. Skipping."
        )
        state_dict = {
            "version": "1.0",
            "new_features": [],
            "config": {k: v for k, v in config.items() if not k.startswith("_")},
            "message": "group_keys/value_cols empty",
        }
        return train_df, val_df, test_df, orig_df, state_dict

    # Validate columns
    missing = [c for c in keys + values if c not in train_df.columns]
    if missing:
        return train_df, val_df, test_df, orig_df, {"error": f"Missing cols: {missing}"}

    # Groupby
    grouped = train_df.groupby(keys)[values]

    if reference_stat == "mean":
        center = grouped.mean()
    elif reference_stat == "median":
        center = grouped.median()
    elif reference_stat == "min":
        center = grouped.min()
    elif reference_stat == "max":
        center = grouped.max()
    else:
        center = grouped.quantile(quantile_value)

    if zscore_method == "std":
        spread = grouped.std()
    else:
        spread = grouped.apply(lambda df: (df - df.median()).abs().median())
        spread = spread * mad_scale

    new_features = []

    center_cols = {col: f"{col}_center" for col in center.columns}
    spread_cols = {col: f"{col}_spread" for col in spread.columns}
    center = center.rename(columns=center_cols)
    spread = spread.rename(columns=spread_cols)
    stats = center.join(spread).reset_index()

    if reference_stat == "quantile":
        stat_suffix = f"q{str(quantile_value).replace('.', '_')}"
    else:
        stat_suffix = reference_stat
    zscore_suffix = "std" if zscore_method == "std" else "mad"

    def process_df(df):
        if df is None:
            return None
        df_out = df.copy()

        # Merge stats
        temp = df[keys].merge(stats, on=keys, how="left")
        # temp has same index as df IF we preserve it?
        # merge resets index if not careful or if relations are 1:N?
        # keys are not unique in df, but unique in stats. M:1 merge.
        # merge preserves order of left key? Not guaranteed?
        # Safer: set index, merge, restore.

        # Actually map is safer and faster for single key. For multi-key, merge is needed.
        # Let's use left join on index.
        temp.index = df.index

        for v in values:
            center_col = f"{v}_center"
            spread_col = f"{v}_spread"

            # If stats missing (unseen group), fillna?
            # Global fallback?
            if temp[center_col].isnull().any() or temp[spread_col].isnull().any():
                if reference_stat == "mean":
                    g_center = train_df[v].mean()
                elif reference_stat == "median":
                    g_center = train_df[v].median()
                elif reference_stat == "min":
                    g_center = train_df[v].min()
                elif reference_stat == "max":
                    g_center = train_df[v].max()
                else:
                    g_center = train_df[v].quantile(quantile_value)

                if zscore_method == "std":
                    g_spread = train_df[v].std()
                else:
                    g_spread = (
                        train_df[v] - train_df[v].median()
                    ).abs().median() * mad_scale

                temp[center_col] = temp[center_col].fillna(g_center)
                temp[spread_col] = temp[spread_col].fillna(g_spread)

            if config["add_group_mean"]:
                if reference_stat == "mean":
                    name = f"{v}_grp_mean"
                else:
                    name = f"{v}_grp_{stat_suffix}"
                df_out[name] = temp[center_col]
                if name not in new_features:
                    new_features.append(name)

            if config["add_centered"]:
                if reference_stat == "mean":
                    name = f"{v}_centered"
                else:
                    name = f"{v}_centered_{stat_suffix}"
                df_out[name] = df[v] - temp[center_col]
                if name not in new_features:
                    new_features.append(name)

            if config["add_zscore"]:
                if reference_stat == "mean" and zscore_method == "std":
                    name = f"{v}_grp_zscore"
                else:
                    name = f"{v}_grp_zscore_{stat_suffix}_{zscore_suffix}"
                # Avoid div by zero
                sigma = temp[spread_col].fillna(0)
                sigma = sigma.replace(0, eps)  # if spread is 0 (constant group)
                df_out[name] = (df[v] - temp[center_col]) / sigma
                if name not in new_features:
                    new_features.append(name)

            if config["add_ratio"]:
                if reference_stat == "mean":
                    name = f"{v}_grp_ratio"
                else:
                    name = f"{v}_grp_ratio_{stat_suffix}"
                mu = temp[center_col].replace(0, eps)
                df_out[name] = df[v] / mu
                if name not in new_features:
                    new_features.append(name)

        return df_out

    # Apply
    # Note: process_df does the merge inside.

    train_df = process_df(train_df)
    test_df = process_df(test_df)
    val_df = process_df(val_df)
    orig_df = process_df(orig_df)

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
        "new_features": new_features,
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
    }

    return train_df, val_df, test_df, orig_df, state_dict
