"""
Numeric Binner Sub-Module

Purpose: Discretize continuous variables into bins (quantiles, uniform, kmeans).
Libraries: sklearn.preprocessing
Parameters:
  - numeric_include: List[str]
  - numeric_exclude: List[str]
  - strategy: "uniform" | "quantile" | "kmeans"
  - n_bins: int
  - encode: "ordinal" | "onehot"
  - drop_original: bool
"""

from pathlib import Path
from typing import Any, Dict, Tuple
import warnings

import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

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
        "strategy": "quantile",
        "n_bins": 5,
        "encode": "ordinal",
        "drop_original": False,
        "use_original_features_only": True,
    }
    validation.validate_config(config, required_params, optional_params)
    validation.validate_choice(
        config["strategy"], ["uniform", "quantile", "kmeans"], "strategy"
    )
    validation.validate_choice(config["encode"], ["ordinal", "onehot"], "encode")

    # 2. Submodule dir
    submodule_dir = artifacts.get_submodule_artifact_dir(artifact_dir, "numeric_binner")
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

    if not numeric_cols:
        return train_df, val_df, test_df, orig_df, {"message": "No numeric cols"}

    # 4. Fit Discretizer
    # sklearn KBinsDiscretizer handles multiple columns at once
    est = KBinsDiscretizer(
        n_bins=config["n_bins"],
        encode=config["encode"],
        strategy=config["strategy"],
        subsample=200_000 if len(train_df) > 200_000 else None,
    )

    # Fit on train
    # WARNING: sklearn output for onehot is sparse matrix by default in older versions,
    # but encode='onehot-dense' exists in newer?
    # Actually encode='onehot' returns sparse. We want dense usually for dataframe.
    # Check sklearn version or convert.

    # Let's force dense output if possible, or convert.
    # In newer sklearn, encode='onehot-dense' is available.
    # Let's try to handle it.

    try:
        est.fit(train_df[numeric_cols])
    except ValueError as e:
        # Fallback for old sklearn or n_bins issues
        warnings.warn(f"Discretizer fit failed: {e}. Skipping binning.")
        return train_df, val_df, test_df, orig_df, {"error": str(e)}

    # Helper to transform
    def transform_and_merge(df, is_train=False):
        if df is None:
            return None

        # Transform
        mat = est.transform(df[numeric_cols])

        # If sparse, densify
        if hasattr(mat, "toarray"):
            mat = mat.toarray()

        # Create column names
        if config["encode"] == "ordinal":
            new_names = [f"{c}_bin" for c in numeric_cols]
            # Convert to int if ordinal
            mat = mat.astype(int)
        else:
            # Onehot: we need feature names.
            # sklearn < 1.0 doesn't support get_feature_names_out properly?
            # Assuming recent sklearn.
            if hasattr(est, "get_feature_names_out"):
                new_names = est.get_feature_names_out(numeric_cols)
            else:
                # Manual fallback
                new_names = []
                for i, col in enumerate(numeric_cols):
                    for b in range(est.n_bins_[i]):
                        new_names.append(f"{col}_bin_{b}")

        df_new = pd.DataFrame(mat, columns=new_names, index=df.index)

        if config["drop_original"]:
            df = df.drop(columns=numeric_cols)

        return pd.concat([df, df_new], axis=1)

    train_df = transform_and_merge(train_df, is_train=True)
    test_df = transform_and_merge(test_df)
    val_df = transform_and_merge(val_df)
    orig_df = transform_and_merge(orig_df)

    # Save fitted
    artifacts.save_fitted_object(est, submodule_dir, "discretizer.pkl")

    # 5. Reports
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
        "bin_edges": [arr.tolist() for arr in est.bin_edges_],
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
    }

    return train_df, val_df, test_df, orig_df, state_dict
