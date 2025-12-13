"""
Drift Detector - Train-Test Distribution Drift Detection

Purpose: Detect features with significantly different distributions between train and test sets
Libraries: scipy.stats, sklearn.ensemble, pandas, numpy
Parameters: drift_metric, max_psi, max_ks, max_pvalue, min_auc, action, max_drop_fraction, exclude_cols
"""

from pathlib import Path
from typing import Any, Dict, Tuple, List
import warnings

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier

from mlarena.preprocessing.utils import (
    validation,
    artifacts,
    dataframe_utils,
    report,
)


def _calculate_psi(train_col: pd.Series, test_col: pd.Series, bins: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI) for a feature.

    PSI = sum((test_pct - train_pct) * ln(test_pct / train_pct))

    Args:
        train_col: Training data column
        test_col: Test data column
        bins: Number of bins for discretization

    Returns:
        PSI value (higher = more drift)
    """
    # Handle missing values
    train_clean = train_col.dropna()
    test_clean = test_col.dropna()

    if len(train_clean) == 0 or len(test_clean) == 0:
        return np.nan

    # Determine if numeric or categorical
    if pd.api.types.is_numeric_dtype(train_col):
        # Numeric: create bins based on train quantiles
        try:
            _, bin_edges = pd.qcut(train_clean, q=bins, retbins=True, duplicates='drop')
            train_binned = pd.cut(train_clean, bins=bin_edges, include_lowest=True)
            test_binned = pd.cut(test_clean, bins=bin_edges, include_lowest=True)
        except (ValueError, TypeError):
            # If binning fails, return NaN
            return np.nan
    else:
        # Categorical: use categories as bins
        train_binned = train_clean
        test_binned = test_clean

    # Calculate distributions
    train_dist = train_binned.value_counts(normalize=True, dropna=False)
    test_dist = test_binned.value_counts(normalize=True, dropna=False)

    # Align distributions
    all_bins = train_dist.index.union(test_dist.index)
    train_pct = train_dist.reindex(all_bins, fill_value=0.001)  # Small epsilon to avoid log(0)
    test_pct = test_dist.reindex(all_bins, fill_value=0.001)

    # Calculate PSI
    psi = np.sum((test_pct - train_pct) * np.log(test_pct / train_pct))

    return psi


def _calculate_ks_statistic(train_col: pd.Series, test_col: pd.Series) -> Tuple[float, float]:
    """
    Calculate Kolmogorov-Smirnov statistic for numeric features.

    Args:
        train_col: Training data column
        test_col: Test data column

    Returns:
        Tuple of (KS statistic, p-value)
    """
    train_clean = train_col.dropna()
    test_clean = test_col.dropna()

    if len(train_clean) == 0 or len(test_clean) == 0:
        return np.nan, np.nan

    try:
        ks_stat, p_value = stats.ks_2samp(train_clean, test_clean)
        return ks_stat, p_value
    except Exception:
        return np.nan, np.nan


def _calculate_chi2_statistic(train_col: pd.Series, test_col: pd.Series) -> Tuple[float, float]:
    """
    Calculate Chi-Square statistic for categorical features.

    Args:
        train_col: Training data column
        test_col: Test data column

    Returns:
        Tuple of (Chi2 statistic, p-value)
    """
    train_clean = train_col.dropna()
    test_clean = test_col.dropna()

    if len(train_clean) == 0 or len(test_clean) == 0:
        return np.nan, np.nan

    # Create contingency table
    train_dist = train_clean.value_counts()
    test_dist = test_clean.value_counts()

    # Align categories
    all_cats = train_dist.index.union(test_dist.index)
    train_counts = train_dist.reindex(all_cats, fill_value=0)
    test_counts = test_dist.reindex(all_cats, fill_value=0)

    # Create contingency table
    contingency = pd.DataFrame({
        'train': train_counts,
        'test': test_counts
    }).T

    try:
        chi2_stat, p_value, _, _ = stats.chi2_contingency(contingency)
        return chi2_stat, p_value
    except Exception:
        return np.nan, np.nan


def _calculate_model_auc(
    train_col: pd.Series,
    test_col: pd.Series,
    random_state: int = 42
) -> float:
    """
    Train a simple model to discriminate train vs test (AUC).
    Higher AUC = more drift (easier to distinguish train from test).

    Args:
        train_col: Training data column
        test_col: Test data column
        random_state: Random state for reproducibility

    Returns:
        AUC score (0.5 = no drift, 1.0 = perfect drift)
    """
    from sklearn.metrics import roc_auc_score

    # Prepare data
    train_clean = train_col.dropna()
    test_clean = test_col.dropna()

    if len(train_clean) == 0 or len(test_clean) == 0:
        return np.nan

    # Combine data
    X_combined = pd.concat([train_clean, test_clean], axis=0).values.reshape(-1, 1)
    y_combined = np.concatenate([
        np.zeros(len(train_clean)),
        np.ones(len(test_clean))
    ])

    # Handle categorical
    if not pd.api.types.is_numeric_dtype(train_col):
        # Simple label encoding
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        X_combined = le.fit_transform(X_combined.ravel()).reshape(-1, 1)

    try:
        # Train simple classifier
        clf = RandomForestClassifier(
            n_estimators=50,
            max_depth=3,
            random_state=random_state,
            n_jobs=-1
        )

        # Use small sample if data is too large
        if len(X_combined) > 10000:
            sample_idx = np.random.RandomState(random_state).choice(
                len(X_combined), 10000, replace=False
            )
            X_combined = X_combined[sample_idx]
            y_combined = y_combined[sample_idx]

        clf.fit(X_combined, y_combined)
        y_pred = clf.predict_proba(X_combined)[:, 1]
        auc = roc_auc_score(y_combined, y_pred)

        return auc
    except Exception:
        return np.nan


def fit_transform(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
    orig_df: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None, Dict[str, Any]]:
    """
    Drift detection preprocessing - detect and optionally remove features with distribution drift.

    Args:
        train_df: Training data
        val_df: Validation data (can be None)
        test_df: Test data
        config: Configuration dictionary with keys:
            - _artifact_dir: Path to save artifacts
            - _dataset: {id_column, target, ignored_columns}
            - drift_metric: Drift detection method (psi|ks|chi2|model_auc)
            - max_psi: Maximum allowed PSI (default: 0.25)
            - max_ks: Maximum allowed KS statistic (default: 0.1)
            - max_pvalue: Maximum allowed p-value for statistical tests (default: 0.01)
            - min_auc: Minimum AUC for drift model (default: 0.6)
            - action: What to do with drifting features (none|drop|flag_only)
            - max_drop_fraction: Maximum fraction of features to drop (default: 0.2)
            - exclude_cols: Columns to exclude from drift detection
            - random_state: Random state for reproducibility (default: 42)
        orig_df: External dataset (can be None) - passed through unchanged

    Returns:
        Tuple of (train_df, val_df, test_df, orig_df, state_dict)

        state_dict contains:
        - version: str - Version of this sub-module
        - drift_results: Dict - Drift statistics per column
        - columns_dropped: List - Columns that were dropped
        - columns_flagged: List - Columns with detected drift
    """
    # 1. Extract config
    artifact_dir = Path(config.get("_artifact_dir", "."))
    dataset_config = config.get("_dataset", {})
    id_column = dataset_config.get("id_column", "id")
    target_column = dataset_config.get("target")
    ignored_columns = dataset_config.get("ignored_columns", [])

    # 2. Validate config
    required_params = []
    optional_params = {
        "drift_metric": "psi",
        "max_psi": 0.25,
        "max_ks": 0.1,
        "max_pvalue": 0.01,
        "min_auc": 0.6,
        "action": "flag_only",
        "max_drop_fraction": 0.2,
        "exclude_cols": [],
        "random_state": 42,
    }
    validation.validate_config(config, required_params, optional_params)

    # Validate drift_metric choice
    validation.validate_choice(
        config["drift_metric"],
        ["psi", "ks", "chi2", "model_auc"],
        "drift_metric"
    )

    # Validate action choice
    validation.validate_choice(
        config["action"],
        ["none", "drop", "flag_only"],
        "action"
    )

    # 3. Create sub-module artifact directory
    submodule_dir = artifacts.get_submodule_artifact_dir(artifact_dir, "drift_detector")

    # 4. Save original DataFrames for reporting
    train_df_original = dataframe_utils.copy_dataframe(train_df)
    test_df_original = dataframe_utils.copy_dataframe(test_df)

    # 5. Determine columns to analyze
    exclude_cols = [id_column, target_column] + ignored_columns + config["exclude_cols"]

    # Get all columns except excluded
    columns_to_check = [col for col in train_df.columns if col not in exclude_cols and col in test_df.columns]

    if len(columns_to_check) == 0:
        warnings.warn("No columns to check for drift (all excluded or not in both train/test)")
        return train_df, val_df, test_df, {
            "version": "1.0",
            "drift_results": {},
            "columns_dropped": [],
            "columns_flagged": [],
            "config": {k: v for k, v in config.items() if not k.startswith("_")},
        }

    # 6. Calculate drift metrics
    drift_results = {}
    drift_metric = config["drift_metric"]

    for col in columns_to_check:
        result = {"column": col, "dtype": str(train_df[col].dtype)}

        if drift_metric == "psi":
            psi = _calculate_psi(train_df[col], test_df[col])
            result["psi"] = float(psi) if not np.isnan(psi) else None
            result["drifted"] = bool(psi > config["max_psi"]) if not np.isnan(psi) else False

        elif drift_metric == "ks":
            # Only for numeric columns
            if pd.api.types.is_numeric_dtype(train_df[col]):
                ks_stat, p_value = _calculate_ks_statistic(train_df[col], test_df[col])
                result["ks_statistic"] = float(ks_stat) if not np.isnan(ks_stat) else None
                result["p_value"] = float(p_value) if not np.isnan(p_value) else None
                result["drifted"] = bool(
                    (ks_stat > config["max_ks"] or p_value < config["max_pvalue"])
                ) if not np.isnan(ks_stat) else False
            else:
                result["ks_statistic"] = None
                result["p_value"] = None
                result["drifted"] = False

        elif drift_metric == "chi2":
            # Only for categorical columns
            if not pd.api.types.is_numeric_dtype(train_df[col]):
                chi2_stat, p_value = _calculate_chi2_statistic(train_df[col], test_df[col])
                result["chi2_statistic"] = float(chi2_stat) if not np.isnan(chi2_stat) else None
                result["p_value"] = float(p_value) if not np.isnan(p_value) else None
                result["drifted"] = bool(p_value < config["max_pvalue"]) if not np.isnan(p_value) else False
            else:
                result["chi2_statistic"] = None
                result["p_value"] = None
                result["drifted"] = False

        elif drift_metric == "model_auc":
            auc = _calculate_model_auc(train_df[col], test_df[col], config["random_state"])
            result["auc"] = float(auc) if not np.isnan(auc) else None
            result["drifted"] = bool(auc > config["min_auc"]) if not np.isnan(auc) else False

        drift_results[col] = result

    # 7. Identify drifted columns
    drifted_cols = [col for col, res in drift_results.items() if res.get("drifted", False)]

    # 8. Apply max_drop_fraction limit
    max_allowed_drops = int(len(columns_to_check) * config["max_drop_fraction"])

    if len(drifted_cols) > max_allowed_drops:
        # Sort by drift severity and take top K
        if drift_metric == "psi":
            drifted_cols_sorted = sorted(
                drifted_cols,
                key=lambda c: drift_results[c].get("psi", 0),
                reverse=True
            )
        elif drift_metric in ["ks", "chi2"]:
            drifted_cols_sorted = sorted(
                drifted_cols,
                key=lambda c: drift_results[c].get("p_value", 1)
            )
        elif drift_metric == "model_auc":
            drifted_cols_sorted = sorted(
                drifted_cols,
                key=lambda c: drift_results[c].get("auc", 0.5),
                reverse=True
            )
        else:
            drifted_cols_sorted = drifted_cols

        drifted_cols = drifted_cols_sorted[:max_allowed_drops]

        warnings.warn(
            f"Drift detected in {len(drifted_cols_sorted)} columns, but max_drop_fraction={config['max_drop_fraction']} "
            f"limits to {max_allowed_drops} columns. Keeping {len(drifted_cols_sorted) - max_allowed_drops} drifted columns."
        )

    # 9. Apply action
    columns_dropped = []
    columns_flagged = drifted_cols.copy()

    if config["action"] == "drop" and drifted_cols:
        train_df = dataframe_utils.safe_drop_columns(train_df, drifted_cols)
        test_df = dataframe_utils.safe_drop_columns(test_df, drifted_cols)
        if val_df is not None:
            val_df = dataframe_utils.safe_drop_columns(val_df, drifted_cols)
        columns_dropped = drifted_cols

    # 10. Save drift report
    drift_report = {
        "drift_metric": drift_metric,
        "columns_analyzed": len(columns_to_check),
        "columns_with_drift": len(drifted_cols),
        "columns_dropped": columns_dropped,
        "columns_flagged": columns_flagged,
        "drift_details": drift_results,
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
    }
    artifacts.save_report(drift_report, submodule_dir, "drift_report.json")

    # 11. Generate and save transformation summary
    transformation_summary = report.create_preprocessing_report(
        train_before=train_df_original,
        train_after=train_df,
        test_before=test_df_original,
        test_after=test_df,
        config=config,
    )
    artifacts.save_report(transformation_summary, submodule_dir, "summary.json")

    # 12. Create state dict
    state_dict = {
        "version": "1.0",
        "drift_metric": drift_metric,
        "drift_results": drift_results,
        "columns_dropped": columns_dropped,
        "columns_flagged": columns_flagged,
        "columns_analyzed": len(columns_to_check),
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
    }

    return train_df, val_df, test_df, orig_df, state_dict
