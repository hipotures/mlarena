"""
Feature Selection Sub-Module

Purpose: Systematically reduce feature dimensionality using various selection methods
Libraries: sklearn.feature_selection, sklearn.ensemble, lightgbm, xgboost, scipy
Parameters:
    - selection_method: variance|cardinality|mi|correlation|model_importance|permutation_importance|null_importance|l1|rfe|none
    - n_features: int|float|None (universal feature selector):
        * 0: pass-through (no selection)
        * 0 < n < 1: keep fraction (e.g., 0.3 = keep 30% best)
        * n >= 1: keep exactly n best features
        * n < 0: drop |n| worst features (e.g., -2 = drop 2 worst)
        * None: threshold-only (use min_importance/min_variance/min_unique_per_million)
    - min_variance: float (threshold for variance method)
    - min_unique_per_million: float (threshold for cardinality method; min unique values per 1M records)
    - min_importance: float (threshold for importance method)
    - importance_model_type: lgbm|xgb|rf
    - n_estimators: int (for model-based methods)
    - max_depth: int (for model-based methods)
    - random_state: int
    - max_drop_fraction: float (max fraction of features to drop in one step)
    - protect_cb_features: bool (keep numeric features ending with "_cb" regardless of selection)
    - optuna_trials: int (LightGBM only; >1 enables Optuna CV tuning)
    - optuna_timeout: float|None (Optuna time budget in seconds; if set, stops study on timeout)
      * If both optuna_trials and optuna_timeout are set, the first limit reached stops the study.
    - optuna_cv_folds: int (CV folds for Optuna tuning)
    - optuna_early_stopping_rounds: int (early stopping rounds during CV)
    - optuna_num_boost_round: int|None (override num_boost_round for tuning)
    - optuna_pruner: str ("median" | "none")
    - use_gpu: bool (LightGBM only)
    - importance_cumulative: float|None (keep top features covering cumulative importance)
    - perm_importance_repeats: int (permutation importance repeats)
    - perm_importance_scoring: str|None (sklearn scoring name)
    - perm_importance_max_samples: int|float|None (subsample rows for permutation importance)
    - null_importance_rounds: int (number of target shuffles)
    - null_importance_quantile: float (quantile cutoff for null importances)
    - optuna_export_trials: bool (export Optuna trials JSON when tuning)
    - optuna_save_storage: bool (save Optuna sqlite storage when tuning)
    - optuna_storage_path: str|None (override sqlite path)
    - optuna_study_name: str (Optuna study name)
    - optuna_load_if_exists: bool (resume existing sqlite study)
"""

from pathlib import Path
from typing import Any, Dict, Tuple, List
import warnings
import json

import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    mutual_info_classif,
    mutual_info_regression,
    RFE,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from scipy.stats import pearsonr

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
    eval_df: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, Dict[str, Any]]:
    """
    Feature selection preprocessing.

    Args:
        train_df: Training data
        val_df: Validation data (can be None)
        test_df: Test data
        config: Configuration dictionary with keys:
            - _artifact_dir: Path to save artifacts
            - _dataset: {id_column, target, ignored_columns, problem_type}
            - selection_method: Method to use for feature selection
            - n_features: Universal feature selector (int|float|None)
            - min_variance: Minimum variance threshold
            - min_importance: Minimum importance threshold
            - importance_model_type: Model type for importance-based selection
            - n_estimators: Number of estimators for model-based methods
            - max_depth: Max depth for model-based methods
            - random_state: Random seed
            - max_drop_fraction: Maximum fraction of features to drop
            - protect_cb_features: Protect numeric columns ending with "_cb"
            - optuna_trials: LightGBM-only Optuna trials (>1 enables tuning)
            - optuna_timeout: Optuna time budget (seconds; if set, stops study on timeout)
              * If both optuna_trials and optuna_timeout are set, the first limit reached stops the study.
            - optuna_cv_folds: CV folds for Optuna tuning
            - optuna_early_stopping_rounds: Early stopping rounds during tuning
            - optuna_num_boost_round: Override num_boost_round for tuning
            - optuna_pruner: "median" | "none"
            - use_gpu: Use GPU for LightGBM (if available)
            - importance_cumulative: Cumulative importance threshold (0-1); mutually exclusive with n_features
            - optuna_export_trials: Export Optuna trials JSON to artifacts
            - optuna_save_storage: Save Optuna sqlite storage to artifacts
            - optuna_storage_path: Custom sqlite path (optional)
            - optuna_study_name: Optuna study name
            - optuna_load_if_exists: Resume study if sqlite exists
        orig_df: External dataset (can be None)
        eval_df: Evaluation dataset (can be None)

    Returns:
        Tuple of (train_df, val_df, test_df, eval_df, orig_df, state_dict)
    """
    # 1. Extract config
    artifact_dir = Path(config.get("_artifact_dir", "."))
    dataset_config = config.get("_dataset", {})
    id_column = dataset_config.get("id_column", "id")
    target_column = dataset_config.get("target")
    ignored_columns = dataset_config.get("ignored_columns", [])
    problem_type = dataset_config.get("problem_type", "binary")

    # 2. Validate config
    required_params = []
    optional_params = {
        "selection_method": "variance",
        "n_features": None,
        "min_variance": 0.01,
        "min_unique_per_million": 5.0,
        "min_importance": 0.001,
        "importance_model_type": "lgbm",
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42,
        "max_drop_fraction": 0.5,
        "protect_cb_features": True,
        "optuna_trials": 1,
        "optuna_timeout": None,
        "optuna_cv_folds": 3,
        "optuna_early_stopping_rounds": 200,
        "optuna_num_boost_round": None,
        "optuna_pruner": "median",
        "use_gpu": False,
        "importance_cumulative": None,
        "optuna_export_trials": True,
        "optuna_save_storage": True,
        "optuna_storage_path": None,
        "optuna_study_name": "feature_selector_lgbm",
        "optuna_load_if_exists": True,
        "perm_importance_repeats": 5,
        "perm_importance_scoring": None,
        "perm_importance_max_samples": None,
        "null_importance_rounds": 10,
        "null_importance_quantile": 0.95,
    }
    validation.validate_config(config, required_params, optional_params)

    # Validate choice parameters
    validation.validate_choice(
        config["selection_method"],
        [
            "variance",
            "cardinality",
            "mi",
            "correlation",
            "model_importance",
            "permutation_importance",
            "null_importance",
            "l1",
            "rfe",
            "none",
        ],
        "selection_method"
    )

    if config["selection_method"] != "none":
        if config["importance_model_type"] is not None:
            validation.validate_choice(
                config["importance_model_type"],
                ["lgbm", "xgb", "rf"],
                "importance_model_type"
            )

        # Validate n_features type
        if config.get("n_features") is not None:
            n_feat = config["n_features"]
            if not isinstance(n_feat, (int, float)):
                raise ValueError(f"n_features must be a number or None, got {type(n_feat).__name__}")

        if config.get("importance_cumulative") is not None:
            validation.validate_numeric_range(
                config["importance_cumulative"],
                min_value=0.0,
                max_value=1.0,
                param_name="importance_cumulative"
            )
            if config.get("n_features") is not None:
                raise ValueError(
                    "n_features and importance_cumulative are mutually exclusive. "
                    "Use only one selection option."
                )
            if config["selection_method"] == "rfe":
                raise ValueError("importance_cumulative is not supported for selection_method=rfe.")

        # Validate numeric ranges
        validation.validate_numeric_range(
            config["max_drop_fraction"],
            min_value=0.0,
            max_value=1.0,
            param_name="max_drop_fraction"
        )

        validation.validate_numeric_range(
            config["perm_importance_repeats"],
            min_value=1,
            max_value=None,
            param_name="perm_importance_repeats",
        )
        if config["perm_importance_max_samples"] is not None:
            max_samples = config["perm_importance_max_samples"]
            if isinstance(max_samples, float):
                validation.validate_numeric_range(
                    max_samples,
                    min_value=0.0,
                    max_value=1.0,
                    param_name="perm_importance_max_samples",
                )
            else:
                validation.validate_numeric_range(
                    max_samples,
                    min_value=1,
                    max_value=None,
                    param_name="perm_importance_max_samples",
                )
        validation.validate_numeric_range(
            config["null_importance_rounds"],
            min_value=1,
            max_value=None,
            param_name="null_importance_rounds",
        )
        validation.validate_numeric_range(
            config["null_importance_quantile"],
            min_value=0.0,
            max_value=1.0,
            param_name="null_importance_quantile",
        )

        validation.validate_numeric_range(
            config["optuna_cv_folds"],
            min_value=2,
            max_value=20,
            param_name="optuna_cv_folds"
        )

        validation.validate_numeric_range(
            config["optuna_trials"],
            min_value=1,
            max_value=1000,
            param_name="optuna_trials"
        )

        if config.get("optuna_timeout") is not None:
            validation.validate_numeric_range(
                config["optuna_timeout"],
                min_value=1,
                max_value=None,
                param_name="optuna_timeout"
            )

        if config["optuna_num_boost_round"] is not None:
            validation.validate_numeric_range(
                config["optuna_num_boost_round"],
                min_value=1,
                max_value=100000,
                param_name="optuna_num_boost_round"
            )

        if config["optuna_pruner"] not in ["median", "none"]:
            raise ValueError(
                "optuna_pruner must be 'median' or 'none'"
            )

        for bool_key in ["optuna_export_trials", "optuna_save_storage", "optuna_load_if_exists", "use_gpu"]:
            if not isinstance(config.get(bool_key), bool):
                raise ValueError(f"{bool_key} must be a boolean")

        if config.get("optuna_storage_path") is not None and not isinstance(config["optuna_storage_path"], str):
            raise ValueError("optuna_storage_path must be a string or null")

        if not isinstance(config.get("optuna_study_name"), str) or not config["optuna_study_name"].strip():
            raise ValueError("optuna_study_name must be a non-empty string")

    # 3. Create sub-module artifact directory
    submodule_dir = artifacts.get_submodule_artifact_dir(artifact_dir, "feature_selector")

    # 4. Save original DataFrames for reporting
    train_df_original = dataframe_utils.copy_dataframe(train_df)
    test_df_original = dataframe_utils.copy_dataframe(test_df)

    # 5. Early exit if method is "none"
    if config["selection_method"] == "none":
        transformation_summary = report.create_preprocessing_report(
            train_before=train_df_original,
            train_after=train_df,
            test_before=test_df_original,
            test_after=test_df,
            config=config,
        )
        artifacts.save_report(transformation_summary, submodule_dir, "summary.json")

        state_dict = {
            "version": "1.0",
            "method": "none",
            "message": "Feature selection skipped (method=none)",
            "config": {k: v for k, v in config.items() if not k.startswith("_")},
        }
        return train_df, val_df, test_df, eval_df, orig_df, state_dict

    # 6. Get feature columns (exclude id, target, ignored)
    exclude_cols = [id_column, target_column] + ignored_columns
    exclude_cols = [col for col in exclude_cols if col]  # Remove None values

    all_feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    numeric_cols = dataframe_utils.get_numeric_columns(train_df, exclude=exclude_cols)

    if not numeric_cols:
        warnings.warn("No numeric columns found for feature selection. Returning unchanged.")
        transformation_summary = report.create_preprocessing_report(
            train_before=train_df_original,
            train_after=train_df,
            test_before=test_df_original,
            test_after=test_df,
            config=config,
        )
        artifacts.save_report(transformation_summary, submodule_dir, "summary.json")

        state_dict = {
            "version": "1.0",
            "method": config["selection_method"],
            "message": "No numeric columns to select from",
            "config": {k: v for k, v in config.items() if not k.startswith("_")},
        }
        return train_df, val_df, test_df, eval_df, orig_df, state_dict

    # 7. Prepare data for selection
    if target_column and target_column in train_df.columns:
        X_train = train_df[numeric_cols].values
        y_train = train_df[target_column].values
    else:
        raise ValueError(f"Target column '{target_column}' not found in training data")

    # 8. Perform feature selection
    optuna_trials_limit = config["optuna_trials"] if config["optuna_trials"] > 1 else None
    optuna_timeout_limit = config["optuna_timeout"] if config.get("optuna_timeout") is not None else None

    selected_features, feature_scores, selection_meta = _select_features(
        X_train=X_train,
        y_train=y_train,
        feature_names=numeric_cols,
        method=config["selection_method"],
        select_features=config["n_features"],
        min_variance=config["min_variance"],
        min_unique_per_million=config["min_unique_per_million"],
        min_importance=config["min_importance"],
        importance_model_type=config["importance_model_type"],
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        random_state=config["random_state"],
        max_drop_fraction=config["max_drop_fraction"],
        problem_type=problem_type,
        optuna_trials=optuna_trials_limit,
        optuna_timeout=optuna_timeout_limit,
        optuna_cv_folds=config["optuna_cv_folds"],
        optuna_early_stopping_rounds=config["optuna_early_stopping_rounds"],
        optuna_num_boost_round=config["optuna_num_boost_round"],
        optuna_pruner=config["optuna_pruner"],
        use_gpu=config["use_gpu"],
        importance_cumulative=config["importance_cumulative"],
        perm_importance_repeats=config["perm_importance_repeats"],
        perm_importance_scoring=config["perm_importance_scoring"],
        perm_importance_max_samples=config["perm_importance_max_samples"],
        null_importance_rounds=config["null_importance_rounds"],
        null_importance_quantile=config["null_importance_quantile"],
        optuna_export_trials=config["optuna_export_trials"],
        optuna_save_storage=config["optuna_save_storage"],
        optuna_storage_path=config["optuna_storage_path"],
        optuna_study_name=config["optuna_study_name"],
        optuna_load_if_exists=config["optuna_load_if_exists"],
        optuna_artifact_dir=submodule_dir,
    )

    # 9. Apply selection to DataFrames
    # Always keep non-numeric columns
    base_keep = [col for col in train_df.columns if col not in numeric_cols]

    # Protect key numeric columns (encoded categoricals ending with "_cb")
    protected_numeric = [col for col in numeric_cols if col.endswith("_cb")] if config.get("protect_cb_features", True) else []

    # Keep non-feature columns + protected numeric + selected numeric features
    keep_cols = base_keep + protected_numeric + selected_features
    keep_cols_unique: List[str] = []
    seen_cols = set()
    for col in keep_cols:
        if col in seen_cols:
            continue
        keep_cols_unique.append(col)
        seen_cols.add(col)
    keep_cols = keep_cols_unique

    train_df = train_df[keep_cols]
    test_df = test_df[[col for col in keep_cols if col in test_df.columns]]
    if val_df is not None:
        val_df = val_df[[col for col in keep_cols if col in val_df.columns]]
    if eval_df is not None:
        eval_df = eval_df[[col for col in keep_cols if col in eval_df.columns]]
    if orig_df is not None:
        orig_df = orig_df[[col for col in keep_cols if col in orig_df.columns]]

    # 10. Save feature importance/scores report (sorted by score desc)
    def _normalize_score(score: float) -> float | None:
        if isinstance(score, (float, np.floating)) and np.isnan(score):
            return None
        return float(score)

    score_items = [
        (feature, _normalize_score(score))
        for feature, score in zip(numeric_cols, feature_scores)
    ]
    score_items_sorted = sorted(
        score_items,
        key=lambda item: (item[1] is None, -(item[1] if item[1] is not None else 0.0)),
    )

    feature_report = {
        "method": config["selection_method"],
        "total_features_before": len(numeric_cols),
        "total_features_after": len(selected_features),
        "features_dropped": len(numeric_cols) - len(selected_features),
        "drop_fraction": (len(numeric_cols) - len(selected_features)) / len(numeric_cols) if numeric_cols else 0,
        "selected_features": selected_features,
        "dropped_features": [col for col in numeric_cols if col not in selected_features],
        "selection_meta": selection_meta,
        "feature_scores": {
            feature: score for feature, score in score_items_sorted
        },
    }
    artifacts.save_report(feature_report, submodule_dir, "feature_selection_report.json")

    # 11. Generate and save transformation report
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
        "method": config["selection_method"],
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
        "features_before": len(numeric_cols),
        "features_after": len(selected_features),
        "selected_features": selected_features,
        "selection_meta": selection_meta,
        "feature_scores_summary": {
            "mean": float(np.nanmean(feature_scores)),
            "std": float(np.nanstd(feature_scores)),
            "min": float(np.nanmin(feature_scores)),
            "max": float(np.nanmax(feature_scores)),
        },
    }

    return train_df, val_df, test_df, eval_df, orig_df, state_dict


def _select_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    method: str,
    select_features: int | float | None,
    min_variance: float,
    min_unique_per_million: float,
    min_importance: float,
    importance_model_type: str,
    n_estimators: int,
    max_depth: int,
    random_state: int,
    max_drop_fraction: float,
    problem_type: str,
    optuna_trials: int | None,
    optuna_timeout: float | None,
    optuna_cv_folds: int,
    optuna_early_stopping_rounds: int,
    optuna_num_boost_round: int | None,
    optuna_pruner: str,
    use_gpu: bool,
    importance_cumulative: float | None,
    perm_importance_repeats: int,
    perm_importance_scoring: str | None,
    perm_importance_max_samples: int | float | None,
    null_importance_rounds: int,
    null_importance_quantile: float,
    optuna_export_trials: bool,
    optuna_save_storage: bool,
    optuna_storage_path: str | None,
    optuna_study_name: str,
    optuna_load_if_exists: bool,
    optuna_artifact_dir: Path | None,
) -> Tuple[List[str], np.ndarray, Dict[str, Any]]:
    """
    Select features using specified method.

    Returns:
        Tuple of (selected_feature_names, feature_scores, selection_meta)
    """
    total_features = X_train.shape[1]
    selection_meta: Dict[str, Any] = {}

    optuna_enabled = (optuna_trials is not None and optuna_trials > 1) or optuna_timeout is not None
    if optuna_enabled and (method != "model_importance" or importance_model_type != "lgbm"):
        warnings.warn(
            "Optuna tuning is only supported for selection_method=model_importance with "
            "importance_model_type=lgbm; ignoring optuna settings."
        )
        optuna_enabled = False

    if optuna_enabled:
        selection_meta["optuna_limits"] = {
            "trials": optuna_trials,
            "timeout_seconds": optuna_timeout,
        }

    # Interpret select_features parameter
    if select_features is None:
        # Threshold-only mode (use min_importance/min_variance)
        n_features_to_select = total_features
    elif select_features == 0:
        # Pass-through (no selection)
        n_features_to_select = total_features
    elif 0 < select_features < 1:
        # Keep fraction
        n_features_to_select = max(1, int(total_features * select_features))
    elif select_features >= 1:
        # Keep exact count
        n_features_to_select = min(int(select_features), total_features)
    elif select_features < 0:
        # Drop N worst
        n_features_to_select = max(1, total_features - abs(int(select_features)))
    else:
        # Should not reach here due to validation
        n_features_to_select = total_features

    # Apply max_drop_fraction constraint
    min_features_to_keep = max(1, int(total_features * (1 - max_drop_fraction)))
    n_features_to_select = max(n_features_to_select, min_features_to_keep)

    def _apply_cumulative_cutoff(feature_scores: np.ndarray) -> None:
        nonlocal n_features_to_select
        if importance_cumulative is None:
            return
        scores = np.array(feature_scores, dtype=float)
        scores = np.nan_to_num(scores, nan=0.0)
        total = float(scores.sum())
        if total <= 0:
            warnings.warn(
                "importance_cumulative requested but total feature importance is 0. "
                "Falling back to max_drop_fraction guard."
            )
            n_features_to_select = max(min_features_to_keep, 1)
        else:
            order = np.argsort(scores)[::-1]
            cumulative = np.cumsum(scores[order]) / total
            cutoff_idx = int(np.searchsorted(cumulative, importance_cumulative, side="left")) + 1
            cutoff_idx = min(cutoff_idx, total_features)
            n_features_to_select = max(cutoff_idx, min_features_to_keep)

        selection_meta["importance_cumulative"] = {
            "target": float(importance_cumulative),
            "selected_features": int(n_features_to_select),
            "min_features_to_keep": int(min_features_to_keep),
        }

    def _build_importance_model() -> Tuple[Any, str]:
        model_type_used = importance_model_type
        if importance_model_type == "rf":
            if problem_type in ["binary", "multiclass"]:
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                    n_jobs=-1,
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                    n_jobs=-1,
                )
        elif importance_model_type == "lgbm":
            try:
                import lightgbm as lgb
            except ImportError:
                warnings.warn("LightGBM not available, falling back to RandomForest for importance model.")
                model_type_used = "rf"
                if problem_type in ["binary", "multiclass"]:
                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=random_state,
                        n_jobs=-1,
                    )
                else:
                    model = RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=random_state,
                        n_jobs=-1,
                    )
                return model, model_type_used
            if problem_type in ["binary", "multiclass"]:
                model = lgb.LGBMClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                    n_jobs=-1,
                    verbosity=-1,
                    device_type="gpu" if use_gpu else "cpu",
                )
            else:
                model = lgb.LGBMRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                    n_jobs=-1,
                    verbosity=-1,
                    device_type="gpu" if use_gpu else "cpu",
                )
        elif importance_model_type == "xgb":
            try:
                import xgboost as xgb
            except ImportError:
                warnings.warn("XGBoost not available, falling back to RandomForest for importance model.")
                model_type_used = "rf"
                if problem_type in ["binary", "multiclass"]:
                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=random_state,
                        n_jobs=-1,
                    )
                else:
                    model = RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=random_state,
                        n_jobs=-1,
                    )
                return model, model_type_used
            if problem_type in ["binary", "multiclass"]:
                model = xgb.XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                    n_jobs=-1,
                    verbosity=0,
                )
            else:
                model = xgb.XGBRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                    n_jobs=-1,
                    verbosity=0,
                )
        else:
            raise ValueError(f"Unknown importance_model_type: {importance_model_type}")

        return model, model_type_used

    # Initialize scores array
    feature_scores = np.zeros(total_features)

    if method == "variance":
        # Variance threshold
        variances = np.var(X_train, axis=0)
        feature_scores = variances
        _apply_cumulative_cutoff(variances)
        selected_mask = variances >= min_variance

        if selected_mask.sum() >= n_features_to_select:
            # Too many (or exact) -> trim to top-K among those above threshold
            candidate_idx = np.where(selected_mask)[0]
            top_idx = candidate_idx[np.argsort(variances[candidate_idx])[::-1][:n_features_to_select]]
            selected_mask = np.zeros(total_features, dtype=bool)
            selected_mask[top_idx] = True
        else:
            # Too few above threshold
            # In threshold-only mode (select_features=None), respect the threshold strictly
            # Otherwise backfill with top-K overall to reach requested count
            if select_features is None:
                # Threshold-only mode: don't backfill, just use what passes threshold
                pass  # selected_mask already set correctly at line 528
            else:
                # Top-K mode: backfill to reach requested count
                top_idx = np.argsort(variances)[::-1][:n_features_to_select]
                selected_mask = np.zeros(total_features, dtype=bool)
                selected_mask[top_idx] = True

    elif method == "cardinality":
        # Cardinality threshold (min unique values per 1M records)
        # Count unique values for each feature
        n_rows = X_train.shape[0]
        n_unique = np.array([len(np.unique(X_train[:, i])) for i in range(total_features)])

        # Normalize to per-million scale
        unique_per_million = (n_unique / n_rows) * 1_000_000
        feature_scores = unique_per_million

        _apply_cumulative_cutoff(unique_per_million)
        selected_mask = unique_per_million >= min_unique_per_million

        if selected_mask.sum() >= n_features_to_select:
            # Too many (or exact) -> trim to top-K among those above threshold
            candidate_idx = np.where(selected_mask)[0]
            top_idx = candidate_idx[np.argsort(unique_per_million[candidate_idx])[::-1][:n_features_to_select]]
            selected_mask = np.zeros(total_features, dtype=bool)
            selected_mask[top_idx] = True
        else:
            # Too few above threshold
            # In threshold-only mode (select_features=None), respect the threshold strictly
            # Otherwise backfill with top-K overall to reach requested count
            if select_features is None:
                # Threshold-only mode: don't backfill, just use what passes threshold
                pass  # selected_mask already set correctly above
            else:
                # Top-K mode: backfill to reach requested count
                top_idx = np.argsort(unique_per_million)[::-1][:n_features_to_select]
                selected_mask = np.zeros(total_features, dtype=bool)
                selected_mask[top_idx] = True

    elif method == "mi":
        # Mutual Information
        # Note: n_jobs added in sklearn 1.5. Fallback for older versions.
        mi_kwargs = {"random_state": random_state}
        try:
            mi_kwargs["n_jobs"] = -1
            if problem_type in ["binary", "multiclass"]:
                mi_scores = mutual_info_classif(X_train, y_train, **mi_kwargs)
            else:
                mi_scores = mutual_info_regression(X_train, y_train, **mi_kwargs)
        except (TypeError, ValueError):
            # Fallback for sklearn < 1.5
            if "n_jobs" in mi_kwargs:
                del mi_kwargs["n_jobs"]
            if problem_type in ["binary", "multiclass"]:
                mi_scores = mutual_info_classif(X_train, y_train, **mi_kwargs)
            else:
                mi_scores = mutual_info_regression(X_train, y_train, **mi_kwargs)

        feature_scores = mi_scores
        _apply_cumulative_cutoff(mi_scores)
        selector = SelectKBest(k=n_features_to_select)
        selector.fit(X_train, y_train)
        selected_mask = selector.get_support()

    elif method == "correlation":
        # Correlation with target
        correlations = np.array([
            abs(pearsonr(X_train[:, i], y_train)[0]) if len(np.unique(X_train[:, i])) > 1 else 0
            for i in range(total_features)
        ])
        feature_scores = correlations
        _apply_cumulative_cutoff(correlations)
        top_indices = np.argsort(correlations)[::-1][:n_features_to_select]
        selected_mask = np.zeros(total_features, dtype=bool)
        selected_mask[top_indices] = True

    elif method == "model_importance":
        # Model-based feature importance
        if importance_model_type == "rf":
            if problem_type in ["binary", "multiclass"]:
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                    n_jobs=-1,
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                    n_jobs=-1,
                )
            model.fit(X_train, y_train)
            importances = model.feature_importances_

        elif importance_model_type == "lgbm":
            if optuna_enabled:
                importances, optuna_summary = _run_optuna_lgbm_importance(
                    X_train=X_train,
                    y_train=y_train,
                    feature_names=feature_names,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                    problem_type=problem_type,
                    optuna_trials=optuna_trials,
                    optuna_timeout=optuna_timeout,
                    optuna_cv_folds=optuna_cv_folds,
                    optuna_early_stopping_rounds=optuna_early_stopping_rounds,
                    optuna_num_boost_round=optuna_num_boost_round,
                    optuna_pruner=optuna_pruner,
                    use_gpu=use_gpu,
                    optuna_export_trials=optuna_export_trials,
                    optuna_save_storage=optuna_save_storage,
                    optuna_storage_path=optuna_storage_path,
                    optuna_study_name=optuna_study_name,
                    optuna_load_if_exists=optuna_load_if_exists,
                    optuna_artifact_dir=optuna_artifact_dir,
                )
                selection_meta.update(optuna_summary)
            else:
                try:
                    import lightgbm as lgb
                    if problem_type in ["binary", "multiclass"]:
                        model = lgb.LGBMClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=random_state,
                            n_jobs=-1,
                            verbosity=-1,
                            device_type="gpu" if use_gpu else "cpu",
                        )
                    else:
                        model = lgb.LGBMRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=random_state,
                            n_jobs=-1,
                            verbosity=-1,
                            device_type="gpu" if use_gpu else "cpu",
                        )
                    model.fit(X_train, y_train)
                    importances = model.feature_importances_
                except ImportError:
                    warnings.warn("LightGBM not available, falling back to RandomForest")
                    return _select_features(
                        X_train, y_train, feature_names, "model_importance",
                        select_features, min_variance, min_importance,
                        "rf", n_estimators, max_depth, random_state,
                        max_drop_fraction, problem_type,
                        optuna_trials, optuna_timeout, optuna_cv_folds,
                        optuna_early_stopping_rounds, optuna_num_boost_round,
                        optuna_pruner, use_gpu,
                        importance_cumulative,
                        perm_importance_repeats,
                        perm_importance_scoring,
                        perm_importance_max_samples,
                        null_importance_rounds,
                        null_importance_quantile,
                        optuna_export_trials,
                        optuna_save_storage,
                        optuna_storage_path,
                        optuna_study_name,
                        optuna_load_if_exists,
                        optuna_artifact_dir
                    )

        elif importance_model_type == "xgb":
            try:
                import xgboost as xgb
                if problem_type in ["binary", "multiclass"]:
                    model = xgb.XGBClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=random_state,
                        n_jobs=-1,
                        verbosity=0,
                    )
                else:
                    model = xgb.XGBRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=random_state,
                        n_jobs=-1,
                        verbosity=0,
                    )
                model.fit(X_train, y_train)
                importances = model.feature_importances_
            except ImportError:
                warnings.warn("XGBoost not available, falling back to RandomForest")
                return _select_features(
                    X_train, y_train, feature_names, "model_importance",
                    select_features, min_variance, min_importance,
                    "rf", n_estimators, max_depth, random_state,
                    max_drop_fraction, problem_type,
                    optuna_trials, optuna_timeout, optuna_cv_folds,
                    optuna_early_stopping_rounds, optuna_num_boost_round,
                    optuna_pruner, use_gpu,
                    importance_cumulative,
                    perm_importance_repeats,
                    perm_importance_scoring,
                    perm_importance_max_samples,
                    null_importance_rounds,
                    null_importance_quantile,
                    optuna_export_trials,
                    optuna_save_storage,
                    optuna_storage_path,
                    optuna_study_name,
                    optuna_load_if_exists,
                    optuna_artifact_dir
                )
        else:
            raise ValueError(f"Unknown importance_model_type: {importance_model_type}")

        feature_scores = importances
        _apply_cumulative_cutoff(importances)
        # Select features using threshold + strict top-K cap
        selected_mask = importances >= min_importance

        if select_features is not None:
            if selected_mask.sum() >= n_features_to_select:
                # Too many (or exact) -> trim to top-K among those above threshold
                candidate_idx = np.where(selected_mask)[0]
                top_idx = candidate_idx[np.argsort(importances[candidate_idx])[::-1][:n_features_to_select]]
                selected_mask = np.zeros(total_features, dtype=bool)
                selected_mask[top_idx] = True
            else:
                # Too few above threshold -> backfill with top-K overall
                top_idx = np.argsort(importances)[::-1][:n_features_to_select]
                selected_mask = np.zeros(total_features, dtype=bool)
                selected_mask[top_idx] = True
        # Threshold-only mode preserved when select_features is None

    elif method == "permutation_importance":
        try:
            from sklearn.inspection import permutation_importance
        except ImportError as exc:  # pragma: no cover
            raise ImportError("permutation_importance requires sklearn.inspection") from exc

        model, model_type_used = _build_importance_model()
        model.fit(X_train, y_train)

        X_perm = X_train
        y_perm = y_train
        max_samples = perm_importance_max_samples
        if max_samples is not None:
            rng = np.random.default_rng(random_state)
            if isinstance(max_samples, float):
                sample_size = max(1, int(len(X_train) * max_samples))
            else:
                sample_size = min(int(max_samples), len(X_train))
            indices = rng.choice(len(X_train), size=sample_size, replace=False)
            X_perm = X_train[indices]
            y_perm = y_train[indices]

        scoring = perm_importance_scoring if perm_importance_scoring else None
        perm_result = permutation_importance(
            model,
            X_perm,
            y_perm,
            n_repeats=perm_importance_repeats,
            random_state=random_state,
            scoring=scoring,
            n_jobs=-1,
        )
        importances = perm_result.importances_mean
        feature_scores = importances
        _apply_cumulative_cutoff(importances)

        selected_mask = importances >= min_importance
        if select_features is not None:
            if selected_mask.sum() >= n_features_to_select:
                candidate_idx = np.where(selected_mask)[0]
                top_idx = candidate_idx[np.argsort(importances[candidate_idx])[::-1][:n_features_to_select]]
                selected_mask = np.zeros(total_features, dtype=bool)
                selected_mask[top_idx] = True
            else:
                top_idx = np.argsort(importances)[::-1][:n_features_to_select]
                selected_mask = np.zeros(total_features, dtype=bool)
                selected_mask[top_idx] = True

        selection_meta["permutation_importance"] = {
            "model_type": model_type_used,
            "n_repeats": int(perm_importance_repeats),
            "scoring": perm_importance_scoring,
            "max_samples": max_samples,
        }

    elif method == "null_importance":
        model, model_type_used = _build_importance_model()
        model.fit(X_train, y_train)

        if not hasattr(model, "feature_importances_"):
            raise ValueError("null_importance requires a model with feature_importances_")

        real_importances = np.asarray(model.feature_importances_, dtype=float)
        null_importances = []
        rng = np.random.default_rng(random_state)

        for i in range(int(null_importance_rounds)):
            y_perm = rng.permutation(y_train)
            null_model, _ = _build_importance_model()
            null_model.fit(X_train, y_perm)
            null_importances.append(np.asarray(null_model.feature_importances_, dtype=float))

        null_matrix = np.vstack(null_importances)
        thresholds = np.quantile(null_matrix, null_importance_quantile, axis=0)

        feature_scores = real_importances
        _apply_cumulative_cutoff(real_importances)

        selected_mask = (real_importances > thresholds) & (real_importances >= min_importance)
        if select_features is not None:
            if selected_mask.sum() >= n_features_to_select:
                candidate_idx = np.where(selected_mask)[0]
                top_idx = candidate_idx[np.argsort(real_importances[candidate_idx])[::-1][:n_features_to_select]]
                selected_mask = np.zeros(total_features, dtype=bool)
                selected_mask[top_idx] = True
            else:
                top_idx = np.argsort(real_importances)[::-1][:n_features_to_select]
                selected_mask = np.zeros(total_features, dtype=bool)
                selected_mask[top_idx] = True

        selection_meta["null_importance"] = {
            "model_type": model_type_used,
            "rounds": int(null_importance_rounds),
            "quantile": float(null_importance_quantile),
        }

    elif method == "l1":
        # L1 regularization
        if problem_type in ["binary", "multiclass"]:
            model = LogisticRegressionCV(
                penalty="l1",
                solver="saga",
                cv=3,
                random_state=random_state,
                n_jobs=-1,
                max_iter=1000,
            )
        else:
            model = LassoCV(
                cv=3,
                random_state=random_state,
                n_jobs=-1,
                max_iter=1000,
            )

        model.fit(X_train, y_train)
        if hasattr(model, "coef_"):
            importances = np.abs(model.coef_).flatten()
        else:
            importances = np.abs(model.coef_[0])

        feature_scores = importances
        _apply_cumulative_cutoff(importances)
        top_indices = np.argsort(importances)[::-1][:n_features_to_select]
        selected_mask = np.zeros(total_features, dtype=bool)
        selected_mask[top_indices] = True

    elif method == "rfe":
        # Recursive Feature Elimination (supports RF, LightGBM, XGBoost)
        if importance_model_type == "lgbm":
            try:
                import lightgbm as lgb
                if problem_type in ["binary", "multiclass"]:
                    estimator = lgb.LGBMClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=random_state,
                        device="gpu" if use_gpu else "cpu",
                        verbosity=-1,
                    )
                else:
                    estimator = lgb.LGBMRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=random_state,
                        device="gpu" if use_gpu else "cpu",
                        verbosity=-1,
                    )
            except ImportError:
                warnings.warn("LightGBM not available, falling back to RandomForest for RFE")
                importance_model_type = "rf"

        if importance_model_type == "xgb":
            try:
                import xgboost as xgb
                if problem_type in ["binary", "multiclass"]:
                    estimator = xgb.XGBClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=random_state,
                        tree_method="hist",
                        device="cuda" if use_gpu else "cpu",
                        verbosity=0,
                    )
                else:
                    estimator = xgb.XGBRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=random_state,
                        tree_method="hist",
                        device="cuda" if use_gpu else "cpu",
                        verbosity=0,
                    )
            except ImportError:
                warnings.warn("XGBoost not available, falling back to RandomForest for RFE")
                importance_model_type = "rf"

        if importance_model_type == "rf":
            # RandomForest (default fallback)
            if problem_type in ["binary", "multiclass"]:
                estimator = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                    n_jobs=-1,
                )
            else:
                estimator = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                    n_jobs=-1,
                )

        rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select)
        rfe.fit(X_train, y_train)
        selected_mask = rfe.support_
        feature_scores = rfe.ranking_  # Lower is better
        selection_meta["rfe_estimator"] = importance_model_type

    else:
        raise ValueError(f"Unknown selection method: {method}")

    # Get selected feature names
    selected_features = [feature_names[i] for i in range(total_features) if selected_mask[i]]

    return selected_features, feature_scores, selection_meta


def _run_optuna_lgbm_importance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    n_estimators: int,
    max_depth: int,
    random_state: int,
    problem_type: str,
    optuna_trials: int | None,
    optuna_timeout: float | None,
    optuna_cv_folds: int,
    optuna_early_stopping_rounds: int,
    optuna_num_boost_round: int | None,
    optuna_pruner: str,
    use_gpu: bool,
    optuna_export_trials: bool,
    optuna_save_storage: bool,
    optuna_storage_path: str | None,
    optuna_study_name: str,
    optuna_load_if_exists: bool,
    optuna_artifact_dir: Path | None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("LightGBM is required for optuna tuning.") from exc

    try:
        from optuna_integration import LightGBMPruningCallback
    except ImportError:
        try:
            from optuna.integration import LightGBMPruningCallback  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "optuna-integration[lightgbm] is required for pruning. "
                "Install optuna-integration==4.6.0."
            ) from exc

    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Optuna is required for tuning. Install optuna==4.6.0.") from exc

    if problem_type == "binary":
        objective = "binary"
        metric = "binary_logloss"
        direction = "minimize"
        stratified = True
        num_class = None
    elif problem_type == "multiclass":
        objective = "multiclass"
        metric = "multi_logloss"
        direction = "minimize"
        stratified = True
        num_class = int(np.unique(y_train).shape[0])
    else:
        objective = "regression"
        metric = "rmse"
        direction = "minimize"
        stratified = False
        num_class = None

    base_params = {
        "objective": objective,
        "metric": metric,
        "verbosity": -1,
        "max_depth": max_depth,
        "num_threads": -1,
        "seed": random_state,
        "feature_fraction_seed": random_state,
        "bagging_seed": random_state,
        "device_type": "gpu" if use_gpu else "cpu",
        "feature_pre_filter": False,
    }
    if num_class:
        base_params["num_class"] = num_class

    num_boost_round = optuna_num_boost_round or n_estimators
    if optuna_pruner == "median":
        pruner = optuna.pruners.MedianPruner()
    else:
        pruner = optuna.pruners.NopPruner()

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)

    def objective_fn(trial: optuna.Trial) -> float:
        params = dict(base_params)
        params.update({
            "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 512, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 0, 10),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
        })

        callbacks = [LightGBMPruningCallback(trial, metric=metric)]
        if optuna_early_stopping_rounds and optuna_early_stopping_rounds > 0:
            callbacks.insert(0, lgb.early_stopping(optuna_early_stopping_rounds))

        cv_results = lgb.cv(
            params,
            dtrain,
            nfold=optuna_cv_folds,
            stratified=stratified,
            shuffle=True,
            seed=random_state,
            num_boost_round=num_boost_round,
            callbacks=callbacks,
        )

        metric_key = f"{metric}-mean"
        if metric_key not in cv_results:
            # Fallback to any matching mean key from LightGBM
            mean_keys = [k for k in cv_results if k.endswith("-mean")]
            metric_match = next((k for k in mean_keys if metric in k), None)
            if metric_match:
                metric_key = metric_match
            elif mean_keys:
                metric_key = mean_keys[0]
            else:
                raise ValueError(f"Missing metric '{metric_key}' in LightGBM CV results.")
        if not cv_results.get(metric_key):
            raise ValueError(f"Missing metric '{metric_key}' in LightGBM CV results.")
        best_score = cv_results[metric_key][-1]
        trial.set_user_attr("best_iteration", len(cv_results[metric_key]))
        return best_score

    storage_uri = None
    storage_path = None
    if optuna_save_storage:
        if optuna_storage_path:
            storage_path = Path(optuna_storage_path)
        elif optuna_artifact_dir is not None:
            storage_path = optuna_artifact_dir / "optuna_study.db"
        else:
            storage_path = Path("optuna_study.db")
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        storage_uri = f"sqlite:///{storage_path}"

    study = optuna.create_study(
        direction=direction,
        pruner=pruner,
        storage=storage_uri,
        study_name=optuna_study_name,
        load_if_exists=optuna_load_if_exists if storage_uri else False,
    )
    study.enqueue_trial({
        "boosting_type": "gbdt",
        "learning_rate": 0.1,
        "num_leaves": 31,
        "min_child_samples": 20,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "lambda_l1": 1e-8,
        "lambda_l2": 1e-8,
        "max_depth": max_depth,
    })
    study.optimize(
        objective_fn,
        n_trials=optuna_trials,
        timeout=optuna_timeout,
    )

    best_params = dict(base_params)
    best_params.update(study.best_trial.params)
    best_iteration = study.best_trial.user_attrs.get("best_iteration", num_boost_round)

    booster = lgb.train(
        best_params,
        dtrain,
        num_boost_round=best_iteration,
    )

    importances = booster.feature_importance(importance_type="split")
    optuna_summary = {
        "optuna": {
            "trials": optuna_trials,
            "timeout_seconds": optuna_timeout,
            "best_score": float(study.best_value),
            "best_params": study.best_trial.params,
            "best_iteration": int(best_iteration),
            "metric": metric,
            "direction": direction,
        }
    }

    if storage_path is not None:
        optuna_summary["optuna"]["storage_path"] = str(storage_path)

    if optuna_export_trials and optuna_artifact_dir is not None:
        optuna_artifact_dir.mkdir(parents=True, exist_ok=True)
        trials_path = optuna_artifact_dir / "optuna_trials.json"

        def _serialize_trial(trial: optuna.Trial) -> Dict[str, Any]:
            return {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "values": trial.values,
                "params": trial.params,
                "distributions": {k: str(v) for k, v in trial.distributions.items()},
                "user_attrs": trial.user_attrs,
                "system_attrs": trial.system_attrs,
                "intermediate_values": {str(k): v for k, v in trial.intermediate_values.items()},
                "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
                "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
            }

        trials_payload = {
            "study": {
                "study_name": study.study_name,
                "direction": direction,
                "best_value": float(study.best_value),
                "best_params": study.best_trial.params,
            },
            "trials": [_serialize_trial(t) for t in study.trials],
        }
        trials_path.write_text(json.dumps(trials_payload, indent=2))
        optuna_summary["optuna"]["trials_export"] = str(trials_path)

    return importances, optuna_summary
