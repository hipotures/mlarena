"""
Feature Selection Sub-Module

Purpose: Systematically reduce feature dimensionality using various selection methods
Libraries: sklearn.feature_selection, sklearn.ensemble, lightgbm, xgboost, scipy
Parameters:
    - selection_method: variance|mi|correlation|model_importance|l1|rfe|none
    - n_features: int|float|None (universal feature selector):
        * 0: pass-through (no selection)
        * 0 < n < 1: keep fraction (e.g., 0.3 = keep 30% best)
        * n >= 1: keep exactly n best features
        * n < 0: drop |n| worst features (e.g., -2 = drop 2 worst)
        * None: threshold-only (use min_importance/min_variance)
    - min_variance: float (threshold for variance method)
    - min_importance: float (threshold for importance method)
    - importance_model_type: lgbm|xgb|rf
    - n_estimators: int (for model-based methods)
    - max_depth: int (for model-based methods)
    - random_state: int
    - max_drop_fraction: float (max fraction of features to drop in one step)
    - protect_cb_features: bool (keep numeric features ending with "_cb" regardless of selection)
"""

from pathlib import Path
from typing import Any, Dict, Tuple, List
import warnings

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
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None, Dict[str, Any]]:
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
        orig_df: External dataset (can be None)

    Returns:
        Tuple of (train_df, val_df, test_df, orig_df, state_dict)
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
        "min_importance": 0.001,
        "importance_model_type": "lgbm",
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42,
        "max_drop_fraction": 0.5,
        "protect_cb_features": True,
    }
    validation.validate_config(config, required_params, optional_params)

    # Validate choice parameters
    validation.validate_choice(
        config["selection_method"],
        ["variance", "mi", "correlation", "model_importance", "l1", "rfe", "none"],
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

        # Validate numeric ranges
        validation.validate_numeric_range(
            config["max_drop_fraction"],
            min_value=0.0,
            max_value=1.0,
            param_name="max_drop_fraction"
        )

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
        return train_df, val_df, test_df, orig_df, state_dict

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
        return train_df, val_df, test_df, orig_df, state_dict

    # 7. Prepare data for selection
    if target_column and target_column in train_df.columns:
        X_train = train_df[numeric_cols].values
        y_train = train_df[target_column].values
    else:
        raise ValueError(f"Target column '{target_column}' not found in training data")

    # 8. Perform feature selection
    selected_features, feature_scores = _select_features(
        X_train=X_train,
        y_train=y_train,
        feature_names=numeric_cols,
        method=config["selection_method"],
        select_features=config["n_features"],
        min_variance=config["min_variance"],
        min_importance=config["min_importance"],
        importance_model_type=config["importance_model_type"],
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        random_state=config["random_state"],
        max_drop_fraction=config["max_drop_fraction"],
        problem_type=problem_type,
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
    if orig_df is not None:
        orig_df = orig_df[[col for col in keep_cols if col in orig_df.columns]]

    # 10. Save feature importance/scores report
    feature_report = {
        "method": config["selection_method"],
        "total_features_before": len(numeric_cols),
        "total_features_after": len(selected_features),
        "features_dropped": len(numeric_cols) - len(selected_features),
        "drop_fraction": (len(numeric_cols) - len(selected_features)) / len(numeric_cols) if numeric_cols else 0,
        "selected_features": selected_features,
        "dropped_features": [col for col in numeric_cols if col not in selected_features],
        "feature_scores": {
            feature: float(score) if not np.isnan(score) else None
            for feature, score in zip(numeric_cols, feature_scores)
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
        "feature_scores_summary": {
            "mean": float(np.nanmean(feature_scores)),
            "std": float(np.nanstd(feature_scores)),
            "min": float(np.nanmin(feature_scores)),
            "max": float(np.nanmax(feature_scores)),
        },
    }

    return train_df, val_df, test_df, orig_df, state_dict


def _select_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    method: str,
    select_features: int | float | None,
    min_variance: float,
    min_importance: float,
    importance_model_type: str,
    n_estimators: int,
    max_depth: int,
    random_state: int,
    max_drop_fraction: float,
    problem_type: str,
) -> Tuple[List[str], np.ndarray]:
    """
    Select features using specified method.

    Returns:
        Tuple of (selected_feature_names, feature_scores)
    """
    total_features = X_train.shape[1]

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

    # Initialize scores array
    feature_scores = np.zeros(total_features)

    if method == "variance":
        # Variance threshold
        variances = np.var(X_train, axis=0)
        feature_scores = variances
        selected_mask = variances >= min_variance

        if selected_mask.sum() >= n_features_to_select:
            # Too many (or exact) -> trim to top-K among those above threshold
            candidate_idx = np.where(selected_mask)[0]
            top_idx = candidate_idx[np.argsort(variances[candidate_idx])[::-1][:n_features_to_select]]
            selected_mask = np.zeros(total_features, dtype=bool)
            selected_mask[top_idx] = True
        else:
            # Too few above threshold -> backfill with top-K overall
            top_idx = np.argsort(variances)[::-1][:n_features_to_select]
            selected_mask = np.zeros(total_features, dtype=bool)
            selected_mask[top_idx] = True

    elif method == "mi":
        # Mutual Information
        if problem_type in ["binary", "multiclass"]:
            mi_scores = mutual_info_classif(X_train, y_train, random_state=random_state)
        else:
            mi_scores = mutual_info_regression(X_train, y_train, random_state=random_state)

        feature_scores = mi_scores
        selector = SelectKBest(k=n_features_to_select)
        selector.fit(X_train, y_train)
        selected_mask = selector.get_support()

    elif method == "correlation":
        # Correlation with target
        correlations = np.array([
            abs(pearsonr(X_train[:, i], y_train)[0]) if len(np.unique(X_train[:, i])) > 1 else 0
            for i in range(n_features)
        ])
        feature_scores = correlations
        top_indices = np.argsort(correlations)[::-1][:n_features_to_select]
        selected_mask = np.zeros(n_features, dtype=bool)
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
            try:
                import lightgbm as lgb
                if problem_type in ["binary", "multiclass"]:
                    model = lgb.LGBMClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=random_state,
                        n_jobs=-1,
                        verbosity=-1,
                    )
                else:
                    model = lgb.LGBMRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=random_state,
                        n_jobs=-1,
                        verbosity=-1,
                    )
                model.fit(X_train, y_train)
                importances = model.feature_importances_
            except ImportError:
                warnings.warn("LightGBM not available, falling back to RandomForest")
                return _select_features(
                    X_train, y_train, feature_names, "model_importance",
                    select_features, min_variance, min_importance,
                    "rf", n_estimators, max_depth, random_state,
                    max_drop_fraction, problem_type
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
                    max_drop_fraction, problem_type
                )
        else:
            raise ValueError(f"Unknown importance_model_type: {importance_model_type}")

        feature_scores = importances
        # Select features using threshold + strict top-K cap
        selected_mask = importances >= min_importance

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
        top_indices = np.argsort(importances)[::-1][:n_features_to_select]
        selected_mask = np.zeros(n_features, dtype=bool)
        selected_mask[top_indices] = True

    elif method == "rfe":
        # Recursive Feature Elimination
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

    else:
        raise ValueError(f"Unknown selection method: {method}")

    # Get selected feature names
    selected_features = [feature_names[i] for i in range(total_features) if selected_mask[i]]

    return selected_features, feature_scores
