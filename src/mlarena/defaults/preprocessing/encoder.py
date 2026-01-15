"""
Categorical Encoding Sub-Module

Purpose: Universal categorical encoding with multiple strategies
Libraries: category_encoders, sklearn.preprocessing, pandas
Parameters: encoding_method, include_cols, exclude_cols, drop_first, handle_unknown, etc.
"""

from pathlib import Path
from typing import Any, Dict, Tuple, List
import inspect
import warnings

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

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
    Categorical encoding with multiple strategies.

    Args:
        train_df: Training data
        val_df: Validation data (can be None)
        test_df: Test data
        config: Configuration dictionary with keys:
            - _artifact_dir: Path to save artifacts
            - _dataset: {id_column, target, ignored_columns}
            - encoding_method: "none|one_hot|ordinal|target_mean|target_mean_oof|catboost|hashing|count|frequency"
            - include_cols: List of columns to encode (None = all categorical)
            - exclude_cols: List of columns to exclude from encoding
            - drop_first: Drop first category in one-hot (default: False)
            - handle_unknown: "ignore|error|use_encoded_value" (default: "ignore")
            - unknown_value: Value for unknown categories (default: -1)
            - hash_dim: Hash space dimension for hashing method (default: 8)
            - target_encoding_smoothing: Smoothing parameter for target encoding (default: 1.0)
            - target_encoding_min_samples: Min samples for target encoding (default: 1)
            - keep_original: Keep original categorical columns (default: False)
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

    # 2. Validate config
    required_params = []
    optional_params = {
        "encoding_method": "one_hot",
        "include_cols": None,
        "exclude_cols": None,
        "max_cardinality": 50,  # Auto-exclude columns with more unique values
        "drop_first": False,
        "handle_unknown": "ignore",
        "unknown_value": -1,
        "hash_dim": 8,
        "target_encoding_smoothing": 1.0,
        "target_encoding_min_samples": 1,
        # category_encoders options
        "ce_handle_unknown": "value",
        "ce_handle_missing": "value",
        "ce_drop_invariant": False,
        # Leave-one-out / Bayesian encoders
        "loo_sigma": None,
        "m_estimate_m": 1.0,
        "m_estimate_sigma": 0.05,
        "james_stein_sigma": 0.05,
        "glmm_sigma": 0.05,
        "glmm_binomial_target": None,
        "woe_sigma": 0.05,
        "woe_regularization": 1.0,
        # BaseN/Binary encoders
        "base_n": 2,
        # Target-mean OOF encoding (KFold)
        "oof_folds": 5,
        "oof_shuffle": True,
        "oof_random_state": 42,
        "oof_feature_prefix": "mean_",
        # Count/Frequency encoding
        "normalize": False,  # If True, return relative frequency (0-1)
        "add_log": False,    # If True, apply log1p to the count
        "min_count": 1,      # Categories with fewer counts are grouped/ignored (treated as 0)
        "keep_original": False,
        "use_original_features_only": True,
    }
    validation.validate_config(config, required_params, optional_params)

    # Validate encoding_method choice
    validation.validate_choice(
        config["encoding_method"],
        [
            "none",
            "one_hot",
            "ordinal",
            "target_mean",
            "target_mean_oof",
            "catboost",
            "hashing",
            "count",
            "frequency",
            "leave_one_out",
            "m_estimate",
            "james_stein",
            "glmm",
            "woe",
            "binary",
            "base_n",
        ],
        "encoding_method"
    )

    # 3. Create sub-module artifact directory
    submodule_dir = artifacts.get_submodule_artifact_dir(artifact_dir, "encoder")

    # 4. Save original DataFrames for reporting
    train_df_original = dataframe_utils.copy_dataframe(train_df)
    test_df_original = dataframe_utils.copy_dataframe(test_df)

    # 5. Determine columns to encode
    encoding_method = config["encoding_method"]
    if encoding_method == "frequency":
        config["normalize"] = True

    # If method is 'none', skip encoding
    if encoding_method == "none":
        state_dict = {
            "version": "1.0",
            "config": {k: v for k, v in config.items() if not k.startswith("_")},
            "encoding_method": "none",
            "encoded_columns": [],
            "message": "No encoding performed (method=none)"
        }

        # Generate report
        summary = report.create_preprocessing_report(
            train_before=train_df_original,
            train_after=train_df,
            test_before=test_df_original,
            test_after=test_df,
            config=config,
        )
        artifacts.save_report(summary, submodule_dir, "summary.json")

        return train_df, val_df, test_df, eval_df, orig_df, state_dict

    # Get categorical columns
    exclude_cols = [id_column, target_column] + ignored_columns
    if config["exclude_cols"]:
        exclude_cols.extend(config["exclude_cols"])

    all_categorical = dataframe_utils.get_categorical_columns(train_df, exclude=exclude_cols)
    use_orig_only = bool(config.get("use_original_features_only"))
    orig_features = config.get("_original_features") if use_orig_only else None

    # Filter by include_cols if specified
    if config["include_cols"]:
        categorical_cols = [col for col in config["include_cols"] if col in all_categorical]
    else:
        categorical_cols = all_categorical

    if use_orig_only:
        categorical_cols = dataframe_utils.filter_original_columns(categorical_cols, orig_features)

    # Auto-detect and exclude high-cardinality columns (unless explicitly included)
    # This prevents explosion of one-hot features for columns like Name, Ticket, etc.
    max_cardinality = config.get("max_cardinality", 50)  # Default threshold

    if not config["include_cols"] and encoding_method in ["one_hot", "ordinal"]:
        high_cardinality_cols = []
        for col in categorical_cols:
            n_unique = train_df[col].nunique()
            if n_unique > max_cardinality:
                high_cardinality_cols.append((col, n_unique))

        if high_cardinality_cols:
            # Exclude them
            high_card_names = [col for col, _ in high_cardinality_cols]
            categorical_cols = [col for col in categorical_cols if col not in high_card_names]

            # Log warning
            warnings.warn(
                f"Excluded {len(high_cardinality_cols)} high-cardinality columns from {encoding_method} encoding "
                f"(>{max_cardinality} unique values): {high_card_names}. "
                f"Consider using 'target_mean', 'catboost', 'hashing', or 'count' for these columns, "
                f"or increase 'max_cardinality' parameter."
            )

    if not categorical_cols:
        state_dict = {
            "version": "1.0",
            "config": {k: v for k, v in config.items() if not k.startswith("_")},
            "encoding_method": encoding_method,
            "encoded_columns": [],
            "message": "No categorical columns found to encode"
        }

        summary = report.create_preprocessing_report(
            train_before=train_df_original,
            train_after=train_df,
            test_before=test_df_original,
            test_after=test_df,
            config=config,
        )
        artifacts.save_report(summary, submodule_dir, "summary.json")

        return train_df, val_df, test_df, eval_df, orig_df, state_dict

    # 6. Perform encoding
    encoded_columns_info = {}

    if encoding_method == "one_hot":
        train_df, val_df, test_df, eval_df, orig_df, encoder_info = _encode_onehot(
            train_df, val_df, test_df, eval_df, orig_df, categorical_cols, config, submodule_dir
        )
        encoded_columns_info = encoder_info

    elif encoding_method == "ordinal":
        train_df, val_df, test_df, eval_df, orig_df, encoder_info = _encode_ordinal(
            train_df, val_df, test_df, eval_df, orig_df, categorical_cols, config, submodule_dir
        )
        encoded_columns_info = encoder_info

    elif encoding_method == "target_mean":
        if target_column is None:
            raise ValueError("target_mean encoding requires target column")
        train_df, val_df, test_df, eval_df, orig_df, encoder_info = _encode_target_mean(
            train_df, val_df, test_df, eval_df, orig_df, categorical_cols, target_column, config, submodule_dir
        )
        encoded_columns_info = encoder_info

    elif encoding_method == "target_mean_oof":
        if target_column is None:
            raise ValueError("target_mean_oof encoding requires target column")
        train_df, val_df, test_df, eval_df, orig_df, encoder_info = _encode_target_mean_oof(
            train_df, val_df, test_df, eval_df, orig_df, categorical_cols, target_column, config, submodule_dir
        )
        encoded_columns_info = encoder_info

    elif encoding_method == "catboost":
        if target_column is None:
            raise ValueError("catboost encoding requires target column")
        train_df, val_df, test_df, eval_df, orig_df, encoder_info = _encode_catboost(
            train_df, val_df, test_df, eval_df, orig_df, categorical_cols, target_column, config, submodule_dir
        )
        encoded_columns_info = encoder_info

    elif encoding_method == "hashing":
        train_df, val_df, test_df, eval_df, orig_df, encoder_info = _encode_hashing(
            train_df, val_df, test_df, eval_df, orig_df, categorical_cols, config, submodule_dir
        )
        encoded_columns_info = encoder_info

    elif encoding_method in ["count", "frequency"]:
        train_df, val_df, test_df, eval_df, orig_df, encoder_info = _encode_count(
            train_df, val_df, test_df, eval_df, orig_df, categorical_cols, config, submodule_dir
        )
        encoded_columns_info = encoder_info
    elif encoding_method == "leave_one_out":
        if target_column is None:
            raise ValueError("leave_one_out encoding requires target column")
        train_df, val_df, test_df, eval_df, orig_df, encoder_info = _encode_category_encoders(
            method_name="leave_one_out",
            encoder_cls_name="LeaveOneOutEncoder",
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            eval_df=eval_df,
            orig_df=orig_df,
            categorical_cols=categorical_cols,
            target_column=target_column,
            config=config,
            submodule_dir=submodule_dir,
            extra_params={
                "sigma": config.get("loo_sigma"),
            },
        )
        encoded_columns_info = encoder_info
    elif encoding_method == "m_estimate":
        if target_column is None:
            raise ValueError("m_estimate encoding requires target column")
        train_df, val_df, test_df, eval_df, orig_df, encoder_info = _encode_category_encoders(
            method_name="m_estimate",
            encoder_cls_name="MEstimateEncoder",
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            eval_df=eval_df,
            orig_df=orig_df,
            categorical_cols=categorical_cols,
            target_column=target_column,
            config=config,
            submodule_dir=submodule_dir,
            extra_params={
                "m": config.get("m_estimate_m"),
                "sigma": config.get("m_estimate_sigma"),
            },
        )
        encoded_columns_info = encoder_info
    elif encoding_method == "james_stein":
        if target_column is None:
            raise ValueError("james_stein encoding requires target column")
        train_df, val_df, test_df, eval_df, orig_df, encoder_info = _encode_category_encoders(
            method_name="james_stein",
            encoder_cls_name="JamesSteinEncoder",
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            eval_df=eval_df,
            orig_df=orig_df,
            categorical_cols=categorical_cols,
            target_column=target_column,
            config=config,
            submodule_dir=submodule_dir,
            extra_params={
                "sigma": config.get("james_stein_sigma"),
            },
        )
        encoded_columns_info = encoder_info
    elif encoding_method == "glmm":
        if target_column is None:
            raise ValueError("glmm encoding requires target column")
        train_df, val_df, test_df, eval_df, orig_df, encoder_info = _encode_category_encoders(
            method_name="glmm",
            encoder_cls_name="GLMMEncoder",
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            eval_df=eval_df,
            orig_df=orig_df,
            categorical_cols=categorical_cols,
            target_column=target_column,
            config=config,
            submodule_dir=submodule_dir,
            extra_params={
                "sigma": config.get("glmm_sigma"),
                "binomial_target": config.get("glmm_binomial_target"),
            },
        )
        encoded_columns_info = encoder_info
    elif encoding_method == "woe":
        problem_type = dataset_config.get("problem_type", "binary")
        if problem_type != "binary":
            raise ValueError(f"woe encoding requires binary classification, got problem_type={problem_type}")
        if target_column is None:
            raise ValueError("woe encoding requires target column")
        train_df, val_df, test_df, eval_df, orig_df, encoder_info = _encode_category_encoders(
            method_name="woe",
            encoder_cls_name="WOEEncoder",
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            eval_df=eval_df,
            orig_df=orig_df,
            categorical_cols=categorical_cols,
            target_column=target_column,
            config=config,
            submodule_dir=submodule_dir,
            extra_params={
                "sigma": config.get("woe_sigma"),
                "regularization": config.get("woe_regularization"),
            },
        )
        encoded_columns_info = encoder_info
    elif encoding_method == "binary":
        train_df, val_df, test_df, eval_df, orig_df, encoder_info = _encode_category_encoders(
            method_name="binary",
            encoder_cls_name="BinaryEncoder",
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            eval_df=eval_df,
            orig_df=orig_df,
            categorical_cols=categorical_cols,
            target_column=None,
            config=config,
            submodule_dir=submodule_dir,
            extra_params={
                "base": 2,
            },
        )
        encoded_columns_info = encoder_info
    elif encoding_method == "base_n":
        train_df, val_df, test_df, eval_df, orig_df, encoder_info = _encode_category_encoders(
            method_name="base_n",
            encoder_cls_name="BaseNEncoder",
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            eval_df=eval_df,
            orig_df=orig_df,
            categorical_cols=categorical_cols,
            target_column=None,
            config=config,
            submodule_dir=submodule_dir,
            extra_params={
                "base": config.get("base_n", 2),
            },
        )
        encoded_columns_info = encoder_info

    # 7. Optionally drop original categorical columns
    if not config["keep_original"]:
        train_df = dataframe_utils.safe_drop_columns(train_df, categorical_cols)
        test_df = dataframe_utils.safe_drop_columns(test_df, categorical_cols)
        if val_df is not None:
            val_df = dataframe_utils.safe_drop_columns(val_df, categorical_cols)
        if orig_df is not None:
            orig_df = dataframe_utils.safe_drop_columns(orig_df, categorical_cols)
        if eval_df is not None:
            eval_df = dataframe_utils.safe_drop_columns(eval_df, categorical_cols)

    # 8. Generate and save report
    summary = report.create_preprocessing_report(
        train_before=train_df_original,
        train_after=train_df,
        test_before=test_df_original,
        test_after=test_df,
        config=config,
    )
    artifacts.save_report(summary, submodule_dir, "summary.json")

    # 9. Create state dict
    state_dict = {
        "version": "1.0",
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
        "encoding_method": encoding_method,
        "encoded_columns": categorical_cols,
        "encoded_columns_info": encoded_columns_info,
        "keep_original": config["keep_original"],
    }

    return train_df, val_df, test_df, eval_df, orig_df, state_dict


def _encode_onehot(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    eval_df: pd.DataFrame | None,
    orig_df: pd.DataFrame | None,
    categorical_cols: List[str],
    config: Dict[str, Any],
    submodule_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, Dict]:
    """One-hot encoding using sklearn."""
    from sklearn.preprocessing import OneHotEncoder

    handle_unknown = "ignore" if config["handle_unknown"] == "ignore" else "error"

    encoder = OneHotEncoder(
        drop="first" if config["drop_first"] else None,
        handle_unknown=handle_unknown,
        sparse_output=False,
        dtype=np.int8
    )

    # Fit on train
    encoder.fit(train_df[categorical_cols])

    # Transform all datasets
    train_encoded = encoder.transform(train_df[categorical_cols])
    test_encoded = encoder.transform(test_df[categorical_cols])
    val_encoded = encoder.transform(val_df[categorical_cols]) if val_df is not None else None
    eval_encoded = encoder.transform(eval_df[categorical_cols]) if eval_df is not None else None
    orig_encoded = encoder.transform(orig_df[categorical_cols]) if orig_df is not None and all(c in orig_df.columns for c in categorical_cols) else None

    # Generate feature names
    feature_names = encoder.get_feature_names_out(categorical_cols)

    # Add to DataFrames using concat for better performance
    train_encoded_df = pd.DataFrame(train_encoded, columns=feature_names, index=train_df.index)
    train_df = pd.concat([train_df, train_encoded_df], axis=1)

    test_encoded_df = pd.DataFrame(test_encoded, columns=feature_names, index=test_df.index)
    test_df = pd.concat([test_df, test_encoded_df], axis=1)

    if val_df is not None:
        val_encoded_df = pd.DataFrame(val_encoded, columns=feature_names, index=val_df.index)
        val_df = pd.concat([val_df, val_encoded_df], axis=1)

    if eval_df is not None:
        eval_encoded_df = pd.DataFrame(eval_encoded, columns=feature_names, index=eval_df.index)
        eval_df = pd.concat([eval_df, eval_encoded_df], axis=1)

    if orig_df is not None and orig_encoded is not None:
        orig_encoded_df = pd.DataFrame(orig_encoded, columns=feature_names, index=orig_df.index)
        orig_df = pd.concat([orig_df, orig_encoded_df], axis=1)

    # Save encoder
    artifacts.save_fitted_object(encoder, submodule_dir, "onehot_encoder.pkl")

    encoder_info = {
        "n_features_out": len(feature_names),
        "feature_names": list(feature_names),
        "drop_first": config["drop_first"],
    }

    return train_df, val_df, test_df, eval_df, orig_df, encoder_info


def _filter_encoder_kwargs(encoder_cls, params: Dict[str, Any]) -> Dict[str, Any]:
    signature = inspect.signature(encoder_cls.__init__)
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
    if accepts_kwargs:
        return params
    return {k: v for k, v in params.items() if k in signature.parameters}


def _encode_category_encoders(
    method_name: str,
    encoder_cls_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    eval_df: pd.DataFrame | None,
    orig_df: pd.DataFrame | None,
    categorical_cols: List[str],
    target_column: str | None,
    config: Dict[str, Any],
    submodule_dir: Path,
    extra_params: Dict[str, Any] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, Dict]:
    try:
        import category_encoders as ce
    except ImportError:
        warnings.warn(
            f"category_encoders not installed, falling back to target_mean for {method_name} encoding"
        )
        if target_column is None:
            raise ImportError("category_encoders not available and no target column for fallback")
        return _encode_target_mean(
            train_df, val_df, test_df, eval_df, orig_df, categorical_cols, target_column, config, submodule_dir
        )

    encoder_cls = getattr(ce, encoder_cls_name, None)
    if encoder_cls is None:
        raise ValueError(f"category_encoders missing encoder {encoder_cls_name}")

    params = {
        "cols": categorical_cols,
        "handle_unknown": config.get("ce_handle_unknown", "value"),
        "handle_missing": config.get("ce_handle_missing", "value"),
        "drop_invariant": config.get("ce_drop_invariant", False),
    }
    if extra_params:
        params.update(extra_params)
    params = _filter_encoder_kwargs(encoder_cls, params)

    encoder = encoder_cls(**params)

    y = train_df[target_column] if target_column is not None else None
    encoder.fit(train_df[categorical_cols], y)

    train_encoded = encoder.transform(train_df[categorical_cols])
    test_encoded = encoder.transform(test_df[categorical_cols])
    val_encoded = encoder.transform(val_df[categorical_cols]) if val_df is not None else None
    eval_encoded = encoder.transform(eval_df[categorical_cols]) if eval_df is not None else None
    orig_encoded = (
        encoder.transform(orig_df[categorical_cols])
        if orig_df is not None and all(c in orig_df.columns for c in categorical_cols)
        else None
    )

    suffix = f"_{method_name}"
    train_encoded = train_encoded.add_suffix(suffix)
    test_encoded = test_encoded.add_suffix(suffix)
    if val_encoded is not None:
        val_encoded = val_encoded.add_suffix(suffix)
    if eval_encoded is not None:
        eval_encoded = eval_encoded.add_suffix(suffix)
    if orig_encoded is not None:
        orig_encoded = orig_encoded.add_suffix(suffix)

    train_df = pd.concat([train_df, train_encoded], axis=1)
    test_df = pd.concat([test_df, test_encoded], axis=1)
    if val_df is not None and val_encoded is not None:
        val_df = pd.concat([val_df, val_encoded], axis=1)
    if eval_df is not None and eval_encoded is not None:
        eval_df = pd.concat([eval_df, eval_encoded], axis=1)
    if orig_df is not None and orig_encoded is not None:
        orig_df = pd.concat([orig_df, orig_encoded], axis=1)

    artifacts.save_fitted_object(encoder, submodule_dir, f"{method_name}_encoder.pkl")

    encoder_info = {
        "n_features_out": int(train_encoded.shape[1]),
        "feature_names": list(train_encoded.columns),
        "method": method_name,
        "params": {k: v for k, v in params.items() if k != "cols"},
    }

    return train_df, val_df, test_df, eval_df, orig_df, encoder_info


def _encode_ordinal(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    eval_df: pd.DataFrame | None,
    orig_df: pd.DataFrame | None,
    categorical_cols: List[str],
    config: Dict[str, Any],
    submodule_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, Dict]:
    """Ordinal encoding using sklearn."""
    from sklearn.preprocessing import OrdinalEncoder

    handle_unknown = "use_encoded_value" if config["handle_unknown"] != "error" else "error"

    encoder = OrdinalEncoder(
        handle_unknown=handle_unknown,
        unknown_value=config["unknown_value"],
        dtype=np.int32
    )

    # Fit on train
    encoder.fit(train_df[categorical_cols])

    # Transform all datasets
    train_df[categorical_cols] = encoder.transform(train_df[categorical_cols])
    test_df[categorical_cols] = encoder.transform(test_df[categorical_cols])
    if val_df is not None:
        val_df[categorical_cols] = encoder.transform(val_df[categorical_cols])
    if eval_df is not None:
        eval_df[categorical_cols] = encoder.transform(eval_df[categorical_cols])
    if orig_df is not None:
        orig_cols = [c for c in categorical_cols if c in orig_df.columns]
        if orig_cols:
            orig_df[orig_cols] = encoder.transform(orig_df[orig_cols])

    # Save encoder
    artifacts.save_fitted_object(encoder, submodule_dir, "ordinal_encoder.pkl")

    encoder_info = {
        "n_features": len(categorical_cols),
        "categories_per_feature": {
            col: list(cats) for col, cats in zip(categorical_cols, encoder.categories_)
        },
        "unknown_value": config["unknown_value"],
    }

    return train_df, val_df, test_df, eval_df, orig_df, encoder_info


def _encode_target_mean(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    eval_df: pd.DataFrame | None,
    orig_df: pd.DataFrame | None,
    categorical_cols: List[str],
    target_column: str,
    config: Dict[str, Any],
    submodule_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, Dict]:
    """Target mean encoding with smoothing."""

    smoothing = config["target_encoding_smoothing"]
    min_samples = config["target_encoding_min_samples"]

    # Global mean
    global_mean = train_df[target_column].mean()

    encodings = {}

    for col in categorical_cols:
        # Compute smoothed target means
        stats = train_df.groupby(col)[target_column].agg(['mean', 'count'])

        # Smoothing formula: (count * mean + smoothing * global_mean) / (count + smoothing)
        smoothed_means = (
            stats['count'] * stats['mean'] + smoothing * global_mean
        ) / (stats['count'] + smoothing)

        # Filter by min_samples
        mask = stats['count'] >= min_samples
        smoothed_means = smoothed_means[mask]

        encodings[col] = smoothed_means.to_dict()

        # Apply encoding
        train_df[f"{col}_te"] = train_df[col].map(encodings[col]).fillna(global_mean)
        test_df[f"{col}_te"] = test_df[col].map(encodings[col]).fillna(global_mean)
        if val_df is not None:
            val_df[f"{col}_te"] = val_df[col].map(encodings[col]).fillna(global_mean)
        if eval_df is not None:
            eval_df[f"{col}_te"] = eval_df[col].map(encodings[col]).fillna(global_mean)
        if orig_df is not None and col in orig_df.columns:
            orig_df[f"{col}_te"] = orig_df[col].map(encodings[col]).fillna(global_mean)

    # Save encodings
    artifacts.save_report(
        {"encodings": {k: {str(key): val for key, val in v.items()} for k, v in encodings.items()},
         "global_mean": global_mean},
        submodule_dir,
        "target_encodings.json"
    )

    encoder_info = {
        "n_features": len(categorical_cols),
        "encoded_feature_names": [f"{col}_te" for col in categorical_cols],
        "global_mean": global_mean,
        "smoothing": smoothing,
    }

    return train_df, val_df, test_df, eval_df, orig_df, encoder_info


def _encode_target_mean_oof(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    eval_df: pd.DataFrame | None,
    orig_df: pd.DataFrame | None,
    categorical_cols: List[str],
    target_column: str,
    config: Dict[str, Any],
    submodule_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, Dict]:
    """
    KFold out-of-fold target mean encoding.

    - Train: encoded with OOF means (no target leakage within folds)
    - Test/Val/Eval/Orig: encoded with full-train means

    Output feature names are `{oof_feature_prefix}{col}` (default: `mean_{col}`).
    """
    from sklearn.model_selection import KFold

    folds = int(config.get("oof_folds", 5))
    if folds < 2:
        raise ValueError(f"oof_folds must be >= 2, got {folds}")

    shuffle = bool(config.get("oof_shuffle", True))
    random_state = int(config.get("oof_random_state", 42))
    prefix = str(config.get("oof_feature_prefix", "mean_"))

    smoothing = float(config["target_encoding_smoothing"])
    min_samples = int(config["target_encoding_min_samples"])

    global_mean = float(train_df[target_column].mean())
    kf = KFold(n_splits=folds, shuffle=shuffle, random_state=random_state if shuffle else None)

    full_train_encodings: Dict[str, Dict] = {}
    oof_feature_names: List[str] = []

    for col in categorical_cols:
        out_col = f"{prefix}{col}"
        oof_feature_names.append(out_col)

        # OOF for train
        oof_values = np.full(len(train_df), global_mean, dtype=np.float32)
        for tr_idx, va_idx in kf.split(train_df):
            tr = train_df.iloc[tr_idx]

            stats = tr.groupby(col)[target_column].agg(["mean", "count"])
            smoothed = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
            if min_samples > 1:
                smoothed = smoothed[stats["count"] >= min_samples]

            mapping = smoothed.to_dict()
            va = train_df.iloc[va_idx]
            oof_values[va_idx] = va[col].map(mapping).fillna(global_mean).astype(np.float32).values

        train_df[out_col] = oof_values

        # Full-train mapping for inference datasets
        full_stats = train_df.groupby(col)[target_column].agg(["mean", "count"])
        full_smoothed = (full_stats["count"] * full_stats["mean"] + smoothing * global_mean) / (full_stats["count"] + smoothing)
        if min_samples > 1:
            full_smoothed = full_smoothed[full_stats["count"] >= min_samples]

        full_mapping = full_smoothed.to_dict()
        full_train_encodings[col] = full_mapping

        test_df[out_col] = test_df[col].map(full_mapping).fillna(global_mean).astype(np.float32)
        if val_df is not None:
            val_df[out_col] = val_df[col].map(full_mapping).fillna(global_mean).astype(np.float32)
        if eval_df is not None:
            eval_df[out_col] = eval_df[col].map(full_mapping).fillna(global_mean).astype(np.float32)
        if orig_df is not None and col in orig_df.columns:
            orig_df[out_col] = orig_df[col].map(full_mapping).fillna(global_mean).astype(np.float32)

    artifacts.save_report(
        {
            "method": "target_mean_oof",
            "folds": folds,
            "shuffle": shuffle,
            "random_state": random_state,
            "feature_prefix": prefix,
            "smoothing": smoothing,
            "min_samples": min_samples,
            "global_mean": global_mean,
            "encodings_full_train": {k: {str(key): val for key, val in v.items()} for k, v in full_train_encodings.items()},
        },
        submodule_dir,
        "target_encodings_oof.json",
    )

    encoder_info = {
        "n_features": len(categorical_cols),
        "encoded_feature_names": oof_feature_names,
        "global_mean": global_mean,
        "smoothing": smoothing,
        "min_samples": min_samples,
        "oof_folds": folds,
        "oof_shuffle": shuffle,
        "oof_random_state": random_state,
        "feature_prefix": prefix,
    }

    return train_df, val_df, test_df, eval_df, orig_df, encoder_info


def _encode_catboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    eval_df: pd.DataFrame | None,
    orig_df: pd.DataFrame | None,
    categorical_cols: List[str],
    target_column: str,
    config: Dict[str, Any],
    submodule_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, Dict]:
    """CatBoost-style target encoding (ordered target statistics)."""
    try:
        from category_encoders import CatBoostEncoder
    except ImportError:
        warnings.warn("category_encoders not installed, falling back to target_mean encoding")
        return _encode_target_mean(train_df, val_df, test_df, eval_df, orig_df, categorical_cols, target_column, config, submodule_dir)

    encoder = CatBoostEncoder(cols=categorical_cols)

    # Fit on train
    encoder.fit(train_df[categorical_cols], train_df[target_column])

    # Transform
    train_encoded = encoder.transform(train_df[categorical_cols])
    test_encoded = encoder.transform(test_df[categorical_cols])
    val_encoded = encoder.transform(val_df[categorical_cols]) if val_df is not None else None
    eval_encoded = encoder.transform(eval_df[categorical_cols]) if eval_df is not None else None
    orig_encoded = encoder.transform(orig_df[categorical_cols]) if orig_df is not None and all(c in orig_df.columns for c in categorical_cols) else None

    # Rename columns
    encoded_cols = [f"{col}_cb" for col in categorical_cols]
    train_df[encoded_cols] = train_encoded.values
    test_df[encoded_cols] = test_encoded.values
    if val_df is not None:
        val_df[encoded_cols] = val_encoded.values
    if eval_df is not None:
        eval_df[encoded_cols] = eval_encoded.values
    if orig_df is not None and orig_encoded is not None:
        orig_df[encoded_cols] = orig_encoded.values

    # Save encoder
    artifacts.save_fitted_object(encoder, submodule_dir, "catboost_encoder.pkl")

    encoder_info = {
        "n_features": len(categorical_cols),
        "encoded_feature_names": encoded_cols,
    }

    return train_df, val_df, test_df, eval_df, orig_df, encoder_info


def _encode_hashing(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    eval_df: pd.DataFrame | None,
    orig_df: pd.DataFrame | None,
    categorical_cols: List[str],
    config: Dict[str, Any],
    submodule_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, Dict]:
    """Feature hashing."""
    from sklearn.feature_extraction import FeatureHasher

    hash_dim = config["hash_dim"]

    all_encoded_features = []

    for col in categorical_cols:
        hasher = FeatureHasher(n_features=hash_dim, input_type='string')

        # Convert to string and reshape
        train_hashed = hasher.transform([[str(x)] for x in train_df[col]]).toarray()
        test_hashed = hasher.transform([[str(x)] for x in test_df[col]]).toarray()
        val_hashed = hasher.transform([[str(x)] for x in val_df[col]]).toarray() if val_df is not None else None
        eval_hashed = hasher.transform([[str(x)] for x in eval_df[col]]).toarray() if eval_df is not None else None
        orig_hashed = hasher.transform([[str(x)] for x in orig_df[col]]).toarray() if orig_df is not None and col in orig_df.columns else None

        # Generate feature names
        feature_names = [f"{col}_hash_{i}" for i in range(hash_dim)]
        all_encoded_features.extend(feature_names)

        # Add to DataFrames
        train_df[feature_names] = train_hashed
        test_df[feature_names] = test_hashed
        if val_df is not None:
            val_df[feature_names] = val_hashed
        if eval_df is not None:
            eval_df[feature_names] = eval_hashed
        if orig_df is not None and orig_hashed is not None:
            orig_df[feature_names] = orig_hashed

    encoder_info = {
        "n_features": len(all_encoded_features),
        "hash_dim": hash_dim,
        "encoded_feature_names": all_encoded_features,
    }

    return train_df, val_df, test_df, eval_df, orig_df, encoder_info


def _encode_count(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    eval_df: pd.DataFrame | None,
    orig_df: pd.DataFrame | None,
    categorical_cols: List[str],
    config: Dict[str, Any],
    submodule_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, Dict]:
    """Count (Frequency) Encoding."""
    
    normalize = config.get("normalize", False)
    add_log = config.get("add_log", False)
    min_count = config.get("min_count", 1)
    
    mapping_storage = {}
    encoded_feature_names = []
    
    for col in categorical_cols:
        # Calculate counts on TRAIN set only
        if normalize:
            counts = train_df[col].value_counts(normalize=True)
        else:
            counts = train_df[col].value_counts(normalize=False)
            
        # Handle min_count (replace rare with 0 or NaN?)
        # Standard practice: unseen categories get 0 count.
        # If we have min_count, we can filter mapping.
        if min_count > 1 and not normalize:
            # Only keep counts >= min_count
            counts = counts[counts >= min_count]
            
        mapping = counts.to_dict()
        mapping_storage[col] = {str(k): v for k, v in mapping.items()}
        
        # Apply mapping
        suffix = "_freq" if normalize else "_count"
        new_col = f"{col}{suffix}"
        encoded_feature_names.append(new_col)
        
        # Function to apply mapping safely
        def apply_map(series):
            mapped = series.map(mapping).fillna(0)
            if add_log:
                return np.log1p(mapped)
            return mapped

        train_df[new_col] = apply_map(train_df[col])
        test_df[new_col] = apply_map(test_df[col])
        
        if val_df is not None:
            val_df[new_col] = apply_map(val_df[col])
        if eval_df is not None:
            eval_df[new_col] = apply_map(eval_df[col])
        if orig_df is not None and col in orig_df.columns:
            orig_df[new_col] = apply_map(orig_df[col])

    # Save artifacts
    artifacts.save_report(
        {
            "method": "count" if not normalize else "frequency",
            "normalize": normalize,
            "add_log": add_log,
            "min_count": min_count,
            "mappings": mapping_storage
        },
        submodule_dir, 
        "count_encodings.json"
    )
    
    encoder_info = {
        "n_features": len(categorical_cols),
        "encoded_feature_names": encoded_feature_names,
        "normalize": normalize,
        "add_log": add_log
    }
    
    return train_df, val_df, test_df, eval_df, orig_df, encoder_info
