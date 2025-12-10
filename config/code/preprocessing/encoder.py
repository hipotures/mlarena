"""
Categorical Encoding Sub-Module

Purpose: Universal categorical encoding with multiple strategies
Libraries: category_encoders, sklearn.preprocessing, pandas
Parameters: encoding_method, include_cols, exclude_cols, drop_first, handle_unknown, etc.
"""

from pathlib import Path
from typing import Any, Dict, Tuple, List
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
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, Dict[str, Any]]:
    """
    Categorical encoding with multiple strategies.

    Args:
        train_df: Training data
        val_df: Validation data (can be None)
        test_df: Test data
        config: Configuration dictionary with keys:
            - _artifact_dir: Path to save artifacts
            - _dataset: {id_column, target, ignored_columns}
            - encoding_method: "none|one_hot|ordinal|target_mean|catboost|hashing"
            - include_cols: List of columns to encode (None = all categorical)
            - exclude_cols: List of columns to exclude from encoding
            - drop_first: Drop first category in one-hot (default: False)
            - handle_unknown: "ignore|error|use_encoded_value" (default: "ignore")
            - unknown_value: Value for unknown categories (default: -1)
            - hash_dim: Hash space dimension for hashing method (default: 8)
            - target_encoding_smoothing: Smoothing parameter for target encoding (default: 1.0)
            - target_encoding_min_samples: Min samples for target encoding (default: 1)
            - keep_original: Keep original categorical columns (default: False)

    Returns:
        Tuple of (train_df, val_df, test_df, state_dict)
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
        "keep_original": False,
    }
    validation.validate_config(config, required_params, optional_params)

    # Validate encoding_method choice
    validation.validate_choice(
        config["encoding_method"],
        ["none", "one_hot", "ordinal", "target_mean", "catboost", "hashing"],
        "encoding_method"
    )

    # 3. Create sub-module artifact directory
    submodule_dir = artifacts.get_submodule_artifact_dir(artifact_dir, "encoder")

    # 4. Save original DataFrames for reporting
    train_df_original = dataframe_utils.copy_dataframe(train_df)
    test_df_original = dataframe_utils.copy_dataframe(test_df)

    # 5. Determine columns to encode
    encoding_method = config["encoding_method"]

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

        return train_df, val_df, test_df, state_dict

    # Get categorical columns
    exclude_cols = [id_column, target_column] + ignored_columns
    if config["exclude_cols"]:
        exclude_cols.extend(config["exclude_cols"])

    all_categorical = dataframe_utils.get_categorical_columns(train_df, exclude=exclude_cols)

    # Filter by include_cols if specified
    if config["include_cols"]:
        categorical_cols = [col for col in config["include_cols"] if col in all_categorical]
    else:
        categorical_cols = all_categorical

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
                f"Consider using 'target_mean', 'catboost', or 'hashing' for these columns, "
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

        return train_df, val_df, test_df, state_dict

    # 6. Perform encoding
    encoded_columns_info = {}

    if encoding_method == "one_hot":
        train_df, val_df, test_df, encoder_info = _encode_onehot(
            train_df, val_df, test_df, categorical_cols, config, submodule_dir
        )
        encoded_columns_info = encoder_info

    elif encoding_method == "ordinal":
        train_df, val_df, test_df, encoder_info = _encode_ordinal(
            train_df, val_df, test_df, categorical_cols, config, submodule_dir
        )
        encoded_columns_info = encoder_info

    elif encoding_method == "target_mean":
        if target_column is None:
            raise ValueError("target_mean encoding requires target column")
        train_df, val_df, test_df, encoder_info = _encode_target_mean(
            train_df, val_df, test_df, categorical_cols, target_column, config, submodule_dir
        )
        encoded_columns_info = encoder_info

    elif encoding_method == "catboost":
        if target_column is None:
            raise ValueError("catboost encoding requires target column")
        train_df, val_df, test_df, encoder_info = _encode_catboost(
            train_df, val_df, test_df, categorical_cols, target_column, config, submodule_dir
        )
        encoded_columns_info = encoder_info

    elif encoding_method == "hashing":
        train_df, val_df, test_df, encoder_info = _encode_hashing(
            train_df, val_df, test_df, categorical_cols, config, submodule_dir
        )
        encoded_columns_info = encoder_info

    # 7. Optionally drop original categorical columns
    if not config["keep_original"]:
        train_df = dataframe_utils.safe_drop_columns(train_df, categorical_cols)
        test_df = dataframe_utils.safe_drop_columns(test_df, categorical_cols)
        if val_df is not None:
            val_df = dataframe_utils.safe_drop_columns(val_df, categorical_cols)

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

    return train_df, val_df, test_df, state_dict


def _encode_onehot(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    categorical_cols: List[str],
    config: Dict[str, Any],
    submodule_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, Dict]:
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

    # Save encoder
    artifacts.save_fitted_object(encoder, submodule_dir, "onehot_encoder.pkl")

    encoder_info = {
        "n_features_out": len(feature_names),
        "feature_names": list(feature_names),
        "drop_first": config["drop_first"],
    }

    return train_df, val_df, test_df, encoder_info


def _encode_ordinal(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    categorical_cols: List[str],
    config: Dict[str, Any],
    submodule_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, Dict]:
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

    # Save encoder
    artifacts.save_fitted_object(encoder, submodule_dir, "ordinal_encoder.pkl")

    encoder_info = {
        "n_features": len(categorical_cols),
        "categories_per_feature": {
            col: list(cats) for col, cats in zip(categorical_cols, encoder.categories_)
        },
        "unknown_value": config["unknown_value"],
    }

    return train_df, val_df, test_df, encoder_info


def _encode_target_mean(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    categorical_cols: List[str],
    target_column: str,
    config: Dict[str, Any],
    submodule_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, Dict]:
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

    return train_df, val_df, test_df, encoder_info


def _encode_catboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    categorical_cols: List[str],
    target_column: str,
    config: Dict[str, Any],
    submodule_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, Dict]:
    """CatBoost-style target encoding (ordered target statistics)."""
    try:
        from category_encoders import CatBoostEncoder
    except ImportError:
        warnings.warn("category_encoders not installed, falling back to target_mean encoding")
        return _encode_target_mean(train_df, val_df, test_df, categorical_cols, target_column, config, submodule_dir)

    encoder = CatBoostEncoder(cols=categorical_cols)

    # Fit on train
    encoder.fit(train_df[categorical_cols], train_df[target_column])

    # Transform
    train_encoded = encoder.transform(train_df[categorical_cols])
    test_encoded = encoder.transform(test_df[categorical_cols])
    val_encoded = encoder.transform(val_df[categorical_cols]) if val_df is not None else None

    # Rename columns
    encoded_cols = [f"{col}_cb" for col in categorical_cols]
    train_df[encoded_cols] = train_encoded.values
    test_df[encoded_cols] = test_encoded.values
    if val_df is not None:
        val_df[encoded_cols] = val_encoded.values

    # Save encoder
    artifacts.save_fitted_object(encoder, submodule_dir, "catboost_encoder.pkl")

    encoder_info = {
        "n_features": len(categorical_cols),
        "encoded_feature_names": encoded_cols,
    }

    return train_df, val_df, test_df, encoder_info


def _encode_hashing(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    categorical_cols: List[str],
    config: Dict[str, Any],
    submodule_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, Dict]:
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

        # Generate feature names
        feature_names = [f"{col}_hash_{i}" for i in range(hash_dim)]
        all_encoded_features.extend(feature_names)

        # Add to DataFrames
        train_df[feature_names] = train_hashed
        test_df[feature_names] = test_hashed
        if val_df is not None:
            val_df[feature_names] = val_hashed

    encoder_info = {
        "n_features": len(all_encoded_features),
        "hash_dim": hash_dim,
        "encoded_feature_names": all_encoded_features,
    }

    return train_df, val_df, test_df, encoder_info
