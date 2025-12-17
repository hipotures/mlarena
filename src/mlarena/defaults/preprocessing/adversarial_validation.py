"""
Adversarial Validation Preprocessing Module

Global preprocessing module that trains binary classifier to detect distribution
shift between train and test sets, then generates sample weights to address the shift.

Features:
- Trains AutoGluon classifier using MLArena model infrastructure
- Generates single-column CSV of sample weights (MLArena format requirement)
- Supports configurable weight transformations
- Returns weights path in state.json for downstream models
- MUST be LAST step in preprocessing chain (no resampling after!)

Weight CSV Format (CRITICAL):
- ONE column only (system uses .iloc[:, 0])
- Column header ignored (can be any name)
- Row order MUST match train_processed.csv
- NO ID column (index-based matching)

Integration with Models:
- Custom models must load weights themselves (not automatic)
- Weights path available in preprocessing state.json
- See docs/submodules/adversarial_validation.md for examples
"""

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

# Module behavior flag: this module does not modify train/test data
# It only generates auxiliary outputs (weights CSV)
PASS_THROUGH = True


def _transform_weights(av_prob: pd.Series, method: str) -> pd.Series:
    """
    Apply configured weight transformation to AV probabilities.

    Args:
        av_prob: Raw probabilities from AV classifier (P(is_test))
        method: Transformation method

    Returns:
        Transformed weights

    Raises:
        ValueError: If method is unknown
    """
    if method == "raw":
        return av_prob

    # Clip to avoid division by zero
    p = av_prob.clip(lower=1e-9, upper=1 - 1e-9)

    if method == "odds_ratio":
        return p / (1 - p)

    elif method == "odds_ratio_capped":
        weights = p / (1 - p)
        return weights.clip(upper=2.0)

    elif method == "odds_ratio_normalized":
        weights = p / (1 - p)
        weights = weights.clip(upper=2.0)
        return weights / weights.mean()

    else:
        valid_methods = ["raw", "odds_ratio", "odds_ratio_capped", "odds_ratio_normalized"]
        raise ValueError(
            f"Invalid weight_transform: {method}. "
            f"Valid options: {valid_methods}"
        )


def fit_transform(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    test_df: pd.DataFrame,
    config: Dict[str, Any],
    orig_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame, Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Generate adversarial validation weights for training data.

    This module:
    1. Drops ID/target columns from train and test
    2. Trains binary classifier to distinguish train from test
    3. Predicts on train data to get AV probabilities
    4. Transforms probabilities to sample weights
    5. Saves weights CSV (single column format)
    6. Returns weights path in state dict

    Args:
        train_df: Training data
        val_df: Validation data (optional, not used)
        test_df: Test data
        config: Configuration dict with keys:
            - presets: AutoGluon preset (default: "medium_quality_faster_train")
            - time_limit: Training time in seconds (default: 600)
            - included_model_types: List of model types or None (default: None)
            - drop_columns: Additional columns to drop (default: [])
            - drop_prefixes: Drop columns whose name starts with any of these prefixes (default: [])
            - weight_transform: Transformation method (default: "odds_ratio_normalized")
            - weights_output_name: Output filename (default: "sample_weights.csv")
            - weight_column_name: Column header in CSV (default: "__sample_weight__")
            - _artifact_dir: Artifact directory (injected by system)
            - _dataset: Dataset metadata (injected by system)
        orig_df: External dataset (optional, not used - pass-through only)

    Returns:
        Tuple of:
            - train_df: Unchanged training data
            - val_df: Unchanged validation data
            - test_df: Unchanged test data
            - orig_df: Unchanged external data (pass-through)
            - state_dict: State with weights_path and statistics

    Raises:
        FileNotFoundError: If av_classifier.py not found
        ValueError: If train/test columns mismatch or row counts mismatch
        RuntimeError: If AV classifier training fails
    """
    # 1. Setup
    artifact_dir = Path(config.get("_artifact_dir", ".")).resolve()
    meta = config.get("_dataset", {})
    id_col = meta.get("id_column", "id")
    target_col = meta.get("target")

    # 2. Determine columns to drop
    drop_cols = set(config.get("drop_columns", []))
    drop_cols.add(id_col)
    if target_col:
        drop_cols.add(target_col)

    drop_prefixes = config.get("drop_prefixes") or []
    if not isinstance(drop_prefixes, list):
        raise ValueError(f"drop_prefixes must be a list of strings, got: {type(drop_prefixes)}")
    for prefix in drop_prefixes:
        if prefix is None:
            continue
        prefix_str = str(prefix)
        if not prefix_str:
            continue
        drop_cols.update([c for c in train_df.columns if str(c).startswith(prefix_str)])
        drop_cols.update([c for c in test_df.columns if str(c).startswith(prefix_str)])

    # 3. Prepare AV datasets
    train_av = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    test_av = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

    # Validate column alignment
    train_cols = set(train_av.columns)
    test_cols = set(test_av.columns)
    if train_cols != test_cols:
        raise ValueError(
            f"Train/test column mismatch:\n"
            f"  Train only: {train_cols - test_cols}\n"
            f"  Test only: {test_cols - train_cols}"
        )

    # Add AV labels
    train_av["__is_test__"] = 0
    test_av["__is_test__"] = 1

    # Concatenate
    av_combined = pd.concat([train_av, test_av], ignore_index=True)

    # 4. Dynamically import av_classifier model (stable path resolution)
    # Use Path(__file__).resolve() to get absolute path of this file
    current_file = Path(__file__).resolve()
    # This file is in: src/mlarena/defaults/preprocessing/adversarial_validation.py
    # Navigate up to repo root: preprocessing -> defaults -> mlarena -> src -> repo_root
    repo_root = current_file.parents[4]

    # Resolve av_classifier model path (check project-local first, then global)
    # Note: This follows same pattern as src/mlarena/modules/model.py:_resolve_model_path
    model_name = "av_classifier"
    project_models_dir = artifact_dir.parents[3] / "code" / "models"  # From artifacts back to project
    project_model_path = project_models_dir / f"{model_name}.py"
    global_model_path = repo_root / "src" / "mlarena" / "defaults" / "models" / f"{model_name}.py"

    # Use global model by default (project-local would take precedence if exists)
    if project_model_path.exists():
        model_path = project_model_path
    elif global_model_path.exists():
        model_path = global_model_path
    else:
        raise FileNotFoundError(
            f"AV classifier model not found. Checked:\n"
            f"  Project-local: {project_model_path}\n"
            f"  Global: {global_model_path}"
        )

    spec = importlib.util.spec_from_file_location(model_name, model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec for {model_path}")

    av_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(av_model)

    # 5. Construct minimal config for model
    av_model_config = {
        "presets": config.get("presets", "medium_quality_faster_train"),
        "time_limit": config.get("time_limit", 600),
        "included_model_types": config.get("included_model_types"),
        "label": "__is_test__",
        "problem_type": "binary",
        "eval_metric": "roc_auc",
        "output_path": artifact_dir / "av_model",
    }

    # 6. Train AV classifier
    try:
        predictor, summary = av_model.train(
            train_df=av_combined,
            val_df=None,
            config=av_model_config,
            artifacts=None,
        )
    except Exception as e:
        raise RuntimeError(f"AV classifier training failed: {e}") from e

    # 7. Predict on train data only
    train_features = train_av.drop(columns=["__is_test__"])
    av_prob = av_model.predict(
        model=predictor,
        test_df=train_features,
        config=av_model_config,
        artifacts=None,
    )

    # 8. Apply weight transformation
    weight_method = config.get("weight_transform", "odds_ratio_normalized")
    weights = _transform_weights(av_prob, weight_method)

    # 9. Create weights DataFrame with SINGLE column (MLArena requirement)
    # CRITICAL: System uses .iloc[:, 0] to extract weights
    # Must have same row count and order as train_processed.csv
    weight_col_name = config.get("weight_column_name", "__sample_weight__")
    weights_df = pd.DataFrame({
        weight_col_name: weights.values
    })

    # Verify row count matches train
    if len(weights_df) != len(train_df):
        raise ValueError(
            f"Weights row count ({len(weights_df)}) doesn't match "
            f"train row count ({len(train_df)})"
        )

    # 10. Save weights CSV (single column, no index)
    weights_filename = config.get("weights_output_name", "sample_weights.csv")
    weights_path = artifact_dir / weights_filename
    weights_df.to_csv(weights_path, index=False)

    # 11. Optionally save raw AV probabilities for debugging
    av_prob_df = pd.DataFrame({
        "av_prob": av_prob.values
    })
    av_prob_path = artifact_dir / "av_predictions.csv"
    av_prob_df.to_csv(av_prob_path, index=False)

    # 12. Build state_dict
    state_dict = {
        "weights_path": str(weights_path),
        "weight_transform": weight_method,
        "av_auc": summary.get("av_auc"),
        "av_rows": summary.get("av_rows"),
        "presets": av_model_config["presets"],
        "time_limit": av_model_config["time_limit"],
        "weight_stats": {
            "mean": float(weights.mean()),
            "std": float(weights.std()),
            "min": float(weights.min()),
            "max": float(weights.max()),
        },
    }

    # 13. Return unchanged dataframes + state
    return train_df, val_df, test_df, orig_df, state_dict
