"""
Diabetes Tail Weighting Preprocessor

Applies higher sample weights to tail samples (high IDs) to address distribution shift.
In many Kaggle competitions, data is sorted chronologically by ID, and test set comes
from a similar time period as the end of train set.

Weighting strategies:
1. Tail weighting: Fixed cutoff ID, samples above get higher weight
2. Percentile weighting: Last X% of samples get higher weight
3. Linear weighting: Linearly increasing weights from start to end

This module saves weights as a single-column CSV that can be loaded by models.

Based on: S5E12 Single CatBoost notebook strategy
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from mlarena.preprocessing.utils import artifacts


def fit_transform(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    test_df: pd.DataFrame,
    config: Dict[str, Any],
    orig_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame, Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Apply tail weighting to training samples based on ID.

    Config keys:
        weighting_strategy (str): "tail" | "percentile" | "linear" (default: "percentile")
        cutoff_id (int): Fixed cutoff ID for "tail" strategy (required if tail)
        tail_percentile (float): Percentile for "percentile" strategy (default: 0.97)
        weight_multiplier (float): Weight for tail samples (default: 5.0)
        weights_output_name (str): Output filename (default: "sample_weights.csv")
        weight_column_name (str): Column header in CSV (default: "sample_weight")

    Returns:
        Tuple of (train_df, val_df, test_df, orig_df, state_dict)

    Note:
        This preprocessor does NOT modify dataframes - it only generates weights CSV.
        Weights path is saved in state_dict for models to load.
    """
    # 1. Extract config
    artifact_dir = Path(config.get("_artifact_dir", "."))
    dataset_meta = config.get("_dataset", {})
    id_col = dataset_meta.get("id_column", "id")

    # Use DataFrame index if ID column not present (removed during preprocessing)
    use_index = id_col not in train_df.columns
    if use_index:
        print(f"⚠️  ID column '{id_col}' not found, using DataFrame index for weighting")
        # Create temporary ID column from index (assuming data is sorted)
        train_df = train_df.copy()
        train_df['__temp_id__'] = train_df.index
        id_col = '__temp_id__'

    # 2. Get configuration
    strategy = config.get("weighting_strategy", "percentile")
    cutoff_id = config.get("cutoff_id")
    tail_percentile = config.get("tail_percentile", 0.97)
    weight_multiplier = config.get("weight_multiplier", 5.0)
    weights_filename = config.get("weights_output_name", "sample_weights.csv")
    weight_col_name = config.get("weight_column_name", "sample_weight")

    # 3. Initialize weights (all 1.0)
    weights = np.ones(len(train_df))

    # 4. Apply weighting strategy
    if strategy == "tail":
        # Fixed cutoff ID
        if cutoff_id is None:
            raise ValueError("cutoff_id is required for 'tail' weighting strategy")

        tail_mask = train_df[id_col] >= cutoff_id
        weights[tail_mask] = weight_multiplier
        tail_count = tail_mask.sum()
        actual_cutoff = cutoff_id

    elif strategy == "percentile":
        # Last X% of samples
        cutoff_id = train_df[id_col].quantile(tail_percentile)
        tail_mask = train_df[id_col] >= cutoff_id
        weights[tail_mask] = weight_multiplier
        tail_count = tail_mask.sum()
        actual_cutoff = cutoff_id

    elif strategy == "linear":
        # Linearly increasing weights
        min_id = train_df[id_col].min()
        max_id = train_df[id_col].max()
        weights = 1.0 + (weight_multiplier - 1.0) * (train_df[id_col] - min_id) / (max_id - min_id)
        tail_count = len(train_df)
        actual_cutoff = None

    else:
        raise ValueError(
            f"Invalid weighting_strategy: '{strategy}'. "
            f"Valid options: 'tail', 'percentile', 'linear'"
        )

    # 5. Create weights DataFrame (single column format)
    weights_df = pd.DataFrame({
        weight_col_name: weights
    })

    # Verify row count matches train
    if len(weights_df) != len(train_df):
        raise ValueError(
            f"Weights row count ({len(weights_df)}) doesn't match "
            f"train row count ({len(train_df)})"
        )

    # 6. Save weights CSV
    weights_path = artifact_dir / weights_filename
    weights_df.to_csv(weights_path, index=False)

    # 7. Build state dict
    state_dict = {
        "version": "1.0",
        "module": "diabetes_weights",
        "weights_path": str(weights_path),
        "weighting_strategy": strategy,
        "cutoff_id": float(actual_cutoff) if actual_cutoff is not None else None,
        "weight_multiplier": weight_multiplier,
        "tail_percentile": tail_percentile if strategy == "percentile" else None,
        "tail_sample_count": int(tail_count) if strategy in ["tail", "percentile"] else len(train_df),
        "tail_percentage": float(tail_count / len(train_df) * 100) if strategy in ["tail", "percentile"] else 100.0,
        "weight_stats": {
            "mean": float(weights.mean()),
            "std": float(weights.std()),
            "min": float(weights.min()),
            "max": float(weights.max()),
        },
    }

    # 8. Save report
    report = {
        "weighting_strategy": strategy,
        "cutoff_id": state_dict["cutoff_id"],
        "weight_multiplier": weight_multiplier,
        "tail_samples": state_dict["tail_sample_count"],
        "tail_percentage": state_dict["tail_percentage"],
        "weight_stats": state_dict["weight_stats"],
        "weights_path": str(weights_path),
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
    }
    artifacts.save_report(report, artifact_dir, "weights_report.json")

    print(f"✅ Tail weighting complete: {strategy} strategy")
    print(f"   Tail samples: {state_dict['tail_sample_count']:,} ({state_dict['tail_percentage']:.2f}%)")
    print(f"   Weight multiplier: {weight_multiplier}x")
    if actual_cutoff is not None:
        print(f"   Cutoff ID: {actual_cutoff:.0f}")

    # Remove temporary ID column if we created it
    if use_index and '__temp_id__' in train_df.columns:
        train_df = train_df.drop(columns=['__temp_id__'])

    # Return unchanged dataframes (weights are saved separately)
    return train_df, val_df, test_df, orig_df, state_dict
