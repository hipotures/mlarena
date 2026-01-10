"""Subsample training rows with optional validation/evaluation splits.

Features:
- train_fraction: Fraction of data for training (0, 1]
- valid_fraction: Fraction of data for validation/tuning [0, 1)
- eval_fraction: Fraction of data for offline evaluation [0, 1)
- Single shuffle, deterministic 4-way split (train/tuning/eval/discard)
- Validation data saved as tuning_processed.csv.gz
- Evaluation data saved as eval_processed.csv.gz
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

# Signals to the framework that this module modifies data
PASS_THROUGH = False


def fit_transform(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
    orig_df: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None, Dict[str, Any]]:
    """
    Subsample training data with optional validation and evaluation splits.

    Args:
        train_df: Original training data
        val_df: Not used (will be replaced with tuning data)
        test_df: Test data for submission (unchanged)
        config: Configuration dict with:
            - train_fraction: Fraction for training (0, 1]
            - valid_fraction: Fraction for validation/tuning [0, 1)
            - test_fraction: Fraction for offline evaluation [0, 1)
            - random_state: Seed for reproducibility
        orig_df: External/original dataset (passed through)

    Returns:
        Tuple of (train_out, tuning_out, test_df, orig_df, state_dict)
    """
    train_fraction = float(config.get("train_fraction", 1.0))
    valid_fraction = float(config.get("valid_fraction", 0.0))
    eval_fraction = float(config.get("eval_fraction", 0.0))
    random_state = config.get("random_state", 42)

    # Validation
    total_fraction = train_fraction + valid_fraction + eval_fraction
    if total_fraction > 1.0:
        raise ValueError(
            f"train_fraction ({train_fraction}) + valid_fraction ({valid_fraction}) "
            f"+ eval_fraction ({eval_fraction}) must be <= 1.0, got {total_fraction}"
        )
    if train_fraction <= 0 or train_fraction > 1:
        raise ValueError(f"train_fraction must be in (0, 1], got {train_fraction}")
    if valid_fraction < 0 or valid_fraction >= 1:
        raise ValueError(f"valid_fraction must be in [0, 1), got {valid_fraction}")
    if eval_fraction < 0 or eval_fraction >= 1:
        raise ValueError(f"eval_fraction must be in [0, 1), got {eval_fraction}")

    # Single shuffle
    shuffled = train_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    n_total = len(shuffled)

    # 4-way split
    n_train = int(n_total * train_fraction)
    n_valid = int(n_total * valid_fraction)
    n_eval = int(n_total * eval_fraction)

    train_out = shuffled.iloc[:n_train].reset_index(drop=True)

    tuning_out = None
    if valid_fraction > 0:
        tuning_out = shuffled.iloc[n_train:n_train + n_valid].reset_index(drop=True)

    eval_out = None
    if eval_fraction > 0:
        eval_out = shuffled.iloc[n_train + n_valid:n_train + n_valid + n_eval].reset_index(drop=True)

    # Save eval data to artifacts and store path in state for model to access
    # Eval data is NOT transformed by subsequent preprocessing steps (true holdout)
    state = {
        "train_fraction": train_fraction,
        "valid_fraction": valid_fraction,
        "eval_fraction": eval_fraction,
        "random_state": random_state,
        "input_rows": n_total,
        "train_rows": len(train_out),
        "tuning_rows": len(tuning_out) if tuning_out is not None else 0,
        "eval_rows": len(eval_out) if eval_out is not None else 0,
        "eval_cols": eval_out.shape[1] if eval_out is not None else 0,
        "discarded_rows": n_total - len(train_out) - (len(tuning_out) if tuning_out is not None else 0) - (len(eval_out) if eval_out is not None else 0),
    }

    if eval_out is not None:
        # Save to artifacts and store path in state
        artifact_dir = config.get("_artifact_dir")
        if artifact_dir:
            eval_path = Path(artifact_dir) / "eval_processed.csv.gz"
            eval_out.to_csv(eval_path, index=False, compression='infer')
            state["eval_path"] = str(eval_path)

    return (
        train_out,
        tuning_out,  # Return tuning data in val_df slot
        test_df.copy(),  # Original submission test data unchanged
        None if orig_df is None else orig_df.copy(),
        state,
    )


def transform(df: pd.DataFrame, state_dict: Dict[str, Any], config: Dict[str, Any]) -> pd.DataFrame:
    """Transform function for test data - pass through unchanged."""
    return df.copy()
