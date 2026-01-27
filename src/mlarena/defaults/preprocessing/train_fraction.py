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
    eval_df: pd.DataFrame | None = None,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame | None,
    pd.DataFrame,
    pd.DataFrame | None,
    pd.DataFrame | None,
    Dict[str, Any],
]:
    """
    Subsample training data with optional validation and evaluation splits.

    Args:
        train_df: Original training data
        val_df: Not used (will be replaced with tuning data)
        test_df: Test data for submission (unchanged)
        config: Configuration dict with:
            - train_fraction: Fraction for training (0, 1]
            - valid_fraction: Fraction for validation/tuning [0, 1)
            - eval_fraction: Fraction for offline evaluation [0, 1)
            - random_state: Seed for reproducibility
        orig_df: External/original dataset (passed through)
        eval_df: Existing evaluation data (passed through or merged)

    Returns:
        Tuple of (train_out, tuning_out, test_df, eval_out, orig_df, state_dict)
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
    shuffled = train_df.sample(frac=1.0, random_state=random_state)
    n_total = len(shuffled)

    # 4-way split
    n_train = int(n_total * train_fraction)
    n_valid = int(n_total * valid_fraction)
    n_eval = int(n_total * eval_fraction)

    train_out = shuffled.iloc[:n_train]

    tuning_out = None
    if valid_fraction > 0:
        tuning_out = shuffled.iloc[n_train : n_train + n_valid]

    eval_out = None
    if eval_fraction > 0:
        eval_out = shuffled.iloc[n_train + n_valid : n_train + n_valid + n_eval]

    # If eval_df was passed in, merge or replace?
    # Usually train_fraction is the generator. If eval_df exists, we append to it?
    # For now, let's assume we append if both exist, or just use the new one.
    if eval_df is not None:
        if eval_out is not None:
            eval_out = pd.concat([eval_df, eval_out], axis=0)
        else:
            eval_out = eval_df

    # State update
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
        "discarded_rows": n_total
        - len(train_out)
        - (len(tuning_out) if tuning_out is not None else 0)
        - (len(eval_out) if eval_out is not None else 0),
    }

    return (
        train_out,
        tuning_out,  # Return tuning data in val_df slot
        test_df.copy(),  # Original submission test data unchanged
        eval_out,  # Return eval data explicitly
        None if orig_df is None else orig_df.copy(),
        state,
    )


def transform(
    df: pd.DataFrame, state_dict: Dict[str, Any], config: Dict[str, Any]
) -> pd.DataFrame:
    """Transform function for test data - pass through unchanged."""
    return df.copy()
