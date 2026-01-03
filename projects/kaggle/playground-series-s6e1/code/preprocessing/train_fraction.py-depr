"""Subsample training rows to a fixed fraction; keep test data unchanged."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd

# Signals to the framework that this module modifies data.
PASS_THROUGH = False


def fit_transform(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
    orig_df: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None, Dict[str, Any]]:
    fraction = float(config.get("train_fraction", 1.0))
    random_state = config.get("random_state", 42)

    if fraction <= 0 or fraction > 1:
        raise ValueError(f"train_fraction must be in (0, 1], got {fraction}")

    if fraction < 1.0:
        sampled_train = train_df.sample(frac=fraction, random_state=random_state).reset_index(drop=True)
    else:
        sampled_train = train_df.copy()

    state = {
        "train_fraction": fraction,
        "random_state": random_state,
        "input_rows": int(len(train_df)),
        "output_rows": int(len(sampled_train)),
    }

    return (
        sampled_train,
        None if val_df is None else val_df.copy(),
        test_df.copy(),
        None if orig_df is None else orig_df.copy(),
        state,
    )


def transform(df: pd.DataFrame, state_dict: Dict[str, Any], config: Dict[str, Any]) -> pd.DataFrame:
    return df.copy()
