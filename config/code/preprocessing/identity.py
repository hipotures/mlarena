"""Identity preprocessing that returns data unchanged.

Provides fit_transform/transform interface expected by the new preprocess templates.
"""
from __future__ import annotations
import pandas as pd
from typing import Any, Dict, Tuple


def fit_transform(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
    orig_df: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None, Dict[str, Any]]:
    state = {"version": "1.0", "transform": "identity"}
    return (
        train_df.copy(),
        None if val_df is None else val_df.copy(),
        test_df.copy(),
        None if orig_df is None else orig_df.copy(),
        state
    )


def transform(df: pd.DataFrame, state_dict: Dict[str, Any], config: Dict[str, Any]) -> pd.DataFrame:
    return df.copy()
