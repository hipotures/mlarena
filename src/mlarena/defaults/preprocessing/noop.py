"""No-op preprocessing sub-module for testing infrastructure."""

from typing import Any, Dict, Tuple
import pandas as pd
from pathlib import Path

from mlarena.preprocessing.utils import report, artifacts, dataframe_utils


def fit_transform(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
    orig_df: pd.DataFrame | None = None,
) -> Tuple[
    pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None, Dict[str, Any]
]:
    """
    Pass-through transformation (does nothing).

    This is a smoke test sub-module to verify the infrastructure works.

    Args:
        train_df: Training data
        val_df: Validation data (can be None)
        test_df: Test data
        config: Configuration dictionary
        orig_df: External dataset (can be None)

    Returns:
        Tuple of (train_df, val_df, test_df, orig_df, state_dict) - unchanged
    """
    artifact_dir = Path(config.get("_artifact_dir", "."))
    submodule_dir = artifacts.get_submodule_artifact_dir(artifact_dir, "noop")

    # Save copies for reporting
    train_df_original = dataframe_utils.copy_dataframe(train_df)
    test_df_original = dataframe_utils.copy_dataframe(test_df)

    # Generate report showing it ran (no transformations applied)
    summary = report.create_preprocessing_report(
        train_before=train_df_original,
        train_after=train_df,
        test_before=test_df_original,
        test_after=test_df,
        config=config,
    )

    # Save report
    artifacts.save_report(summary, submodule_dir, "noop_report.json")

    # Create state dict
    state_dict = {
        "version": "1.0",
        "message": "No-op preprocessing completed successfully",
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
        "train_shape": list(train_df.shape),
        "test_shape": list(test_df.shape),
        "orig_shape": list(orig_df.shape) if orig_df is not None else None,
    }

    return train_df, val_df, test_df, orig_df, state_dict
