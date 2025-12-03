"""Checks feature-selection behavior in fe_set2."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_fe_set2():
    path = Path(__file__).resolve().parents[1] / "config" / "code" / "preprocessing" / "fe_set2.py"
    spec = importlib.util.spec_from_file_location("fe_set2", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_stage1_feature_selection_drops_constant_redundant_and_correlated():
    fe_set2 = _load_fe_set2()

    rows = 30
    num_a = np.linspace(0, 1, rows)
    num_b = num_a * 2 + 0.01  # Strong positive correlation with num_a

    train_df = pd.DataFrame(
        {
            "id": np.arange(rows),
            "target": [0, 1] * (rows // 2),
            "const_col": 1,
            "cat_anchor": ["x", "y"] * (rows // 2),
            "cat_redundant": ["x", "y"] * (rows // 2),
            "num_a": num_a,
            "num_b": num_b,
        }
    )
    test_df = train_df.drop(columns=["target"]).copy()

    config = {
        "_dataset": {
            "target": "target",
            "id_column": "id",
            "ignored_columns": [],
        },
        "target_encoding_features": False,
        # Disable extra feature generators/target encoding to isolate selection logic
        "fill_missing": True,
        "string_clean": True,
        "quantile_bins": [],
        "uniform_bins": [],
        "log1p_bins": [],
        "round_multipliers": [],
        "digit_mods": [],
        "pairwise_cat_limit": 0,
        "target_encoding": {"enabled": False},
        "categorical_overrides": [],
        "numeric_overrides": [],
        "correlation_threshold": 0.8,
        "rfe_enabled": False,
    }

    train_out, _, test_out, state = fe_set2.fit_transform(train_df, None, test_df, config)

    final_cols = set(state["final_columns"])
    dropped = {"const_col", "cat_redundant", "num_b"}

    # Dropped columns should not appear in selected features
    assert dropped.isdisjoint(final_cols)

    # Non-dropped signal columns should remain
    assert {"cat_anchor", "num_a", "target"} <= final_cols

    # Summary should reflect that we dropped three columns
    summary = state["drop_summary_stage1"]
    assert summary.get("dropped_count") == 3

    # Outputs should not contain dropped columns but should keep id passthrough
    for col in dropped:
        assert col not in train_out.columns
        assert col not in test_out.columns
    assert "id" in train_out.columns
    assert "id" in test_out.columns
