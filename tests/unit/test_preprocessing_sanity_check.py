import pandas as pd
import pytest

from mlarena.defaults.preprocessing import sanity_check


def _base_config(tmp_path):
    return {
        "_artifact_dir": str(tmp_path),
        "_dataset": {"id_column": "id", "target": "target", "ignored_columns": []},
        "min_unique_fraction": 0.6,
        "max_missing_fraction": 0.5,
        "drop_duplicates": True,
        "ignore_columns": [],
    }


def test_sanity_check_drops_constant_and_missing(tmp_path):
    train = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "target": [0, 1, 0],
            "constant": [1, 1, 1],
            "mostly_nan": [1.0, None, None],
            "keep": [0.1, 0.2, 0.3],
        }
    )
    test = train.drop(columns=["target"]).copy()

    cfg = _base_config(tmp_path)
    train_out, _, test_out, _, state = sanity_check.fit_transform(
        train_df=train, val_df=None, test_df=test, config=cfg, orig_df=None
    )

    assert "constant" not in train_out.columns
    assert "mostly_nan" not in train_out.columns
    assert "keep" in test_out.columns
    assert state["issues_found"]["high_missing_columns"]
    assert state["columns_dropped"]


def test_sanity_check_reserved_columns_raise(tmp_path):
    train = pd.DataFrame({"id": [1], "target": [0], "sample_weight": [1.0]})
    test = train.drop(columns=["target"])
    cfg = _base_config(tmp_path)
    with pytest.raises(RuntimeError):
        sanity_check.fit_transform(train_df=train, val_df=None, test_df=test, config=cfg, orig_df=None)
