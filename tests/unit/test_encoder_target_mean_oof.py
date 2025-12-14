import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_encoder_module():
    repo_root = Path(__file__).resolve().parents[2]
    encoder_path = repo_root / "config" / "code" / "preprocessing" / "encoder.py"
    spec = importlib.util.spec_from_file_location("encoder", encoder_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_target_mean_oof_no_leakage(tmp_path):
    encoder = _load_encoder_module()

    train_df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "cat": ["A", "A", "B", "B"],
            "target": [0, 0, 1, 1],
        }
    )
    test_df = pd.DataFrame({"id": [10, 11], "cat": ["A", "B"]})

    cfg = {
        "_artifact_dir": str(tmp_path),
        "_dataset": {"id_column": "id", "target": "target", "ignored_columns": []},
        "encoding_method": "target_mean_oof",
        "include_cols": ["cat"],
        "keep_original": True,
        "oof_folds": 2,
        "oof_shuffle": False,  # deterministic split for this test
        "oof_random_state": 42,
        "oof_feature_prefix": "mean_",
        "target_encoding_smoothing": 0.0,
        "target_encoding_min_samples": 1,
    }

    train_out, _, test_out, _, state = encoder.fit_transform(
        train_df=train_df,
        val_df=None,
        test_df=test_df,
        config=cfg,
        orig_df=None,
    )

    assert state["encoding_method"] == "target_mean_oof"
    assert "mean_cat" in train_out.columns
    assert "mean_cat" in test_out.columns

    # With 2-fold no-shuffle KFold, each category is only present in the opposite fold's train split,
    # so every train row should fall back to global mean (0.5) rather than leaking its own label.
    assert np.allclose(train_out["mean_cat"].values, 0.5)
    assert np.allclose(test_out["mean_cat"].values, np.array([0.0, 1.0]))

    # Artifact should be written
    assert (tmp_path / "submodules" / "encoder" / "target_encodings_oof.json").exists()
