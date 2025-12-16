import pandas as pd

from mlarena.defaults.preprocessing import encoder


def _config(tmp_path):
    return {
        "_artifact_dir": str(tmp_path),
        "_dataset": {"id_column": "id", "target": "target", "ignored_columns": []},
        "encoding_method": "target_mean",
        "include_cols": ["cat"],
        "target_encoding_smoothing": 1.0,
        "target_encoding_min_samples": 2,
        "keep_original": False,
    }


def test_target_mean_respects_min_samples(tmp_path):
    train = pd.DataFrame(
        {"id": [1, 2, 3, 4], "cat": ["A", "A", "B", "C"], "target": [0, 0, 1, 1]}
    )
    test = pd.DataFrame({"id": [10, 11], "cat": ["A", "B"]})

    cfg = _config(tmp_path)
    # Category C has only 1 sample, should fall back to global mean
    train_out, _, test_out, _, state = encoder.fit_transform(
        train_df=train,
        val_df=None,
        test_df=test,
        config=cfg,
        orig_df=None,
    )

    global_mean = train["target"].mean()
    assert "cat_te" in train_out.columns
    assert state["encoding_method"] == "target_mean"
    assert train_out.loc[train["cat"] == "C", "cat_te"].iloc[0] == global_mean


def test_target_mean_oof_multiple_columns(tmp_path):
    train = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "c1": ["A", "A", "B", "B"],
            "c2": ["X", "Y", "X", "Y"],
            "target": [0, 0, 1, 1],
        }
    )
    test = pd.DataFrame({"id": [10, 11], "c1": ["A", "B"], "c2": ["X", "Y"]})

    cfg = {
        "_artifact_dir": str(tmp_path),
        "_dataset": {"id_column": "id", "target": "target", "ignored_columns": []},
        "encoding_method": "target_mean_oof",
        "include_cols": ["c1", "c2"],
        "keep_original": False,
        "oof_folds": 2,
        "oof_shuffle": False,
        "oof_random_state": 0,
        "oof_feature_prefix": "mean_",
        "target_encoding_smoothing": 0.0,
        "target_encoding_min_samples": 1,
    }

    train_out, _, test_out, _, state = encoder.fit_transform(
        train_df=train,
        val_df=None,
        test_df=test,
        config=cfg,
        orig_df=None,
    )
    assert {"mean_c1", "mean_c2"}.issubset(train_out.columns)
    assert {"mean_c1", "mean_c2"}.issubset(test_out.columns)
    assert state["encoding_method"] == "target_mean_oof"
