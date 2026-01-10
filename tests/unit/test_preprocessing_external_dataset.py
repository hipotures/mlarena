from pathlib import Path

import pandas as pd
import pytest

from mlarena.defaults.preprocessing import external_dataset


def _cfg(tmp_path, mode="align", mapping=None, source_flag=None):
    return {
        "_artifact_dir": str(tmp_path / "proj" / "experiments" / "chain" / "step" / "artifacts" / "preprocess"),
        "_dataset": {"id_column": "id", "target": "target", "ignored_columns": []},
        "orig_path": "data/external.csv",
        "mode": mode,
        "column_mapping": mapping or {},
        "source_flag": source_flag,
    }


def test_external_dataset_align(tmp_path):
    project_root = Path(tmp_path / "proj")
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": [1, 2], "f1": [0, 1], "target": [0, 1]}).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [3], "f1": [2]}).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [10, 11], "f1": [5, 6], "target": [1, 0]}).to_csv(data_dir / "external.csv", index=False)

    cfg = _cfg(project_root, mode="align")

    train_out, _, test_out, _, orig_out, state = external_dataset.fit_transform(
        train_df=pd.read_csv(data_dir / "train.csv"),
        val_df=None,
        test_df=pd.read_csv(data_dir / "test.csv"),
        config=cfg,
        orig_df=None,
    )

    assert list(train_out.columns) == ["f1", "id", "target"] or set(train_out.columns) == {"id", "f1", "target"}
    assert orig_out is not None
    assert state["alignment"]["matched_columns"] >= 2


def test_external_dataset_union_with_mapping_and_source(tmp_path):
    project_root = Path(tmp_path / "proj")
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": [1], "f1": [0], "target": [0]}).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": [2], "f1": [1]}).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": [10], "feat": [5], "target": [1]}).to_csv(data_dir / "external.csv", index=False)

    cfg = _cfg(project_root, mode="union", mapping={"feat": "f1"}, source_flag="source")

    train_out, _, test_out, _, orig_out, _ = external_dataset.fit_transform(
        train_df=pd.read_csv(data_dir / "train.csv"),
        val_df=None,
        test_df=pd.read_csv(data_dir / "test.csv"),
        config=cfg,
        orig_df=None,
    )

    assert "source" in orig_out.columns
    assert set(train_out.columns) == {"id", "f1", "target", "source"}
    assert set(test_out.columns) == {"id", "f1", "source"}


def test_external_dataset_missing_orig_path(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["orig_path"] = "missing.csv"
    with pytest.raises(FileNotFoundError):
        external_dataset.fit_transform(
            train_df=pd.DataFrame({"id": [1], "target": [0]}),
            val_df=None,
            test_df=pd.DataFrame({"id": [2]}),
            config=cfg,
            orig_df=None,
        )
