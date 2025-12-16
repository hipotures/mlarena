from pathlib import Path

import pandas as pd
import pytest

from kaggle_tools import submission as submission_mod


def _make_sample(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": [1, 2], "target": [0, 1]}).to_csv(path, index=False)


def test_create_submission_tracks_and_saves(tmp_path, monkeypatch):
    project_root = tmp_path / "proj"
    submissions_dir = project_root / "submissions"
    sample_path = project_root / "data" / "sample_submission.csv"
    _make_sample(sample_path)

    preds = pd.DataFrame({"id": [1, 2], "target": [0, 1]})

    logger_calls = {}
    tracker_calls = {}

    class DummyLogger:
        def __init__(self, root):
            logger_calls["root"] = root

        def log_experiment(self, **kwargs):
            logger_calls["kwargs"] = kwargs
            return {"experiment_id": "exp-1", "git": {"hash": "abc"}}

    class DummyTracker:
        def __init__(self, root):
            tracker_calls["root"] = root

        def add_submission(self, **kwargs):
            tracker_calls["kwargs"] = kwargs
            return {"id": 99, "experiment_id": kwargs.get("experiment_id"), "feature_count": kwargs.get("feature_count")}

    monkeypatch.setattr(submission_mod, "ExperimentLogger", DummyLogger)
    monkeypatch.setattr(submission_mod, "SubmissionsTracker", DummyTracker)

    artifact = submission_mod.create_submission(
        predictions=preds,
        project_root=project_root,
        competition_name="demo-comp",
        submissions_dir=submissions_dir,
        sample_submission_path=sample_path,
        filename_prefix="unit",
        metric_name="auc",
        metric_value=0.9,
        model_name="demo-model",
        local_cv_score=0.8,
        cv_std=0.01,
        notes="test",
        config={"foo": "bar"},
        feature_count=2,
    )

    assert artifact.path.exists()
    saved = pd.read_csv(artifact.path)
    assert list(saved.columns) == ["id", "target"]
    assert tracker_calls["kwargs"]["feature_count"] == 2
    assert tracker_calls["kwargs"]["experiment_id"] == "exp-1"
    assert artifact.tracker_entry["id"] == 99
    assert artifact.experiment["git"]["hash"] == "abc"


def test_create_submission_requires_test_ids_when_no_sample(tmp_path):
    project_root = tmp_path / "proj"
    preds = pd.Series([0.1, 0.2])
    with pytest.raises(ValueError):
        submission_mod.create_submission(
            predictions=preds,
            project_root=project_root,
            competition_name="demo-comp",
            submissions_dir=project_root / "submissions",
            sample_submission_path=project_root / "missing.csv",
        )


def test_validate_submission_rejects_mismatch(tmp_path):
    sample = tmp_path / "sample.csv"
    sub = tmp_path / "sub.csv"
    pd.DataFrame({"id": [1, 2], "target": [0, 1]}).to_csv(sample, index=False)
    pd.DataFrame({"id": [1, 3], "target": [0, 1]}).to_csv(sub, index=False)

    with pytest.raises(AssertionError):
        submission_mod.validate_submission(sub, sample)
