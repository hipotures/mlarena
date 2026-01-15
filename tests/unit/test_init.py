import builtins
from pathlib import Path

import pytest

from mlarena.utils.init.core import init_project


def _stub_detection(monkeypatch):
    """Avoid hitting network/AI helpers."""
    monkeypatch.setattr("mlarena.utils.init.core.fetch_kaggle_evaluation", lambda *a, **k: "")
    monkeypatch.setattr(
        "mlarena.utils.init.core.detect_problem_type_and_metric",
        lambda *a, **k: ("binary", "roc_auc", True, {}),
    )
    monkeypatch.setattr("mlarena.utils.init.core.validate_and_fix_metric", lambda *a, **k: ("roc_auc", None))


def test_init_creates_project_structure(tmp_path, monkeypatch):
    calls = {}

    def fake_structure(root: Path, console):
        calls["structure"] = root
        for d in ["data", "code/utils", "templates"]:
            (root / d).mkdir(parents=True, exist_ok=True)

    def fake_copy(root: Path, console):
        calls["templates"] = True

    _stub_detection(monkeypatch)
    monkeypatch.setattr("mlarena.utils.init.core.create_directory_structure", fake_structure)
    monkeypatch.setattr("mlarena.utils.init.core.copy_templates", fake_copy)
    monkeypatch.setattr("mlarena.utils.init.core.download_kaggle_data", lambda *a, **k: True)

    project_root = tmp_path / "projects" / "kaggle" / "demo"
    result = init_project(
        project_root=project_root,
        competition_slug="demo",
        skip_download=True,
        target_column="target",
        problem_type="binary",
        metric="roc_auc",
        id_column="id",
    )

    assert result["success"] is True
    assert (project_root / "data").exists()
    assert calls.get("structure") == project_root
    assert calls.get("templates") is True


def test_init_respects_existing_project(tmp_path, monkeypatch):
    _stub_detection(monkeypatch)
    project_root = tmp_path / "projects" / "kaggle" / "demo"
    (project_root / "code" / "utils").mkdir(parents=True, exist_ok=True)
    (project_root / "code" / "__init__.py").write_text("")
    (project_root / "code" / "utils" / "__init__.py").write_text("")
    (project_root / "code" / "utils" / "config.py").write_text(
        "\n".join(
            [
                "TARGET_COLUMN = 'y'",
                "AUTOGLUON_PROBLEM_TYPE = 'binary'",
                "AUTOGLUON_EVAL_METRIC = 'roc_auc'",
                "ID_COLUMN = 'id'",
            ]
        )
    )

    result = init_project(
        project_root=project_root,
        competition_slug="demo",
        skip_download=True,
        target_column="y",
        problem_type="binary",
        metric="roc_auc",
    )

    assert result["success"] is True
    assert result["stats"].get("already_initialized") is True


def test_init_downloads_data_when_requested(tmp_path, monkeypatch):
    _stub_detection(monkeypatch)
    called = {}

    def fake_download(comp, root, console):
        called["slug"] = comp
        return True

    monkeypatch.setattr("mlarena.utils.init.core.download_kaggle_data", fake_download)
    def fake_structure(root, console):
        for d in ["data", "code/utils", "templates"]:
            (root / d).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("mlarena.utils.init.core.copy_templates", lambda *a, **k: None)
    monkeypatch.setattr("mlarena.utils.init.core.create_directory_structure", fake_structure)

    project_root = tmp_path / "proj"
    res = init_project(
        project_root=project_root,
        competition_slug="my-comp",
        skip_download=False,
        target_column="target",
        problem_type="binary",
        metric="roc_auc",
    )

    assert res["success"] is True
    assert called.get("slug") == "my-comp"
