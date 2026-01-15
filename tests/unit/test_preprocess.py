import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from mlarena.core.experiment import ExperimentState
from mlarena.core.module import ModuleContext
from mlarena.modules.preprocess import PreprocessModule


def _make_project(tmp_path):
    project_root = tmp_path / "projects" / "kaggle" / "demo"
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "feat": [1.0, None, 3.0],
            "dropme": [0, 0, 1],
            "target": [0, 1, 0],
        }
    )
    test = pd.DataFrame({"id": [10, 11], "feat": [None, 2.5], "dropme": [1, 0]})
    train.to_csv(data_dir / "train.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)

    (project_root / "code" / "utils").mkdir(parents=True, exist_ok=True)
    (project_root / "code" / "utils" / "__init__.py").write_text("")
    (project_root / "code" / "utils" / "config.py").write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "PROJECT_ROOT = Path(__file__).parent.parent.parent",
                "DATA_DIR = PROJECT_ROOT / 'data'",
                "TRAIN_PATH = DATA_DIR / 'train.csv'",
                "TEST_PATH = DATA_DIR / 'test.csv'",
                "TARGET_COLUMN = 'target'",
                "ID_COLUMN = 'id'",
                "AUTOGLUON_PROBLEM_TYPE = 'binary'",
                "AUTOGLUON_EVAL_METRIC = 'roc_auc'",
                "AUTOGLUON_PRESET = 'medium'",
                "AUTOGLUON_TIME_LIMIT = 5",
                "IGNORED_COLUMNS = []",
                "SUBMISSION_PROBAS = True",
                "COMPETITION_NAME = 'demo-comp'",
            ]
        )
    )
    return project_root


def _patch_templates(monkeypatch, template_dict):
    class StubLoader:
        def __init__(self, *a, **k):
            pass

        def load(self, name):
            return template_dict.get(name, {})

    # Patch TemplateLoader where it is defined to cover local imports
    monkeypatch.setattr("mlarena.core.config.TemplateLoader", StubLoader)
    # Patch template_loader.load_templates used inside execute/can_run
    monkeypatch.setitem(
        __import__("sys").modules,
        "template_loader",
        type("TL", (), {"load_templates": lambda *_a, **_k: (template_dict, [])}),
    )


def _make_context(project_root: Path, state: ExperimentState, module_name: str = "preprocess") -> ModuleContext:
    artifact_dir = state.experiment_dir / "artifacts" / module_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return ModuleContext(
        project_name="demo",
        project_root=project_root,
        experiment_id=state.experiment_id,
        experiment_dir=state.experiment_dir,
        artifact_dir=artifact_dir,
        cli_args={},
        state=state,
        config_module=None,
    )


def test_preprocess_applies_template_and_saves(monkeypatch, tmp_path):
    template = {
        "basic": {
            "drop_columns": ["dropme"],
            "fillna": {"feat": 0},
        }
    }
    _patch_templates(monkeypatch, template)
    project_root = _make_project(tmp_path)
    state = ExperimentState.load_or_create(project_root, "demo")
    ctx = _make_context(project_root, state)

    cfg = SimpleNamespace(
        PROJECT_ROOT=project_root,
        DATA_DIR=project_root / "data",
        TRAIN_PATH=project_root / "data" / "train.csv",
        TEST_PATH=project_root / "data" / "test.csv",
        TARGET_COLUMN="target",
        ID_COLUMN="id",
        AUTOGLUON_PROBLEM_TYPE="binary",
        AUTOGLUON_EVAL_METRIC="roc_auc",
        AUTOGLUON_PRESET="medium",
        AUTOGLUON_TIME_LIMIT=5,
        IGNORED_COLUMNS=[],
        SUBMISSION_PROBAS=True,
        COMPETITION_NAME="demo-comp",
    )
    monkeypatch.setattr("mlarena.modules.preprocess.load_project_config", lambda root: cfg)
    monkeypatch.setattr("mlarena.modules.preprocess.data_paths", lambda config: (config.TRAIN_PATH, config.TEST_PATH))
    real_read = pd.read_csv

    def fake_read(path, *a, **k):
        p = Path(path)
        if p.name == "train.csv":
            return pd.DataFrame(
                {"id": [1, 2, 3], "feat": [1.0, None, 3.0], "dropme": [0, 0, 1], "target": [0, 1, 0]}
            )
        if p.name == "test.csv":
            return pd.DataFrame({"id": [10, 11], "feat": [None, 2.5], "dropme": [1, 0]})
        return real_read(path, *a, **k)

    monkeypatch.setattr("pandas.read_csv", fake_read)

    module = PreprocessModule(ctx)
    module.set_invocation_params({"preprocess_template": "basic"})
    result = module.execute()

    assert result.success is True
    train_path = Path(result.payload["train_processed"])
    test_path = Path(result.payload["test_processed"])
    assert train_path.exists() and test_path.exists()

    train_df = pd.read_csv(train_path)
    assert "dropme" not in train_df.columns
    assert train_df["feat"].isna().sum() == 0


def test_preprocess_uses_cache(monkeypatch, tmp_path):
    template = {"cached": {}}
    _patch_templates(monkeypatch, template)
    project_root = _make_project(tmp_path)
    state = ExperimentState.load_or_create(project_root, "demo")
    ctx = _make_context(project_root, state)

    # Seed cached outputs
    processed_dir = ctx.artifact_dir
    processed_dir.mkdir(parents=True, exist_ok=True)
    for name in ["train_processed.csv.gz", "test_processed.csv.gz"]:
        (processed_dir / name).write_text("id,feat\n1,0")
    module = PreprocessModule(ctx)
    module.set_invocation_params({"preprocess_template": "cached", "cache": True})
    res = module.execute()
    assert res.payload.get("cached") is True


def test_preprocess_requires_template(monkeypatch, tmp_path):
    _patch_templates(monkeypatch, {})
    project_root = _make_project(tmp_path)
    state = ExperimentState.load_or_create(project_root, "demo")
    ctx = _make_context(project_root, state)
    module = PreprocessModule(ctx)

    with pytest.raises(ValueError):
        module.execute()
