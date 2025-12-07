from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from mlarena.core.experiment import ExperimentState
from mlarena.core.module import ModuleContext
from mlarena.modules.tune import TuneModule


class StubStudy:
    def __init__(self):
        self.best_params = {"lr": 0.1}
        self.best_value = 0.9

    def optimize(self, fn, n_trials, show_progress_bar=False):
        # Do not call fn to avoid AutoGluon dependency
        self.best_value = 0.9


class StubOptuna:
    def create_study(self, direction="maximize"):
        return StubStudy()


def _patch_optuna(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "optuna", StubOptuna())


def _patch_templates(monkeypatch):
    monkeypatch.setattr("mlarena.modules.tune.TemplateLoader", lambda *a, **k: type("TL", (), {"load": lambda self, name: {"search_space": {}}})())


def _make_project(tmp_path):
    project_root = tmp_path / "projects" / "kaggle" / "demo"
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": [1, 2, 3], "f": [0.1, 0.2, 0.3], "target": [0, 1, 0]}).to_csv(data_dir / "train.csv", index=False)
    (project_root / "code" / "utils").mkdir(parents=True, exist_ok=True)
    (project_root / "code" / "utils" / "__init__.py").write_text("")
    (project_root / "code" / "utils" / "config.py").write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "PROJECT_ROOT = Path(__file__).parent.parent.parent",
                "DATA_DIR = PROJECT_ROOT / 'data'",
                "TRAIN_PATH = DATA_DIR / 'train.csv'",
                "TARGET_COLUMN = 'target'",
                "AUTOGLUON_PROBLEM_TYPE = 'binary'",
                "AUTOGLUON_EVAL_METRIC = 'roc_auc'",
            ]
        )
    )
    return project_root


def _ctx(project_root, state):
    art = state.experiment_dir / "artifacts" / "tune"
    art.mkdir(parents=True, exist_ok=True)
    return ModuleContext(
        project_name="demo",
        project_root=project_root,
        experiment_id=state.experiment_id,
        experiment_dir=state.experiment_dir,
        artifact_dir=art,
        cli_args={},
        state=state,
        config_module=None,
    )


def test_tune_records_best_params(monkeypatch, tmp_path):
    _patch_optuna(monkeypatch)
    _patch_templates(monkeypatch)
    project_root = _make_project(tmp_path)
    state = ExperimentState.load_or_create(project_root, "demo")
    ctx = _ctx(project_root, state)
    module = TuneModule(ctx)
    module.set_invocation_params({"tune_template": "dummy", "n_trials": 1, "time_limit": 1})

    res = module.execute()
    assert res.success is True
    assert res.payload["best_params"] == {"lr": 0.1}
    assert Path(ctx.artifact_dir / "tune_result.json").exists()


def test_tune_fails_without_target(monkeypatch, tmp_path):
    _patch_optuna(monkeypatch)
    _patch_templates(monkeypatch)
    project_root = tmp_path / "projects" / "kaggle" / "demo"
    (project_root / "data").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": [1, 2], "f": [0, 1]}).to_csv(project_root / "data" / "train.csv", index=False)
    state = ExperimentState.load_or_create(project_root, "demo")
    cfg = SimpleNamespace(
        PROJECT_ROOT=project_root,
        DATA_DIR=project_root / "data",
        TRAIN_PATH=project_root / "data" / "train.csv",
        TARGET_COLUMN="target",
        AUTOGLUON_PROBLEM_TYPE="binary",
        AUTOGLUON_EVAL_METRIC="roc_auc",
    )
    monkeypatch.setattr("mlarena.modules.tune.load_project_config", lambda root: cfg)
    monkeypatch.setattr("mlarena.modules.tune.data_paths", lambda config: (config.TRAIN_PATH, None))
    real_read = pd.read_csv

    def fake_read(path, *a, **k):
        p = Path(path)
        if p.name == "train.csv":
            return pd.DataFrame({"id": [1, 2], "f": [0, 1]})
        return real_read(path, *a, **k)

    monkeypatch.setattr("pandas.read_csv", fake_read)
    ctx = _ctx(project_root, state)
    module = TuneModule(ctx)
    res = module.execute()
    assert res.success is False
