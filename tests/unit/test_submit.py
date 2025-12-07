import builtins
from pathlib import Path

import pandas as pd

from mlarena.core.experiment import ExperimentState, ModuleEntry
from mlarena.core.module import ModuleContext
from mlarena.modules.submit import SubmitModule


def _ctx(project_root, state):
    art = state.experiment_dir / "artifacts" / "submit"
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


def _make_config(project_root):
    code_dir = project_root / "code" / "utils"
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / "__init__.py").write_text("")
    (code_dir / "config.py").write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "PROJECT_ROOT = Path(__file__).parent.parent.parent",
                "COMPETITION_NAME = 'demo-comp'",
                "DATA_DIR = PROJECT_ROOT / 'data'",
                "TRAIN_PATH = DATA_DIR / 'train.csv'",
                "TARGET_COLUMN = 'y'",
                "ID_COLUMN = 'id'",
                "IGNORED_COLUMNS = []",
            ]
        )
    )


def test_submit_calls_kaggle(monkeypatch, tmp_path):
    project_root = tmp_path / "projects" / "kaggle" / "demo"
    project_root.mkdir(parents=True, exist_ok=True)
    _make_config(project_root)

    # dummy train for feature count
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": [1], "feat": [0], "y": [1]}).to_csv(data_dir / "train.csv", index=False)

    submission_file = project_root / "sub.csv"
    pd.DataFrame({"id": [1], "y": [0.5]}).to_csv(submission_file, index=False)

    state = ExperimentState.load_or_create(project_root, "demo")
    state.modules["predict"] = ModuleEntry(name="predict", status="completed", payload={"submission_file": str(submission_file)})
    state.save()

    ctx = _ctx(project_root, state)
    module = SubmitModule(ctx)
    module.set_invocation_params({"auto_submit": True})

    called = {}
    monkeypatch.setattr("subprocess.check_call", lambda args: called.setdefault("args", args))

    res = module.execute()
    assert res.success is True
    assert called["args"][0:4] == ["kaggle", "competitions", "submit", "-c"]


def test_submit_skips_when_flag(monkeypatch, tmp_path):
    project_root = tmp_path / "projects" / "kaggle" / "demo"
    project_root.mkdir(parents=True, exist_ok=True)
    submission_file = project_root / "sub.csv"
    submission_file.write_text("id,y\n1,0")
    state = ExperimentState.load_or_create(project_root, "demo")
    state.modules["predict"] = ModuleEntry(name="predict", status="completed", payload={"submission_file": str(submission_file)})
    ctx = _ctx(project_root, state)
    module = SubmitModule(ctx)
    module.set_invocation_params({"skip_submit": True})
    res = module.execute()
    assert res.success is True
    assert res.payload["submitted"] is False


def test_submit_requires_predict(tmp_path):
    project_root = tmp_path / "projects" / "kaggle" / "demo"
    state = ExperimentState.load_or_create(project_root, "demo")
    ctx = _ctx(project_root, state)
    module = SubmitModule(ctx)
    res = module.execute()
    assert res.success is False
