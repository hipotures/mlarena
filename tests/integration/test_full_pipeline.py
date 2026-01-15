from pathlib import Path

import pandas as pd

from mlarena.cli.main import main
from mlarena.core.experiment import ExperimentState
from mlarena.core.registry import ModuleRegistry


def _write_demo_project(project_root: Path):
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train = pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4, 5],
            "feature": [0.1, 0.2, 0.3, 0.4, 0.5],
            "target": [0, 1, 0, 1, 0],
        }
    )
    test = pd.DataFrame({"PassengerId": [6, 7], "feature": [0.25, 0.35]})
    train.to_csv(data_dir / "train.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)

    code_dir = project_root / "code" / "utils"
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / "__init__.py").write_text("")
    (code_dir / "config.py").write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "PROJECT_ROOT = Path(__file__).parent.parent.parent",
                "DATA_DIR = PROJECT_ROOT / 'data'",
                "TRAIN_PATH = DATA_DIR / 'train.csv'",
                "TEST_PATH = DATA_DIR / 'test.csv'",
                "TARGET_COLUMN = 'target'",
                "ID_COLUMN = 'PassengerId'",
                "AUTOGLUON_PROBLEM_TYPE = 'binary'",
                "AUTOGLUON_EVAL_METRIC = 'roc_auc'",
                "AUTOGLUON_PRESET = 'medium'",
                "AUTOGLUON_TIME_LIMIT = 10",
                "IGNORED_COLUMNS = []",
                "SUBMISSION_PROBAS = True",
                "COMPETITION_NAME = 'demo-comp'",
            ]
        )
    )


def test_full_pipeline_predict(monkeypatch, tmp_path, mock_autogluon, capsys):
    ModuleRegistry.clear()
    ModuleRegistry.discover(force_reload=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("mlarena.cli.main.REPO_ROOT", tmp_path)
    project = "demo"
    project_root = tmp_path / "projects" / "kaggle" / project
    _write_demo_project(project_root)

    exit_code = main(["predict", f"project={project}", "predict.predict_suffix=it"])
    assert exit_code == 0

    exp_root = project_root / "experiments"
    exp_dirs = list(exp_root.glob("exp-*"))
    assert exp_dirs
    state = ExperimentState.load_or_create(exp_root.parent, project, experiment_id=exp_dirs[0].name)
    assert state.modules["model"].status == "completed"
    assert state.modules["predict"].status == "completed"

    payload = state.modules["predict"].payload
    assert Path(payload["submission_file"]).exists()
