from pathlib import Path

import pandas as pd

from mlarena.cli.main import main
from mlarena.core.experiment import ExperimentState


def _write_minimal_project(root: Path):
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train = pd.DataFrame({"id": [1, 2], "target": [0, 1]})
    test = pd.DataFrame({"id": [3, 4]})
    train.to_csv(data_dir / "train.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)

    code_dir = root / "code" / "utils"
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


def test_cli_lists_modules(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    exit_code = main(["--project", "demo", "modules"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "eda" in out and "model" in out


def test_cli_runs_eda(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    project_root = tmp_path / "projects" / "kaggle" / "demo"
    _write_minimal_project(project_root)

    exit_code = main(["--project", "demo", "eda", "--eda-notes", "hello"])
    assert exit_code == 0

    exp_root = project_root / "experiments"
    exp_dirs = list(exp_root.glob("exp-*"))
    assert exp_dirs
    state = ExperimentState.load_or_create(exp_root.parent, "demo", experiment_id=exp_dirs[0].name)
    assert state.modules["eda"].status == "completed"

    out = capsys.readouterr().out
    assert "[ok] eda" in out
