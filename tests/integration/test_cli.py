import json
from pathlib import Path

import pandas as pd

from mlarena.cli.main import main
from mlarena.core.experiment import ExperimentState
from mlarena.core.registry import ModuleRegistry


def _write_minimal_project(root: Path):
    """Write minimal project structure for testing."""
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

    # Create other required directories
    (root / "experiments").mkdir(parents=True, exist_ok=True)
    (root / "submissions").mkdir(parents=True, exist_ok=True)
    (root / "templates").mkdir(parents=True, exist_ok=True)


def _ensure_registry():
    """Reset and repopulate module registry to include all built-ins."""
    ModuleRegistry.clear()
    ModuleRegistry.discover(force_reload=True)


def test_cli_lists_modules(monkeypatch, tmp_path, capsys):
    _ensure_registry()
    monkeypatch.chdir(tmp_path)
    exit_code = main(["modules", "--project", "demo"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "eda" in out and "model" in out


def test_cli_runs_eda(monkeypatch, tmp_path, capsys):
    _ensure_registry()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("mlarena.cli.main.REPO_ROOT", tmp_path)
    project_root = tmp_path / "projects" / "kaggle" / "demo"
    _write_minimal_project(project_root)

    # Avoid ydata_profiling imports to keep tests warning-free and fast
    def _stub_profile(df, title, html_path, json_path):
        html_path.write_text("<html></html>")
        json_path.write_text("{}")
        return {"summary": {"rows": len(df)}, "html": str(html_path), "json": str(json_path)}

    monkeypatch.setattr("mlarena.modules.eda._safe_profile", _stub_profile)

    exit_code = main(["eda", "--project", "demo", "--eda-notes", "hello"])
    assert exit_code == 0

    exp_dir = project_root / "experiments" / "eda"
    assert exp_dir.exists()
    state = ExperimentState.load_or_create(project_root, "demo", experiment_id="eda", setup_module_name="eda")
    assert state.modules["eda"].status == "completed"

    summary_path = exp_dir / "artifacts" / "eda" / "eda_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary.get("notes") == "hello"
