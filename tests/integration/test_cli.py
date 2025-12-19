import json
from pathlib import Path

import pandas as pd

from mlarena.cli.main import main
from mlarena.core.experiment import ExperimentState
from mlarena.core.module import ModuleResult
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


def _write_minimal_preprocess(root: Path):
    """Add a minimal preprocessing template + module for CLI tests."""
    templates_dir = root / "templates" / "preprocess"
    templates_dir.mkdir(parents=True, exist_ok=True)
    (templates_dir / "cli_minimal.yaml").write_text("module: minimal_preprocess\nconfig: {}\n")

    preprocess_dir = root / "code" / "preprocessing"
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    (preprocess_dir / "minimal_preprocess.py").write_text(
        "\n".join(
            [
                "PASS_THROUGH = True",
                "",
                "def fit_transform(train_df, val_df, test_df, config, orig_df=None):",
                "    state = {\"note\": \"pass_through\"}",
                "    if orig_df is not None:",
                "        return train_df, val_df, test_df, orig_df, state",
                "    return train_df, val_df, test_df, state",
            ]
        )
    )


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


def test_cli_rejects_unknown_command(monkeypatch, tmp_path, capsys):
    _ensure_registry()
    monkeypatch.chdir(tmp_path)
    exit_code = main(["preproc", "--project", "demo"])
    assert exit_code == 1
    out = capsys.readouterr().out
    assert "Unknown command: preproc" in out


def test_cli_preprocess_chain_unpacking(monkeypatch, tmp_path):
    _ensure_registry()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("mlarena.cli.main.REPO_ROOT", tmp_path)
    project_root = tmp_path / "projects" / "kaggle" / "demo"
    _write_minimal_project(project_root)
    _write_minimal_preprocess(project_root)

    exit_code = main(["preprocess", "--project", "demo", "preprocess_template=cli_minimal"])
    assert exit_code == 0

    output_path = (
        project_root
        / "experiments"
        / "pre-cli_minimal"
        / "0-cli_minimal"
        / "artifacts"
        / "preprocess"
        / "train_processed.csv"
    )
    assert output_path.exists()


def _mark_setup_completed(project_root: Path) -> None:
    for module_name in ("init", "eda"):
        state = ExperimentState.load_or_create(
            project_root=project_root,
            project_name="demo",
            experiment_id=module_name,
            setup_module_name=module_name,
        )
        state.complete_module(module_name, payload={"status": "completed"})
        state.save()


def test_cli_auto_flow_respects_time_limit(monkeypatch, tmp_path, mock_autogluon):
    _ensure_registry()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("mlarena.cli.main.REPO_ROOT", tmp_path)
    project_root = tmp_path / "projects" / "kaggle" / "demo"
    _write_minimal_project(project_root)
    _write_minimal_preprocess(project_root)
    _mark_setup_completed(project_root)

    def _fake_fetch_score(self):
        self.context.artifact_dir.mkdir(parents=True, exist_ok=True)
        marker = self.context.artifact_dir / "fetch_score.txt"
        marker.write_text("skipped in test")
        return ModuleResult(success=True, payload={"score": 0.0}, artifacts=[marker])

    monkeypatch.setattr("mlarena.modules.fetch_score.FetchScoreModule.execute", _fake_fetch_score)

    exit_code = main(
        [
            "--project",
            "demo",
            "--model-template",
            "baseline",
            "--preprocess-template",
            "cli_minimal",
            "--time-limit",
            "36000",
            "--skip-submit",
            "--wait-seconds",
            "0",
        ]
    )
    assert exit_code == 0

    exp_dirs = list((project_root / "experiments").glob("exp-*"))
    assert exp_dirs
    state = ExperimentState.load_or_create(project_root, "demo", experiment_id=exp_dirs[0].name)
    assert state.modules["model"].payload["time_limit"] == 36000


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

    exit_code = main(["eda", "--project", "demo", "eda.eda_notes=hello"])
    assert exit_code == 0

    exp_dir = project_root / "experiments" / "eda"
    assert exp_dir.exists()
    state = ExperimentState.load_or_create(project_root, "demo", experiment_id="eda", setup_module_name="eda")
    assert state.modules["eda"].status == "completed"

    summary_path = exp_dir / "artifacts" / "eda" / "eda_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary.get("notes") == "hello"
