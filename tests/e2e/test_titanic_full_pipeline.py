import os
import shutil
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from mlarena.cli.main import main
from mlarena.core.registry import ModuleRegistry


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.skipif(os.environ.get("MLA_E2E") != "1", reason="Set MLA_E2E=1 to run Titanic e2e"),
]


@pytest.fixture(scope="module")
def titanic_e2e_setup(tmp_path_factory):
    source = Path("/home/xai/ml/kaggle/projects/kaggle/titanic")
    if not source.exists():
        pytest.skip("Titanic project not available")

    root = tmp_path_factory.mktemp("e2e_titanic")
    target = root / "projects" / "kaggle" / "titanic"
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns(
            "experiments",
            "submissions",
            "AutogluonModels*",
            "*.pkl",
            "*.log",
        ),
    )
    return root, target


def _reset_artifacts(project_root: Path) -> None:
    shutil.rmtree(project_root / "experiments", ignore_errors=True)
    shutil.rmtree(project_root / "submissions", ignore_errors=True)
    (project_root / "experiments").mkdir(parents=True, exist_ok=True)
    (project_root / "submissions").mkdir(parents=True, exist_ok=True)


def test_titanic_predict_e2e(monkeypatch, titanic_e2e_setup, mock_autogluon):
    repo_root, project_root = titanic_e2e_setup
    _reset_artifacts(project_root)
    ModuleRegistry.clear()
    ModuleRegistry.discover(force_reload=True)
    monkeypatch.chdir(repo_root)
    monkeypatch.setattr("mlarena.cli.main.REPO_ROOT", repo_root)

    def _fake_profile(df, title, html_path, json_path):
        html_path.write_text("<html></html>")
        json_path.write_text("{}")
        return {"summary": {"rows": len(df)}, "html": str(html_path), "json": str(json_path)}

    monkeypatch.setattr("mlarena.modules.eda._safe_profile", _fake_profile)

    # Keep model fast; avoid global templates
    monkeypatch.setattr(
        "mlarena.modules.model.TemplateLoader.load",
        lambda self, name: {"preset": "fast", "time_limit": 1, "use_gpu": 0},
    )

    eda_code = main(["eda", "--project", "titanic", "--eda-notes", "e2e-fast"])
    assert eda_code == 0
    exit_code = main(["predict", "--project", "titanic", "--predict-suffix", "e2e", "--force"])
    assert exit_code == 0

    submissions = list((project_root / "experiments").glob("exp-*/artifacts/predict/submission-e2e.csv"))
    assert submissions, "prediction artifact not found"
    df = pd.read_csv(submissions[0])
    assert df.shape[0] == 418


def test_titanic_preprocess_then_model(monkeypatch, titanic_e2e_setup, mock_autogluon):
    repo_root, project_root = titanic_e2e_setup
    _reset_artifacts(project_root)
    ModuleRegistry.clear()
    ModuleRegistry.discover(force_reload=True)
    monkeypatch.chdir(repo_root)
    monkeypatch.setattr("mlarena.cli.main.REPO_ROOT", repo_root)

    def _fake_templates(template_type, project_root, suppress_warnings=True):
        assert template_type == "preprocess"
        return {"baseline": {"fillna": {"Age": 0}, "drop_columns": ["Cabin"]}}, []

    # Stub template_loader module so preprocess avoids production templates
    fake_module = types.SimpleNamespace(load_templates=_fake_templates)
    monkeypatch.setitem(sys.modules, "template_loader", fake_module)

    # Model template stub
    monkeypatch.setattr(
        "mlarena.modules.model.TemplateLoader.load",
        lambda self, name: {"preset": "fast", "time_limit": 1, "use_gpu": 0},
    )

    exit_code = main(
        ["preprocess", "--project", "titanic", "--preprocess-template", "baseline", "--force"]
    )
    assert exit_code == 0

    processed_train = project_root / "experiments" / "pre-baseline" / "artifacts" / "preprocess" / "train_processed.csv"
    processed_test = project_root / "experiments" / "pre-baseline" / "artifacts" / "preprocess" / "test_processed.csv"
    assert processed_train.exists() and processed_test.exists()

    # Use fixed experiment id so predict can reuse state
    exp_id = "exp-pre-e2e"
    exit_code = main(
        [
            "model",
            "--project",
            "titanic",
            "--preprocess-template",
            "baseline",
            "--model-template",
            "dev-gpu",
            "--experiment-id",
            exp_id,
            "--force",
        ]
    )
    assert exit_code == 0

    # Predict on the same experiment to validate end-to-end artifacts
    pred_exit = main(
        [
            "predict",
            "--project",
            "titanic",
            "--experiment-id",
            exp_id,
            "--predict-suffix",
            "pre-e2e",
            "--force",
        ]
    )
    assert pred_exit == 0

    model_runs = sorted(project_root.glob(f"experiments/{exp_id}/artifacts/model/train_used.csv"))
    assert model_runs, "train_used.csv missing after model run"
    train_df = pd.read_csv(model_runs[-1])
    assert "Cabin" not in train_df.columns
    if "Age" in train_df.columns:
        assert train_df["Age"].isna().sum() == 0

    submission_files = list((project_root / "experiments" / exp_id / "artifacts" / "predict").glob("submission-pre-e2e.csv"))
    assert submission_files, "predict artifact missing for preprocess pipeline"
    sub_df = pd.read_csv(submission_files[0])
    assert sub_df.shape[0] == 418


def test_titanic_reproducibility(monkeypatch, titanic_e2e_setup, mock_autogluon):
    repo_root, project_root = titanic_e2e_setup
    _reset_artifacts(project_root)
    ModuleRegistry.clear()
    ModuleRegistry.discover(force_reload=True)
    monkeypatch.chdir(repo_root)
    monkeypatch.setattr("mlarena.cli.main.REPO_ROOT", repo_root)
    monkeypatch.setattr(
        "mlarena.modules.model.TemplateLoader.load",
        lambda self, name: {"preset": "fast", "time_limit": 1, "use_gpu": 0},
    )

    # Ensure leaderboard contains score_val so local_cv is set
    def _leaderboard_with_score(self, silent=True):
        return pd.DataFrame([{"model": "best", "score_val": 0.5}])

    monkeypatch.setattr(mock_autogluon, "leaderboard", _leaderboard_with_score, raising=False)

    first = main(["model", "--project", "titanic", "--model-template", "dev-gpu", "--experiment-id", "exp-repro-1", "--force"])
    second = main(["model", "--project", "titanic", "--model-template", "dev-gpu", "--experiment-id", "exp-repro-2", "--force"])
    assert first == 0 and second == 0

    state1 = project_root / "experiments" / "exp-repro-1" / "state.json"
    state2 = project_root / "experiments" / "exp-repro-2" / "state.json"
    assert state1.exists() and state2.exists()

    import json

    cv1 = json.loads(state1.read_text())["modules"]["model"]["payload"]["local_cv"]
    cv2 = json.loads(state2.read_text())["modules"]["model"]["payload"]["local_cv"]
    assert cv1 == cv2 == 0.5


def test_titanic_validation_errors(monkeypatch, titanic_e2e_setup, mock_autogluon):
    repo_root, project_root = titanic_e2e_setup
    _reset_artifacts(project_root)
    ModuleRegistry.clear()
    ModuleRegistry.discover(force_reload=True)
    monkeypatch.chdir(repo_root)
    monkeypatch.setattr("mlarena.cli.main.REPO_ROOT", repo_root)

    def _missing_template(self, name):
        raise FileNotFoundError("template missing for test")

    monkeypatch.setattr("mlarena.modules.model.TemplateLoader.load", _missing_template)

    exit_code = main(["predict", "--project", "titanic", "--predict-suffix", "err", "--force"])
    assert exit_code == 1

    failed_states = list(project_root.glob("experiments/exp-*/state.json"))
    assert failed_states, "expected experiment state to be written on failure"
    last_state = failed_states[-1].read_text()
    assert "failed" in last_state or "template missing" in last_state
