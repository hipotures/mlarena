from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from mlarena.core.experiment import ExperimentState, ModuleEntry
from mlarena.modules.predict import PredictModule


def _make_context(state, project_root, config_module, module_name="predict"):
    artifact_dir = state.experiment_dir / "artifacts" / module_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    from mlarena.core.module import ModuleContext

    return ModuleContext(
        project_name="demo",
        project_root=project_root,
        experiment_id=state.experiment_id,
        experiment_dir=state.experiment_dir,
        artifact_dir=artifact_dir,
        cli_args={},
        state=state,
        config_module=config_module,
    )


def test_predict_generates_submission(context_factory, sample_config_with_data, monkeypatch, mock_autogluon):
    state = ExperimentState.load_or_create(sample_config_with_data.PROJECT_ROOT, "demo")
    model_artifact = state.experiment_dir / "artifacts" / "model"
    model_artifact.mkdir(parents=True, exist_ok=True)
    # Dummy file so path exists
    (model_artifact / "predictor.pkl").write_text("stub")

    state.modules["model"] = ModuleEntry(
        name="model",
        status="completed",
        payload={"model_artifact": str(model_artifact)},
    )
    state.save()

    # Dummy predictor injected directly
    class StubPredictor:
        problem_type = "binary"

        def predict(self, df):
            return pd.Series([1] * len(df))

        def predict_proba(self, df, as_multiclass=False):
            return pd.Series([0.7] * len(df))

    monkeypatch.setattr("mlarena.modules.predict._load_predictor", lambda path: StubPredictor())

    ctx = _make_context(state, sample_config_with_data.PROJECT_ROOT, sample_config_with_data)
    module = PredictModule(ctx)
    module.set_invocation_params({"predict_suffix": "unit"})

    res = module.execute()
    assert res.success is True
    submission = Path(res.payload["submission_file"])
    assert submission.exists()
    df = pd.read_csv(submission)
    assert list(df.columns) == ["PassengerId", "target"]
    assert len(df) == 2


def test_predict_fails_without_model(context_factory, project_root):
    state = ExperimentState.load_or_create(project_root, "no-model")
    cfg = SimpleNamespace(
        PROJECT_ROOT=project_root,
        DATA_DIR=project_root / "data",
        TRAIN_PATH=project_root / "data" / "train.csv",
        TEST_PATH=project_root / "data" / "test.csv",
        TARGET_COLUMN="target",
        ID_COLUMN="id",
        SUBMISSION_PROBAS=False,
    )
    ctx = _make_context(state, project_root, cfg)
    module = PredictModule(ctx)

    res = module.execute()
    assert res.success is False
    assert "model" in (res.error or "")
