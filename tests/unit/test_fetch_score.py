from pathlib import Path

from mlarena.core.experiment import ExperimentState, ModuleEntry
from mlarena.core.module import ModuleContext
from mlarena.modules.fetch_score import FetchScoreModule


def _ctx(project_root, state):
    art = state.experiment_dir / "artifacts" / "fetch-score"
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


def test_fetch_score_uses_placeholder(monkeypatch, tmp_path):
    project_root = tmp_path / "projects" / "kaggle" / "demo"
    state = ExperimentState.load_or_create(project_root, "demo")
    ctx = _ctx(project_root, state)
    module = FetchScoreModule(ctx)
    module.set_invocation_params({"score_placeholder": 0.123})
    monkeypatch.setattr(FetchScoreModule, "_fetch_latest_submission", lambda self, comp: {"publicScore": "0.456"})
    res = module.execute()
    assert res.success is True
    assert res.payload["score"] == 0.456
    assert Path(ctx.artifact_dir / "fetch_score.txt").exists()


def test_fetch_score_handles_missing(monkeypatch, tmp_path):
    project_root = tmp_path / "projects" / "kaggle" / "demo"
    state = ExperimentState.load_or_create(project_root, "demo")
    ctx = _ctx(project_root, state)
    module = FetchScoreModule(ctx)
    monkeypatch.setattr(FetchScoreModule, "_fetch_latest_submission", lambda self, comp: None)
    res = module.execute()
    assert res.success is False
    assert res.payload["score"] is None
