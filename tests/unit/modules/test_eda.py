from pathlib import Path

from mlarena.core.experiment import ExperimentState
from mlarena.modules.eda import EDAModule


def test_eda_creates_summary(context_factory, sample_config_with_data):
    state = ExperimentState.load_or_create(sample_config_with_data.PROJECT_ROOT, "demo")
    ctx = context_factory("eda", state=state, config_module=sample_config_with_data)
    module = EDAModule(ctx)
    module.set_invocation_params({"eda_notes": "sanity"})

    result = module.execute()
    assert result.success is True
    assert "summary_file" in result.payload
    assert Path(result.payload["summary_file"]).exists()
    target_info = result.payload.get("target") or {}
    assert target_info.get("unique") == 2


def test_eda_handles_missing_data(context_factory, project_root):
    state = ExperimentState.load_or_create(project_root, "demo-missing")
    ctx = context_factory("eda-missing", state=state, config_module=None)
    module = EDAModule(ctx)
    result = module.execute()

    assert result.success is True
    status = result.payload.get("status")
    if status == "skipped":
        assert result.payload.get("reason") == "train/test missing"
    else:
        # Current behavior: generate minimal artifacts even when data present
        summary_file = Path(result.payload.get("summary_file", ""))
        assert summary_file.exists()
