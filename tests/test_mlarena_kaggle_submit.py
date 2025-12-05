import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlarena.core.experiment import ExperimentState, ModuleEntry  # noqa: E402
from mlarena.core.module import ModuleContext, BaseModule, ModuleResult  # noqa: E402
from mlarena.core.pipeline import PipelineExecutor  # noqa: E402
from mlarena.modules.submit import SubmitModule  # noqa: E402
from mlarena.modules.fetch_score import FetchScoreModule  # noqa: E402


pytestmark = pytest.mark.skipif(
    os.environ.get("MLA_E2E_KAGGLE") != "1",
    reason="Set MLA_E2E_KAGGLE=1 to run real Kaggle submit/fetch test (Titanic). Requires kaggle CLI auth and network.",
)


def _make_titanic_submission(path: Path):
    ids = list(range(892, 892 + 418))
    df = pd.DataFrame({"PassengerId": ids, "Survived": [0] * len(ids)})
    df.to_csv(path, index=False)


class _DummyPredict(BaseModule):
    name = "predict"
    description = "dummy predict"
    dependencies = set()

    def execute(self) -> ModuleResult:
        # Injected later with submission path
        path = getattr(self, "submission_path", None)
        return ModuleResult(success=True, payload={"submission_file": str(path)} if path else {})


def test_kaggle_submit_and_fetch(tmp_path, monkeypatch):
    project = "titanic-test"
    project_root = tmp_path / "projects" / "kaggle" / project
    project_root.mkdir(parents=True, exist_ok=True)

    # Create dummy config for COMPETITION_NAME
    code_dir = project_root / "code" / "utils"
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / "__init__.py").write_text("")
    (code_dir / "config.py").write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "PROJECT_ROOT = Path(__file__).parent.parent.parent",
                "COMPETITION_NAME = 'titanic'",
            ]
        )
    )

    # Prepare submission file
    submission_path = project_root / "dummy_submission.csv"
    _make_titanic_submission(submission_path)

    state = ExperimentState.load_or_create(project_root, project)
    state.modules["predict"] = ModuleEntry(
        name="predict",
        status="completed",
        payload={"submission_file": str(submission_path)},
    )
    state.save()

    # Build module contexts
    submit_ctx = ModuleContext(
        project_name=project,
        project_root=project_root,
        experiment_id=state.experiment_id,
        experiment_dir=state.experiment_dir,
        artifact_dir=state.experiment_dir / "artifacts" / "submit",
        cli_args={},
        state=state,
        config_module=None,
    )
    fetch_ctx = ModuleContext(
        project_name=project,
        project_root=project_root,
        experiment_id=state.experiment_id,
        experiment_dir=state.experiment_dir,
        artifact_dir=state.experiment_dir / "artifacts" / "fetch-score",
        cli_args={},
        state=state,
        config_module=None,
    )

    submit_module = SubmitModule(submit_ctx)
    fetch_module = FetchScoreModule(fetch_ctx)

    dummy_predict = _DummyPredict(
        ModuleContext(
            project_name=project,
            project_root=project_root,
            experiment_id=state.experiment_id,
            experiment_dir=state.experiment_dir,
            artifact_dir=state.experiment_dir / "artifacts" / "predict",
            cli_args={},
            state=state,
            config_module=None,
        )
    )
    dummy_predict.submission_path = submission_path

    modules = {"predict": dummy_predict, "submit": submit_module, "fetch-score": fetch_module}
    executor = PipelineExecutor(modules)

    results = executor.run_module("fetch-score", skip_deps=False, force=True)

    if not results["submit"].success:
        # Likely network/DNS issue in CI sandbox; skip gracefully
        pytest.skip(f"Submit failed (probably offline): {results['submit'].error}")

    assert results["fetch-score"].success
    assert results["fetch-score"].payload.get("latest_submission") is not None
