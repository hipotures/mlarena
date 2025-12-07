from pathlib import Path

import pandas as pd

from mlarena.core.experiment import ExperimentState, ModuleEntry
from mlarena.core.module import ModuleContext
from mlarena.modules.stack import StackModule


def _ctx(project_root, state):
    art = state.experiment_dir / "artifacts" / "stack"
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


def test_stack_averages_predictions(tmp_path):
    project_root = tmp_path / "projects" / "kaggle" / "demo"
    project_root.mkdir(parents=True, exist_ok=True)
    pred1 = project_root / "p1.csv"
    pred2 = project_root / "p2.csv"
    pd.DataFrame({"id": [1, 2], "target": [0.2, 0.6]}).to_csv(pred1, index=False)
    pd.DataFrame({"id": [1, 2], "target": [0.4, 0.2]}).to_csv(pred2, index=False)

    state = ExperimentState.load_or_create(project_root, "demo")
    ctx = _ctx(project_root, state)
    module = StackModule(ctx)
    module.set_invocation_params({"prediction_files": [str(pred1), str(pred2)], "id_column": "id", "target_column": "target"})
    res = module.execute()

    assert res.success is True
    stacked = Path(res.payload["stacked_submission"])
    df = pd.read_csv(stacked)
    assert list(df["target"]) == [0.3, 0.4]


def test_stack_fails_when_no_predictions(tmp_path):
    project_root = tmp_path / "projects" / "kaggle" / "demo"
    state = ExperimentState.load_or_create(project_root, "demo")
    ctx = _ctx(project_root, state)
    module = StackModule(ctx)
    res = module.execute()
    assert res.success is False
