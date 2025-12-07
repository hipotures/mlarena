import json
from types import SimpleNamespace

import pandas as pd
import yaml

from mlarena.core.experiment import ExperimentState
from mlarena.modules.model import ModelModule


def test_model_respects_template(context_factory, sample_config_with_data, mock_autogluon):
    """Test that ModelModule correctly loads and applies YAML templates."""
    # Create templates directory and write YAML template
    tpl_dir = sample_config_with_data.PROJECT_ROOT / "templates"
    tpl_dir.mkdir(parents=True, exist_ok=True)

    tpl_data = {
        "templates": {
            "unit-model": {
                "preset": "high",
                "time_limit": 5,
                "use_gpu": 0,
                "hyperparameters": {"GBM": {"max_depth": 2}}
            }
        }
    }
    (tpl_dir / "model.yaml").write_text(yaml.dump(tpl_data))

    state = ExperimentState.load_or_create(sample_config_with_data.PROJECT_ROOT, "demo")
    ctx = context_factory("model", state=state, config_module=sample_config_with_data)
    module = ModelModule(ctx)
    module.set_invocation_params({"model_template": "unit-model"})

    result = module.execute()
    assert result.success is True
    assert result.payload["preset"] == "high"
    assert result.payload["time_limit"] == 5
    assert result.payload["use_gpu"] == 0
    assert result.payload["hyperparameters"]["GBM"]["max_depth"] == 2


def test_model_fails_when_target_missing(context_factory, project_root, mock_autogluon):
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"PassengerId": [1, 2], "feature": [0.1, 0.2]}).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"PassengerId": [3], "feature": [0.5]}).to_csv(data_dir / "test.csv", index=False)

    cfg = SimpleNamespace(
        PROJECT_ROOT=project_root,
        DATA_DIR=data_dir,
        TRAIN_PATH=data_dir / "train.csv",
        TEST_PATH=data_dir / "test.csv",
        TARGET_COLUMN="target",
        ID_COLUMN="PassengerId",
        AUTOGLUON_PROBLEM_TYPE="binary",
        AUTOGLUON_EVAL_METRIC="roc_auc",
        AUTOGLUON_PRESET="medium",
        AUTOGLUON_TIME_LIMIT=5,
        SUBMISSION_PROBAS=True,
        COMPETITION_NAME="demo-comp",
        IGNORED_COLUMNS=[],
    )

    state = ExperimentState.load_or_create(project_root, "demo-missing-target")
    ctx = context_factory("model-missing", state=state, config_module=cfg)
    module = ModelModule(ctx)

    result = module.execute()
    assert result.success is False
    assert "TARGET_COLUMN" in (result.error or "")
