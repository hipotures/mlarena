from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml

from mlarena.core.experiment import ExperimentState
from mlarena.modules.model import ModelModule


def test_model_respects_template(context_factory, sample_config_with_data, mock_autogluon):
    """Test that ModelModule correctly loads and applies YAML templates."""
    # Create templates directory and write YAML template
    tpl_dir = sample_config_with_data.PROJECT_ROOT / "templates" / "model"
    tpl_dir.mkdir(parents=True, exist_ok=True)

    tpl_data = {
        "model": "autogluon_baseline",
        "config": {
            "preset": "high",
            "time_limit": 5,
            "use_gpu": 0,
            "hyperparameters": {"GBM": {"max_depth": 2}},
        },
    }
    (tpl_dir / "unit-model.yaml").write_text(yaml.dump(tpl_data))

    state = ExperimentState.load_or_create(sample_config_with_data.PROJECT_ROOT, "demo")
    ctx = context_factory("model", state=state, config_module=sample_config_with_data)
    module = ModelModule(ctx)
    module.set_invocation_params({"model_template": "unit-model"})

    result = module.execute()
    assert result.success is True
    assert result.payload["preset"] == "high"
    assert result.payload["time_limit"] == 5
    assert result.payload["use_gpu"] == 0
    model_path = Path(result.payload["model_artifact"])
    predictor = mock_autogluon.store[str(model_path)]
    assert predictor.hyperparameters["GBM"]["max_depth"] == 2


def test_model_hpo_template(context_factory, sample_config_with_data, mock_autogluon):
    """Ensure HPO preset + search space settings are forwarded to AutoGluon."""
    tpl_dir = sample_config_with_data.PROJECT_ROOT / "templates" / "model"
    hpo_dir = tpl_dir / "hpo"
    hpo_dir.mkdir(parents=True, exist_ok=True)

    # Minimal HPO preset
    hpo_preset = {
        "hpo": {"num_trials": 1, "scheduler": "local", "searcher": "auto"},
        "search_space": {"GBM": {"max_depth": [3, 5, "int"]}},
    }
    (hpo_dir / "mini.yaml").write_text(yaml.safe_dump(hpo_preset))

    tpl_data = {
        "model": "autogluon_baseline",
        "hpo_preset": "mini",
        # Extend/override search space at template level to ensure merge works
        "search_space": {"GBM": {"num_leaves": [20, 40, "int"]}},
        "config": {"preset": "medium", "time_limit": 5, "use_gpu": 0},
    }
    (tpl_dir / "unit-hpo.yaml").write_text(yaml.safe_dump(tpl_data))

    state = ExperimentState.load_or_create(sample_config_with_data.PROJECT_ROOT, "demo")
    ctx = context_factory("model", state=state, config_module=sample_config_with_data)
    module = ModelModule(ctx)
    module.set_invocation_params({"model_template": "unit-hpo"})

    result = module.execute()
    assert result.success is True

    model_path = Path(result.payload["model_artifact"])
    predictor = mock_autogluon.store[str(model_path)]
    # Search space merged under GBM
    assert "GBM" in predictor.hyperparameters
    assert "num_leaves" in predictor.hyperparameters["GBM"]
    # HPO kwargs forwarded
    hpo_kwargs = predictor.extra_fit_kwargs.get("hyperparameter_tune_kwargs", {})
    assert hpo_kwargs.get("num_trials") == 1


def test_model_respects_sample_weights(context_factory, sample_config_with_data, mock_autogluon):
    """Ensure sample_weight_strategy column is passed through to AutoGluon."""
    tpl_dir = sample_config_with_data.PROJECT_ROOT / "templates" / "model"
    tpl_dir.mkdir(parents=True, exist_ok=True)
    tpl_data = {
        "model": "autogluon_baseline",
        "config": {
            "preset": "medium",
            "time_limit": 5,
            "use_gpu": 0,
            "sample_weight_strategy": "sample_weight",
        },
    }
    (tpl_dir / "unit-weights.yaml").write_text(yaml.safe_dump(tpl_data))

    # Add weights column to training data
    train_df = pd.read_csv(sample_config_with_data.TRAIN_PATH)
    train_df["sample_weight"] = [1.0, 2.0, 1.5, 1.0]
    train_df.to_csv(sample_config_with_data.TRAIN_PATH, index=False)

    state = ExperimentState.load_or_create(sample_config_with_data.PROJECT_ROOT, "demo")
    ctx = context_factory("model", state=state, config_module=sample_config_with_data)
    module = ModelModule(ctx)
    module.set_invocation_params({"model_template": "unit-weights"})

    result = module.execute()
    assert result.success is True

    model_path = Path(result.payload["model_artifact"])
    predictor = mock_autogluon.store[str(model_path)]
    assert predictor.sample_weight == "sample_weight"
    assert "sample_weight" in predictor.df.columns
    assert predictor.df["sample_weight"].tolist() == [1.0, 2.0, 1.5, 1.0]


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
