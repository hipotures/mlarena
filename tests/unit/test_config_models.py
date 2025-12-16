from pathlib import Path

import pytest
from pydantic import ValidationError

from kaggle_tools.config_models import (
    DatasetConfig,
    Hyperparameters,
    ModelConfig,
    OptunaConfig,
    PreprocessingConfig,
    SystemConfig,
)


def test_dataset_config_defaults_and_required():
    cfg = DatasetConfig(
        train_path=Path("/tmp/train.csv"),
        test_path=Path("/tmp/test.csv"),
        target="target",
        id_column="id",
        sample_submission_path=Path("/tmp/sample.csv"),
    )
    assert cfg.ignored_columns == []
    assert cfg.submission_probas is True
    assert cfg.sample_weight_strategy is None

    with pytest.raises(ValidationError):
        DatasetConfig(
            train_path=Path("/tmp/train.csv"),
            test_path=Path("/tmp/test.csv"),
            target="target",
            id_column="id",
            sample_submission_path=None,  # type: ignore[arg-type]
        )


def test_system_and_hyperparameters_defaults():
    sys_cfg = SystemConfig(
        project_root=Path("/tmp/proj"),
        code_dir=Path("/tmp/proj/code"),
        experiment_dir=Path("/tmp/proj/exp"),
        artifact_dir=Path("/tmp/proj/art"),
        model_path=Path("/tmp/proj/model"),
        template="baseline",
        experiment_id="exp-1",
    )
    assert sys_cfg.random_seed == 42
    assert sys_cfg.use_gpu is False

    hyper = Hyperparameters()
    assert hyper.use_gpu is False
    assert hyper.presets is None


def test_preprocessing_and_optuna_defaults():
    prep = PreprocessingConfig()
    assert prep.storage_format == "parquet"
    assert prep.save_transformed_data is True
    assert prep.feat_stage == {}
    assert prep.cv_stage == {}

    opt = OptunaConfig()
    assert opt.direction == "maximize"
    assert opt.n_trials == 50
    assert opt.cv_folds == 5


def test_model_config_validation(monkeypatch):
    sys_cfg = SystemConfig(
        project_root=Path("/tmp/proj"),
        code_dir=Path("/tmp/proj/code"),
        experiment_dir=Path("/tmp/proj/exp"),
        artifact_dir=Path("/tmp/proj/art"),
        model_path=Path("/tmp/proj/model"),
        template="baseline",
        experiment_id="exp-1",
    )
    data_cfg = DatasetConfig(
        train_path=Path("/tmp/train.csv"),
        test_path=Path("/tmp/test.csv"),
        target="target",
        id_column="id",
        sample_submission_path=Path("/tmp/sample.csv"),
    )

    model_cfg = ModelConfig(system=sys_cfg, dataset=data_cfg)
    assert model_cfg.hyperparameters.presets is None
    assert model_cfg.preprocessing.storage_format == "parquet"
    assert model_cfg.optuna.direction == "maximize"

    with pytest.raises(ValidationError):
        ModelConfig(system=sys_cfg)  # type: ignore[arg-type]
