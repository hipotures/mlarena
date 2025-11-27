"""
Typed configuration objects shared between the new ML runner and model modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ExtraModel(BaseModel):
    """Base Pydantic model that accepts additional keys for forward compatibility."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class SystemConfig(ExtraModel):
    """Paths and runtime metadata injected by the runner."""

    project_root: Path
    code_dir: Path
    experiment_dir: Path
    artifact_dir: Path
    model_path: Path
    template: str
    experiment_id: str
    random_seed: int = 42
    use_gpu: bool = False


class DatasetConfig(ExtraModel):
    """Competition-specific dataset metadata."""

    train_path: Path
    test_path: Path
    target: str
    id_column: str
    ignored_columns: List[str] = Field(default_factory=list)
    sample_submission_path: Path
    problem_type: Optional[str] = None
    metric: Optional[str] = None
    submission_probas: bool = True


class PreprocessingConfig(ExtraModel):
    """Feature engineering configuration."""

    feat_stage: Dict[str, Any] = Field(default_factory=dict)
    cv_stage: Dict[str, Any] = Field(default_factory=dict)
    storage_format: str = "parquet"  # parquet or csv
    save_transformed_data: bool = True
    compression: str = "snappy"


class OptunaConfig(ExtraModel):
    """Optuna hyperparameter tuning configuration."""

    storage: str = ""  # "sqlite:///experiments/{exp_id}/artifacts/optuna/study.db"
    study_name: str = ""
    direction: str = "maximize"  # or "minimize"
    n_trials: int = 50
    timeout: Optional[int] = None  # seconds
    n_jobs: int = 1
    cv_folds: int = 5
    early_stopping_rounds: int = 50
    sampler: str = "TPESampler"
    pruner: str = "MedianPruner"
    param_space: Dict[str, Dict[str, List]] = Field(default_factory=dict)


class Hyperparameters(ExtraModel):
    """General-purpose bucket for template-controlled training knobs."""

    presets: Optional[str] = None
    time_limit: Optional[int] = None
    use_gpu: bool = False
    excluded_models: Optional[List[str]] = None

    # NEW: Optuna preset reference
    preset: Optional[str] = None  # References configs/presets/{preset}.yaml


class ModelConfig(ExtraModel):
    """
    Structured configuration object passed to every model.

    Additional model-specific settings can be stored under the ``model`` key or as
    extra attributes (thanks to ``extra="allow"``).
    """

    system: SystemConfig
    dataset: DatasetConfig
    hyperparameters: Hyperparameters = Field(default_factory=Hyperparameters)
    model: Dict[str, Any] = Field(default_factory=dict)

    # NEW fields for Optuna system
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    optuna: OptunaConfig = Field(default_factory=OptunaConfig)

    def as_plain_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""

        return self.model_dump(mode="json")
