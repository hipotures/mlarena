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


class Hyperparameters(ExtraModel):
    """General-purpose bucket for template-controlled training knobs."""

    presets: Optional[str] = None
    time_limit: Optional[int] = None
    use_gpu: bool = False
    excluded_models: Optional[List[str]] = None


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

    def as_plain_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""

        return self.model_dump(mode="json")
