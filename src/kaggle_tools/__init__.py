"""
Kaggle Tools - Shared utilities for Kaggle competition projects.

This package provides common functionality for:
- Submission creation and validation
- Experiment tracking
- Project initialization
- Automated workflows
- Feature engineering with data leakage protection
- Hyperparameter tuning with Optuna
- Model ensembling and stacking
"""

__version__ = "0.2.0"

from .config_models import (
    DatasetConfig,
    Hyperparameters,
    ModelConfig,
    SystemConfig,
    PreprocessingConfig,
    OptunaConfig,
)
from .submission import create_submission, validate_submission

# Import submodules for convenient access
from . import preprocessing
from . import optuna
from . import stacking

__all__ = [
    "create_submission",
    "validate_submission",
    "DatasetConfig",
    "Hyperparameters",
    "ModelConfig",
    "SystemConfig",
    "PreprocessingConfig",
    "OptunaConfig",
    "preprocessing",
    "optuna",
    "stacking",
]
