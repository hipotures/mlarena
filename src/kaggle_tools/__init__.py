"""
Kaggle Tools - Shared utilities for Kaggle competition projects.

This package provides common functionality for:
- Submission creation and validation
- Experiment tracking
- Project initialization
- Automated workflows
"""

__version__ = "0.2.0"

from .config_models import DatasetConfig, Hyperparameters, ModelConfig, SystemConfig
from .submission import create_submission, validate_submission

__all__ = [
    "create_submission",
    "validate_submission",
    "DatasetConfig",
    "Hyperparameters",
    "ModelConfig",
    "SystemConfig",
]
