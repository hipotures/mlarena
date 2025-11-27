"""
Optuna hyperparameter tuning module.

This module provides:
- StudyManager: Optuna study lifecycle management with SQLite storage
- CVObjective: Cross-validation objective functions with pruning
- Parameter space generators for XGBoost, LightGBM, CatBoost
"""

from kaggle_tools.optuna.study_manager import StudyManager
from kaggle_tools.optuna.objective import CVObjective
from kaggle_tools.optuna.param_spaces import (
    xgboost_param_space,
    lightgbm_param_space,
    catboost_param_space,
    suggest_from_config,
)

__all__ = [
    "StudyManager",
    "CVObjective",
    "xgboost_param_space",
    "lightgbm_param_space",
    "catboost_param_space",
    "suggest_from_config",
]
