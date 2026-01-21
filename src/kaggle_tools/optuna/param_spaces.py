"""
Model-specific parameter space definitions for Optuna.

This module provides parameter space generators for XGBoost, LightGBM, and CatBoost
that can be configured via YAML presets.
"""

from typing import Any, Dict, List

import optuna


def suggest_from_config(trial: optuna.Trial, param_name: str, param_spec: List) -> Any:
    """
    Parse parameter specification from YAML config and suggest value.

    Format: [min, max, type] or [min, max, type, step]
    Type: "int", "float", "log", "categorical"

    Examples:
        [0.001, 0.3, "log"] -> trial.suggest_float(param_name, 0.001, 0.3, log=True)
        [3, 10, "int"] -> trial.suggest_int(param_name, 3, 10)
        [0.5, 1.0, "float"] -> trial.suggest_float(param_name, 0.5, 1.0)
        ["gbtree", "dart"] -> trial.suggest_categorical(param_name, ["gbtree", "dart"])

    Args:
        trial: Optuna trial object
        param_name: Parameter name
        param_spec: Parameter specification from config

    Returns:
        Suggested parameter value

    Raises:
        ValueError: If param_spec format is invalid
    """
    if not isinstance(param_spec, list):
        raise ValueError(f"Parameter spec must be a list, got: {type(param_spec)}")

    # Categorical parameter (list of choices)
    if len(param_spec) == 2 and isinstance(param_spec[0], str):
        return trial.suggest_categorical(param_name, param_spec)

    if len(param_spec) < 3:
        raise ValueError(
            f"Parameter spec must be [min, max, type] or [choices], got: {param_spec}"
        )

    min_val, max_val, param_type = param_spec[:3]
    step = param_spec[3] if len(param_spec) > 3 else None

    # Convert to numeric (YAML can return strings)
    if param_type in ["int", "float", "log"]:
        min_val = float(min_val)
        max_val = float(max_val)
        if step is not None:
            step = float(step)

    if param_type == "int":
        min_val, max_val = int(min_val), int(max_val)
        if step is not None:
            step = int(step)
            return trial.suggest_int(param_name, min_val, max_val, step=step)
        else:
            return trial.suggest_int(param_name, min_val, max_val)
    elif param_type == "float":
        if step is not None:
            return trial.suggest_float(param_name, min_val, max_val, step=step)
        else:
            return trial.suggest_float(param_name, min_val, max_val)
    elif param_type == "log":
        # Optuna requires low > 0 for log sampling
        if min_val <= 0:
            min_val = max(min_val, 1e-6)
        return trial.suggest_float(param_name, min_val, max_val, log=True)
    else:
        raise ValueError(f"Unknown param type: {param_type}")


def xgboost_param_space(trial: optuna.Trial, config: Dict[str, List]) -> Dict[str, Any]:
    """
    Generate XGBoost parameter space from config.

    Args:
        trial: Optuna trial object
        config: Parameter space configuration (from YAML)

    Returns:
        Dictionary of XGBoost hyperparameters

    Example:
        >>> config = {
        ...     "learning_rate": [0.001, 0.3, "log"],
        ...     "max_depth": [3, 10, "int"],
        ...     "subsample": [0.5, 1.0, "float"],
        ... }
        >>> params = xgboost_param_space(trial, config)
    """
    params = {}

    # Sample hyperparameters from config
    for param_name, param_spec in config.items():
        params[param_name] = suggest_from_config(trial, param_name, param_spec)

    # Fixed parameters (can be overridden in config)
    defaults = {
        "objective": "binary:logistic",  # Override based on problem type
        "tree_method": "hist",
        "random_state": 42,
        "enable_categorical": True,  # Native categorical support (XGBoost 1.5+)
    }

    # Add defaults if not in params
    for key, value in defaults.items():
        if key not in params:
            params[key] = value

    return params


def lightgbm_param_space(
    trial: optuna.Trial, config: Dict[str, List]
) -> Dict[str, Any]:
    """
    Generate LightGBM parameter space from config.

    Args:
        trial: Optuna trial object
        config: Parameter space configuration (from YAML)

    Returns:
        Dictionary of LightGBM hyperparameters

    Example:
        >>> config = {
        ...     "learning_rate": [0.001, 0.3, "log"],
        ...     "num_leaves": [15, 255, "int"],
        ...     "feature_fraction": [0.5, 1.0, "float"],
        ... }
        >>> params = lightgbm_param_space(trial, config)
    """
    params = {}

    # Sample hyperparameters from config
    for param_name, param_spec in config.items():
        params[param_name] = suggest_from_config(trial, param_name, param_spec)

    # Fixed parameters
    defaults = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "random_state": 42,
    }

    for key, value in defaults.items():
        if key not in params:
            params[key] = value

    return params


def catboost_param_space(
    trial: optuna.Trial, config: Dict[str, List]
) -> Dict[str, Any]:
    """
    Generate CatBoost parameter space from config.

    Args:
        trial: Optuna trial object
        config: Parameter space configuration (from YAML)

    Returns:
        Dictionary of CatBoost hyperparameters

    Example:
        >>> config = {
        ...     "learning_rate": [0.001, 0.3, "log"],
        ...     "depth": [4, 10, "int"],
        ...     "l2_leaf_reg": [1.0, 10.0, "float"],
        ... }
        >>> params = catboost_param_space(trial, config)
    """
    params = {}

    # Sample hyperparameters from config
    for param_name, param_spec in config.items():
        params[param_name] = suggest_from_config(trial, param_name, param_spec)

    # Fixed parameters
    defaults = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "verbose": False,
        "random_state": 42,
    }

    for key, value in defaults.items():
        if key not in params:
            params[key] = value

    return params
