from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from rich.console import Console

from kaggle_tools.config_models import ModelConfig
from kaggle_tools.optuna import CVObjective, StudyManager, xgboost_param_space
from mlarena.defaults.models.boosting_utils import (
    build_leaderboard,
    cast_categorical_columns,
    prepare_training_data,
    resolve_metric,
    score_dataset,
)


DEFAULT_PARAM_SPACE: Dict[str, list] = {
    "learning_rate": [0.01, 0.3, "log"],
    "max_depth": [3, 10, "int"],
    "subsample": [0.6, 1.0, "float"],
    "colsample_bytree": [0.6, 1.0, "float"],
    "min_child_weight": [1, 10, "int"],
    "reg_alpha": [0.0, 1.0, "float"],
    "reg_lambda": [0.5, 5.0, "log"],
}


def _align_categories(train_df: pd.DataFrame, other_df: pd.DataFrame) -> pd.DataFrame:
    if other_df is None:
        return other_df
    other_df = other_df.copy()
    cat_cols = train_df.select_dtypes(include=["category"]).columns
    for col in cat_cols:
        if col in other_df.columns:
            other_df[col] = pd.Categorical(
                other_df[col], categories=train_df[col].cat.categories
            )
    return other_df


def _xgb_defaults(
    problem_type: str,
    use_gpu: bool,
    random_seed: int,
    num_class: Optional[int],
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "random_state": random_seed,
        "enable_categorical": True,
        "tree_method": "gpu_hist" if use_gpu else "hist",
    }
    if problem_type == "regression":
        params.setdefault("objective", "reg:squarederror")
    elif problem_type == "multiclass":
        params.setdefault("objective", "multi:softprob")
        if num_class:
            params["num_class"] = num_class
    else:
        params.setdefault("objective", "binary:logistic")
    return params


def train(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    config: ModelConfig,
    artifacts: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    import xgboost as xgb

    verbosity = int(getattr(config.hyperparameters, "verbosity", 2) or 2)
    console = Console(quiet=verbosity == 0)

    prepared = prepare_training_data(train_df, config, artifacts, console)
    train_data = cast_categorical_columns(prepared.train_data)
    tuning_data = (
        cast_categorical_columns(prepared.tuning_data)
        if prepared.tuning_data is not None
        else None
    )
    eval_data = (
        cast_categorical_columns(prepared.eval_data)
        if prepared.eval_data is not None
        else None
    )
    tuning_data = (
        _align_categories(train_data, tuning_data) if tuning_data is not None else None
    )
    eval_data = (
        _align_categories(train_data, eval_data) if eval_data is not None else None
    )

    target_column = prepared.target_column
    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]

    metric_name, score_fn, needs_proba, _ = resolve_metric(config, y_train)
    problem_type = str(config.dataset.problem_type or "").strip().lower() or "binary"
    if problem_type not in {"binary", "multiclass", "regression"}:
        problem_type = "binary"
        if y_train.nunique(dropna=True) > 2:
            problem_type = "multiclass"

    num_class = (
        int(y_train.nunique(dropna=True)) if problem_type == "multiclass" else None
    )
    model_class = (
        xgb.XGBRegressor if problem_type == "regression" else xgb.XGBClassifier
    )

    optuna_cfg = config.optuna
    if not optuna_cfg.storage:
        optuna_cfg.storage = (
            f"sqlite:///{config.system.artifact_dir / 'optuna' / 'study.db'}"
        )
    if not optuna_cfg.study_name:
        optuna_cfg.study_name = f"{config.system.experiment_id}_xgboost"

    param_space = (
        optuna_cfg.param_space.get("xgboost") if optuna_cfg.param_space else None
    )
    if not param_space:
        param_space = DEFAULT_PARAM_SPACE

    use_gpu = bool(config.hyperparameters.use_gpu or config.system.use_gpu)

    def _param_space(trial):
        params = xgboost_param_space(trial, param_space)
        params.update(
            _xgb_defaults(problem_type, use_gpu, config.system.random_seed, num_class)
        )
        return params

    best_params: Dict[str, Any] = {}
    best_value: Optional[float] = None
    optuna_enabled = optuna_cfg.n_trials and optuna_cfg.n_trials > 0

    if optuna_enabled:
        optuna_cfg.direction = "maximize"
        optuna_dir = config.system.artifact_dir / "optuna"
        optuna_dir.mkdir(parents=True, exist_ok=True)
        best_params_file = optuna_dir / "best_params_xgboost.json"

        if best_params_file.exists():
            console.print(
                f"[green]✓[/green] Loading cached params from {best_params_file}"
            )
            best_params = json.loads(best_params_file.read_text())
        else:
            console.print("[cyan]i[/cyan] Running Optuna for XGBoost")
            study_manager = StudyManager(optuna_cfg)
            study = study_manager.create_or_load_study()

            objective = CVObjective(
                model_class=model_class,
                train_df=train_data,
                target_col=target_column,
                param_space_fn=_param_space,
                metric_fn=score_fn,
                cv_folds=optuna_cfg.cv_folds,
                early_stopping_rounds=optuna_cfg.early_stopping_rounds,
                random_seed=config.system.random_seed,
                stratified=problem_type in {"binary", "multiclass"},
            )

            study.optimize(
                objective,
                n_trials=optuna_cfg.n_trials,
                timeout=optuna_cfg.timeout,
                n_jobs=optuna_cfg.n_jobs,
            )

            best_params = study.best_params
            best_value = (
                float(study.best_value) if study.best_value is not None else None
            )
            best_params_file.write_text(json.dumps(best_params, indent=2))
            study_manager.export_trials_dataframe(optuna_dir / "trials_xgboost.csv")
    else:
        console.print("[yellow]⚠[/yellow] Optuna disabled (n_trials <= 0)")

    n_estimators = int(
        getattr(config.hyperparameters, "n_estimators", None)
        or config.model.get("n_estimators", 1000)
    )
    early_stopping_rounds = int(
        getattr(config.hyperparameters, "early_stopping_rounds", None)
        or config.model.get("early_stopping_rounds", 50)
    )

    params = _xgb_defaults(problem_type, use_gpu, config.system.random_seed, num_class)
    params.update(best_params)

    model = model_class(**params, n_estimators=n_estimators)

    eval_set = None
    if tuning_data is not None:
        X_tune = tuning_data.drop(columns=[target_column])
        y_tune = tuning_data[target_column]
        eval_set = [(X_tune, y_tune)]

    fit_kwargs: Dict[str, Any] = {
        "sample_weight": prepared.sample_weight,
        "verbose": verbosity > 1,
    }
    if eval_set:
        fit_kwargs.update(
            {
                "eval_set": eval_set,
                "early_stopping_rounds": early_stopping_rounds,
            }
        )

    model.fit(X_train, y_train, **fit_kwargs)

    if best_value is None:
        if tuning_data is not None:
            best_value = score_dataset(
                model, tuning_data, target_column, score_fn, needs_proba
            )
        else:
            best_value = score_dataset(
                model, train_data, target_column, score_fn, needs_proba
            )

    eval_score = None
    if eval_data is not None:
        eval_score = score_dataset(
            model, eval_data, target_column, score_fn, needs_proba
        )

    model_dir = Path(config.system.model_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pkl"
    import joblib

    joblib.dump(model, model_path)

    training_summary = {
        "local_cv_score": best_value,
        "best_params": best_params,
        "model_path": str(model_dir),
        "eval_score": eval_score,
        "metric": metric_name,
        "leaderboard": build_leaderboard("xgboost", best_value, metric_name),
        **prepared.meta,
    }

    return model, training_summary
