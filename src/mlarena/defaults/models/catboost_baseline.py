from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from rich.console import Console

from kaggle_tools.config_models import ModelConfig
from kaggle_tools.optuna import CVObjective, StudyManager, catboost_param_space
from mlarena.defaults.models.boosting_utils import (
    build_leaderboard,
    detect_categorical_features,
    prepare_training_data,
    resolve_metric,
    score_dataset,
)


DEFAULT_PARAM_SPACE: Dict[str, list] = {
    "learning_rate": [0.01, 0.3, "log"],
    "depth": [4, 10, "int"],
    "l2_leaf_reg": [1.0, 10.0, "float"],
    "subsample": [0.5, 1.0, "float"],
}


def _cat_defaults(
    problem_type: str,
    metric_name: str,
    use_gpu: bool,
    random_seed: int,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "random_seed": random_seed,
        "verbose": False,
    }
    params["task_type"] = "GPU" if use_gpu else "CPU"

    if problem_type == "regression":
        params.setdefault("loss_function", "MAE" if metric_name in {"mae", "mean_absolute_error"} else "RMSE")
        params.setdefault("eval_metric", params["loss_function"])
    elif problem_type == "multiclass":
        params.setdefault("loss_function", "MultiClass")
        params.setdefault("eval_metric", "MultiClass")
    else:
        params.setdefault("loss_function", "Logloss")
        params.setdefault("eval_metric", "AUC" if "auc" in metric_name else "Logloss")
    return params


def train(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    config: ModelConfig,
    artifacts: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    import catboost as cb

    verbosity = int(getattr(config.hyperparameters, "verbosity", 2) or 2)
    console = Console(quiet=verbosity == 0)

    prepared = prepare_training_data(train_df, config, artifacts, console)
    train_data = prepared.train_data
    tuning_data = prepared.tuning_data
    eval_data = prepared.eval_data

    target_column = prepared.target_column
    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]

    metric_name, score_fn, needs_proba, _ = resolve_metric(config, y_train)
    problem_type = str(config.dataset.problem_type or "").strip().lower() or "binary"
    if problem_type not in {"binary", "multiclass", "regression"}:
        problem_type = "binary"
        if y_train.nunique(dropna=True) > 2:
            problem_type = "multiclass"

    model_class = cb.CatBoostRegressor if problem_type == "regression" else cb.CatBoostClassifier

    use_gpu = bool(config.hyperparameters.use_gpu or config.system.use_gpu)
    cat_features = detect_categorical_features(X_train)
    if cat_features and verbosity > 0:
        console.print(f"[cyan]i[/cyan] CatBoost detected {len(cat_features)} categorical features")

    optuna_cfg = config.optuna
    if not optuna_cfg.storage:
        optuna_cfg.storage = f"sqlite:///{config.system.artifact_dir / 'optuna' / 'study.db'}"
    if not optuna_cfg.study_name:
        optuna_cfg.study_name = f"{config.system.experiment_id}_catboost"

    param_space = optuna_cfg.param_space.get("catboost") if optuna_cfg.param_space else None
    if not param_space:
        param_space = DEFAULT_PARAM_SPACE

    def _param_space(trial):
        params = catboost_param_space(trial, param_space)
        params.update(_cat_defaults(problem_type, metric_name, use_gpu, config.system.random_seed))
        return params

    best_params: Dict[str, Any] = {}
    best_value: Optional[float] = None
    optuna_enabled = optuna_cfg.n_trials and optuna_cfg.n_trials > 0

    if optuna_enabled:
        optuna_cfg.direction = "maximize"
        optuna_dir = config.system.artifact_dir / "optuna"
        optuna_dir.mkdir(parents=True, exist_ok=True)
        best_params_file = optuna_dir / "best_params_catboost.json"

        if best_params_file.exists():
            console.print(f"[green]✓[/green] Loading cached params from {best_params_file}")
            best_params = json.loads(best_params_file.read_text())
        else:
            console.print("[cyan]i[/cyan] Running Optuna for CatBoost")
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
                model_kwargs={"cat_features": cat_features},
            )

            study.optimize(
                objective,
                n_trials=optuna_cfg.n_trials,
                timeout=optuna_cfg.timeout,
                n_jobs=optuna_cfg.n_jobs,
            )

            best_params = study.best_params
            best_value = float(study.best_value) if study.best_value is not None else None
            best_params_file.write_text(json.dumps(best_params, indent=2))
            study_manager.export_trials_dataframe(optuna_dir / "trials_catboost.csv")
    else:
        console.print("[yellow]⚠[/yellow] Optuna disabled (n_trials <= 0)")

    iterations = int(
        getattr(config.hyperparameters, "iterations", None)
        or getattr(config.hyperparameters, "n_estimators", None)
        or config.model.get("iterations", config.model.get("n_estimators", 1000))
    )
    early_stopping_rounds = int(
        getattr(config.hyperparameters, "early_stopping_rounds", None)
        or config.model.get("early_stopping_rounds", 50)
    )

    params = _cat_defaults(problem_type, metric_name, use_gpu, config.system.random_seed)
    params.update(best_params)
    params["verbose"] = 100 if verbosity > 1 else False

    model = model_class(
        **params,
        iterations=iterations,
        cat_features=cat_features,
    )

    eval_set = None
    if tuning_data is not None:
        X_tune = tuning_data.drop(columns=[target_column])
        y_tune = tuning_data[target_column]
        eval_set = (X_tune, y_tune)

    fit_kwargs: Dict[str, Any] = {
        "sample_weight": prepared.sample_weight,
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
            best_value = score_dataset(model, tuning_data, target_column, score_fn, needs_proba)
        else:
            best_value = score_dataset(model, train_data, target_column, score_fn, needs_proba)

    eval_score = None
    if eval_data is not None:
        eval_score = score_dataset(model, eval_data, target_column, score_fn, needs_proba)

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
        "leaderboard": build_leaderboard("catboost", best_value, metric_name),
        **prepared.meta,
    }

    return model, training_summary
