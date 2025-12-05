"""Hyperparameter tuning module using Optuna + AutoGluon."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from mlarena.core.module import BaseModule, ModuleResult
from mlarena.core.registry import ModuleRegistry
from mlarena.core.config import TemplateLoader
from mlarena.utils.project import data_paths, load_project_config


@ModuleRegistry.register
class TuneModule(BaseModule):
    name = "tune"
    description = "Hyperparameter tuning"
    dependencies = {"preprocess"}

    @classmethod
    def register_cli_args(cls, parser) -> None:
        parser.add_argument("--n-trials", type=int, default=10, help="Number of tuning trials.")
        parser.add_argument("--time-limit", type=int, default=60, help="Per-trial time limit (seconds).")
        parser.add_argument("--tune-template", type=str, default="tune", help="Template name with search space.")

    def execute(self) -> ModuleResult:
        artifact_dir: Path = self.context.artifact_dir
        artifact_dir.mkdir(parents=True, exist_ok=True)

        try:
            import optuna
        except Exception as exc:
            marker = artifact_dir / "tune_failed.txt"
            marker.write_text(f"Optuna missing: {exc}")
            return ModuleResult(success=False, error="optuna missing", artifacts=[marker])

        config = self.context.config_module or load_project_config(self.context.project_root)
        train_path, _ = data_paths(config)
        if not train_path.exists():
            marker = artifact_dir / "tune_failed.txt"
            marker.write_text("train.csv missing")
            return ModuleResult(success=False, error="train missing", artifacts=[marker])

        train_df = pd.read_csv(train_path)
        target = getattr(config, "TARGET_COLUMN", None)
        if target not in train_df.columns:
            marker = artifact_dir / "tune_failed.txt"
            marker.write_text("TARGET_COLUMN missing")
            return ModuleResult(success=False, error="target missing", artifacts=[marker])

        template_name = self.invocation_params.get("tune_template", "tune")
        template_cfg = TemplateLoader(self.context.project_root).load(template_name)
        search_space = template_cfg.get("search_space", {})

        n_trials = int(self.invocation_params.get("n_trials", 10))
        per_trial_limit = int(self.invocation_params.get("time_limit", 60))

        def objective(trial: optuna.Trial) -> float:
            params = {}
            for key, spec in search_space.items():
                if spec.get("type") == "float":
                    params[key] = trial.suggest_float(key, spec["low"], spec["high"], log=spec.get("log", False))
                elif spec.get("type") == "int":
                    params[key] = trial.suggest_int(key, spec["low"], spec["high"], log=spec.get("log", False))
                elif spec.get("type") == "categorical":
                    params[key] = trial.suggest_categorical(key, spec["choices"])
            try:
                from autogluon.tabular import TabularPredictor
            except Exception as exc:
                raise optuna.TrialPruned(f"autogluon unavailable: {exc}")

            # Small subset for speed
            sample_df = train_df.sample(min(len(train_df), 300), random_state=trial.number)
            predictor = TabularPredictor(
                label=target,
                problem_type=getattr(config, "AUTOGLUON_PROBLEM_TYPE", None),
                eval_metric=getattr(config, "AUTOGLUON_EVAL_METRIC", None),
                path=str(artifact_dir / f"trial_{trial.number}"),
            )
            predictor.fit(sample_df, presets="medium", time_limit=per_trial_limit, hyperparameters=params or None)
            leaderboard = predictor.leaderboard(silent=True)
            best_score = float(leaderboard.iloc[0]["score"]) if "score" in leaderboard.columns else 0.0
            return best_score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_value = study.best_value
        result_path = artifact_dir / "tune_result.json"
        result_path.write_text(json.dumps({"best_params": best_params, "best_value": best_value}, indent=2))

        return ModuleResult(
            success=True,
            payload={"best_params": best_params, "best_value": best_value, "template": template_name},
            artifacts=[result_path],
        )
