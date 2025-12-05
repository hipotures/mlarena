"""Model training module."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from mlarena.core.module import BaseModule, ModuleResult
from mlarena.core.registry import ModuleRegistry
from mlarena.core.config import TemplateLoader
from mlarena.utils.project import data_paths, load_project_config


def _load_processed_or_raw(context, config) -> pd.DataFrame:
    # Prefer processed artifact
    preprocess_entry = context.state.modules.get("preprocess")
    processed = getattr(preprocess_entry, "payload", {}) if preprocess_entry else {}
    path = None
    if processed and processed.get("train_processed"):
        path = Path(processed["train_processed"])
    if path is None or not Path(path).exists():
        path, _ = data_paths(config)
    return pd.read_csv(path)


@ModuleRegistry.register
class ModelModule(BaseModule):
    name = "model"
    description = "Train model"
    dependencies = {"preprocess"}

    @classmethod
    def register_cli_args(cls, parser) -> None:
        parser.add_argument("--time-limit", type=int, default=None, help="Optional time limit override.")
        parser.add_argument("--preset", type=str, default=None, help="AutoGluon preset override.")
        parser.add_argument("--use-gpu", type=int, choices=[0, 1], default=None, help="Force GPU usage for AutoGluon.")
        parser.add_argument("--model-template", default="dev-gpu", help="Model template name (for hyperparameters).")

    def execute(self) -> ModuleResult:
        artifact_dir: Path = self.context.artifact_dir
        artifact_dir.mkdir(parents=True, exist_ok=True)
        config = self.context.config_module or load_project_config(self.context.project_root)

        train_df = _load_processed_or_raw(self.context, config)
        target = getattr(config, "TARGET_COLUMN", None)
        if target is None or target not in train_df.columns:
            marker = artifact_dir / "model_failed.txt"
            marker.write_text("TARGET_COLUMN missing; aborting model step.")
            return ModuleResult(success=False, error="TARGET_COLUMN missing", artifacts=[marker])

        preset = self.invocation_params.get("preset") or getattr(config, "AUTOGLUON_PRESET", "medium")
        time_limit = self.invocation_params.get("time_limit") or getattr(config, "AUTOGLUON_TIME_LIMIT", None)
        use_gpu_param = self.invocation_params.get("use_gpu")
        template_name = self.invocation_params.get("model_template", "dev-gpu")

        template_cfg = TemplateLoader(self.context.project_root).load(template_name)
        if "preset" in template_cfg and not self.invocation_params.get("preset"):
            preset = template_cfg["preset"]
        if "time_limit" in template_cfg and not self.invocation_params.get("time_limit"):
            time_limit = template_cfg["time_limit"]
        if "use_gpu" in template_cfg and use_gpu_param is None:
            use_gpu_param = template_cfg["use_gpu"]

        try:
            from autogluon.tabular import TabularPredictor
        except Exception as exc:  # pragma: no cover - dependency issue
            marker = artifact_dir / "model_failed.txt"
            marker.write_text(f"AutoGluon not available: {exc}")
            return ModuleResult(success=False, error="autogluon missing", artifacts=[marker])

        label = target
        problem_type = getattr(config, "AUTOGLUON_PROBLEM_TYPE", None)
        eval_metric = getattr(config, "AUTOGLUON_EVAL_METRIC", None)
        id_col = getattr(config, "ID_COLUMN", None)
        if id_col and id_col in train_df.columns:
            train_df = train_df.drop(columns=[id_col])

        train_path = artifact_dir / "train_used.csv"
        train_df.to_csv(train_path, index=False)

        ag_path = artifact_dir / "AutogluonModels"
        predictor = TabularPredictor(label=label, problem_type=problem_type, eval_metric=eval_metric, path=str(ag_path))
        hyperparams: Dict[str, Any] = {}
        hyperparams.update(template_cfg.get("hyperparameters", {}))
        ag_args_fit = {}
        if use_gpu_param is not None:
            ag_args_fit["num_gpus"] = 1 if use_gpu_param else 0

        predictor.fit(
            train_df,
            presets=preset,
            time_limit=time_limit,
            ag_args_fit=ag_args_fit or None,
            hyperparameters=hyperparams or None,
        )

        lb_path = artifact_dir / "leaderboard.csv"
        leaderboard = predictor.leaderboard(silent=True)
        leaderboard.to_csv(lb_path, index=False)

        info = predictor.info()
        best_model = info.get("best_model")

        return ModuleResult(
            success=True,
            payload={
                "model_artifact": str(ag_path),
                "leaderboard": str(lb_path),
                "best_model": best_model,
                "preset": preset,
                "time_limit": time_limit,
                "use_gpu": use_gpu_param,
                "template": template_name,
                "hyperparameters": hyperparams,
            },
            artifacts=[ag_path, lb_path, train_path],
        )
