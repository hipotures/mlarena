"""Model training module."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from rich.console import Console

# pandas imported in execute()

console = Console()

from mlarena.core.module import BaseModule, ModuleResult
from mlarena.core.registry import ModuleRegistry
from mlarena.core.config import TemplateLoader
from mlarena.utils.project import data_paths, load_project_config


def _load_processed_or_raw(context, config, preprocess_template: str | None = None):
    """
    Load preprocessed data from experiments/pre-{template}/ or raw data.

    Args:
        context: Module context
        config: Project config
        preprocess_template: Name of preprocessing template (e.g., 'baseline', 'av_weights')

    Returns:
        Tuple of (train_df, test_df, sample_weight_df or None)
    """
    import pandas as pd
    import json

    sample_weight = None

    if preprocess_template:
        # Load from experiments/pre-{template}/artifacts/preprocess/
        preprocess_exp_dir = context.project_root / "experiments" / f"pre-{preprocess_template}"
        train_path = preprocess_exp_dir / "artifacts" / "preprocess" / "train_processed.csv"
        test_path = preprocess_exp_dir / "artifacts" / "preprocess" / "test_processed.csv"

        if not train_path.exists():
            raise FileNotFoundError(
                f"Preprocessed data not found: {train_path}\n"
                f"Run: python scripts/mla.py preprocess --project {context.project_name} --preprocess-template {preprocess_template}"
            )

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path) if test_path.exists() else None

        # Check if preprocessing created sample weights (e.g., av_weights)
        state_path = preprocess_exp_dir / "state.json"
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)

            preprocess_payload = state.get("modules", {}).get("preprocess", {}).get("payload", {})
            custom_state = preprocess_payload.get("custom_module_state", {})
            weights_path_str = custom_state.get("weights_path")

            if weights_path_str:
                weights_path = Path(weights_path_str)
                if weights_path.exists():
                    sample_weight = pd.read_csv(weights_path)
    else:
        # Use raw data
        train_path, test_path = data_paths(config)
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path) if test_path.exists() else None

    return train_df, test_df, sample_weight


def _resolve_model_path(project_root: Path, model_name: str) -> Path:
    """Resolve model file: check project-local first, then global.

    Args:
        project_root: Project root directory
        model_name: Model name (e.g., 'autogluon_baseline', 'autogluon_av_weights')

    Returns:
        Path to model file

    Raises:
        RuntimeError: If model exists in both local and global (ambiguity)
        FileNotFoundError: If model not found anywhere
    """
    repo_root = Path(__file__).resolve().parents[3]  # src/mlarena/modules/model.py -> ../../.. -> repo root

    local_path = project_root / "code" / "models" / f"{model_name}.py"
    global_path = repo_root / "config" / "code" / "models" / f"{model_name}.py"

    local_exists = local_path.exists()
    global_exists = global_path.exists()

    # Ambiguity detection
    if local_exists and global_exists:
        raise RuntimeError(
            f"Model '{model_name}' exists both locally and globally:\n"
            f"  Local:  {local_path}\n"
            f"  Global: {global_path}\n"
            "Rename or remove one to avoid ambiguity."
        )

    # Priority: local > global
    if local_exists:
        return local_path
    if global_exists:
        return global_path

    # Not found
    raise FileNotFoundError(
        f"Model file not found for '{model_name}'. Checked:\n"
        f"  Local:  {local_path}\n"
        f"  Global: {global_path}"
    )


def _load_model_module(project_root: Path, model_name: str):
    """Load model Python file as module using importlib.util.

    Args:
        project_root: Project root directory
        model_name: Model name to load

    Returns:
        Loaded module object with train() and predict() functions
    """
    import importlib.util

    model_path = _resolve_model_path(project_root, model_name)
    console.print(f"[dim]Loading model from: {model_path.relative_to(Path.cwd())}[/dim]")

    spec = importlib.util.spec_from_file_location(model_name, model_path)
    if spec is None:
        raise RuntimeError(f"Unable to create module spec for {model_path}")

    module = importlib.util.module_from_spec(spec)

    if not spec.loader:
        raise RuntimeError(f"Unable to load module spec for {model_path}")

    spec.loader.exec_module(module)
    return module


@ModuleRegistry.register
class ModelModule(BaseModule):
    name = "model"
    description = "Train model"
    dependencies = set()  # No dependencies - preprocessing is optional and in separate exp-{id}

    @classmethod
    def register_cli_args(cls, parser) -> None:
        parser.add_argument("--time-limit", type=int, default=None, help="Optional time limit override.")
        parser.add_argument("--preset", type=str, default=None, help="AutoGluon preset override.")
        parser.add_argument("--use-gpu", type=int, choices=[0, 1], default=None, help="Force GPU usage for AutoGluon.")
        parser.add_argument("--model-template", default="dev-gpu", help="Model template name (for hyperparameters).")
        parser.add_argument("--preprocess-template", type=str, default=None, help="Preprocessing template to use (e.g., baseline, av_weights). If not specified, uses raw data.")

    def _build_model_config(self, template_cfg: Dict[str, Any], config_module, preset: str, time_limit: int, use_gpu_param: bool, artifact_dir: Path):
        """Build ModelConfig object for custom model interface."""
        from kaggle_tools.config_models import ModelConfig, Hyperparameters, DatasetConfig, SystemConfig

        train_path, test_path = data_paths(config_module)

        return ModelConfig(
            hyperparameters=Hyperparameters(
                presets=preset,
                time_limit=time_limit,
                use_gpu=use_gpu_param if use_gpu_param is not None else False,
                **template_cfg.get("hyperparameters", {})
            ),
            dataset=DatasetConfig(
                train_path=train_path,
                test_path=test_path,
                target=getattr(config_module, "TARGET_COLUMN"),
                id_column=getattr(config_module, "ID_COLUMN", "id"),
                problem_type=getattr(config_module, "AUTOGLUON_PROBLEM_TYPE", None),
                metric=getattr(config_module, "AUTOGLUON_EVAL_METRIC", None),
                ignored_columns=getattr(config_module, "IGNORED_COLUMNS", []),
                sample_submission_path=getattr(config_module, "SAMPLE_SUBMISSION_PATH", test_path.parent / "sample_submission.csv"),
                submission_probas=getattr(config_module, "SUBMISSION_PROBAS", False),
            ),
            system=SystemConfig(
                project_root=self.context.project_root,
                code_dir=self.context.project_root / "code",
                experiment_dir=self.context.experiment_dir,
                artifact_dir=artifact_dir,
                model_path=artifact_dir / "model",
                template=self.invocation_params.get("model_template", "dev-gpu"),
                experiment_id=self.context.experiment_id,
                random_seed=getattr(config_module, "RANDOM_SEED", 42),
                use_gpu=use_gpu_param if use_gpu_param is not None else False,
            ),
            model=template_cfg.get("model", {}),
        )

    def execute(self) -> ModuleResult:
        import pandas as pd
        from rich.table import Table

        template_name = self.invocation_params.get("model_template", "dev-gpu")

        # Handle --model-template list
        if template_name == "list":
            loader = TemplateLoader(self.context.project_root, template_type="model")
            templates = loader.list_available_with_source()

            table = Table(title="Available Model Templates", show_header=True)
            table.add_column("Template Name", style="cyan")
            table.add_column("Source", justify="center")

            for tpl in templates:
                name = tpl["name"]
                source = tpl["source"]

                if source == "global":
                    source_display = "🅶"
                    style = None
                elif source == "local":
                    source_display = "🅻"
                    style = None
                else:  # conflict
                    source_display = "🅶🅻"
                    style = "on grey23"  # dark grey background

                table.add_row(name, source_display, style=style)

            console.print(table)
            console.print(f"\n[dim]Usage: --model-template <name>[/dim]")
            console.print(f"[dim]🅶 = global template  🅻 = local template  🅶🅻 = name conflict (local overrides)[/dim]")
            return ModuleResult(
                success=True,
                payload={"templates": [t["name"] for t in templates]},
            )

        artifact_dir: Path = self.context.artifact_dir
        artifact_dir.mkdir(parents=True, exist_ok=True)
        config = self.context.config_module or load_project_config(self.context.project_root)

        preprocess_template = self.invocation_params.get("preprocess_template")
        train_df, test_df, sample_weight = _load_processed_or_raw(self.context, config, preprocess_template)
        target = getattr(config, "TARGET_COLUMN", None)
        if target is None or target not in train_df.columns:
            marker = artifact_dir / "model_failed.txt"
            marker.write_text("TARGET_COLUMN missing; aborting model step.")
            return ModuleResult(success=False, error="TARGET_COLUMN missing", artifacts=[marker])

        preset = self.invocation_params.get("preset") or getattr(config, "AUTOGLUON_PRESET", "medium")
        time_limit = self.invocation_params.get("time_limit") or getattr(config, "AUTOGLUON_TIME_LIMIT", None)
        use_gpu_param = self.invocation_params.get("use_gpu")

        loader = TemplateLoader(self.context.project_root, template_type="model")
        template_cfg = loader.load(template_name)

        # Check if template was found (when explicitly specified and not using defaults)
        if not template_cfg and template_name not in ["dev-gpu", "cpu-dev-5m"]:
            available = loader.list_available()
            console.print(f"\n[red]✗ Template '{template_name}' not found[/red]\n")

            if available:
                from rich.table import Table
                table = Table(title="Available Model Templates")
                table.add_column("Template Name", style="cyan")
                for tpl in available:
                    table.add_row(tpl)
                console.print(table)
                console.print(f"\n[dim]Usage: --model-template <name>[/dim]")
            else:
                console.print("[yellow]No templates found in config/templates/model.yaml[/yellow]")

            return ModuleResult(
                success=False,
                error=f"template_not_found: {template_name}",
            )

        if "preset" in template_cfg and not self.invocation_params.get("preset"):
            preset = template_cfg["preset"]
        if "time_limit" in template_cfg and not self.invocation_params.get("time_limit"):
            time_limit = template_cfg["time_limit"]
        if "use_gpu" in template_cfg and use_gpu_param is None:
            use_gpu_param = template_cfg["use_gpu"]

        # Check if template specifies a custom model implementation
        model_implementation = template_cfg.get("model")

        if model_implementation:
            # === DYNAMIC MODEL LOADING PATH ===
            console.print(f"[cyan]Using model implementation: {model_implementation}[/cyan]")

            # Load custom model module
            model_module = _load_model_module(self.context.project_root, model_implementation)

            # Build ModelConfig for model interface
            model_config = self._build_model_config(template_cfg, config, preset, time_limit, use_gpu_param, artifact_dir)

            # Call train()
            console.print("[green]Training model...[/green]")
            train_result = model_module.train(
                train_df=train_df,
                val_df=None,
                config=model_config,
                artifacts=None,
            )

            # Handle return: (model, summary) tuple or just model
            if isinstance(train_result, tuple) and len(train_result) == 2:
                predictor, training_summary = train_result
            else:
                predictor, training_summary = train_result, {}

            local_cv = training_summary.get("local_cv")

            # Call predict()
            if test_df is not None:
                console.print("[green]Generating predictions...[/green]")
                predictions = model_module.predict(
                    model=predictor,
                    test_df=test_df,
                    config=model_config,
                    artifacts=None,
                )

                # Save predictions
                submission_path = artifact_dir / "submission.csv"
                predictions.to_csv(submission_path, index=False)
                console.print(f"[green]✓ Predictions saved: {submission_path.relative_to(self.context.project_root)}[/green]")
            else:
                submission_path = None

            # Save leaderboard if model has it
            lb_path = None
            if hasattr(predictor, "leaderboard"):
                try:
                    lb_path = artifact_dir / "leaderboard.csv"
                    leaderboard = predictor.leaderboard(silent=True)
                    leaderboard.to_csv(lb_path, index=False)
                except Exception:
                    pass

            return ModuleResult(
                success=True,
                payload={
                    "model_implementation": model_implementation,
                    "local_cv": local_cv,
                    "training_summary": training_summary,
                    "preset": preset,
                    "time_limit": time_limit,
                    "use_gpu": use_gpu_param,
                    "template": template_name,
                    "preprocess_template": preprocess_template,
                },
                artifacts=[f for f in [artifact_dir / "model", lb_path, submission_path] if f and f.exists()],
            )

        else:
            # === FALLBACK: INLINE AUTOGLUON BASELINE ===
            console.print("[dim]Using default AutoGluon baseline (no template.model specified)[/dim]")

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

            fit_kwargs = {
                "presets": preset,
                "time_limit": time_limit,
                "ag_args_fit": ag_args_fit or None,
                "hyperparameters": hyperparams or None,
            }

            predictor.fit(train_df, **fit_kwargs)

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
                    "preprocess_template": preprocess_template,  # Track which preprocessing was used
                },
                artifacts=[ag_path, lb_path, train_path],
            )
