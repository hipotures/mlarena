"""Model training module."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console

# pandas imported in execute()

console = Console()

from mlarena.core.module import BaseModule, ModuleResult
from mlarena.core.registry import ModuleRegistry
from mlarena.core.config import TemplateLoader
from mlarena.utils.project import data_paths, load_project_config


def _load_processed_or_raw(
    context,
    config,
    preprocess_template: str | None = None,
    *,
    preprocess_exp_dir: str | Path | None = None,
):
    """
    Load preprocessed data from experiments (supports chains) or raw data.

    Args:
        context: Module context
        config: Project config
        preprocess_template: Name of preprocessing template (e.g., 'baseline', 'av_weights')
        preprocess_exp_dir: Optional explicit experiment directory for the final preprocessing step
            (e.g., experiments/pre-ps5e12_drift_mi/7-imbalance_handler)

    Returns:
        Tuple of (train_df, test_df, sample_weight_df or None, orig_df or None)
    """
    import pandas as pd
    import json

    sample_weight = None

    def _load_from_exp_dir(exp_dir: Path):
        train_path = exp_dir / "artifacts" / "preprocess" / "train_processed.csv.gz"
        test_path = exp_dir / "artifacts" / "preprocess" / "test_processed.csv.gz"
        orig_path = exp_dir / "artifacts" / "preprocess" / "orig_processed.csv.gz"

        if not train_path.exists():
            return None, None, None, None

        train_df_local = pd.read_csv(train_path, compression='infer')
        test_df_local = pd.read_csv(test_path, compression='infer') if test_path.exists() else None
        orig_df_local = pd.read_csv(orig_path, compression='infer') if orig_path.exists() else None

        sample_weight_local = None
        state_path = exp_dir / "state.json"
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)

            # For preprocessing chains, module name is not "preprocess" but the actual step name
            # Get the first module (should be only one in single-step state)
            modules = state.get("modules", {})
            if modules:
                # Try "preprocess" first (legacy single-step), then first module (chain step)
                preprocess_module = modules.get("preprocess") or next(iter(modules.values()))
                preprocess_payload = preprocess_module.get("payload", {})
            else:
                preprocess_payload = {}

            custom_state = preprocess_payload.get("custom_module_state", {})
            weights_path_str = custom_state.get("weights_path")

            if weights_path_str:
                weights_path = Path(weights_path_str)
                # Handle both absolute and relative paths
                if not weights_path.is_absolute():
                    weights_path = context.project_root / weights_path
                if weights_path.exists():
                    sample_weight_local = pd.read_csv(weights_path, compression='infer')

        return train_df_local, test_df_local, sample_weight_local, orig_df_local

    if preprocess_template:
        # 1) Prefer explicit experiment dir (for chains)
        if preprocess_exp_dir:
            exp_dir = Path(preprocess_exp_dir)
            train_df, test_df, sample_weight, orig_df = _load_from_exp_dir(exp_dir)
            if train_df is not None:
                return train_df, test_df, sample_weight, orig_df

        # 2) Try default single-step location (legacy)
        default_dir = context.project_root / "experiments" / f"pre-{preprocess_template}"
        train_df, test_df, sample_weight, orig_df = _load_from_exp_dir(default_dir)
        if train_df is not None:
            return train_df, test_df, sample_weight, orig_df

        # 3) Fallback: search meta-template chain dir (pre-{template}) and use last step
        chain_dir = context.project_root / "experiments" / f"pre-{preprocess_template}"
        if chain_dir.exists() and chain_dir.is_dir():
            # Subdirs are named "<idx>-<module>"; pick the highest idx
            candidates = []
            for subdir in chain_dir.iterdir():
                if not subdir.is_dir():
                    continue
                try:
                    idx_str = subdir.name.split("-")[0]
                    idx = int(idx_str)
                except Exception:
                    continue
                candidates.append((idx, subdir))
            if candidates:
                candidates.sort(reverse=True)
                _, latest_dir = candidates[0]
                train_df, test_df, sample_weight, orig_df = _load_from_exp_dir(latest_dir)
                if train_df is not None:
                    return train_df, test_df, sample_weight, orig_df

            # Also handle hashed chain dirs: pre-{template}/{hash}/<idx>-<module>
            hashed_latest = []
            for hash_dir in chain_dir.iterdir():
                if not hash_dir.is_dir():
                    continue
                step_candidates = []
                for step_dir in hash_dir.iterdir():
                    if not step_dir.is_dir():
                        continue
                    try:
                        idx_str = step_dir.name.split("-")[0]
                        idx = int(idx_str)
                    except Exception:
                        continue
                    step_candidates.append((idx, step_dir))
                if step_candidates:
                    step_candidates.sort(reverse=True)
                    _, latest_step = step_candidates[0]
                    hashed_latest.append(latest_step)
            if hashed_latest:
                latest_dir = max(hashed_latest, key=lambda p: p.stat().st_mtime)
                train_df, test_df, sample_weight, orig_df = _load_from_exp_dir(latest_dir)
                if train_df is not None:
                    return train_df, test_df, sample_weight, orig_df

        # 4) Fallback: search chain directories containing this template (pick most recent)
        experiments_dir = context.project_root / "experiments"
        candidates = list(experiments_dir.glob(f"pre-*/[0-9]*-{preprocess_template}"))
        candidates += list(experiments_dir.glob(f"pre-*/*/[0-9]*-{preprocess_template}"))
        candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
        for exp_dir in candidates:
            train_df, test_df, sample_weight, orig_df = _load_from_exp_dir(exp_dir)
            if train_df is not None:
                return train_df, test_df, sample_weight, orig_df

        raise FileNotFoundError(
            f"Preprocessed data not found for template '{preprocess_template}'.\n"
            f"Searched: {default_dir}/artifacts/preprocess/ and any chain step '*-{preprocess_template}'.\n"
            f"Run: python scripts/mla.py preprocess --project {context.project_name} --preprocess-template {preprocess_template}"
        )
    else:
        # Use raw data (no orig in raw data)
        train_path, test_path = data_paths(config)
        train_df = pd.read_csv(train_path, compression='infer')
        test_df = pd.read_csv(test_path, compression='infer') if test_path.exists() else None
        orig_df = None

    return train_df, test_df, sample_weight, orig_df


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
    global_path = repo_root / "src" / "mlarena" / "defaults" / "models" / f"{model_name}.py"

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
    try:
        display_path = model_path.relative_to(Path.cwd())
    except ValueError:
        # Model is outside cwd (e.g., global defaults), use absolute path
        display_path = model_path
    console.print(f"[dim]Loading model from: {display_path}[/dim]")

    spec = importlib.util.spec_from_file_location(model_name, model_path)
    if spec is None:
        raise RuntimeError(f"Unable to create module spec for {model_path}")

    module = importlib.util.module_from_spec(spec)

    if not spec.loader:
        raise RuntimeError(f"Unable to load module spec for {model_path}")

    spec.loader.exec_module(module)
    return module


def _print_training_summary(
    project_root: Path,
    experiment_id: str,
    project_name: str,
    template_name: str,
    preset: str,
    time_limit: int,
    use_gpu: bool,
    local_cv_score: float | None,
    best_model: str | None,
    preprocess_template: str | None,
    leaderboard_path: Path | None,
):
    """Print rich training summary with next steps."""
    from rich.table import Table
    from mlarena.core.module import print_next_steps

    # Training configuration table
    config_table = Table(title="Training Configuration", show_header=True)
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")

    config_table.add_row("Template", template_name)
    config_table.add_row("Preset", preset)
    config_table.add_row("Time Limit", f"{time_limit}s" if time_limit else "None")
    config_table.add_row("GPU", "Yes" if use_gpu else "No")
    if preprocess_template:
        config_table.add_row("Preprocessing", preprocess_template)
    if best_model:
        config_table.add_row("Best Model", best_model)
    if local_cv_score is not None:
        config_table.add_row("Local CV", f"{local_cv_score:.6f}")

    console.print("\n")
    console.print(config_table)

    # Artifacts
    console.print(f"\n[bold green]✓[/bold green] Training completed:")
    if leaderboard_path and leaderboard_path.exists():
        console.print(f"  Leaderboard: [cyan]{leaderboard_path.relative_to(project_root)}[/cyan]")


@ModuleRegistry.register
class ModelModule(BaseModule):
    """Train a model using a YAML template and optional preprocessing artifacts."""

    name = "model"
    description = "Train model"
    dependencies = set()  # No automatic dependencies - preprocessing is optional and loaded by name

    def _build_model_config(self, template_cfg: Dict[str, Any], config_module, preset: str, time_limit: int, use_gpu_param: bool, artifact_dir: Path):
        """Build ModelConfig object for custom model interface."""
        from kaggle_tools.config_models import ModelConfig, Hyperparameters, DatasetConfig, SystemConfig

        train_path, test_path = data_paths(config_module)

        # Extract model config (if it's a dict, not a string filename)
        model_cfg = template_cfg.get("model", {})
        if isinstance(model_cfg, str):
            # "model" is a filename reference, not config - use empty dict
            model_cfg = {}
        # Allow lightweight model-specific overrides from templates
        if "merge_orig" in template_cfg:
            model_cfg["merge_orig"] = template_cfg.get("merge_orig")

        # Build Hyperparameters with top-level fit args + model-specific hyperparams
        hyperparams_dict = {
            "presets": preset,
            "time_limit": time_limit,
            "use_gpu": use_gpu_param if use_gpu_param is not None else False,
        }

        # Add top-level fit args from template
        if "excluded_model_types" in template_cfg:
            hyperparams_dict["excluded_models"] = template_cfg["excluded_model_types"]
        if "included_model_types" in template_cfg:
            hyperparams_dict["included_model_types"] = template_cfg["included_model_types"]
        fit_args = template_cfg.get("fit_args")
        if fit_args is not None:
            if not isinstance(fit_args, dict):
                raise ValueError("fit_args must be a dict")
            hyperparams_dict["fit_args"] = fit_args

        # NEW: Load HPO preset if specified
        hpo_preset_name = template_cfg.get("hpo_preset")
        if hpo_preset_name:
            from pathlib import Path
            import yaml
            from rich.console import Console

            console = Console()
            repo_root = Path(__file__).resolve().parents[3]

            # Try global preset first
            hpo_preset_path = repo_root / "src" / "mlarena" / "templates" / "model" / "hpo" / f"{hpo_preset_name}.yaml"

            # Check project-local override
            if not hpo_preset_path.exists():
                project_hpo_path = self.context.project_root / "templates" / "model" / "hpo" / f"{hpo_preset_name}.yaml"
                if project_hpo_path.exists():
                    hpo_preset_path = project_hpo_path
                else:
                    raise FileNotFoundError(
                        f"HPO preset '{hpo_preset_name}' not found. "
                        f"Checked: {hpo_preset_path} and {project_hpo_path}"
                    )

            # Load preset YAML
            hpo_preset_data = yaml.safe_load(hpo_preset_path.read_text()) or {}

            # Extract HPO config defaults
            hpo_config = hpo_preset_data.get("hpo", {})
            hpo_tune_kwargs = {
                "num_trials": hpo_config.get("num_trials", 50),
                "scheduler": hpo_config.get("scheduler", "local"),
                "searcher": hpo_config.get("searcher", "auto"),
            }

            # Extract search space
            hpo_search_space = hpo_preset_data.get("search_space", {})

            # Allow template to override HPO defaults
            if "num_trials" in template_cfg:
                hpo_tune_kwargs["num_trials"] = template_cfg["num_trials"]
            if "scheduler" in template_cfg:
                hpo_tune_kwargs["scheduler"] = template_cfg["scheduler"]
            if "searcher" in template_cfg:
                hpo_tune_kwargs["searcher"] = template_cfg["searcher"]

            # Allow template to extend/override search space
            if "search_space" in template_cfg:
                template_space = template_cfg["search_space"]
                for model_type, params in template_space.items():
                    if model_type not in hpo_search_space:
                        hpo_search_space[model_type] = {}
                    hpo_search_space[model_type].update(params)

            # Add to hyperparams_dict
            hyperparams_dict["hyperparameter_tune_kwargs"] = hpo_tune_kwargs
            hyperparams_dict["search_space"] = hpo_search_space

            # Display HPO configuration (will also appear in console)
            console.print(f"\n[cyan]HPO Configuration:[/cyan]")
            console.print(f"  [bold]Preset:[/bold] {hpo_preset_name}")
            console.print(f"  [bold]Trials:[/bold] {hpo_tune_kwargs['num_trials']}")
            console.print(f"  [bold]Scheduler:[/bold] {hpo_tune_kwargs['scheduler']}")
            console.print(f"  [bold]Searcher:[/bold] {hpo_tune_kwargs['searcher']}")
            console.print(f"  [bold]Models:[/bold] {list(hpo_search_space.keys())}\n")

        # Add model-specific hyperparameters (e.g., NN_TORCH, GBM configs)
        hyperparams_dict.update(template_cfg.get("hyperparameters", {}))

        template_seed = template_cfg.get("random_seed")
        if template_seed is None:
            template_seed = template_cfg.get("seed")
        random_seed = template_seed if template_seed is not None else getattr(config_module, "RANDOM_SEED", 42)

        return ModelConfig(
            hyperparameters=Hyperparameters(**hyperparams_dict),
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
                sample_weight_strategy=template_cfg.get("sample_weight_strategy"),
                weight_evaluation=template_cfg.get("weight_evaluation"),
            ),
            system=SystemConfig(
                project_root=self.context.project_root,
                code_dir=self.context.project_root / "code",
                experiment_dir=self.context.experiment_dir,
                artifact_dir=artifact_dir,
                model_path=artifact_dir / "model",
                template=self.invocation_params.get("model_template", "gpu-dev-5m"),
                experiment_id=self.context.experiment_id,
                random_seed=random_seed,
                use_gpu=use_gpu_param if use_gpu_param is not None else False,
            ),
            model=model_cfg,
        )

    def execute(self) -> ModuleResult:
        """
        Train a model based on the selected template and persist artifacts.

        Returns:
            ModuleResult containing leaderboard path, submission path, and CV metrics.

        Raises:
            FileNotFoundError: When preprocessing artifacts are missing.
            RuntimeError: When template resolution fails.
        """
        import pandas as pd
        from rich.table import Table

        template_name = self.invocation_params.get("model_template", "gpu-dev-5m")

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
        preprocess_exp_dir = self.invocation_params.get("preprocess_exp_dir")

        # Priority order (lowest to highest): defaults -> template -> CLI args -> convenience flags
        # 1. Start with defaults
        preset = getattr(config, "AUTOGLUON_PRESET", "medium")
        time_limit = getattr(config, "AUTOGLUON_TIME_LIMIT", None)
        use_gpu_param = None

        loader = TemplateLoader(self.context.project_root, template_type="model")
        template_cfg = loader.load(template_name)

        # Check if template was found (when explicitly specified and not using defaults)
        if not template_cfg and template_name not in ["gpu-dev-5m", "cpu-dev-5m"]:
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

        # Flatten config dict into template_cfg for backward compatibility
        if "config" in template_cfg:
            config_dict = template_cfg.pop("config")
            # Merge config_dict into template_cfg, but don't override existing top-level keys
            for key, value in config_dict.items():
                if key not in template_cfg:
                    template_cfg[key] = value

        # 2. Override with template values
        preset = template_cfg.get("preset") or template_cfg.get("presets") or preset
        time_limit = template_cfg.get("time_limit") or time_limit
        use_gpu_from_template = template_cfg.get("use_gpu")
        if use_gpu_from_template is not None:
            use_gpu_param = use_gpu_from_template

        # 3. Override with CLI args if provided
        if self.invocation_params.get("preset"):
            preset = self.invocation_params.get("preset")
        if self.invocation_params.get("time_limit"):
            time_limit = self.invocation_params.get("time_limit")
        if self.invocation_params.get("use_gpu") is not None:
            use_gpu_param = self.invocation_params.get("use_gpu")

        # 4. Convenience flags have highest priority (override everything)
        if self.invocation_params.get("dev"):
            preset = "medium"
            time_limit = 300
            use_gpu_param = 0
            self.invocation_params["_convenience_flag"] = "dev"
            # Update invocation_params so header shows final values
            self.invocation_params["preset"] = preset
            self.invocation_params["time_limit"] = time_limit
            self.invocation_params["use_gpu"] = use_gpu_param
        elif self.invocation_params.get("smoke"):
            preset = "medium"
            time_limit = 60
            use_gpu_param = 0
            self.invocation_params["_convenience_flag"] = "smoke"
            # Update invocation_params so header shows final values
            self.invocation_params["preset"] = preset
            self.invocation_params["time_limit"] = time_limit
            self.invocation_params["use_gpu"] = use_gpu_param

        # Allow template to supply default preprocess template (overridable by CLI)
        if not preprocess_template:
            preprocess_template = template_cfg.get("preprocess_template")

        train_df, test_df, sample_weight, orig_df = _load_processed_or_raw(
            self.context,
            config,
            preprocess_template,
            preprocess_exp_dir=preprocess_exp_dir,
        )
        target = getattr(config, "TARGET_COLUMN", None)
        if target is None or target not in train_df.columns:
            marker = artifact_dir / "model_failed.txt"
            marker.write_text("TARGET_COLUMN missing; aborting model step.")
            return ModuleResult(success=False, error="TARGET_COLUMN missing", artifacts=[marker])

        # Load model implementation (defaults to autogluon_baseline)
        model_implementation = template_cfg.get("model", "autogluon_baseline")
        console.print(f"[cyan]Using model implementation: {model_implementation}[/cyan]")

        # Load model module
        model_module = _load_model_module(self.context.project_root, model_implementation)

        # Build ModelConfig for model interface
        model_config = self._build_model_config(template_cfg, config, preset, time_limit, use_gpu_param, artifact_dir)

        # Build artifacts dict with orig_df and sample_weight
        artifacts = {}
        if orig_df is not None:
            artifacts['orig_df'] = orig_df
        if sample_weight is not None:
            artifacts['sample_weight'] = sample_weight

        # Call train()
        console.print("[green]Training model...[/green]")
        train_result = model_module.train(
            train_df=train_df,
            val_df=None,
            config=model_config,
            artifacts=artifacts if artifacts else None,  # Backward compat: None if empty
        )

        # Handle return: (model, summary) tuple or just model
        if isinstance(train_result, tuple) and len(train_result) == 2:
            predictor, training_summary = train_result
        else:
            predictor, training_summary = train_result, {}

        local_cv_score = training_summary.get("local_cv_score")

        # Save leaderboard if returned by train() (extract before JSON serialization)
        lb_path = None
        leaderboard = training_summary.pop("leaderboard", None)  # Remove from dict to avoid JSON error
        if leaderboard is not None:
            try:
                lb_path = artifact_dir / "leaderboard.csv.gz"
                leaderboard.to_csv(lb_path, index=False, compression='infer')
            except Exception:
                pass

        # Print training summary
        _print_training_summary(
            project_root=self.context.project_root,
            experiment_id=self.context.experiment_id,
            project_name=self.context.project_name,
            template_name=template_name,
            preset=preset,
            time_limit=time_limit,
            use_gpu=use_gpu_param,
            local_cv_score=local_cv_score,
            best_model=model_implementation,
            preprocess_template=preprocess_template,
            leaderboard_path=lb_path,
        )

        return ModuleResult(
            success=True,
            payload={
                "model_implementation": model_implementation,
                "model_artifact": str(artifact_dir / "model"),  # Required by predict module
                "local_cv_score": local_cv_score,
                "training_summary": training_summary,
                "preset": preset,
                "time_limit": time_limit,
                "use_gpu": use_gpu_param,
                "template": template_name,
                "preprocess_template": preprocess_template,
                "leaderboard": str(lb_path) if lb_path else None,
            },
            artifacts=[f for f in [artifact_dir / "model", lb_path] if f and f.exists()],
        )
