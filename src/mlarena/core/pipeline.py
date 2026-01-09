"""
Pipeline execution logic for MLArena.

Resolves module dependencies and coordinates state updates.
"""

import traceback
from datetime import datetime as dt
from typing import Dict, List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .module import BaseModule, ModuleResult
from .experiment import OverwriteLockedError
from .display import print_module_header, print_module_footer, format_path_relative, extract_template_overrides
from pathlib import Path


class PipelineExecutor:
    """Resolve dependencies and run MLArena modules with consistent UX output."""

    def __init__(self, modules: Dict[str, BaseModule]):
        """
        Initialize the executor.

        Args:
            modules: Mapping of module name to instantiated ``BaseModule``.
        """
        self.modules = modules

    def _collect_module_header_data(self, module_name: str, module: BaseModule) -> dict:
        """
        Collect template, invocation overrides, and IO path metadata for header display.

        Args:
            module_name: Name of the module being prepared.
            module: Instantiated module object.

        Returns:
            Dictionary with template name/config, CLI overrides, and resolved input/output paths.
        """
        invocation = getattr(module, "invocation_params", {})

        # For already-completed modules, get invocation from state if invocation_params is empty
        if not invocation:
            state_entry = module.context.state.modules.get(module_name)
            if state_entry and hasattr(state_entry, "invocation") and state_entry.invocation:
                invocation = state_entry.invocation

        # Get template info
        template_name = invocation.get(f"{module_name}_template")
        template_config = {}
        cli_overrides = {}

        if template_name:
            try:
                import sys
                from pathlib import Path as P
                REPO_ROOT = P(__file__).resolve().parents[3]
                sys.path.insert(0, str(REPO_ROOT / "scripts"))
                from template_loader import load_templates

                templates, _ = load_templates(module_name, module.context.project_root, suppress_warnings=True)
                template_config = templates.get(template_name, {})

                if invocation.get("dev"):
                    invocation["preset"] = "medium"
                    invocation["time_limit"] = 300
                    invocation["use_gpu"] = 0
                    invocation["_convenience_flag"] = "dev"
                elif invocation.get("smoke"):
                    invocation["preset"] = "medium"
                    invocation["time_limit"] = 60
                    invocation["use_gpu"] = 0
                    invocation["_convenience_flag"] = "smoke"

                cli_overrides = extract_template_overrides(template_config.get("config", {}), invocation)
            except Exception:
                pass
        # If preprocess provides an explicit template config (e.g., in preprocess-tune),
        # merge it into the header so overrides like use_original_features_only are visible.
        if module_name == "preprocess" and invocation.get("preprocess_template_config"):
            override_tpl = invocation.get("preprocess_template_config") or {}
            if isinstance(override_tpl, dict):
                base_cfg = template_config.get("config", {}) if isinstance(template_config, dict) else {}
                override_cfg = override_tpl.get("config", {}) if isinstance(override_tpl.get("config", {}), dict) else {}
                merged_cfg = {**base_cfg, **override_cfg}
                template_config = dict(template_config) if isinstance(template_config, dict) else {}
                if override_tpl.get("module"):
                    template_config["module"] = override_tpl.get("module")
                template_config["config"] = merged_cfg

        input_paths = {}
        output_paths = {}

        if module_name == "init":
            model_tpl = invocation.get("model_template")
            preprocess_tpl = invocation.get("preprocess_template")

            if model_tpl or preprocess_tpl:
                input_paths["pipeline"] = "auto-flow"
                if model_tpl:
                    input_paths["model_template"] = model_tpl
                if preprocess_tpl:
                    input_paths["preprocess_template"] = preprocess_tpl

                modules = ["init", "eda", "prep", "model"]
                if not invocation.get("skip_submit"):
                    modules.extend(["submit", "fetch"])
                flow = " → ".join(modules)
                output_paths["flow"] = flow

        elif module_name == "preprocess":
            input_source = invocation.get("input_source")
            chain_exp_id = invocation.get("chain_exp_id", "")

            if input_source:
                input_paths["from"] = input_source
                if chain_exp_id:
                    input_paths["train"] = f"experiments/{chain_exp_id}/{input_source}/artifacts/preprocess/train_processed.csv.gz"
                    input_paths["test"] = f"experiments/{chain_exp_id}/{input_source}/artifacts/preprocess/test_processed.csv.gz"
                else:
                    input_paths["train"] = f"experiments/pre-{input_source}/artifacts/preprocess/train_processed.csv.gz"
                    input_paths["test"] = f"experiments/pre-{input_source}/artifacts/preprocess/test_processed.csv.gz"
            else:
                try:
                    from mlarena.utils.project import data_paths, load_project_config
                    config = module.context.config_module or load_project_config(module.context.project_root)
                    train_path, test_path = data_paths(config)
                    input_paths["train"] = format_path_relative(train_path, module.context.project_root)
                    input_paths["test"] = format_path_relative(test_path, module.context.project_root)

                    config_dict = template_config.get("config", {})
                    if config_dict and "orig_path" in config_dict:
                        orig_path = config_dict["orig_path"]
                        input_paths["original"] = format_path_relative(orig_path, module.context.project_root)
                except Exception:
                    pass

            experiment_id = module.context.experiment_id
            output_paths["train"] = f"experiments/{experiment_id}/artifacts/preprocess/train_processed.csv.gz"
            output_paths["test"] = f"experiments/{experiment_id}/artifacts/preprocess/test_processed.csv.gz"

        elif module_name == "model":
            preprocess_template = invocation.get("preprocess_template") or template_config.get("preprocess_template")
            preprocess_exp_dir = invocation.get("preprocess_exp_dir")
            shapes_meta = None

            def _load_shapes_from_state(exp_dir: Path):
                state_path = exp_dir / "state.json"
                if not state_path.exists():
                    return None
                try:
                    import json
                    with open(state_path) as f:
                        state = json.load(f)
                    payload = state.get("modules", {}).get("preprocess", {}).get("payload", {})
                    shapes = payload.get("shapes", {})
                    train_after = shapes.get("train_after")
                    test_after = shapes.get("test_after")
                    tuning_after = shapes.get("tuning_after")

                    custom_state = payload.get("custom_module_state", {})
                    eval_rows = custom_state.get("eval_rows")

                    result = {}
                    if train_after:
                        result["train"] = train_after
                    if test_after:
                        result["test"] = test_after
                    if tuning_after:
                        result["tuning"] = tuning_after
                    if eval_rows:
                        result["eval"] = (eval_rows, None)

                    return result if result else None
                except Exception:
                    return None
                return None

            def _load_shapes_from_chain_root(root: Path):
                shapes = _load_shapes_from_state(root)
                if shapes:
                    return shapes
                candidates = []
                for subdir in root.iterdir():
                    if not subdir.is_dir():
                        continue
                    try:
                        idx = int(subdir.name.split("-")[0])
                    except Exception:
                        continue
                    candidates.append((idx, subdir))
                if candidates:
                    candidates.sort(reverse=True)
                    _, latest_dir = candidates[0]
                    shapes = _load_shapes_from_state(latest_dir)
                    if shapes:
                        return shapes
                return None

            def _resolve_preprocess_dir():
                if preprocess_exp_dir:
                    return Path(preprocess_exp_dir)
                if preprocess_template:
                    default_dir = module.context.project_root / f"experiments/pre-{preprocess_template}"
                    if default_dir.exists():
                        return default_dir
                    experiments_dir = module.context.project_root / "experiments"
                    candidates = sorted(
                        experiments_dir.glob(f"pre-*/[0-9]*-{preprocess_template}"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    if candidates:
                        return candidates[0]
                return None

            pp_dir = _resolve_preprocess_dir()
            if pp_dir:
                train_pp = pp_dir / "artifacts" / "preprocess" / "train_processed.csv.gz"
                test_pp = pp_dir / "artifacts" / "preprocess" / "test_processed.csv.gz"
                tuning_pp = pp_dir / "artifacts" / "preprocess" / "tuning_processed.csv.gz"
                eval_pp = pp_dir / "artifacts" / "preprocess" / "eval_processed.csv.gz"

                input_paths["train"] = format_path_relative(train_pp, module.context.project_root)
                input_paths["test"] = format_path_relative(test_pp, module.context.project_root)

                if tuning_pp.exists():
                    input_paths["tuning"] = format_path_relative(tuning_pp, module.context.project_root)
                if eval_pp.exists():
                    input_paths["eval"] = format_path_relative(eval_pp, module.context.project_root)

                shapes_meta = _load_shapes_from_chain_root(pp_dir)
                if shapes_meta:
                    input_paths["__shapes__"] = shapes_meta
            else:
                input_paths["train"] = "data/train.csv.gz"
                input_paths["test"] = "data/test.csv.gz"
                shapes_meta = None

            experiment_id = module.context.experiment_id
            output_paths["model"] = f"experiments/{experiment_id}/artifacts/"
            output_paths["submission"] = "submissions/"

        elif module_name == "preprocess-tune":
            model_tpl = invocation.get("model_template")
            if model_tpl:
                input_paths["model_template"] = model_tpl
            super_chain_path = invocation.get("super_chain")
            if super_chain_path:
                input_paths["super_chain"] = super_chain_path
            study_name = invocation.get("study_name")
            if study_name:
                input_paths["study_name"] = str(study_name)
            n_trials = invocation.get("n_trials")
            if n_trials is not None:
                input_paths["n_trials"] = str(n_trials)
            optuna_workers = invocation.get("optuna_workers")
            if optuna_workers is not None:
                input_paths["optuna_workers"] = str(optuna_workers)
            max_trial_sec = invocation.get("max_trial_sec")
            if max_trial_sec is not None:
                input_paths["max_trial_sec"] = str(max_trial_sec)

        elif module_name == "eda":
            input_paths["train"] = "data/train.csv.gz"
            input_paths["test"] = "data/test.csv.gz"

            experiment_id = module.context.experiment_id
            output_paths["summary"] = f"experiments/{experiment_id}/eda_summary.txt"

        elif module_name == "predict":
            experiment_id = invocation.get("experiment_id")
            if not experiment_id:
                predict_state = module.context.state.modules.get("predict")
                if predict_state and hasattr(predict_state, "invocation") and predict_state.invocation:
                    experiment_id = predict_state.invocation.get("experiment_id")
            if not experiment_id:
                experiment_id = module.context.experiment_id

            if experiment_id:
                model_state_path = module.context.project_root / "experiments" / experiment_id / "state.json"
                preprocess_template = None
                preprocess_exp_dir = None
                shapes_meta = None
                model_template = None

                if model_state_path.exists():
                    try:
                        import json
                        with open(model_state_path) as f:
                            state = json.load()

                        model_payload = state.get("modules", {}).get("model", {}).get("payload", {})
                        model_invocation = state.get("modules", {}).get("model", {}).get("invocation", {})

                        preprocess_template = model_payload.get("preprocess_template") or model_invocation.get("preprocess_template")
                        preprocess_exp_dir = model_invocation.get("preprocess_exp_dir")
                        model_template = model_payload.get("template") or model_invocation.get("model_template")
                    except Exception:
                        pass

                input_paths["model"] = f"experiments/{experiment_id}/artifacts/model"

                if preprocess_template:
                    if preprocess_exp_dir:
                        pp_dir = Path(preprocess_exp_dir)
                    else:
                        pp_dir = module.context.project_root / f"experiments/pre-{preprocess_template}"

                    test_pp = pp_dir / "artifacts" / "preprocess" / "test_processed.csv.gz"
                    if test_pp.exists():
                        input_paths["test"] = format_path_relative(test_pp, module.context.project_root)

                        def _load_shapes_from_state(exp_dir: Path):
                            state_path = exp_dir / "state.json"
                            if not state_path.exists():
                                return None
                            try:
                                import json
                                with open(state_path) as f:
                                    state = json.load()
                                payload = state.get("modules", {}).get("preprocess", {}).get("payload", {})
                                shapes = payload.get("shapes", {})
                                test_after = shapes.get("test_after")
                                return {"test": test_after} if test_after else None
                            except Exception:
                                return None

                        shapes_meta = _load_shapes_from_state(pp_dir)
                        if shapes_meta:
                            input_paths["__shapes__"] = shapes_meta
                    else:
                        input_paths["test"] = "data/test.csv.gz"
                else:
                    input_paths["test"] = "data/test.csv.gz"

                if preprocess_template:
                    input_paths["preprocessing"] = preprocess_template

                if model_template:
                    input_paths["model_template"] = model_template

        elif module_name == "submit":
            predict_payload = module.context.state.modules.get("predict")
            if predict_payload and hasattr(predict_payload, "payload") and predict_payload.payload:
                submission_file = predict_payload.payload.get("submission_file")
                if submission_file:
                    input_paths["submission"] = format_path_relative(submission_file, module.context.project_root)
            output_paths["kaggle"] = "Kaggle API upload"

        elif module_name == "fetch-score":
            experiment_id = invocation.get("experiment_id")
            if experiment_id:
                input_paths["submission"] = f"experiments/{experiment_id}/"
                output_paths["score"] = "Public leaderboard score"

        return {
            "template_name": template_name,
            "template_config": template_config,
            "cli_overrides": cli_overrides,
            "input_paths": input_paths,
            "output_paths": output_paths,
        }

    def _resolve_execution_order(self, target_module: str) -> List[str]:
        """
        Topologically sort dependencies for a target module.

        Args:
            target_module: Module to execute.

        Returns:
            Ordered list of module names where dependencies precede dependents.

        Raises:
            KeyError: When a referenced dependency is missing.
            ValueError: On cyclic dependency detection.
        """
        if target_module not in self.modules:
            raise KeyError(f"Module '{target_module}' is not registered in pipeline.")

        order: List[str] = []
        temp: set[str] = set()
        perm: set[str] = set()

        def visit(name: str) -> None:
            if name in perm:
                return
            if name in temp:
                raise ValueError(f"Cyclic dependency detected at '{name}'.")
            temp.add(name)
            module = self.modules.get(name)
            if module is None:
                raise KeyError(f"Dependency '{name}' not found.")
            for dep in module.dependencies:
                if dep not in self.modules:
                    raise KeyError(f"Dependency '{dep}' required by '{name}' is missing.")
                visit(dep)
            perm.add(name)
            temp.remove(name)
            order.append(name)

        visit(target_module)
        return order

    def run_module(self, module_name: str, force: bool = False, skip_deps: bool = False) -> Dict[str, ModuleResult]:
        """
        Execute a module (and optionally its dependencies) with rich header/footer output.

        Args:
            module_name: Name of the primary module to run.
            force: Re-run the target even if previously completed.
            skip_deps: Skip dependency resolution and run only the target module.

        Returns:
            Mapping of module name to ``ModuleResult`` for executed steps.
        """
        results: Dict[str, ModuleResult] = {}
        execution_plan = [module_name] if skip_deps else self._resolve_execution_order(module_name)
        executed_in_session = set()

        # Cleanup stale "running" states from interrupted executions
        for name in execution_plan:
            module = self.modules[name]
            state_entry = module.context.state.modules.get(name)
            if state_entry and state_entry.status == "running":
                # Mark as failed - was interrupted
                module.context.state.fail_module(name, "Interrupted - marked as failed on restart")
                module.context.state.save()

        for name in execution_plan:
            module = self.modules[name]
            state_entry = module.context.state.modules.get(name)
            already_completed = state_entry and state_entry.status == "completed"

            # Validate module can run (before checking completed status)
            can_run, reason = module.can_run()
            if not can_run:
                module.context.state.fail_module(name, reason)
                module.context.state.save()
                results[name] = ModuleResult(success=False, error=reason)
                break

            # Handle wait time between submission and score fetching
            if name == "fetch-score" and "submit" in executed_in_session:
                wait_seconds = 30
                if module.context.config:
                    wait_seconds = getattr(module.context.config, "wait_seconds", 30)
                
                if wait_seconds > 0:
                    console = Console(force_terminal=True)
                    console.print(f"\n[dim]Waiting {wait_seconds}s for Kaggle processing (submit → fetch-score)...[/dim]")
                    import time
                    time.sleep(wait_seconds)

            # Check for overwrite lock when forcing completed module
            if already_completed and force and name == module_name:
                is_locked, lock_metadata = module.context.state.is_locked(name)
                if is_locked:
                    lock_path = module.context.state.get_lock_path(name)
                    # Convert to relative path for user-friendly error message
                    try:
                        relative_lock_path = lock_path.relative_to(module.context.state.project_root)
                    except ValueError:
                        relative_lock_path = lock_path
                    raise OverwriteLockedError(name, relative_lock_path, lock_metadata)

            # Handle already completed modules
            # force only applies to target module, not dependencies
            if already_completed and not (force and name == module_name):
                # Init module should always run to show project status (even if completed)
                if name == "init":
                    # Let init module execute to show config
                    pass
                elif name == module_name:
                    # Target module is already completed - show full header/footer with "Already completed" status
                    console = Console(force_terminal=True)

                    # Collect header data (same as normal execution)
                    try:
                        header_data = self._collect_module_header_data(name, module)
                    except Exception as e:
                        console.print(f"[dim red]Data collection error: {e}[/dim red]")
                        header_data = {}

                    # Get started_at from state_entry
                    started_at = state_entry.started_at or dt.now().isoformat()

                    # Display module header with collected data
                    try:
                        print_module_header(
                            module_name=name,
                            started_at=started_at,
                            experiment_id=module.context.experiment_id,
                            template_name=header_data.get("template_name"),
                            template_config=header_data.get("template_config"),
                            cli_overrides=header_data.get("cli_overrides"),
                            cli_invocation=getattr(module, "invocation_params", {}),
                            input_paths=header_data.get("input_paths"),
                            output_paths=header_data.get("output_paths"),
                            project_root=module.context.project_root,
                            project_name=module.context.project_name,
                            console=console
                        )
                    except Exception as e:
                        # Fallback to simple header if display fails
                        console.print(f"[dim red]Header display error: {e}[/dim red]")
                        start_time = dt.now().strftime("%Y-%m-%d %H:%M:%S")
                        header_content = f"[bold]Started:[/bold] {start_time}"
                        console.print(Panel(header_content, title=f"[bold cyan]{name.upper()}[/bold cyan]", border_style="cyan"))

                    # Calculate duration
                    completed_time_raw = state_entry.finished_at or state_entry.started_at or dt.now().isoformat()
                    try:
                        started_dt = dt.fromisoformat(started_at.replace('Z', '+00:00'))
                        finished_dt = dt.fromisoformat(completed_time_raw.replace('Z', '+00:00'))
                        duration = (finished_dt - started_dt).total_seconds()
                    except Exception:
                        duration = 0.0

                    payload = state_entry.payload or {}
                    output_paths = {}
                    metrics = {}
                    shapes = payload.get("shapes")

                    if name == "preprocess":
                        is_pass_through = shapes and shapes.get("pass_through", False)

                        if not is_pass_through:
                            if "train_processed" in payload:
                                output_paths["train"] = format_path_relative(payload["train_processed"], module.context.project_root)
                            if "test_processed" in payload:
                                output_paths["test"] = format_path_relative(payload["test_processed"], module.context.project_root)
                            if "orig_processed" in payload and payload["orig_processed"]:
                                output_paths["orig"] = format_path_relative(payload["orig_processed"], module.context.project_root)
                            if "tuning_processed" in payload and payload["tuning_processed"]:
                                output_paths["tuning"] = format_path_relative(payload["tuning_processed"], module.context.project_root)

                        if "custom_module_state" in payload and "weights_path" in payload["custom_module_state"]:
                            output_paths["weights"] = format_path_relative(payload["custom_module_state"]["weights_path"], module.context.project_root)
                        if "custom_module_state" in payload and "av_stats" in payload["custom_module_state"]:
                            metrics["av_stats"] = payload["custom_module_state"]["av_stats"]
                            if "output_rows" in payload["custom_module_state"]:
                                metrics["clipped_rows"] = payload["custom_module_state"]["output_rows"]
                    elif name == "model":
                        if "model_artifact" in payload:
                            output_paths["model"] = format_path_relative(payload.get("model_artifact"), module.context.project_root)
                        if "leaderboard" in payload:
                            output_paths["leaderboard"] = format_path_relative(payload.get("leaderboard"), module.context.project_root)
                        if "submission_file" in payload:
                            output_paths["submission"] = format_path_relative(payload.get("submission_file"), module.context.project_root)
                        if "local_cv_score" in payload:
                            metrics["local_cv_score"] = payload["local_cv_score"]
                        if "best_model" in payload:
                            metrics["best_model"] = payload["best_model"]
                    elif name == "eda":
                        if "summary_file" in payload and isinstance(payload.get("summary_file"), str):
                            output_paths["summary"] = format_path_relative(payload.get("summary_file"), module.context.project_root)
                        if "train_profile_path" in payload:
                            output_paths["train_profile"] = format_path_relative(payload.get("train_profile_path"), module.context.project_root)
                        if "test_profile_path" in payload:
                            output_paths["test_profile"] = format_path_relative(payload.get("test_profile_path"), module.context.project_root)

                        if "train_profile" in payload and isinstance(payload["train_profile"], dict):
                            train_prof = payload["train_profile"]
                            if "summary" in train_prof and "table" in train_prof["summary"]:
                                table = train_prof["summary"]["table"]
                                metrics["train_shape"] = f"{table.get('n', 0):,} × {table.get('n_var', 0)}"
                                if table.get("n_cells_missing", 0) > 0:
                                    metrics["train_missing"] = f"{table.get('n_cells_missing'):,} cells ({table.get('p_cells_missing', 0)*100:.1f}%)"

                        if "test_profile" in payload and isinstance(payload["test_profile"], dict):
                            test_prof = payload["test_profile"]
                            if "summary" in test_prof and "table" in test_prof["summary"]:
                                table = test_prof["summary"]["table"]
                                metrics["test_shape"] = f"{table.get('n', 0):,} × {table.get('n_var', 0)}"

                        if "target" in payload:
                            metrics["target"] = payload["target"]

                    elif name == "predict":
                        if "submission_file" in payload:
                            output_paths["submission"] = format_path_relative(payload["submission_file"], module.context.project_root)

                    elif name == "submit":
                        if "submission_file" in payload:
                            output_paths["submission"] = format_path_relative(payload["submission_file"], module.context.project_root)
                        if "local_cv_score" in payload:
                            metrics["local_cv_score"] = payload["local_cv_score"]
                        if "public_score" in payload and payload["public_score"]:
                            metrics["public_score"] = payload["public_score"]

                    elif name == "fetch-score":
                        if "score" in payload and payload["score"]:
                            metrics["public_score"] = payload["score"]
                        submission_info = payload.get("matched_submission") or payload.get("latest_submission")
                        if submission_info:
                            metrics["submission_id"] = submission_info
                        if payload.get("target_submission"):
                            metrics["submission_target"] = payload["target_submission"]

                    metrics["status"] = "✓ [red]Already completed[/red]"

                    try:
                        print_module_footer(
                            module_name=name,
                            finished_at=completed_time_raw,
                            duration=duration,
                            output_paths=output_paths,
                            metrics=metrics,
                            shapes=shapes,
                            project_root=module.context.project_root,
                            project_name=module.context.project_name,
                            experiment_id=module.context.experiment_id,
                            cli_invocation=getattr(module, "invocation_params", {}),
                            console=console
                        )
                    except Exception as e:
                        console.print(f"[dim red]Footer display error: {e}[/dim red]")

                    console.print(f"\n[dim]Use [cyan]--force[/cyan] to re-run this module[/dim]\n")

                    results[name] = ModuleResult(success=True, payload=state_entry.payload)
                    continue
                else:
                    results[name] = ModuleResult(success=True, payload=state_entry.payload)
                    continue

            defer_save = name == "init" and not module.context.project_root.exists()

            module.context.state.start_module(name, getattr(module, "invocation_params", {}))
            if not defer_save:
                module.context.state.save()

            module_state = module.context.state.modules.get(name)
            console = Console(force_terminal=True)

            try:
                header_data = self._collect_module_header_data(name, module)
            except Exception as e:
                console.print(f"[dim red]Data collection error: {e}[/dim red]")
                header_data = {}

            if module_state and hasattr(module_state, 'started_at'):
                started_at = module_state.started_at
            else:
                started_at = dt.now().isoformat()

            try:
                print_module_header(
                    module_name=name,
                    started_at=started_at,
                    experiment_id=module.context.experiment_id,
                    template_name=header_data.get("template_name"),
                    template_config=header_data.get("template_config"),
                    cli_overrides=header_data.get("cli_overrides"),
                    cli_invocation=getattr(module, "invocation_params", {}),
                    input_paths=header_data.get("input_paths"),
                    output_paths=header_data.get("output_paths"),
                    project_root=module.context.project_root,
                    project_name=module.context.project_name,
                    console=console
                )
            except Exception as e:
                console.print(f"[dim red]Header display error: {e}[/dim red]")
                import traceback
                console.print(f"[dim]{traceback.format_exc()}[/dim]")
                start_time = dt.now().strftime("%Y-%m-%d %H:%M:%S")
                header_content = f"[bold]Started:[/bold] {start_time}"
                console.print(Panel(header_content, title=f"[bold cyan]{name.upper()}[/bold cyan]", border_style="cyan"))

            try:
                outcome = module.execute()
            except Exception as exc:
                import traceback
                error_msg = f"{exc}"
                detail = traceback.format_exc()
                module.context.state.fail_module(name, error_msg)
                if not defer_save or module.context.project_root.exists():
                    module.context.state.save()

                module_state_after = module.context.state.modules.get(name)
                try:
                    if module_state_after:
                        started_at_str = module_state_after.started_at or dt.now().isoformat()
                        finished_at_str = module_state_after.finished_at or dt.now().isoformat()
                    else:
                        started_at_str = dt.now().isoformat()
                        finished_at_str = dt.now().isoformat()
                    started_at = dt.fromisoformat(started_at_str.replace('Z', '+00:00'))
                    finished_at = dt.fromisoformat(finished_at_str.replace('Z', '+00:00'))
                    duration = (finished_at - started_at).total_seconds()

                    print_module_footer(
                        module_name=name,
                        finished_at=finished_at_str,
                        duration=duration,
                        error=error_msg,
                        project_root=module.context.project_root,
                        project_name=module.context.project_name,
                        experiment_id=module.context.experiment_id,
                        cli_invocation=getattr(module, "invocation_params", {}),
                        console=console
                    )
                except:
                    pass

                results[name] = ModuleResult(success=False, error=detail)
                break

            if outcome.success:
                module.context.state.complete_module(name, outcome.payload)
                if not defer_save or module.context.project_root.exists():
                    module.context.state.save()

                lock_param = getattr(module, "invocation_params", {}).get("lock", False)
                if lock_param:
                    lock_metadata = {}
                    if name == "preprocess":
                        template = getattr(module, "invocation_params", {}).get("preprocess_template")
                        if template:
                            lock_metadata["template"] = template
                    elif name == "model":
                        template = getattr(module, "invocation_params", {}).get("model_template")
                        if template:
                            lock_metadata["template"] = template

                    module.context.state.create_lock(name, lock_metadata)

                module_state_after = module.context.state.modules.get(name)
                try:
                    if module_state_after:
                        started_at_str = module_state_after.started_at or dt.now().isoformat()
                        finished_at_str = module_state_after.finished_at or dt.now().isoformat()
                    else:
                        started_at_str = dt.now().isoformat()
                        finished_at_str = dt.now().isoformat()
                    started_at = dt.fromisoformat(started_at_str.replace('Z', '+00:00'))
                    finished_at = dt.fromisoformat(finished_at_str.replace('Z', '+00:00'))
                    duration = (finished_at - started_at).total_seconds()

                    payload = outcome.payload or {}
                    output_paths = {}
                    metrics = {}
                    shapes = payload.get("shapes")

                    if name == "preprocess":
                        is_pass_through = shapes and shapes.get("pass_through", False)

                        if not is_pass_through:
                            if "train_processed" in payload:
                                output_paths["train"] = format_path_relative(payload["train_processed"], module.context.project_root)
                            if "test_processed" in payload:
                                output_paths["test"] = format_path_relative(payload["test_processed"], module.context.project_root)
                            if "orig_processed" in payload and payload["orig_processed"]:
                                output_paths["orig"] = format_path_relative(payload["orig_processed"], module.context.project_root)
                            if "tuning_processed" in payload and payload["tuning_processed"]:
                                output_paths["tuning"] = format_path_relative(payload["tuning_processed"], module.context.project_root)

                        if "custom_module_state" in payload and "weights_path" in payload["custom_module_state"]:
                            output_paths["weights"] = format_path_relative(payload["custom_module_state"]["weights_path"], module.context.project_root)
                        if "custom_module_state" in payload and "av_stats" in payload["custom_module_state"]:
                            metrics["av_stats"] = payload["custom_module_state"]["av_stats"]
                            if "output_rows" in payload["custom_module_state"]:
                                metrics["clipped_rows"] = payload["custom_module_state"]["output_rows"]
                    elif name == "model":
                        if "model_artifact" in payload:
                            output_paths["model"] = format_path_relative(payload.get("model_artifact"), module.context.project_root)
                        if "leaderboard" in payload:
                            output_paths["leaderboard"] = format_path_relative(payload.get("leaderboard"), module.context.project_root)
                        if "submission_file" in payload:
                            output_paths["submission"] = format_path_relative(payload.get("submission_file"), module.context.project_root)
                        if "local_cv_score" in payload:
                            metrics["local_cv_score"] = payload["local_cv_score"]
                        if "best_model" in payload:
                            metrics["best_model"] = payload["best_model"]
                    elif name == "eda":
                        if "summary_file" in payload and isinstance(payload.get("summary_file"), str):
                            output_paths["summary"] = format_path_relative(payload.get("summary_file"), module.context.project_root)
                        if "train_profile_path" in payload:
                            output_paths["train_profile"] = format_path_relative(payload.get("train_profile_path"), module.context.project_root)
                        if "test_profile_path" in payload:
                            output_paths["test_profile"] = format_path_relative(payload.get("test_profile_path"), module.context.project_root)

                        if "train_profile" in payload and isinstance(payload["train_profile"], dict):
                            train_prof = payload["train_profile"]
                            if "summary" in train_prof and "table" in train_prof["summary"]:
                                table = train_prof["summary"]["table"]
                                metrics["train_shape"] = f"{table.get('n', 0):,} × {table.get('n_var', 0)}"
                                if table.get("n_cells_missing", 0) > 0:
                                    metrics["train_missing"] = f"{table.get('n_cells_missing'):,} cells ({table.get('p_cells_missing', 0)*100:.1f}%)"

                        if "test_profile" in payload and isinstance(payload["test_profile"], dict):
                            test_prof = payload["test_profile"]
                            if "summary" in test_prof and "table" in test_prof["summary"]:
                                table = test_prof["summary"]["table"]
                                metrics["test_shape"] = f"{table.get('n', 0):,} × {table.get('n_var', 0)}"

                        if "target" in payload:
                            metrics["target"] = payload["target"]

                    elif name == "predict":
                        if "submission_file" in payload:
                            output_paths["submission"] = format_path_relative(payload["submission_file"], module.context.project_root)

                    elif name == "submit":
                        if "submission_file" in payload:
                            output_paths["submission"] = format_path_relative(payload["submission_file"], module.context.project_root)
                        if "local_cv_score" in payload:
                            metrics["local_cv_score"] = payload["local_cv_score"]
                        if "public_score" in payload and payload["public_score"]:
                            metrics["public_score"] = payload["public_score"]

                    elif name == "fetch-score":
                        if "score" in payload and payload["score"]:
                            metrics["public_score"] = payload["score"]
                        submission_info = payload.get("matched_submission") or payload.get("latest_submission")
                        if submission_info:
                            metrics["submission_id"] = submission_info
                        if payload.get("target_submission"):
                            metrics["submission_target"] = payload["target_submission"]

                    print_module_footer(
                        module_name=name,
                        finished_at=finished_at_str,
                        duration=duration,
                        output_paths=output_paths,
                        metrics=metrics,
                        shapes=shapes,
                        project_root=module.context.project_root,
                        project_name=module.context.project_name,
                        experiment_id=module.context.experiment_id,
                        cli_invocation=getattr(module, "invocation_params", {}),
                        console=console
                    )

                    from mlarena.core.module import print_next_steps
                    print_next_steps(name, module.context.project_name, module.context.experiment_id, console)
                except Exception as e:
                    pass

                results[name] = outcome
                if outcome.success:
                    executed_in_session.add(name)
            else:
                module.context.state.fail_module(name, outcome.error or "unknown error")
                if not defer_save or module.context.project_root.exists():
                    module.context.state.save()

                module_state_after = module.context.state.modules.get(name)
                try:
                    if module_state_after:
                        started_at_str = module_state_after.started_at or dt.now().isoformat()
                        finished_at_str = module_state_after.finished_at or dt.now().isoformat()
                    else:
                        started_at_str = dt.now().isoformat()
                        finished_at_str = dt.now().isoformat()
                    started_at = dt.fromisoformat(started_at_str.replace('Z', '+00:00'))
                    finished_at = dt.fromisoformat(finished_at_str.replace('Z', '+00:00'))
                    duration = (finished_at - started_at).total_seconds()

                    print_module_footer(
                        module_name=name,
                        finished_at=finished_at_str,
                        duration=duration,
                        error=outcome.error or "unknown error",
                        project_root=module.context.project_root,
                        project_name=module.context.project_name,
                        experiment_id=module.context.experiment_id,
                        cli_invocation=getattr(module, "invocation_params", {}),
                        console=console
                    )
                except:
                    pass

                results[name] = outcome
                break

        return results
