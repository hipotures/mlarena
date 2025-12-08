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
from .display import print_module_header, print_module_footer, format_path_relative, extract_template_overrides
from pathlib import Path


class PipelineExecutor:
    def __init__(self, modules: Dict[str, BaseModule]):
        self.modules = modules

    def _collect_module_header_data(self, module_name: str, module: BaseModule) -> dict:
        """Collect template and path data for module header display."""
        invocation = getattr(module, "invocation_params", {})

        # Get template info
        template_name = invocation.get(f"{module_name}_template")
        template_config = {}
        cli_overrides = {}

        if template_name:
            # Load template configuration
            try:
                import sys
                from pathlib import Path as P
                REPO_ROOT = P(__file__).resolve().parents[3]
                sys.path.insert(0, str(REPO_ROOT / "scripts"))
                from template_loader import load_templates

                templates, _ = load_templates(module_name, module.context.project_root, suppress_warnings=True)
                template_config = templates.get(template_name, {}).get("config", {})

                # Extract CLI overrides
                cli_overrides = extract_template_overrides(template_config, invocation)
            except Exception:
                pass  # Template loading failed, continue with empty config

        # Collect input/output paths based on module type
        input_paths = {}
        output_paths = {}

        if module_name == "init":
            # For init in auto-flow, show pipeline configuration
            model_tpl = invocation.get("model_template")
            preprocess_tpl = invocation.get("preprocess_template")

            if model_tpl or preprocess_tpl:
                # This is auto-flow
                input_paths["pipeline"] = "auto-flow"
                if model_tpl:
                    input_paths["model_template"] = model_tpl
                if preprocess_tpl:
                    input_paths["preprocess_template"] = preprocess_tpl

                # Show which modules will run (use short names for compactness)
                modules = ["init", "eda", "prep", "model"]
                if not invocation.get("skip_submit"):
                    modules.extend(["submit", "fetch"])
                # Format as compact flow
                flow = " → ".join(modules)
                output_paths["flow"] = flow

        elif module_name == "preprocess":
            # Check if part of chain (input_source in invocation)
            input_source = invocation.get("input_source")

            # Input paths for preprocessing
            if input_source:
                # Part of chain: show previous step as input
                input_paths["from"] = f"pre-{input_source}"
                input_paths["train"] = f"experiments/pre-{input_source}/artifacts/preprocess/train_processed.csv"
                input_paths["test"] = f"experiments/pre-{input_source}/artifacts/preprocess/test_processed.csv"
            else:
                # First step: show raw data
                try:
                    from mlarena.utils.project import data_paths, load_project_config
                    config = module.context.config_module or load_project_config(module.context.project_root)
                    train_path, test_path = data_paths(config)
                    input_paths["train"] = format_path_relative(train_path, module.context.project_root)
                    input_paths["test"] = format_path_relative(test_path, module.context.project_root)

                    # Check for original dataset in template config
                    if template_config and "orig_path" in template_config:
                        orig_path = template_config["orig_path"]
                        input_paths["original"] = format_path_relative(orig_path, module.context.project_root)
                except Exception:
                    pass

            # Output paths for preprocessing (always to current experiment)
            experiment_id = module.context.experiment_id
            output_paths["train"] = f"experiments/{experiment_id}/artifacts/preprocess/train_processed.csv"
            output_paths["test"] = f"experiments/{experiment_id}/artifacts/preprocess/test_processed.csv"

        elif module_name == "model":
            # Input paths for model
            input_paths["train"] = "data/train.csv"
            input_paths["test"] = "data/test.csv"

            # Check if using preprocessed data
            preprocess_template = invocation.get("preprocess_template") or template_config.get("preprocess_template")
            if preprocess_template:
                input_paths["preprocessed"] = f"experiments/pre-{preprocess_template}/artifacts/preprocess/"

            # Output paths for model
            experiment_id = module.context.experiment_id
            output_paths["model"] = f"experiments/{experiment_id}/artifacts/"
            output_paths["submission"] = "submissions/"

        elif module_name == "eda":
            # Input paths for EDA
            input_paths["train"] = "data/train.csv"
            input_paths["test"] = "data/test.csv"

            # Output paths for EDA
            experiment_id = module.context.experiment_id
            output_paths["summary"] = f"experiments/{experiment_id}/eda_summary.txt"

        elif module_name == "predict":
            # Input paths for predict
            experiment_id = invocation.get("experiment_id")
            if experiment_id:
                input_paths["model"] = f"experiments/{experiment_id}/artifacts/"
                input_paths["test"] = "data/test.csv"
                output_paths["predictions"] = f"experiments/{experiment_id}/predictions.csv"

        elif module_name == "submit":
            # Input paths for submit - get actual file from predict payload
            predict_payload = module.context.state.modules.get("predict")
            if predict_payload and hasattr(predict_payload, "payload") and predict_payload.payload:
                submission_file = predict_payload.payload.get("submission_file")
                if submission_file:
                    input_paths["submission"] = format_path_relative(submission_file, module.context.project_root)
            output_paths["kaggle"] = "Kaggle API upload"

        elif module_name == "fetch-score":
            # Input paths for fetch-score
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
        results: Dict[str, ModuleResult] = {}
        execution_plan = [module_name] if skip_deps else self._resolve_execution_order(module_name)

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

            # Handle already completed modules
            # force only applies to target module, not dependencies
            if already_completed and not (force and name == module_name):
                # Init module should always run to show project status (even if completed)
                if name == "init":
                    # Let init module execute to show config
                    pass
                elif name == module_name:
                    # Target module is already completed - show warning and skip
                    console = Console(force_terminal=True)
                    completed_time_raw = state_entry.finished_at or state_entry.started_at or "unknown"

                    # Format timestamp to seconds (remove microseconds and timezone)
                    if completed_time_raw != "unknown":
                        try:
                            dt_obj = dt.fromisoformat(completed_time_raw.replace('Z', '+00:00'))
                            completed_time = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
                        except Exception:
                            completed_time = completed_time_raw
                    else:
                        completed_time = completed_time_raw

                    info_table = Table(show_header=False, box=None)
                    info_table.add_column(style="bold")
                    info_table.add_column(style="green")
                    info_table.add_row("Status:", "Already completed")
                    info_table.add_row("Completed at:", completed_time)

                    console.print(Panel(
                        info_table,
                        title=f"[bold yellow]{name.upper()}[/bold yellow]",
                        border_style="yellow"
                    ))
                    console.print(f"\n[dim]Use [cyan]--force[/cyan] to re-run this module[/dim]\n")

                    results[name] = ModuleResult(success=True, payload=state_entry.payload)
                    continue
                else:
                    # Dependency is completed - skip silently
                    results[name] = ModuleResult(success=True, payload=state_entry.payload)
                    continue

            defer_save = name == "init" and not module.context.project_root.exists()

            module.context.state.start_module(name, getattr(module, "invocation_params", {}))
            if not defer_save:
                module.context.state.save()

            # Get module state for header/footer
            module_state = module.context.state.modules.get(name)
            console = Console(force_terminal=True)

            # Collect header data (template, paths, etc.)
            try:
                header_data = self._collect_module_header_data(name, module)
            except Exception as e:
                console.print(f"[dim red]Data collection error: {e}[/dim red]")
                header_data = {}

            # Get started_at from module_state
            if module_state and hasattr(module_state, 'started_at'):
                started_at = module_state.started_at
            else:
                started_at = dt.now().isoformat()

            # Display module header with collected data
            try:
                print_module_header(
                    module_name=name,
                    started_at=started_at,
                    experiment_id=module.context.experiment_id,
                    template_name=header_data.get("template_name"),
                    template_config=header_data.get("template_config"),
                    cli_overrides=header_data.get("cli_overrides"),
                    input_paths=header_data.get("input_paths"),
                    output_paths=header_data.get("output_paths"),
                    project_root=module.context.project_root,
                    console=console
                )
            except Exception as e:
                # Fallback to simple header if display fails
                console.print(f"[dim red]Header display error: {e}[/dim red]")
                import traceback
                console.print(f"[dim]{traceback.format_exc()}[/dim]")
                start_time = dt.now().strftime("%Y-%m-%d %H:%M:%S")
                header_content = f"[bold]Started:[/bold] {start_time}"
                console.print(Panel(header_content, title=f"[bold cyan]{name.upper()}[/bold cyan]", border_style="cyan"))

            try:
                outcome = module.execute()
            except Exception as exc:  # pragma: no cover - defensive
                error_msg = f"{exc}"
                detail = traceback.format_exc()
                module.context.state.fail_module(name, error_msg)
                if not defer_save or module.context.project_root.exists():
                    module.context.state.save()

                # Display error footer
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
                        console=console
                    )
                except:
                    pass  # Skip footer if there's an error

                results[name] = ModuleResult(success=False, error=detail)
                break

            if outcome.success:
                module.context.state.complete_module(name, outcome.payload)
                if not defer_save or module.context.project_root.exists():
                    module.context.state.save()

                # Display success footer
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

                    # Extract data for footer from payload
                    payload = outcome.payload or {}
                    output_paths = {}
                    metrics = {}
                    shapes = payload.get("shapes")

                    # Module-specific path extraction
                    if name == "preprocess":
                        if "train_processed" in payload:
                            output_paths["train"] = format_path_relative(payload["train_processed"], module.context.project_root)
                        if "test_processed" in payload:
                            output_paths["test"] = format_path_relative(payload["test_processed"], module.context.project_root)
                        if "custom_module_state" in payload and "weights_path" in payload["custom_module_state"]:
                            output_paths["weights"] = payload["custom_module_state"]["weights_path"]
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
                        if "local_cv" in payload:
                            metrics["local_cv"] = payload["local_cv"]
                        if "best_model" in payload:
                            metrics["best_model"] = payload["best_model"]
                    elif name == "eda":
                        # Only add string paths, skip dict data
                        if "summary_file" in payload and isinstance(payload.get("summary_file"), str):
                            output_paths["summary"] = format_path_relative(payload.get("summary_file"), module.context.project_root)
                        if "train_profile_path" in payload:
                            output_paths["train_profile"] = format_path_relative(payload.get("train_profile_path"), module.context.project_root)
                        if "test_profile_path" in payload:
                            output_paths["test_profile"] = format_path_relative(payload.get("test_profile_path"), module.context.project_root)

                        # Extract key statistics from train/test profiles
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
                        # Only show submission file (predictions and submission_file are the same)
                        if "submission_file" in payload:
                            output_paths["submission"] = format_path_relative(payload["submission_file"], module.context.project_root)

                    elif name == "submit":
                        if "submission_file" in payload:
                            output_paths["submission"] = format_path_relative(payload["submission_file"], module.context.project_root)
                        if "local_cv" in payload:
                            metrics["local_cv"] = payload["local_cv"]
                        if "public_score" in payload and payload["public_score"]:
                            metrics["public_score"] = payload["public_score"]

                    elif name == "fetch-score":
                        if "score" in payload and payload["score"]:
                            metrics["public_score"] = payload["score"]
                        if "latest_submission" in payload:
                            metrics["submission_id"] = payload["latest_submission"]

                    print_module_footer(
                        module_name=name,
                        finished_at=finished_at_str,
                        duration=duration,
                        output_paths=output_paths,
                        metrics=metrics,
                        shapes=shapes,
                        project_root=module.context.project_root,
                        console=console
                    )

                    # Print next steps after footer (for all modules)
                    from mlarena.core.module import print_next_steps
                    print_next_steps(name, module.context.project_name, module.context.experiment_id, console)
                except Exception as e:
                    # Skip footer if there's an error
                    pass

                results[name] = outcome
            else:
                module.context.state.fail_module(name, outcome.error or "unknown error")
                if not defer_save or module.context.project_root.exists():
                    module.context.state.save()

                # Display error footer
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
                        console=console
                    )
                except:
                    pass  # Skip footer if there's an error

                results[name] = outcome
                break

        return results
