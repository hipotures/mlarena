"""
CLI entry point for MLArena.

Provides dynamic module subcommands discovered via ModuleRegistry and
executes them through the PipelineExecutor with dependency handling.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from mlarena.core.config import load_pipeline_def
from mlarena.core.experiment import ExperimentState
from mlarena.core.module import ModuleContext, ModuleResult
from mlarena.core.pipeline import PipelineExecutor
from mlarena.core.registry import ModuleRegistry
from mlarena.utils.project import load_project_config
from mlarena.core.conf import GlobalConfig, get_config


# Repository root (resolve to absolute path)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _parse_preprocess_templates(template_arg: str, project_root: Path) -> Tuple[List[str], str, bool]:
    """
    Resolve a preprocess template argument into an execution chain.

    Args:
        template_arg: Raw CLI value (single name, comma-separated list, or meta-template).
        project_root: Root of the target Kaggle project.

    Returns:
        Tuple where:
            templates: Ordered list of template names to execute.
            chain_exp_id: Experiment identifier for the chain (e.g., ``pre-full-pipeline``).
            is_meta: Whether the argument resolved to a meta-template chain.

    Raises:
        ValueError: If a meta-template declares a non-list ``chain`` field.

    Examples:
        >>> _parse_preprocess_templates("baseline", Path("."))[0]
        ['baseline']
        >>> chain = _parse_preprocess_templates("noop,encoder", Path("."))  # doctest: +ELLIPSIS
        >>> chain[0]
        ['noop', 'encoder']
    """
    if template_arg is None:
        template_arg = "baseline"

    # Split by comma and strip whitespace
    templates = [t.strip() for t in template_arg.split(",")]

    # Check if single template is a meta-template (has "chain" key)
    if len(templates) == 1:
        try:
            import sys
            sys.path.insert(0, str(REPO_ROOT / "scripts"))
            from template_loader import load_templates

            all_templates, _ = load_templates("preprocess", project_root, suppress_warnings=True)
            template_config = all_templates.get(templates[0], {})

            # If template has "chain" key, it's a meta-template
            if "chain" in template_config:
                chain = template_config["chain"]
                if not isinstance(chain, list):
                    raise ValueError(f"Meta-template '{templates[0]}' chain must be a list")

                # Meta-template: use template name as experiment ID
                chain_exp_id = f"pre-{templates[0]}"
                return chain, chain_exp_id, True
        except Exception:
            pass  # If template loading fails, treat as regular template

    # CLI chain or single template
    if len(templates) == 1:
        # Single template (not meta)
        chain_exp_id = f"pre-{templates[0]}"
    else:
        # CLI chain: create hash from template list
        chain_str = ",".join(templates)
        chain_hash = hashlib.md5(chain_str.encode()).hexdigest()[:8]
        chain_exp_id = f"pre-chain-{chain_hash}"

    return templates, chain_exp_id, False


def _build_parser() -> argparse.ArgumentParser:
    """
    Build a simplified top-level CLI parser.
    """
    parser = argparse.ArgumentParser(prog="mla", description="MLArena pipeline runner")
    parser.add_argument("command", nargs="?", help="Module to run (e.g., model, preprocess) or 'modules' to list.")
    parser.add_argument("--project", "-p", help="Project name (e.g., Titanic)")
    parser.add_argument("--experiment-id", "-e", help="Experiment ID to resume")
    parser.add_argument("--profile", "-s", help="Config profile (e.g., smoke, dev)")
    parser.add_argument("--force", "-f", action="store_true", help="Re-run completed modules")
    return parser


def _build_module_context(
    project_root: Path,
    project: str,
    module_name: str,
    config: GlobalConfig,
    experiment_id: Optional[str] = None,
    config_module=None,
    pipeline_def: Optional[Dict] = None,
    argv: Optional[List[str]] = None,
) -> ModuleContext:
    """
    Build an isolated execution context for a module.
    """
    # Determine experiment_id if not provided
    if experiment_id is None:
        if module_name == "init":
            experiment_id = "init"
        elif module_name == "eda":
            experiment_id = "eda"
        elif module_name == "preprocess":
            experiment_id = "pre-baseline"  # default
        else:
            from datetime import datetime as dt
            experiment_id = f"exp-{dt.utcnow().strftime('%Y%m%d-%H%M%S')}"

    # Load or create state for this module
    setup_module_name = module_name if module_name in ("init", "eda") else None

    state = ExperimentState.load_or_create(
        project_root=project_root,
        project_name=project,
        experiment_id=experiment_id,
        pipeline=pipeline_def or {},
        run_invocation={"argv": argv or [], "module": module_name},
        create_dirs=True,
        setup_module_name=setup_module_name,
    )

    artifact_dir = state.experiment_dir / "artifacts" / module_name

    return ModuleContext(
        project_name=project,
        project_root=project_root,
        experiment_id=state.experiment_id,
        experiment_dir=state.experiment_dir,
        artifact_dir=artifact_dir,
        cli_args={},
        state=state,
        config_module=config_module,
        config=config,
    )


def _build_contexts(project_root: Path, project: str, state: ExperimentState, config_module, config: GlobalConfig) -> Dict[str, ModuleContext]:
    """
    Build contexts for all registered modules sharing a single experiment.
    """
    contexts: Dict[str, ModuleContext] = {}
    for name in ModuleRegistry.available():
        artifact_dir = state.experiment_dir / "artifacts" / name
        contexts[name] = ModuleContext(
            project_name=project,
            project_root=project_root,
            experiment_id=state.experiment_id,
            experiment_dir=state.experiment_dir,
            artifact_dir=artifact_dir,
            cli_args={},
            state=state,
            config_module=config_module,
            config=config,
        )
    return contexts


def run_preprocess_chain(
    project_root: Path,
    project_name: str,
    config: GlobalConfig,
    config_module,
    pipeline_def: Dict,
    argv: List[str],
    console,
) -> Tuple[int, Dict[str, ModuleResult], Optional[Path], List[str]]:
    """
    Execute a chain of preprocessing templates.
    Returns (exit_code, results, final_preprocess_exp_dir, resolved_templates).
    """
    resolved_preprocess_template = config.preprocess_template or "baseline"
    preprocess_templates, chain_exp_id, is_meta = _parse_preprocess_templates(resolved_preprocess_template, project_root)
    
    results = {}
    force = config.force

    if len(preprocess_templates) > 1 or is_meta:
        console.print(f"\n[bold cyan]Preprocessing Chain:[/bold cyan] {' → '.join(preprocess_templates)}")
        console.print(f"[dim]Chain experiment: {chain_exp_id}[/dim]\n")

    for idx, tpl_name in enumerate(preprocess_templates):
        submodule_exp_id = f"{idx}-{tpl_name}"
        full_exp_id = f"{chain_exp_id}/{submodule_exp_id}"
        input_source_idx = idx - 1
        input_source = f"{input_source_idx}-{preprocess_templates[input_source_idx]}" if idx > 0 else None

        exp_dir = project_root / "experiments" / chain_exp_id / submodule_exp_id
        state_file = exp_dir / "state.json"

        already_completed = False
        module_payload = {}
        saved_input_source = None

        if state_file.exists():
            with open(state_file) as f:
                saved_state = json.load(f)
                module_entry = saved_state.get("modules", {}).get("preprocess", {})
                already_completed = module_entry.get("status") == "completed"
                module_payload = module_entry.get("payload", {})
                saved_input_source = module_payload.get("input_source")

        if already_completed and not force:
            if saved_input_source == input_source:
                console.print(f"[dim]✓ preprocess ({tpl_name}) already completed, skipping[/dim]")
                results[f"preprocess-{tpl_name}"] = ModuleResult(success=True, payload=module_payload)
                continue

        context = _build_module_context(
            project_root=project_root,
            project=project_name,
            module_name="preprocess",
            config=config,
            experiment_id=full_exp_id,
            config_module=config_module,
            pipeline_def=pipeline_def,
            argv=argv,
        )

        module_cls = ModuleRegistry.get("preprocess")
        module = module_cls(context)

        is_last_in_chain = (idx == len(preprocess_templates) - 1)
        module.set_invocation_params({
            "preprocess_template": tpl_name,
            "input_source": input_source,
            "chain_exp_id": chain_exp_id,
            "is_last_in_chain": is_last_in_chain,
            "force": force,
        })

        executor = PipelineExecutor({"preprocess": module})
        module_results = executor.run_module("preprocess", force=force, skip_deps=False)

        result = module_results.get("preprocess")
        results[f"preprocess-{tpl_name}"] = result

        if not result or not result.success:
            return 1, results, None, preprocess_templates

    final_preprocess_exp_dir = None
    if preprocess_templates:
        final_step_id = f"{len(preprocess_templates) - 1}-{preprocess_templates[-1]}"
        final_preprocess_exp_dir = project_root / "experiments" / chain_exp_id / final_step_id
        
    return 0, results, final_preprocess_exp_dir, preprocess_templates


def run_auto_flow(
    project_root: Path,
    project_name: str,
    config: GlobalConfig,
    argv: Optional[List[str]] = None,
) -> int:
    """
    Execute the full MLArena pipeline end-to-end using GlobalConfig.
    """
    from rich.console import Console
    import time
    import json
    from mlarena.core.module import ModuleResult
    from mlarena.utils.project import load_project_config

    console = Console(force_terminal=True)
    results: Dict[str, ModuleResult] = {}

    model_template = config.model_template
    force = config.force
    skip_submit = config.skip_submit
    skip_git = config.skip_git
    wait_seconds = config.wait_seconds

    # Full sequence with smart checking
    setup_modules = ["init", "eda", "preprocess"]
    pipeline_modules = ["model", "predict", "submit", "fetch-score"]

    if force:
        console.print(f"\n[dim]Force mode: ON[/dim]\n")

    # Load project config module
    config_module = None
    pipeline_def = {}
    if project_root.exists():
        try:
            config_module = load_project_config(project_root)
            pipeline_def, _ = load_pipeline_def("default", project_root=project_root)
        except Exception:
            pass 

    # Resolve preprocess template
    resolved_preprocess_template = config.preprocess_template
    if resolved_preprocess_template is None:
        try:
            import sys
            sys.path.insert(0, str(REPO_ROOT / "scripts"))
            from template_loader import load_templates

            model_templates, _ = load_templates("model", project_root, suppress_warnings=True)
            model_tpl_cfg = model_templates.get(model_template, {})
            resolved_preprocess_template = model_tpl_cfg.get("preprocess_template", "baseline")
        except Exception:
            resolved_preprocess_template = "baseline"

    # Phase 1: Setup modules
    for module_name in ["init", "eda"]:
        check_exp_id = module_name
        exp_dir = project_root / "experiments" / check_exp_id
        state_file = exp_dir / "state.json"

        already_completed = False
        module_payload = {}
        if state_file.exists():
            with open(state_file) as f:
                saved_state = json.load(f)
                module_entry = saved_state.get("modules", {}).get(module_name, {})
                already_completed = module_entry.get("status") == "completed"
                module_payload = module_entry.get("payload", {})

        if already_completed and not force:
            console.print(f"[dim]✓ {module_name} already completed (exp: {check_exp_id}), skipping[/dim]")
            results[module_name] = ModuleResult(success=True, payload=module_payload)
            continue

        context = _build_module_context(
            project_root=project_root,
            project=project_name,
            module_name=module_name,
            config=config,
            experiment_id=check_exp_id,
            config_module=config_module,
            pipeline_def=pipeline_def,
            argv=argv,
        )

        module_cls = ModuleRegistry.get(module_name)
        module = module_cls(context)

        if module_name == "init":
            module.set_invocation_params({
                "model_template": model_template,
                "preprocess_template": resolved_preprocess_template,
                "force": force,
            })
        else:
            module.set_invocation_params({"force": force})

        executor = PipelineExecutor({module_name: module})
        module_results = executor.run_module(module_name, force=force, skip_deps=False)

        result = module_results.get(module_name)
        results[module_name] = result

        if not result or not result.success:
            console.print(f"\n[red]✗ Auto-flow stopped at {module_name}[/red]")
            return 1

    # Phase 1b: Preprocessing chain
    exit_code, chain_results, final_preprocess_exp_dir, preprocess_templates = run_preprocess_chain(
        project_root, project_name, config, config_module, pipeline_def, argv or [], console
    )
    results.update(chain_results)
    if exit_code != 0:
        return exit_code

    # Phase 2: Pipeline modules
    model_context = _build_module_context(
        project_root=project_root,
        project=project_name,
        module_name="model",
        config=config,
        experiment_id=None,
        config_module=config_module,
        pipeline_def=pipeline_def,
        argv=argv,
    )

    shared_experiment_id = model_context.experiment_id
    model_cls = ModuleRegistry.get("model")
    model_module = model_cls(model_context)

    final_preprocess_template = preprocess_templates[-1] if preprocess_templates else resolved_preprocess_template
    
    model_params = {
        "model_template": model_template,
        "preprocess_template": final_preprocess_template,
        "preprocess_exp_dir": str(final_preprocess_exp_dir) if final_preprocess_exp_dir else None,
        "force": force,
    }
    # Backward compat for some model modules that still use dict
    model_params.update(config.model) 
    # Use common values as fallback if needed? Modules should handle it.
    
    model_module.set_invocation_params(model_params)

    executor = PipelineExecutor({"model": model_module})
    module_results = executor.run_module("model", force=force, skip_deps=True)
    result = module_results.get("model")
    results["model"] = result

    if not result or not result.success:
        return 1

    remaining_modules = ["predict", "submit", "fetch-score"]
    all_modules = {"model": model_module}

    for module_name in remaining_modules:
        context = _build_module_context(
            project_root=project_root,
            project=project_name,
            module_name=module_name,
            config=config,
            experiment_id=shared_experiment_id,
            config_module=config_module,
            pipeline_def=pipeline_def,
            argv=argv,
        )

        module_cls = ModuleRegistry.get(module_name)
        module = module_cls(context)

        if module_name == "submit":
            module.set_invocation_params({"skip_submit": skip_submit})
        elif module_name == "predict":
            module.set_invocation_params({
                "preprocess_template": final_preprocess_template,
                "preprocess_exp_dir": str(final_preprocess_exp_dir) if final_preprocess_exp_dir else None,
            })

        all_modules[module_name] = module

    executor = PipelineExecutor(all_modules)
    for module_name in remaining_modules:
        if module_name == "fetch-score" and wait_seconds > 0:
            console.print(f"\n[dim]Waiting {wait_seconds}s for Kaggle processing...[/dim]")
            time.sleep(wait_seconds)

        module_results = executor.run_module(module_name, force=force, skip_deps=True)
        result = module_results.get(module_name)
        results[module_name] = result

        if not result or not result.success:
            return 1

    if not skip_git:
        _create_auto_flow_commit(project_name, results, console)

    console.print("\n[bold green]✓ Auto-flow completed successfully[/bold green]\n")
    return 0

    # All modules succeeded - create git commit
    if not skip_git:
        _create_auto_flow_commit(project_name, results, console)

    console.print("\n[bold green]✓ Auto-flow completed successfully[/bold green]\n")
    return 0


def _create_auto_flow_commit(
    project_name: str,
    results: Dict[str, "ModuleResult"],
    console,
):
    """
    Create a git commit summarizing an auto-flow run.

    Args:
        project_name: Kaggle project slug.
        results: Mapping of module names to their execution outcomes.
        console: Rich console for user feedback.

    Raises:
        subprocess.CalledProcessError: If git add/commit fails unexpectedly.
    """
    import subprocess

    # Extract scores from module payloads
    local_cv_score = None
    public_score = None

    model_result = results.get("model")
    if model_result and model_result.payload:
        local_cv = model_result.payload.get("local_cv_score") or model_result.payload.get("local_cv")

    fetch_result = results.get("fetch-score")
    if fetch_result and fetch_result.payload:
        public_score = fetch_result.payload.get("score")

    # Build commit message (list executed modules)
    executed_modules = [m for m in results.keys() if results[m] and results[m].success]
    flow_desc = "→".join(executed_modules)
    parts = [f"auto-flow({project_name}): {flow_desc}"]

    if local_cv_score is not None:
        parts.append(f"local {local_cv_score:.3f}")

    if public_score is not None:
        parts.append(f"public {public_score:.3f}")
    else:
        parts.append("public pending")

    message = " | ".join(parts)

    # Stage changes (project directory only)
    project_root = REPO_ROOT / "projects" / "kaggle" / project_name

    try:
        # Stage project directory
        subprocess.run(
            ["git", "add", str(project_root)],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )

        # Check if there are staged changes
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=REPO_ROOT,
        )

        if result.returncode == 0:
            console.print("[yellow]No staged changes; skipping git commit[/yellow]")
            return

        # Create commit
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )

        console.print(f"\n[green]✓ Git commit created:[/green] [dim]{message}[/dim]")

    except subprocess.CalledProcessError as e:
        console.print(f"\n[yellow]⚠ Git commit failed: {e}[/yellow]")
        console.print("[dim]You can commit manually if needed[/dim]")


def _convert_dash_args_to_overrides(args_list: List[str]) -> List[str]:
    """
    Convert --flag value arguments to key=value format for OmegaConf.
    Maps common parameters to their config sections.
    """
    # Map common flags to their config paths
    COMMON_PARAMS = {"time_limit", "use_gpu", "preset", "seed"}

    result = []
    i = 0
    while i < len(args_list):
        arg = args_list[i]

        # Already in key=value format (with or without --)
        if "=" in arg:
            if arg.startswith("--"):
                key_val = arg.lstrip("-")
                key, val = key_val.split("=", 1)
                key = key.replace("-", "_")
            else:
                key, val = arg.split("=", 1)
                key = key.replace("-", "_")

            # Prefix with common. if needed
            if key in COMMON_PARAMS and not key.startswith("common."):
                key = f"common.{key}"

            result.append(f"{key}={val}")
            i += 1
            continue

        # --flag value format
        if arg.startswith("--"):
            key = arg.lstrip("-").replace("-", "_")

            # Check if next item is the value
            if i + 1 < len(args_list) and not args_list[i + 1].startswith("-"):
                value = args_list[i + 1]

                # Prefix with common. if needed
                if key in COMMON_PARAMS:
                    key = f"common.{key}"

                result.append(f"{key}={value}")
                i += 2
                continue
            else:
                # Boolean flag
                if key in COMMON_PARAMS:
                    key = f"common.{key}"
                result.append(f"{key}=true")
                i += 1
                continue

        # Plain value without flag
        result.append(arg)
        i += 1

    return result


def main(argv: List[str] | None = None) -> int:
    """
    Main CLI entry point using GlobalConfig.
    """
    argv = argv or sys.argv[1:]

    # Discover modules
    if not ModuleRegistry.available():
        ModuleRegistry.discover()

    parser = _build_parser()
    args, overrides = parser.parse_known_args(argv)

    # Convert --flag value to key=value for OmegaConf
    overrides = _convert_dash_args_to_overrides(overrides)

    command = args.command
    # If command looks like an override (key=value), it's not a command
    if command and "=" in command:
        overrides.insert(0, command)
        command = None

    # Detect project
    project = args.project
    
    # If project not in flags, try to find it in overrides (e.g., project=Titanic)
    if not project:
        # Simple heuristic: look for project=... in overrides
        for override in overrides:
            if override.startswith("project="):
                project = override.split("=", 1)[1]
                break

    if not project and command != "modules":
        parser.print_help()
        print("\n[error] --project is required")
        return 1

    # Build Unified Config
    try:
        # Combine profile and other overrides
        if args.profile:
            overrides.insert(0, f"profile={args.profile}")
        if args.force:
            overrides.append("force=true")
        if args.experiment_id:
            overrides.append(f"experiment_id={args.experiment_id}")
            
        config = get_config(project or "default", overrides)
    except Exception as e:
        print(f"[error] Config error: {e}")
        return 1

    # Handle 'modules' command
    if command == "modules":
        available = sorted(ModuleRegistry.available())
        print("\n".join(available))
        return 0

    project_root = REPO_ROOT / "projects" / "kaggle" / project
    
    # Auto-flow: no command provided
    if command is None:
        return run_auto_flow(
            project_root=project_root,
            project_name=project,
            config=config,
            argv=argv,
        )

    # Single module execution
    if command not in ModuleRegistry.available():
        print(f"[error] Unknown module: {command}")
        return 1

    # Load project config and pipeline
    config_module = None
    pipeline_def = {}
    if project_root.exists():
        config_module = load_project_config(project_root)
        pipeline_def, _ = load_pipeline_def("default", project_root=project_root)
    elif command != "init":
        print(f"[error] Project '{project}' not initialized. Run: mla init --project {project}")
        return 1

    # Special case: preprocess chain (even if run as single command)
    if command == "preprocess":
        from rich.console import Console
        console = Console(force_terminal=True)
        
        # Check if project exists
        if not project_root.exists():
            print(f"[error] Project '{project}' not initialized. Run: mla init --project {project}")
            return 1
            
        exit_code, _, _ = run_preprocess_chain(
            project_root, project, config, config_module, pipeline_def, argv, console
        )
        return exit_code

    state = ExperimentState.load_or_create(
        project_root=project_root,
        project_name=project,
        experiment_id=config.experiment_id,
        pipeline=pipeline_def,
        run_invocation={"argv": argv, "cli_args": vars(args)},
        create_dirs=True,
        setup_module_name=command if command in ("init", "eda") else None,
    )

    contexts = _build_contexts(project_root, project, state, config_module, config)
    
    modules = {}
    for name in ModuleRegistry.available():
        module_cls = ModuleRegistry.get(name)
        module = module_cls(contexts[name])
        if name == command:
            # Bridging: provide a combined dict for modules still using invocation_params
            params = getattr(config, name.replace("-", "_"), {})
            if not isinstance(params, dict):
                params = {}
            else:
                params = params.copy()
            
            # Inject top-level config fields that modules expect
            for field in ["model_template", "preprocess_template", "force", "skip_submit"]:
                if hasattr(config, field):
                    params[field] = getattr(config, field)
            
            # Also inject common fields (these override template defaults if explicitly set via CLI)
            for field, value in config.common.model_dump().items():
                if value is not None:
                    params[field] = value
                    
            module.set_invocation_params(params)
        modules[name] = module

    executor = PipelineExecutor(modules)
    results = executor.run_module(command, force=config.force, skip_deps=config.skip_deps)

    last = results.get(command)
    return 0 if (last and last.success) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
