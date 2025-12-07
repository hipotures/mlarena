"""
CLI entry point for MLArena.

Provides dynamic module subcommands discovered via ModuleRegistry and
executes them through the PipelineExecutor with dependency handling.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

from mlarena.core.config import load_pipeline_def
from mlarena.core.experiment import ExperimentState
from mlarena.core.module import ModuleContext
from mlarena.core.pipeline import PipelineExecutor
from mlarena.core.registry import ModuleRegistry
from mlarena.utils.project import load_project_config


# Repository root (resolve to absolute path)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


COMMON_FLAGS = [
    ("--project", "-p", {"required": False, "help": "Competition/project name (projects/kaggle/<name>)"}),
    ("--experiment-id", "-e", {"help": "Existing experiment id to resume"}),
    ("--force", "-f", {"action": "store_true", "help": "Re-run completed modules"}),
    ("--skip-deps", None, {"action": "store_true", "help": "Do not run dependencies automatically"}),
]


def _add_common(subparser: argparse.ArgumentParser) -> List[str]:
    dests: List[str] = []
    for long, short, kwargs in COMMON_FLAGS:
        opts = [long] + ([short] if short else [])
        action = subparser.add_argument(*opts, **kwargs)
        dests.append(action.dest)
    return dests


def _build_parser(module_arg_map: Dict[str, List[str]]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mla", description="MLArena pipeline runner")
    parser.add_argument("--show-payload", action="store_true", help="Print raw module payloads (debug)")

    # Auto-flow arguments (top-level, before subparsers)
    parser.add_argument("--project", "-p", help="Project name (enables auto-flow if no module specified)")
    parser.add_argument("--model-template", default="baseline", help="Model template for auto-flow")
    parser.add_argument("--preprocess-template", default="baseline", help="Preprocessing template for auto-flow")
    parser.add_argument("--force", "-f", action="store_true", help="Force re-run ALL modules from scratch")
    parser.add_argument("--auto-submit", action="store_true", help="Skip confirmation prompts")
    parser.add_argument("--skip-git", action="store_true", help="Skip automatic git commit")
    parser.add_argument("--wait-seconds", type=int, default=30, help="Seconds to wait before fetch-score")

    subparsers = parser.add_subparsers(dest="command", required=False)

    modules_parser = subparsers.add_parser("modules", help="List available modules")
    modules_parser.add_argument("--project", "-p", required=False, help="(optional) project for compatibility")

    for name in sorted(ModuleRegistry.available()):
        module_cls = ModuleRegistry.get(name)
        sub = subparsers.add_parser(name, help=module_cls.description or "")
        common_dests = set(_add_common(sub))
        before = {a.dest for a in sub._actions}
        if hasattr(module_cls, "register_cli_args"):
            module_cls.register_cli_args(sub)
        after = {a.dest for a in sub._actions}
        module_arg_map[name] = sorted([d for d in (after - before) if d not in common_dests])
    return parser


def _extract_module_params(args: argparse.Namespace, module_arg_map: Dict[str, List[str]]) -> Dict[str, object]:
    params: Dict[str, object] = {}

    # Add common flags (force, skip_deps, etc.)
    common_flag_names = ["force", "skip_deps"]
    for key in common_flag_names:
        if hasattr(args, key):
            params[key] = getattr(args, key)

    # Add module-specific args
    mod_args = module_arg_map.get(args.command, [])
    for key in mod_args:
        if hasattr(args, key):
            params[key] = getattr(args, key)

    return params


def _build_module_context(
    project_root: Path,
    project: str,
    module_name: str,
    experiment_id: Optional[str] = None,
    config_module=None,
    pipeline_def: Optional[Dict] = None,
    argv: Optional[List[str]] = None,
) -> ModuleContext:
    """
    Build context for a single module with its own experiment state.

    For auto-flow, each setup module (init/eda/preprocess) gets its own experiment_id.
    """
    # Determine experiment_id if not provided
    if experiment_id is None:
        if module_name == "init":
            experiment_id = "init"
        elif module_name == "eda":
            experiment_id = "eda"
        elif module_name == "preprocess":
            # Will be set by caller (e.g., "pre-baseline")
            experiment_id = "pre-baseline"  # default
        else:
            # Generate new experiment_id for other modules
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
    )


def _build_contexts(project_root: Path, project: str, state: ExperimentState, config_module) -> Dict[str, ModuleContext]:
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
        )
    return contexts


def run_auto_flow(
    project_root: Path,
    project_name: str,
    model_template: str = "baseline",
    preprocess_template: str = "baseline",
    force: bool = False,
    auto_submit: bool = False,
    skip_git: bool = False,
    wait_seconds: int = 30,
    argv: Optional[List[str]] = None,
) -> int:
    """
    Run full auto-flow: init → eda → preprocess → model → predict → submit → fetch-score.

    Returns:
        Exit code (0 = success, 1 = failure)
    """
    from rich.console import Console
    import time
    import json
    from mlarena.core.module import ModuleResult
    from mlarena.utils.project import load_project_config

    console = Console(force_terminal=True)
    results: Dict[str, ModuleResult] = {}

    # Full sequence with smart checking
    setup_modules = ["init", "eda", "preprocess"]
    pipeline_modules = ["model", "predict", "submit", "fetch-score"]

    console.print("\n[bold cyan]AUTO-FLOW PIPELINE[/bold cyan]")
    console.print(f"Model template: [yellow]{model_template}[/yellow]")
    console.print(f"Preprocess template: [yellow]{preprocess_template}[/yellow]")
    console.print(f"Force mode: [yellow]{'ON' if force else 'OFF'}[/yellow]\n")

    # Load config (may not exist for init module)
    config_module = None
    pipeline_def = {}
    if project_root.exists():
        try:
            config_module = load_project_config(project_root)
            pipeline_def, _ = load_pipeline_def("default", project_root=project_root)
        except Exception:
            pass  # Init will create project

    # Phase 1: Setup modules (init/eda/preprocess) - smart checking
    for module_name in setup_modules:
        # Determine experiment ID for checking
        if module_name == "init":
            check_exp_id = "init"
        elif module_name == "eda":
            check_exp_id = "eda"
        elif module_name == "preprocess":
            check_exp_id = f"pre-{preprocess_template}"
        else:
            check_exp_id = None

        # Check if already completed
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

        # Run module
        console.print(f"\n[bold]Running {module_name}...[/bold]")

        # Build context for this module
        context = _build_module_context(
            project_root=project_root,
            project=project_name,
            module_name=module_name,
            experiment_id=check_exp_id,
            config_module=config_module,
            pipeline_def=pipeline_def,
            argv=argv,
        )

        # Create module instance
        module_cls = ModuleRegistry.get(module_name)
        module = module_cls(context)

        # Set invocation params
        if module_name == "preprocess":
            module.set_invocation_params({
                "preprocess_template": preprocess_template,
                "force": force,
            })
        else:
            module.set_invocation_params({"force": force})

        # Create executor and run
        executor = PipelineExecutor({module_name: module})
        module_results = executor.run_module(module_name, force=force, skip_deps=False)

        result = module_results.get(module_name)
        results[module_name] = result

        # Stop on failure
        if not result or not result.success:
            console.print(f"\n[red]✗ Auto-flow stopped at {module_name}[/red]")
            if result and result.error:
                console.print(f"[red]Error: {result.error}[/red]\n")
            return 1

    # Reload config after init/eda (project now exists)
    if project_root.exists() and config_module is None:
        config_module = load_project_config(project_root)
        pipeline_def, _ = load_pipeline_def("default", project_root=project_root)

    # Phase 2: Pipeline modules (model onwards) - create single experiment
    # First run model to get experiment_id, then run rest with that experiment_id

    # Step 1: Run model first to create experiment_id
    console.print(f"\n[bold]Running model...[/bold]")

    model_context = _build_module_context(
        project_root=project_root,
        project=project_name,
        module_name="model",
        experiment_id=None,  # Create new
        config_module=config_module,
        pipeline_def=pipeline_def,
        argv=argv,
    )

    shared_experiment_id = model_context.experiment_id

    model_cls = ModuleRegistry.get("model")
    model_module = model_cls(model_context)
    model_module.set_invocation_params({
        "model_template": model_template,
        "preprocess_template": preprocess_template,
        "force": force,
    })

    # Run model (no dependencies)
    executor = PipelineExecutor({"model": model_module})
    module_results = executor.run_module("model", force=force, skip_deps=False)

    result = module_results.get("model")
    results["model"] = result

    if not result or not result.success:
        console.print(f"\n[red]✗ Auto-flow stopped at model[/red]")
        if result and result.error:
            console.print(f"[red]Error: {result.error}[/red]\n")
        return 1

    # Step 2: Run predict/submit/fetch-score with shared experiment_id and ALL modules
    # Create contexts for all remaining modules
    remaining_modules = ["predict", "submit", "fetch-score"]
    all_modules = {}

    # Add model to modules dict (for dependency resolution)
    all_modules["model"] = model_module

    for module_name in remaining_modules:
        context = _build_module_context(
            project_root=project_root,
            project=project_name,
            module_name=module_name,
            experiment_id=shared_experiment_id,
            config_module=config_module,
            pipeline_def=pipeline_def,
            argv=argv,
        )

        module_cls = ModuleRegistry.get(module_name)
        module = module_cls(context)

        if module_name == "submit":
            module.set_invocation_params({"auto_submit": auto_submit})

        all_modules[module_name] = module

    # Create single executor with all modules
    executor = PipelineExecutor(all_modules)

    # Run each module in sequence (dependencies resolved automatically)
    for module_name in remaining_modules:
        # Wait before fetch-score
        if module_name == "fetch-score" and wait_seconds > 0:
            console.print(f"\n[dim]Waiting {wait_seconds}s for Kaggle processing...[/dim]")
            time.sleep(wait_seconds)

        console.print(f"\n[bold]Running {module_name}...[/bold]")

        module_results = executor.run_module(module_name, force=force, skip_deps=False)

        result = module_results.get(module_name)
        results[module_name] = result

        # Stop on failure
        if not result or not result.success:
            console.print(f"\n[red]✗ Auto-flow stopped at {module_name}[/red]")
            if result and result.error:
                console.print(f"[red]Error: {result.error}[/red]\n")
            return 1

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
    """Create git commit for auto-flow with scores."""
    import subprocess

    # Extract scores from module payloads
    local_cv = None
    public_score = None

    model_result = results.get("model")
    if model_result and model_result.payload:
        local_cv = model_result.payload.get("local_cv")

    fetch_result = results.get("fetch-score")
    if fetch_result and fetch_result.payload:
        public_score = fetch_result.payload.get("score")

    # Build commit message (list executed modules)
    executed_modules = [m for m in results.keys() if results[m] and results[m].success]
    flow_desc = "→".join(executed_modules)
    parts = [f"auto-flow({project_name}): {flow_desc}"]

    if local_cv is not None:
        parts.append(f"local {local_cv:.3f}")

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


def main(argv: List[str] | None = None) -> int:
    argv = argv or sys.argv[1:]

    # Discover modules only if registry is empty (first run or after clear in tests)
    if not ModuleRegistry.available():
        ModuleRegistry.discover()

    module_arg_map: Dict[str, List[str]] = {}
    parser = _build_parser(module_arg_map)
    args = parser.parse_args(argv)

    # Detect auto-flow: --project provided but no subcommand
    is_auto_flow = False
    if args.command is None:
        if not args.project:
            parser.print_help()
            print("\n[error] Either provide a module name or --project for auto-flow")
            return 1
        is_auto_flow = True

    # Handle auto-flow
    if is_auto_flow:
        project_root = REPO_ROOT / "projects" / "kaggle" / args.project
        return run_auto_flow(
            project_root=project_root,
            project_name=args.project,
            model_template=args.model_template,
            preprocess_template=args.preprocess_template,
            force=args.force,
            auto_submit=args.auto_submit,
            skip_git=args.skip_git,
            wait_seconds=args.wait_seconds,
            argv=argv,
        )

    if args.command == "modules":
        preferred_order = [
            "init",
            "eda",
            "preprocess",
            "feat",
            "tune",
            "model",
            "predict",
            "stack",
            "submit",
            "fetch-score",
        ]

        available = set(ModuleRegistry.available())
        ordered = [m for m in preferred_order if m in available]
        # add any others that might be present
        for name in sorted(available):
            if name not in ordered:
                ordered.append(name)

        print("\n".join(ordered))
        return 0

    project_root = REPO_ROOT / "projects" / "kaggle" / args.project

    # Setup modules (init/eda) use fixed directories
    setup_module_name = args.command if args.command in ("init", "eda") else None
    init_mode = args.command == "init"

    if init_mode:
        config_module = None
        pipeline_def = {}
    else:
        if not project_root.exists():
            print(f"[error] Project '{args.project}' not initialized. Run: mla init --project {args.project}")
            return 1
        config_module = load_project_config(project_root)
        pipeline_def, pipeline_warnings = load_pipeline_def("default", project_root=project_root)
        for w in pipeline_warnings:
            print(f"[warn] {w}")

    # For preprocess module, use pre-{template} as experiment_id
    experiment_id = args.experiment_id
    if args.command == "preprocess" and not experiment_id:
        preprocess_template = getattr(args, "preprocess_template", None)
        if preprocess_template:
            experiment_id = f"pre-{preprocess_template}"

    state = ExperimentState.load_or_create(
        project_root=project_root,
        project_name=args.project,
        experiment_id=experiment_id,
        pipeline=pipeline_def,
        run_invocation={"argv": argv, "cli_args": vars(args)},
        create_dirs=True,  # Always create state - init creates project first
        setup_module_name=setup_module_name,
    )

    contexts = _build_contexts(project_root, args.project, state, config_module)

    modules = {}
    for name in ModuleRegistry.available():
        module_cls = ModuleRegistry.get(name)
        module = module_cls(contexts[name])
        if name == args.command:
            module.set_invocation_params(_extract_module_params(args, module_arg_map))
        modules[name] = module

    executor = PipelineExecutor(modules)
    results = executor.run_module(args.command, force=args.force, skip_deps=args.skip_deps)

    # Don't show console reporting - modules display their own Rich output
    # Only show errors
    for mod_name, result in results.items():
        if result.error:
            print(f"[fail] {mod_name}")
            print(f"  error: {result.error}")
        if args.show_payload and result.payload:
            print(f"  payload: {result.payload}")

    last = results.get(args.command)

    return 0 if (last and last.success) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
