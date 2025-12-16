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
from typing import Dict, List, Optional, Tuple

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


def _add_common(subparser: argparse.ArgumentParser) -> List[str]:
    """
    Attach shared CLI flags to a module subparser.

    Args:
        subparser: The argparse subparser for a specific module.

    Returns:
        List of destination names for the added arguments.
    """
    dests: List[str] = []
    for long, short, kwargs in COMMON_FLAGS:
        opts = [long] + ([short] if short else [])
        action = subparser.add_argument(*opts, **kwargs)
        dests.append(action.dest)
    return dests


def _build_parser(module_arg_map: Dict[str, List[str]]) -> argparse.ArgumentParser:
    """
    Build the top-level CLI parser and all module subcommands.

    Args:
        module_arg_map: Mutable mapping to record module-specific argument dest names.

    Returns:
        Configured argparse parser ready for CLI parsing.
    """
    parser = argparse.ArgumentParser(prog="mla", description="MLArena pipeline runner")
    parser.add_argument("--show-payload", action="store_true", help="Print raw module payloads (debug)")

    # Auto-flow arguments (top-level, before subparsers)
    parser.add_argument("--project", "-p", help="Project name (enables auto-flow if no module specified)")
    parser.add_argument("--model-template", default="baseline", help="Model template for auto-flow")
    parser.add_argument("--preprocess-template", default=None, help="Preprocessing template for auto-flow (defaults to model template's preprocess if set, else baseline)")
    parser.add_argument("--force", "-f", action="store_true", help="Force re-run ALL modules from scratch")
    parser.add_argument("--skip-submit", action="store_true", help="Skip Kaggle submission (save submission file only)")
    parser.add_argument("--skip-git", action="store_true", help="Skip automatic git commit")
    parser.add_argument("--wait-seconds", type=int, default=30, help="Seconds to wait before fetch-score")
    parser.add_argument("--dev", "-d", action="store_true",
                       help="Development mode: preset=medium, time_limit=300s, use_gpu=0")
    parser.add_argument("--smoke", "-s", action="store_true",
                       help="Smoke test mode: preset=medium, time_limit=60s, use_gpu=0")

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


def _validate_convenience_flags(args: argparse.Namespace) -> None:
    """
    Validate --dev/--smoke usage and raise on conflicts.

    Args:
        args: Parsed argparse namespace.

    Raises:
        ValueError: When mutually exclusive flags are combined or overridden explicitly.
    """
    # Check mutual exclusivity
    if getattr(args, "dev", False) and getattr(args, "smoke", False):
        raise ValueError(
            "Cannot use --dev and --smoke together. Choose one:\n"
            "  --dev:   5-minute development iteration\n"
            "  --smoke: 1-minute smoke test"
        )

    # Check for conflicts with explicit config flags
    convenience_flag = None
    if getattr(args, "dev", False):
        convenience_flag = "--dev"
    elif getattr(args, "smoke", False):
        convenience_flag = "--smoke"

    if convenience_flag:
        conflicting = []
        if getattr(args, "preset", None) is not None:
            conflicting.append("--preset")
        if getattr(args, "time_limit", None) is not None:
            conflicting.append("--time-limit")
        if getattr(args, "use_gpu", None) is not None:
            conflicting.append("--use-gpu")

        if conflicting:
            raise ValueError(
                f"Cannot use {convenience_flag} with explicit config flags: {', '.join(conflicting)}\n"
                f"{convenience_flag} already sets these values. Use one or the other."
            )


def _extract_module_params(args: argparse.Namespace, module_arg_map: Dict[str, List[str]]) -> Dict[str, object]:
    """
    Collect invocation parameters for a module from parsed CLI args.

    Args:
        args: Parsed argparse namespace.
        module_arg_map: Mapping of module name to its dedicated argument dests.

    Returns:
        Dictionary of parameters to persist in state and pass to the module.
    """
    params: Dict[str, object] = {}

    # Apply convenience flags FIRST (before extracting CLI args)
    if getattr(args, "dev", False):
        params["preset"] = "high"
        params["time_limit"] = 300
        params["use_gpu"] = 0
        params["_convenience_flag"] = "dev"  # Track for display
    elif getattr(args, "smoke", False):
        params["preset"] = "medium"
        params["time_limit"] = 60
        params["use_gpu"] = 0
        params["_convenience_flag"] = "smoke"  # Track for display

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
    Build an isolated execution context for a module.

    For auto-flow, each setup module (init/eda/preprocess) gets its own experiment
    identifier to preserve caching semantics.

    Args:
        project_root: Root directory of the Kaggle project.
        project: Project slug (used for state metadata).
        module_name: Name of the module being executed.
        experiment_id: Optional explicit experiment identifier.
        config_module: Loaded project config module, if available.
        pipeline_def: Pipeline definition snapshot for state recording.
        argv: Raw argv list for invocation tracking.

    Returns:
        Fully populated ModuleContext instance.
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
    """
    Build contexts for all registered modules sharing a single experiment.

    Args:
        project_root: Root directory of the Kaggle project.
        project: Project slug.
        state: Loaded ExperimentState object.
        config_module: Project config module.

    Returns:
        Mapping of module name to its ModuleContext.
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
        )
    return contexts


def run_auto_flow(
    project_root: Path,
    project_name: str,
    model_template: str = "baseline",
    preprocess_template: Optional[str] = None,
    force: bool = False,
    skip_submit: bool = False,
    skip_git: bool = False,
    wait_seconds: int = 30,
    argv: Optional[List[str]] = None,
    dev: bool = False,
    smoke: bool = False,
) -> int:
    """
    Execute the full MLArena pipeline end-to-end.

    The auto-flow runs ``init → eda → preprocess (chains supported) → model → predict → submit → fetch-score``
    with smart skipping for already-completed setup modules unless ``force`` is provided.

    Args:
        project_root: Root directory of the Kaggle project.
        project_name: Project slug (used for state and commits).
        model_template: Model template name to train.
        preprocess_template: Optional preprocessing template or chain override.
        force: Re-run modules even if previously completed.
        skip_submit: Skip Kaggle submission while still producing prediction CSV.
        skip_git: Disable automatic commit after successful flow.
        wait_seconds: Delay before fetching the public score.
        argv: Original argv for invocation tracking.
        dev: Apply fast development overrides (preset=high, time_limit=300, use_gpu=0).
        smoke: Apply ultra-fast smoke overrides (preset=medium, time_limit=60, use_gpu=0).

    Returns:
        Integer exit code (``0`` on success, ``1`` on failure).

    Examples:
        >>> run_auto_flow(Path("/tmp/proj"), "demo", model_template="cpu-dev-5m", skip_submit=True)  # doctest: +SKIP
        0
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

    # Pipeline info now shown in INIT module header
    if force:
        console.print(f"\n[dim]Force mode: ON[/dim]\n")

    # Load config (may not exist for init module)
    config_module = None
    pipeline_def = {}
    if project_root.exists():
        try:
            config_module = load_project_config(project_root)
            pipeline_def, _ = load_pipeline_def("default", project_root=project_root)
        except Exception:
            pass  # Init will create project

    # Resolve preprocess template: CLI override > model template link > baseline
    resolved_preprocess_template = preprocess_template
    if resolved_preprocess_template is None:
        try:
            import sys

            sys.path.insert(0, str(REPO_ROOT / "scripts"))
            from template_loader import load_templates  # type: ignore

            model_templates, _ = load_templates("model", project_root, suppress_warnings=True)
            model_tpl_cfg = model_templates.get(model_template, {})
            resolved_preprocess_template = model_tpl_cfg.get("preprocess_template", "baseline")
        except Exception:
            resolved_preprocess_template = "baseline"

    # Phase 1: Setup modules (init/eda/preprocess chain) - smart checking

    # Run init and eda first
    for module_name in ["init", "eda"]:
        # Determine experiment ID for checking
        check_exp_id = module_name  # "init" or "eda"

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
        if module_name == "init":
            # Init needs to know about pipeline templates for header display
            module.set_invocation_params({
                "model_template": model_template,
                "preprocess_template": resolved_preprocess_template,
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

    # Now handle preprocessing chain
    preprocess_templates, chain_exp_id, is_meta = _parse_preprocess_templates(resolved_preprocess_template, project_root)

    # Print chain info
    if len(preprocess_templates) > 1 or is_meta:
        console.print(f"\n[bold cyan]Preprocessing Chain:[/bold cyan] {' → '.join(preprocess_templates)}")
        console.print(f"[dim]Chain experiment: {chain_exp_id}[/dim]\n")

    for idx, tpl_name in enumerate(preprocess_templates):
        # Sub-module experiment ID: chain_exp_id/{idx}-{submodule}
        # Index prevents duplicate names when same template used multiple times
        submodule_exp_id = f"{idx}-{tpl_name}"
        full_exp_id = f"{chain_exp_id}/{submodule_exp_id}"

        input_source_idx = idx - 1
        input_source = f"{input_source_idx}-{preprocess_templates[input_source_idx]}" if idx > 0 else None

        # Check if already completed
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

        # Smart cache: skip if completed AND input_source matches
        if already_completed and not force:
            if saved_input_source == input_source:
                console.print(f"[dim]✓ preprocess ({tpl_name}) already completed (exp: {full_exp_id}), skipping[/dim]")
                results[f"preprocess-{tpl_name}"] = ModuleResult(success=True, payload=module_payload)
                continue
            else:
                console.print(f"[yellow]⚠ preprocess ({tpl_name}) input changed, re-running[/yellow]")

        # Build context for this preprocessing step
        # Use full_exp_id which includes chain directory
        context = _build_module_context(
            project_root=project_root,
            project=project_name,
            module_name="preprocess",
            experiment_id=full_exp_id,
            config_module=config_module,
            pipeline_def=pipeline_def,
            argv=argv,
        )

        # Create module instance
        module_cls = ModuleRegistry.get("preprocess")
        module = module_cls(context)

        # Set invocation params BEFORE creating executor (can_run() needs them)
        is_last_in_chain = (idx == len(preprocess_templates) - 1)
        module.set_invocation_params({
            "preprocess_template": tpl_name,
            "input_source": input_source,
            "chain_exp_id": chain_exp_id,
            "is_last_in_chain": is_last_in_chain,
            "force": force,
        })

        # Create executor and run
        executor = PipelineExecutor({"preprocess": module})
        module_results = executor.run_module("preprocess", force=force, skip_deps=False)

        result = module_results.get("preprocess")
        results[f"preprocess-{tpl_name}"] = result

        # FAIL FAST: Stop on first failure
        if not result or not result.success:
            console.print(f"\n[red]✗ Auto-flow stopped at preprocess step: {tpl_name}[/red]")
            if result and result.error:
                console.print(f"[red]Error: {result.error}[/red]\n")
            return 1

    # Determine final preprocessing experiment directory (chain-aware)
    final_preprocess_exp_dir = None
    if preprocess_templates:
        final_step_id = f"{len(preprocess_templates) - 1}-{preprocess_templates[-1]}"
        final_preprocess_exp_dir = project_root / "experiments" / chain_exp_id / final_step_id

    # Reload config after init/eda/preprocess (ensure fresh config)
    if project_root.exists():
        config_module = load_project_config(project_root)
        pipeline_def, _ = load_pipeline_def("default", project_root=project_root)

    # Phase 2: Pipeline modules (model onwards) - create single experiment
    # First run model to get experiment_id, then run rest with that experiment_id

    # Step 1: Run model first to create experiment_id
    # (header is now printed by pipeline)

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

    # Model uses LAST template in preprocessing chain
    final_preprocess_template = preprocess_templates[-1] if preprocess_templates else resolved_preprocess_template

    # Build model invocation params
    model_params = {
        "model_template": model_template,
        "preprocess_template": final_preprocess_template,
        "preprocess_exp_dir": str(final_preprocess_exp_dir) if final_preprocess_exp_dir else None,
        "force": force,
    }

    # Apply convenience flags
    if dev:
        model_params.update({
            "preset": "high",
            "time_limit": 300,
            "use_gpu": 0,
            "_convenience_flag": "dev"
        })
    elif smoke:
        model_params.update({
            "preset": "medium",
            "time_limit": 60,
            "use_gpu": 0,
            "_convenience_flag": "smoke"
        })

    model_module.set_invocation_params(model_params)

    # Run model (skip dependencies - preprocessing already completed)
    executor = PipelineExecutor({"model": model_module})
    module_results = executor.run_module("model", force=force, skip_deps=True)

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
            module.set_invocation_params({"skip_submit": skip_submit})
        elif module_name == "predict":
            module.set_invocation_params({
                "preprocess_template": final_preprocess_template,
                "preprocess_exp_dir": str(final_preprocess_exp_dir) if final_preprocess_exp_dir else None,
            })

        all_modules[module_name] = module

    # Create single executor with all modules
    executor = PipelineExecutor(all_modules)

    # Run each module in sequence (skip dependencies - model already completed)
    for module_name in remaining_modules:
        # Wait before fetch-score
        if module_name == "fetch-score" and wait_seconds > 0:
            console.print(f"\n[dim]Waiting {wait_seconds}s for Kaggle processing...[/dim]")
            time.sleep(wait_seconds)

        # (header is now printed by pipeline)

        module_results = executor.run_module(module_name, force=force, skip_deps=True)

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
    """
    Parse CLI arguments, resolve module dispatch, and execute the requested pipeline.

    Args:
        argv: Optional argv override for programmatic execution.

    Returns:
        Integer exit code from the invoked module or auto-flow.

    Examples:
        >>> main(["modules"])  # doctest: +ELLIPSIS
        0
    """
    argv = argv or sys.argv[1:]

    # Discover modules only if registry is empty (first run or after clear in tests)
    if not ModuleRegistry.available():
        ModuleRegistry.discover()

    module_arg_map: Dict[str, List[str]] = {}
    parser = _build_parser(module_arg_map)
    args = parser.parse_args(argv)

    # Validate convenience flags early
    try:
        _validate_convenience_flags(args)
    except ValueError as e:
        print(f"[error] {e}")
        return 1

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
            skip_submit=args.skip_submit,
            skip_git=args.skip_git,
            wait_seconds=args.wait_seconds,
            argv=argv,
            dev=args.dev,
            smoke=args.smoke,
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

    # Handle preprocess chain manually (similar to auto-flow)
    if args.command == "preprocess":
        from rich.console import Console
        from mlarena.core.module import ModuleResult

        console = Console(force_terminal=True)
        preprocess_template_arg = getattr(args, "preprocess_template", "baseline")
        preprocess_templates, chain_exp_id, is_meta = _parse_preprocess_templates(preprocess_template_arg, project_root)

        # Print chain info
        if len(preprocess_templates) > 1 or is_meta:
            console.print(f"\n[bold cyan]Preprocessing Chain:[/bold cyan] {' → '.join(preprocess_templates)}")
            console.print(f"[dim]Chain experiment: {chain_exp_id}[/dim]\n")

        results_dict = {}
        for idx, tpl_name in enumerate(preprocess_templates):
            # Sub-module experiment ID: chain_exp_id/{idx}-{submodule}
            # Index prevents duplicate names when same template used multiple times
            submodule_exp_id = f"{idx}-{tpl_name}"
            full_exp_id = f"{chain_exp_id}/{submodule_exp_id}"
            input_source_idx = idx - 1
            input_source = f"{input_source_idx}-{preprocess_templates[input_source_idx]}" if idx > 0 else None

            # Create state for this preprocessing step
            state = ExperimentState.load_or_create(
                project_root=project_root,
                project_name=args.project,
                experiment_id=full_exp_id,
                pipeline=pipeline_def,
                run_invocation={"argv": argv, "cli_args": vars(args), "template": tpl_name, "input_source": input_source},
                create_dirs=True,
                setup_module_name=None,  # preprocess is not a setup module
            )

            # Build context for this step (use full_exp_id which includes chain directory)
            context = _build_module_context(
                project_root=project_root,
                project=args.project,
                module_name="preprocess",
                experiment_id=full_exp_id,
                config_module=config_module,
                pipeline_def=pipeline_def,
                argv=argv,
            )

            # Create module
            module_cls = ModuleRegistry.get("preprocess")
            module = module_cls(context)

            # Extract params and add input_source, chain_exp_id, and is_last_in_chain
            params = _extract_module_params(args, module_arg_map)
            params["preprocess_template"] = tpl_name
            params["input_source"] = input_source
            params["chain_exp_id"] = chain_exp_id
            is_last_in_chain = (idx == len(preprocess_templates) - 1)
            params["is_last_in_chain"] = is_last_in_chain
            module.set_invocation_params(params)

            # Run module
            executor = PipelineExecutor({"preprocess": module})
            module_results = executor.run_module("preprocess", force=args.force, skip_deps=args.skip_deps)

            result = module_results.get("preprocess")
            results_dict[f"preprocess-{tpl_name}"] = result

            # Fail fast
            if not result or not result.success:
                console.print(f"\n[red]✗ Preprocess chain stopped at: {tpl_name}[/red]")
                if result and result.error:
                    console.print(f"[red]Error: {result.error}[/red]\n")
                return 1

        # All preprocessing steps succeeded
        return 0

    # For non-preprocess modules, use standard flow
    experiment_id = args.experiment_id

    # If an experiment_id is provided for downstream modules, skip deps by default
    # to avoid requiring preprocess/model templates again.
    if args.command in ("predict", "submit", "fetch-score", "stack") and experiment_id:
        args.skip_deps = True

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
