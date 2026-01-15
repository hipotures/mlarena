"""
CLI entry point for MLArena.

Provides dynamic module subcommands discovered via ModuleRegistry and
executes them through the PipelineExecutor with dependency handling.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
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


def _parse_preprocess_templates(template_arg: str, project_root: Path) -> Tuple[List[str], List[Dict], str, str, bool]:
    """
    Resolve a preprocess template argument into an execution chain with semantic hashing.
    """
    if template_arg is None:
        template_arg = "baseline"

    templates = [t.strip() for t in template_arg.split(",")]

    from mlarena.core.config import TemplateLoader
    loader = TemplateLoader(project_root, template_type="preprocess")

    if len(templates) == 1:
        template_config = loader.load(templates[0])

        if template_config and "chain" in template_config:
            chain = template_config["chain"]
            if not isinstance(chain, list):
                raise ValueError(f"Meta-template '{templates[0]}' chain must be a list")

            templates = chain
            chain_exp_id = f"pre-{template_arg}"
            is_meta = True
        else:
            chain_exp_id = f"pre-{templates[0]}"
            is_meta = False
    else:
        chain_str = ",".join(templates)
        chain_hash_id = hashlib.md5(chain_str.encode()).hexdigest()[:8]
        chain_exp_id = f"pre-chain-{chain_hash_id}"
        is_meta = False

    template_configs = []
    for tpl_name in templates:
        tpl_cfg = loader.load(tpl_name)
        if not tpl_cfg:
            raise ValueError(f"Template '{tpl_name}' not found in preprocess templates")
        template_configs.append(tpl_cfg)

    from mlarena.utils.hash_utils import compute_chain_hash
    combined_hash, individual_hashes = compute_chain_hash(template_configs, project_root)

    return templates, template_configs, chain_exp_id, combined_hash, is_meta


def _is_within_path(base: Path, target: Path) -> bool:
    try:
        target.relative_to(base)
        return True
    except ValueError:
        return False


def _safe_clear_preprocess_chain_dir(chain_base_dir: Path, project_root: Path, console) -> None:
    """
    Clear prior preprocess step outputs for a chain when --force is used.

    Safety rules:
    - Only operates inside <project_root>/experiments/<chain>/<hash>
    - The <hash> directory must be a 12-character hex string (0-f).
    - Refuses to delete if path is too shallow or outside experiments.
    """
    if not chain_base_dir.exists():
        return

    experiments_root = (project_root / "experiments").resolve()
    project_root_resolved = project_root.resolve()

    try:
        chain_resolved = chain_base_dir.resolve()
    except FileNotFoundError:
        return

    if not _is_within_path(experiments_root, chain_resolved):
        console.print(f"[yellow]⚠ Refusing to clean path outside experiments: {chain_base_dir}[/yellow]")
        return

    if chain_resolved == experiments_root:
        return

    rel_parts = chain_resolved.relative_to(experiments_root).parts
    if len(rel_parts) < 2:
        # Must be at least experiments/pre-template/hash
        return

    # Security: Only allow deleting if the directory name is a 12-char hex hash
    hash_name = chain_resolved.name
    if not re.match(r"^[0-9a-f]{12}$", hash_name):
        return

    if _is_within_path(project_root_resolved, chain_resolved):
        rel_chain = chain_resolved.relative_to(project_root_resolved)
    else:
        rel_chain = chain_resolved

    try:
        console.print(f"[dim]Force mode: removing entire hash directory {rel_chain}[/dim]")
        # shutil.rmtree is recursive
        import shutil
        shutil.rmtree(chain_resolved)
    except Exception as exc:
        console.print(f"[yellow]⚠ Failed to remove {rel_chain}: {exc}[/yellow]")


def _validate_setup_modules(
    project_root: Path,
    console,
) -> Tuple[bool, Optional[str]]:
    """
    Validate that required setup modules (init, eda) have been completed.
    """
    import json

    init_state_file = project_root / "experiments" / "init" / "state.json"
    if not init_state_file.exists():
        return False, (
            "Project initialization not found.\n"
            f"Run: mla init --project {project_root.name}"
        )

    try:
        with open(init_state_file) as f:
            init_state = json.load(f)
            init_module = init_state.get("modules", {}).get("init", {})
            init_status = init_module.get("status")

            if init_status != "completed":
                return False, (
                    f"Project initialization incomplete (status: {init_status}).\n"
                    f"Run: mla init --project {project_root.name} --force"
                )
    except (json.JSONDecodeError, KeyError) as e:
        return False, (
            f"Invalid init state file: {e}\n"
            f"Run: mla init --project {project_root.name} --force"
        )

    eda_state_file = project_root / "experiments" / "eda" / "state.json"
    if not eda_state_file.exists():
        return False, (
            "Exploratory data analysis not found.\n"
            f"Run: mla eda --project {project_root.name}"
        )

    try:
        with open(eda_state_file) as f:
            eda_state = json.load(f)
            eda_module = eda_state.get("modules", {}).get("eda", {})
            eda_status = eda_module.get("status")

            if eda_status != "completed":
                return False, (
                    f"Exploratory data analysis incomplete (status: {eda_status}).\n"
                    f"Run: mla eda --project {project_root.name} --force"
                )
    except (json.JSONDecodeError, KeyError) as e:
        return False, (
            f"Invalid eda state file: {e}\n"
            f"Run: mla eda --project {project_root.name} --force"
        )

    return True, None


def _resolve_preprocess_tune_study_name(config: GlobalConfig, project_root: Path) -> str:
    """Resolve study name for preprocess tuning (used for default exp-id)."""
    section = getattr(config, "preprocess_tune", {}) or {}
    if isinstance(section, dict):
        if section.get("study_name"):
            return str(section["study_name"])
        if section.get("experiment_prefix"):
            return str(section["experiment_prefix"])
    if hasattr(config, "study_name") and getattr(config, "study_name"):
        return str(getattr(config, "study_name"))

    super_chain_path = None
    if isinstance(section, dict):
        super_chain_path = section.get("super_chain")
    if not super_chain_path:
        super_chain_path = REPO_ROOT / "conf" / "preprocess" / "super_chain_optuna.yaml"
    else:
        super_chain_path = Path(super_chain_path)
        if not super_chain_path.is_absolute():
            super_chain_path = REPO_ROOT / super_chain_path

    try:
        import yaml
        if Path(super_chain_path).exists():
            payload = yaml.safe_load(Path(super_chain_path).read_text()) or {}
            tune_cfg = payload.get("tune", {}) or {}
            if tune_cfg.get("study_name"):
                return str(tune_cfg["study_name"])
            if payload.get("experiment_prefix"):
                return str(payload["experiment_prefix"])
    except Exception:
        pass

    return "optuna_preprocess"


def _resolve_preprocess_tune_model_template(config: GlobalConfig, project_root: Path) -> Optional[str]:
    section = getattr(config, "preprocess_tune", {}) or {}
    super_chain_path = section.get("super_chain")
    if not super_chain_path:
        super_chain_path = REPO_ROOT / "conf" / "preprocess" / "super_chain_optuna.yaml"
    try:
        import yaml
        payload = yaml.safe_load(Path(super_chain_path).read_text()) or {}
        eval_cfg = payload.get("evaluation", {}) or {}
        model_tpl = eval_cfg.get("model")
        if model_tpl:
            return str(model_tpl)
    except Exception:
        pass
    return None


def _build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    """
    Build a simplified top-level CLI parser.
    """
    parser = argparse.ArgumentParser(prog="mla", description="MLArena pipeline runner", add_help=add_help)
    # Command is handled manually to avoid greedy consumption of flag values
    parser.add_argument("--project", "-p", help="Project name (e.g., Titanic)")
    parser.add_argument("--exp-id", "-e", help="Experiment ID to resume")
    parser.add_argument("--profile", "-s", help="Config profile (e.g., smoke, dev)")
    parser.add_argument("--force", "-f", action="store_true", help="Re-run completed modules")
    parser.add_argument("--json-output", action="store_true", help="Output results in JSON format (silent mode)")
    parser.add_argument("--mcts", action="store_true", help="Use MCTS for search")
    parser.add_argument("--mcts-live", "--live", action="store_true", dest="mcts_live", help="Enable live tree visualization for MCTS")
    parser.add_argument("--telegram-test", action="store_true", help="Send a test Telegram message on startup")
    parser.add_argument("--classic", action="store_true", help="Execute chain step-by-step using intermediate files (legacy mode)")
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
    create_dirs: bool = True,
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
        create_dirs=create_dirs,
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
    standalone: bool = True,
) -> Tuple[int, Dict[str, ModuleResult], Optional[Path], List[str]]:
    """
    Execute a chain of preprocessing templates with content-addressed caching.
    """
    resolved_preprocess_template = config.preprocess_template or "baseline"

    preprocess_templates, template_configs, chain_exp_id, combined_hash, is_meta = \
        _parse_preprocess_templates(resolved_preprocess_template, project_root)

    chain_base_dir = project_root / "experiments" / chain_exp_id / combined_hash

    results = {}
    force = config.force
    preprocess_cfg = config.preprocess if isinstance(config.preprocess, dict) else {}
    quiet_preprocess_panel = bool(
        preprocess_cfg.get("quiet_preprocess_panel")
        or preprocess_cfg.get("quiet_preprocess_panels")
        or preprocess_cfg.get("quiet_preprocess")
    )

    if len(preprocess_templates) > 1 or is_meta:
        from rich.panel import Panel
        from rich.text import Text
        from rich.console import Group
        from rich.align import Align

        flow_items = []
        for i, tpl in enumerate(preprocess_templates):
            flow_items.append(Panel(f"[bold cyan]{tpl}[/]", expand=False, border_style="dim"))
            if i < len(preprocess_templates) - 1:
                flow_items.append(Text("↓", style="bold yellow"))

        # Helper to align everything center
        aligned_items = [Align.center(item) for item in flow_items]

        console.print(Panel(
            Group(
                Align.center(f"[dim]Chain: {chain_exp_id} | Hash: {combined_hash}[/dim]"),
                Text(""),
                *aligned_items
            ),
            title="[bold cyan]Preprocessing Flow[/bold cyan]",
            border_style="cyan",
            expand=False
        ))

    # Clean prior state if forced before saving new configs
    if force and chain_base_dir.exists():
        _safe_clear_preprocess_chain_dir(chain_base_dir, project_root, console)

    if not (chain_base_dir / "config").exists():
        from mlarena.utils.hash_utils import save_canonical_configs, compute_chain_hash

    # Check for FUSED execution mode
    # Check both config.preprocess.fuse (yaml) and potential cli override if exposed
    fuse_mode = False
    if isinstance(preprocess_cfg, dict):
        fuse_mode = preprocess_cfg.get("fuse", False)
    
    # Fused mode optimization: Run the whole chain in one go
    if fuse_mode and len(preprocess_templates) > 1 and not force:
         # Check if final result exists, if so return immediately
         final_step_id = f"{len(preprocess_templates) - 1}-{preprocess_templates[-1]}"
         final_exp_dir = chain_base_dir / final_step_id
         final_state_file = final_exp_dir / "state.json"
         
         if final_state_file.exists():
            try:
                with open(final_state_file) as f:
                    final_state = json.load(f)
                    module_entry = final_state.get("modules", {}).get("preprocess", {})
                    if module_entry.get("status") == "completed":
                         console.print(f"\n[bold yellow]Using cached FUSED preprocessed data[/bold yellow]")
                         # Fill results for all steps (virtual)
                         for idx, tpl in enumerate(preprocess_templates):
                             results[f"preprocess-{tpl}"] = ModuleResult(success=True, payload={"cached": True})
                         return 0, results, final_exp_dir, preprocess_templates
            except:
                pass

    if fuse_mode and len(preprocess_templates) > 1:
        console.print(f"[bold magenta]⚡ Fused Mode Enabled: Executing {len(preprocess_templates)} steps in-memory[/bold magenta]")
        
        final_idx = len(preprocess_templates) - 1
        final_tpl_name = preprocess_templates[final_idx]
        final_sub_id = f"{final_idx}-{final_tpl_name}"
        full_exp_id = f"{chain_exp_id}/{combined_hash}/{final_sub_id}"
        
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

        module.set_invocation_params({
            "fused_chain_configs": template_configs,
            "fused_chain_names": preprocess_templates,
            "chain_root_dir": str(chain_base_dir),
            "preprocess_template": final_tpl_name, # For locking/logging purposes
            "force": force,
            "lock": config.lock,
            "quiet_preprocess_panel": quiet_preprocess_panel,
        })

        executor = PipelineExecutor({"preprocess": module})
        # We only run the final step "module", but inside it executes everything
        module_results = executor.run_module("preprocess", force=force, skip_deps=False, console=console)
        result = module_results.get("preprocess")
        
        # Populate results for all steps to satisfy the contract
        for tpl in preprocess_templates:
            results[f"preprocess-{tpl}"] = result
            
        final_preprocess_exp_dir = chain_base_dir / final_sub_id
        return (0 if result.success else 1), results, final_preprocess_exp_dir, preprocess_templates


    # Determine if we should run in UNIFIED PIPELINE mode (now the default)
    # Disabled only if --classic is present or explicitly disabled in config
    use_classic = bool(getattr(config, "classic", False)) or "--classic" in argv
    use_pipeline = not use_classic
    
    # Check for deprecated or unknown flags in argv
    if "--pipeline" in argv:
        console.print("[bold red]✗ Error:[/bold red] The [yellow]--pipeline[/yellow] flag has been removed. Unified Pipeline is now the [bold]default[/bold] mode. Use [cyan]--classic[/cyan] for legacy behavior.")
        return 1, {}, None, []

    if standalone:
        allowed_flags = [
            "--classic", "--force", "--lock", "--cache", "--preprocess-template",
            "--project", "-p", "--exp-id", "--smoke", "--dev", "--preset",
            "--time-limit", "--use-gpu", "--seed", "--mla-retention"
        ]
        unsupported_flags = [a for a in argv if a.startswith("--") and a not in allowed_flags]
        if unsupported_flags:
            console.print(f"[bold red]✗ Error:[/bold red] Unknown preprocessing arguments: [yellow]{', '.join(unsupported_flags)}[/yellow]")
            return 1, {}, None, []

    if use_pipeline and len(preprocess_templates) > 1:
        console.print(f"[bold magenta]⚡ Unified Pipeline Enabled: Executing {len(preprocess_templates)} steps in-memory[/bold magenta]")
        
        # Prepare single execution context pointing directly to the last step
        # This prevents an extra 'artifacts' folder at the root level
        final_tpl = preprocess_templates[-1]
        final_idx = len(preprocess_templates) - 1
        last_step_subpath = f"{final_idx}-{final_tpl}"
        
        # Build context for 'preprocess' module pointing to the LAST step dir
        context = _build_module_context(
            project_root=project_root,
            project=project_name,
            module_name="preprocess",
            config=config,
            experiment_id=f"{chain_exp_id}/{combined_hash}/{last_step_subpath}", 
            config_module=config_module,
            pipeline_def=pipeline_def,
            argv=argv,
        )
        
        module_cls = ModuleRegistry.get("preprocess")
        module = module_cls(context)
        
        # Pass the full list of templates as 'steps'
        module.set_invocation_params({
            "preprocess_template": resolved_preprocess_template,
            "steps": preprocess_templates,
            "quiet_preprocess_panel": quiet_preprocess_panel,
            "cache": getattr(config, "cache", False),
            "classic": use_classic # Pass classic flag status
        })
        
        executor = PipelineExecutor({"preprocess": module})
        module_results = executor.run_module("preprocess", force=force, skip_deps=False, console=console)
        result = module_results.get("preprocess")
        
        # Map result to all virtual steps for the return value
        for tpl in preprocess_templates:
            results[f"preprocess-{tpl}"] = result
            
        # The physical results are in the last step folder (handled inside preprocess.py)
        final_exp_dir = chain_base_dir / last_step_subpath
        return (0 if result.success else 1), results, final_exp_dir, preprocess_templates

    execution_start_idx = 0

    if not force and chain_base_dir.exists():
        all_completed = True
        resume_from_idx = None

        for idx, tpl_name in enumerate(preprocess_templates):
            submodule_exp_id = f"{idx}-{tpl_name}"
            exp_dir = chain_base_dir / submodule_exp_id
            state_file = exp_dir / "state.json"

            if not state_file.exists():
                all_completed = False
                resume_from_idx = idx
                break

            try:
                with open(state_file) as f:
                    saved_state = json.load(f)
                    module_entry = saved_state.get("modules", {}).get("preprocess", {})
                    status = module_entry.get("status")

                    if status != "completed":
                        all_completed = False
                        resume_from_idx = idx
                        break

                    results[f"preprocess-{tpl_name}"] = ModuleResult(
                        success=True,
                        payload=module_entry.get("payload", {})
                    )
            except (json.JSONDecodeError, KeyError):
                all_completed = False
                resume_from_idx = idx
                break

        if all_completed:
            from mlarena.core.display import print_module_header, print_module_footer
            from datetime import datetime

            first_template = preprocess_templates[0]
            first_config = template_configs[0] if template_configs else {}
            display_exp_id = f"{chain_exp_id}/{combined_hash}/0-{first_template}"

            started_at = datetime.now().isoformat()
            print_module_header(
                module_name="preprocess",
                started_at=started_at,
                experiment_id=display_exp_id,
                template_name=first_template,
                template_config=first_config,
                project_root=project_root,
                project_name=project_name,
                cli_invocation={
                    "quiet_preprocess_panel": quiet_preprocess_panel,
                    "json_output": config.json_output
                },
                console=console
            )

            finished_at = datetime.now().isoformat()
            final_step_id = f"{len(preprocess_templates) - 1}-{preprocess_templates[-1]}"
            final_preprocess_exp_dir = chain_base_dir / final_step_id
            footer_exp_id = f"{chain_exp_id}/{combined_hash}/{final_step_id}"
            input_source = None
            if len(preprocess_templates) > 1:
                input_source = f"{len(preprocess_templates) - 2}-{preprocess_templates[-2]}"

            final_state_file = final_preprocess_exp_dir / "state.json"
            output_paths = {}
            shapes = None
            cli_invocation = {
                "chain_exp_id": f"{chain_exp_id}/{combined_hash}",
                "input_source": input_source,
                "quiet_preprocess_panel": quiet_preprocess_panel,
                "json_output": config.json_output,
            }
            if final_state_file.exists():
                try:
                    final_state = json.loads(final_state_file.read_text())
                    final_payload = final_state.get("modules", {}).get("preprocess", {}).get("payload", {})
                    if "train_processed" in final_payload:
                        output_paths["train_processed"] = final_payload["train_processed"]
                    if "test_processed" in final_payload:
                        output_paths["test_processed"] = final_payload["test_processed"]
                    if "orig_processed" in final_payload and final_payload["orig_processed"]:
                        output_paths["orig_processed"] = final_payload["orig_processed"]
                    if "tuning_processed" in final_payload and final_payload["tuning_processed"]:
                        output_paths["tuning_processed"] = final_payload["tuning_processed"]
                    custom_state = final_payload.get("custom_module_state", {})
                    if custom_state.get("eval_path"):
                        output_paths["eval_processed"] = custom_state["eval_path"]
                    shapes = final_payload.get("shapes")
                except:
                    pass

            print_module_footer(
                module_name="preprocess",
                finished_at=finished_at,
                duration=0.0,
                output_paths=output_paths,
                shapes=shapes,
                metrics={"cache_status": f"✓ Cached (hash: {combined_hash})"},
                project_root=project_root,
                project_name=project_name,
                experiment_id=footer_exp_id,
                cli_invocation=cli_invocation,
                console=console
            )

            return 0, results, final_preprocess_exp_dir, preprocess_templates

        elif resume_from_idx is not None and resume_from_idx > 0:
            console.print(f"\n[yellow]Resuming from step {resume_from_idx}: {preprocess_templates[resume_from_idx]}[/yellow]")
            execution_start_idx = resume_from_idx

    for idx in range(execution_start_idx, len(preprocess_templates)):
        tpl_name = preprocess_templates[idx]
        submodule_exp_id = f"{idx}-{tpl_name}"
        full_exp_id = f"{chain_exp_id}/{combined_hash}/{submodule_exp_id}"
        input_source_idx = idx - 1
        input_source = f"{input_source_idx}-{preprocess_templates[input_source_idx]}" if idx > 0 else None

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
            "chain_exp_id": f"{chain_exp_id}/{combined_hash}",
            "is_last_in_chain": is_last_in_chain,
            "force": force,
            "json_output": config.json_output,
            "lock": config.lock,
            "quiet_preprocess_panel": quiet_preprocess_panel,
        })

        executor = PipelineExecutor({"preprocess": module})
        module_results = executor.run_module("preprocess", force=force, skip_deps=False, console=console)

        result = module_results.get("preprocess")
        results[f"preprocess-{tpl_name}"] = result

        if not result or not result.success:
            return 1, results, None, preprocess_templates

    final_preprocess_exp_dir = None
    if preprocess_templates:
        final_step_id = f"{len(preprocess_templates) - 1}-{preprocess_templates[-1]}"
        final_preprocess_exp_dir = chain_base_dir / final_step_id

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

    console = Console(force_terminal=True, quiet=config.json_output)
    results: Dict[str, ModuleResult] = {}

    model_template = config.model_template
    force = config.force
    skip_submit = config.skip_submit
    skip_git = config.skip_git
    wait_seconds = config.wait_seconds

    # Pipeline sequence (prerequisites validated separately)
    pipeline_modules = ["model", "predict", "submit", "fetch-score"]

    if force:
        console.print(f"\n[dim]Force mode: ON[/dim]\n")

    # Validate prerequisites (init, eda must be run manually)
    console.print("[bold cyan]Validating prerequisites...[/bold cyan]")

    is_valid, error_msg = _validate_setup_modules(project_root, console)

    if not is_valid:
        console.print(f"\n[bold red]✗ Prerequisites validation failed:[/bold red]\n")
        console.print(f"[yellow]{error_msg}[/yellow]\n")
        return 1

    console.print("[dim]✓ Prerequisites validated (init, eda completed)[/dim]\n")

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
    if not resolved_preprocess_template:
        try:
            from mlarena.core.config import TemplateLoader
            loader = TemplateLoader(project_root, template_type="model")
            model_tpl_cfg = loader.load(model_template)
            resolved_preprocess_template = model_tpl_cfg.get("preprocess_template")
        except Exception:
            resolved_preprocess_template = None

    # Phase 1: Preprocessing chain
    preprocess_templates = []
    final_preprocess_exp_dir = None
    if resolved_preprocess_template:
        config.preprocess_template = resolved_preprocess_template
        exit_code, chain_results, final_preprocess_exp_dir, preprocess_templates = run_preprocess_chain(
            project_root, project_name, config, config_module, pipeline_def, argv or [], console, standalone=False
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
        experiment_id=config.experiment_id,
        config_module=config_module,
        pipeline_def=pipeline_def,
        argv=argv,
    )

    shared_experiment_id = model_context.experiment_id
    model_cls = ModuleRegistry.get("model")
    model_module = model_cls(model_context)

    # Use the full resolved template name (e.g. the chain name) for reporting
    final_preprocess_template = resolved_preprocess_template
    
    model_params = {
        "model_template": model_template,
        "preprocess_template": final_preprocess_template,
        "preprocess_exp_dir": str(final_preprocess_exp_dir) if final_preprocess_exp_dir else None,
        "force": force,
        "json_output": config.json_output,
        "lock": config.lock,
    }
    # Backward compat for some model modules that still use dict
    model_params.update(config.model) 
    for field, value in config.common.model_dump().items():
        if value is not None:
            model_params[field] = value
    
    model_module.set_invocation_params(model_params)

    executor = PipelineExecutor({"model": model_module})
    module_results = executor.run_module("model", force=force, skip_deps=True, console=console)
    result = module_results.get("model")
    results["model"] = result

    if not result or not result.success:
        return 1

    remaining_modules = ["predict"]
    if not skip_submit:
        remaining_modules.extend(["submit", "fetch-score"])
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

        # Inject json_output into all modules
        invocation_params = {"json_output": config.json_output}

        if module_name == "submit":
            invocation_params["skip_submit"] = skip_submit
            module.set_invocation_params(invocation_params)
        elif module_name == "predict":
            invocation_params.update({
                "preprocess_template": final_preprocess_template,
                "preprocess_exp_dir": str(final_preprocess_exp_dir) if final_preprocess_exp_dir else None,
            })
            module.set_invocation_params(invocation_params)
        else:
            module.set_invocation_params(invocation_params)

        all_modules[module_name] = module

    executor = PipelineExecutor(all_modules)
    for module_name in remaining_modules:
        if module_name == "fetch-score" and wait_seconds > 0:
            console.print(f"\n[dim]Waiting {wait_seconds}s for Kaggle processing...[/dim]")
            time.sleep(wait_seconds)

        module_results = executor.run_module(module_name, force=force, skip_deps=True, console=console)
        result = module_results.get(module_name)
        results[module_name] = result

        if not result or not result.success:
            return 1

    if not skip_git:
        _create_auto_flow_commit(project_name, results, console)

    if not config.json_output:
        console.print("\n[bold green]✓ Auto-flow completed successfully[/bold green]\n")
    return 0

    # All modules succeeded - create git commit
    if not skip_git:
        _create_auto_flow_commit(project_name, results, console)

    if not config.json_output:
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
        local_cv_score = model_result.payload.get("local_cv_score") or model_result.payload.get("local_cv")

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
    from rich.console import Console
    
    argv = argv or sys.argv[1:]
    json_output_enabled = "--json-output" in argv
    console = Console(force_terminal=True, quiet=json_output_enabled)

    if json_output_enabled:
        import logging
        import warnings
        import os
        # Silence major loggers
        logging.getLogger("autogluon").setLevel(logging.ERROR)
        logging.getLogger("mlarena").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore")
        
        # Redirect stderr to devnull
        sys.stderr = open(os.devnull, 'w')
    
    # Discover modules
    if not ModuleRegistry.available():
        ModuleRegistry.discover()

    # Pre-scan for module help request
    # Detect if user is asking for help on a specific module (e.g. "mla exp --help")
    # In this case, we disable the main parser help so the module can show its own help.
    aliases = {"exp": "experiments", "sub": "submissions", "pre": "preprocess"}
    module_help_requested = False
    
    if "-h" in argv or "--help" in argv:
        for arg in argv:
            potential_cmd = aliases.get(arg, arg)
            if potential_cmd in ("experiments", "submissions", "queue"):
                module_help_requested = True
                break

    parser = _build_parser(add_help=not module_help_requested)
    args, overrides = parser.parse_known_args(argv)

    # Convert --flag value to key=value for OmegaConf
    overrides = _convert_dash_args_to_overrides(overrides)
    
    # Merge argparse known flags into overrides so they reach GlobalConfig
    for key, value in vars(args).items():
        if value is True: # Only add enabled boolean flags
            if not any(o.startswith(f"{key}=") for o in overrides):
                overrides.append(f"{key}=true")
        elif value is not None and not isinstance(value, bool):
            # For non-boolean args like --project or --profile, add if not already in overrides
            if not any(o.startswith(f"{key}=") for o in overrides):
                overrides.append(f"{key}={value}")

    # Detect command from overrides (first non-override argument)
    command = None
    if overrides and not overrides[0].startswith("-") and "=" not in overrides[0]:
        # First override is a plain value (potential command)
        potential_command = overrides[0]
        # Check if it's a valid module or 'modules' command or 'queue' command
        potential_command = overrides[0]
        
        # Resolve aliases
        if potential_command == "exp":
            command = "experiments"
            overrides.pop(0)
            if "exp" in argv:
                argv[argv.index("exp")] = "experiments"
        elif potential_command == "sub":
            command = "submissions"
            overrides.pop(0)
            if "sub" in argv:
                argv[argv.index("sub")] = "submissions"
        elif potential_command == "pre":
            command = "preprocess"
            overrides.pop(0)
            if "pre" in argv:
                argv[argv.index("pre")] = "preprocess"
        elif potential_command in ("modules", "queue") or potential_command in ModuleRegistry.available():
            command = overrides.pop(0)
        else:
            available = ", ".join(sorted(ModuleRegistry.available()))
            console.print(f"[bold red]✗ Error:[/bold red] Unknown command: [yellow]{potential_command}[/yellow]")
            if available:
                console.print(f"[dim]Available modules: {available}[/dim]")
            return 1

    # Detect project
    project = args.project

    # Handle preprocess tune subcommand: "pre tune" or "pre ... tune"
    if command == "preprocess" and "tune" in overrides:
        overrides = [o for o in overrides if o != "tune"]
        command = "preprocess-tune"
    
    # If project not in flags, try to find it in overrides (e.g., project=Titanic)
    if not project:
        # Simple heuristic: look for project=... in overrides
        for override in overrides:
            if override.startswith("project="):
                project = override.split("=", 1)[1]
                break

    # If help requested for a module, allow missing project (use dummy)
    if not project and module_help_requested:
        project = "_help_mode_dummy_"

    if not project and command != "modules":
        parser.print_help()
        console.print("\n[bold red]✗ Error:[/bold red] [yellow]--project[/yellow] is required")
        return 1

    # Build Unified Config
    try:
        # Combine profile and other overrides
        if args.profile:
            overrides.insert(0, f"profile={args.profile}")
        if args.force:
            overrides.append("force=true")
        if args.json_output:
            overrides.append("json_output=true")
        if args.exp_id:
            overrides.append(f"experiment_id={args.exp_id}")
        if args.mcts:
            overrides.append("mcts.enabled=true")
        if args.mcts_live:
            overrides.append("mcts_live=true")
            
        config = get_config(project or "default", overrides)
    except Exception as e:
        console.print(f"[bold red]✗ Config error:[/bold red] {e}")
        return 1

    # Handle 'modules' command
    if command == "modules":
        available = sorted(ModuleRegistry.available())
        console.print("\n".join(available))
        return 0

    # Handle 'queue' command - delegate to standalone script
    if command == "queue":
        # Note: Task Queue manages computation. For submissions, see scripts/submission_queue.py
        # Build arguments for task_queue.py (expects --project before subcommand)
        # Format: task_queue.py --project PROJECT subcommand [args]
        queue_args = ["--project", project] + [arg for arg in sys.argv[1:] if arg != "queue" and arg not in ["-p", "--project", project]]
        result = subprocess.run(
            ["python", "scripts/task_queue.py"] + queue_args,
            cwd=REPO_ROOT
        )
        return result.returncode

    project_root = REPO_ROOT / "projects" / "kaggle" / project

    # Default experiment id for preprocess-tune
    if command == "preprocess-tune" and not config.experiment_id:
        study_name = _resolve_preprocess_tune_study_name(config, project_root)
        config.experiment_id = f"optuna_{study_name}"
    
    # Auto-flow: no command provided
    if command is None:
        try:
            return run_auto_flow(
                project_root=project_root,
                project_name=project,
                config=config,
                argv=argv,
            )
        except Exception as e:
            if e.__class__.__name__ == "OverwriteLockedError":
                from rich.console import Console
                console = Console(force_terminal=True, quiet=config.json_output)
                console.print(f"[bold red]✗ {str(e)}[/bold red]")
                return 1
            raise

    # Single module execution
    if command not in ModuleRegistry.available():
        console.print(f"[bold red]✗ Error:[/bold red] Unknown module: [yellow]{command}[/yellow]")
        return 1

    # Load project config and pipeline
    config_module = None
    pipeline_def = {}
    if project_root.exists():
        config_module = load_project_config(project_root)
        pipeline_def, _ = load_pipeline_def("default", project_root=project_root)
    elif command != "init" and project != "_help_mode_dummy_":
        console.print(f"[bold red]✗ Error:[/bold red] Project '[yellow]{project}[/yellow]' not initialized. Run: [cyan]mla init --project {project}[/cyan]")
        return 1

    # Special case: list-style admin commands (avoid creating experiment dirs)
    if command in {"submissions", "experiments"}:
        context = _build_module_context(
            project_root=project_root,
            project=project,
            module_name=command,
            config=config,
            experiment_id=f"cli-{command}",
            config_module=config_module,
            pipeline_def=pipeline_def,
            argv=argv,
            create_dirs=False,
        )

        module_cls = ModuleRegistry.get(command)
        module = module_cls(context)
        params = getattr(config, command.replace("-", "_"), {})
        if not isinstance(params, dict):
            params = {}
        else:
            params = params.copy()

        # Inject top-level config fields
        for field in ["force", "json_output", "lock"]:
            if hasattr(config, field):
                params[field] = getattr(config, field)

        module.set_invocation_params(params)
        result = module.execute()
        return 0 if (result and result.success) else 1

    # Special case: preprocess chain (even if run as single command)
    if command == "preprocess":
        from rich.console import Console
        console = Console(force_terminal=True, quiet=config.json_output)

        # Check if project exists
        if not project_root.exists():
            console.print(f"[bold red]✗ Error:[/bold red] Project '[yellow]{project}[/yellow]' not initialized. Run: [cyan]mla init --project {project}[/cyan]")
            return 1

        try:
            exit_code, _, _, _ = run_preprocess_chain(
                project_root, project, config, config_module, pipeline_def, argv, console
            )
            return exit_code
        except Exception as e:
            if e.__class__.__name__ == "OverwriteLockedError":
                from rich.console import Console
                console = Console(force_terminal=True, quiet=config.json_output)
                console.print(f"[bold red]✗ {str(e)}[/bold red]")
                return 1
            raise

    # Special case: modules requiring auto-preprocessing
    # Resolve and run preprocessing chain before module execution
    final_preprocess_template = None
    final_preprocess_exp_dir = None
    if command in ["model", "predict", "submit", "fetch-score", "tune", "stack"]:
        from rich.console import Console
        console = Console(force_terminal=True, quiet=config.json_output)

        # Resolve preprocess template (CLI arg or from model template)
        resolved_preprocess_template = config.preprocess_template
        if not resolved_preprocess_template:
            try:
                from mlarena.core.config import TemplateLoader
                loader = TemplateLoader(project_root, template_type="model")
                model_tpl_cfg = loader.load(config.model_template)
                resolved_preprocess_template = model_tpl_cfg.get("preprocess_template")
            except Exception:
                resolved_preprocess_template = None

        # Run preprocessing chain if template exists
        preprocess_templates = []
        if resolved_preprocess_template:
            config.preprocess_template = resolved_preprocess_template
            exit_code, chain_results, final_preprocess_exp_dir, preprocess_templates = run_preprocess_chain(
                project_root, project, config, config_module, pipeline_def, argv, console, standalone=False
            )
            if exit_code != 0:
                return exit_code

            # Use the full resolved template name (e.g. the chain name) for reporting
            final_preprocess_template = resolved_preprocess_template

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
        
        # Inject params into ALL modules so dependencies (like model) receive configuration
        # Bridging: provide a combined dict for modules still using invocation_params
        params = getattr(config, name.replace("-", "_"), {})
        if not isinstance(params, dict):
            params = {}
        else:
            params = params.copy()
        
        # Inject top-level config fields that modules expect
        for field in ["model_template", "preprocess_template", "force", "json_output", "skip_submit", "lock", "mcts_live", "n_trials"]:
            if hasattr(config, field):
                params[field] = getattr(config, field)
        
        # Explicitly handle renamed mcts_enabled -> mcts for module compatibility
        if hasattr(config, "mcts"):
            params["mcts"] = config.mcts

        # Inject all fields from 'mcts' section if mcts is active
        if config.mcts and hasattr(config, "mcts_section"):
            if isinstance(config.mcts_section, dict):
                for k, v in config.mcts_section.items():
                    params[f"mcts.{k}"] = v

        # Also inject common fields (these override template defaults if explicitly set via CLI)
        for field, value in config.common.model_dump().items():
            if value is not None:
                params[field] = value

        # Override with auto-resolved preprocessing params (for model command)
        if name == "model" and final_preprocess_template:
            params["preprocess_template"] = final_preprocess_template
            params["preprocess_exp_dir"] = str(final_preprocess_exp_dir) if final_preprocess_exp_dir else None

        if name == "preprocess-tune":
            section = getattr(config, "preprocess_tune", {}) or {}
            if "model_template" not in section:
                resolved_tpl = _resolve_preprocess_tune_model_template(config, project_root)
                if resolved_tpl:
                    params["model_template"] = resolved_tpl

        module.set_invocation_params(params)
        modules[name] = module

    executor = PipelineExecutor(modules)
    try:
        results = executor.run_module(command, force=config.force, skip_deps=config.skip_deps, console=console)
        last = results.get(command)
        return 0 if (last and last.success) else 1
    except Exception as e:
        if e.__class__.__name__ == "OverwriteLockedError":
            from rich.console import Console
            console = Console(force_terminal=True)
            console.print(f"[bold red]✗ {str(e)}[/bold red]")
            return 1
        raise


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
