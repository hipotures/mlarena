"""
CLI entry point for MLArena.

Provides dynamic module subcommands discovered via ModuleRegistry and
executes them through the PipelineExecutor with dependency handling.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

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
    subparsers = parser.add_subparsers(dest="command", required=True)

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
    mod_args = module_arg_map.get(args.command, [])
    for key in mod_args:
        if hasattr(args, key):
            params[key] = getattr(args, key)
    return params


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


def main(argv: List[str] | None = None) -> int:
    argv = argv or sys.argv[1:]

    # Reset registry to avoid duplicate registration across repeated invocations (tests).
    ModuleRegistry.clear()
    ModuleRegistry.discover()

    module_arg_map: Dict[str, List[str]] = {}
    parser = _build_parser(module_arg_map)
    args = parser.parse_args(argv)

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

    state = ExperimentState.load_or_create(
        project_root=project_root,
        project_name=args.project,
        experiment_id=args.experiment_id,
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

    # Simple console reporting (only show summary for non-init modules)
    # Init module already prints its own Rich output, so we don't want to clutter
    if args.command != "init":
        for mod_name, result in results.items():
            status = "ok" if result.success else "fail"
            print(f"[{status}] {mod_name}")
            if args.show_payload and result.payload:
                print(f"  payload: {result.payload}")
            if result.error:
                print(f"  error: {result.error}")

    last = results.get(args.command)

    return 0 if (last and last.success) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
