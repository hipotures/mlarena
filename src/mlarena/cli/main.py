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


GLOBAL_ARGS = {"project", "experiment_id", "force", "skip_deps", "command"}


def _build_parser(module_arg_map: Dict[str, List[str]]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mla", description="MLArena pipeline runner")
    parser.add_argument("--project", "-p", required=True, help="Competition/project name (projects/kaggle/<name>)")
    parser.add_argument("--experiment-id", "-e", help="Existing experiment id to resume")
    parser.add_argument("--force", "-f", action="store_true", help="Re-run completed modules")
    parser.add_argument("--skip-deps", action="store_true", help="Do not run dependencies automatically")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Special helper command
    subparsers.add_parser("modules", help="List available modules")

    for name in sorted(ModuleRegistry.available()):
        module_cls = ModuleRegistry.get(name)
        sub = subparsers.add_parser(name, help=module_cls.description or "")
        before = {a.dest for a in sub._actions}
        if hasattr(module_cls, "register_cli_args"):
            module_cls.register_cli_args(sub)
        after = {a.dest for a in sub._actions}
        module_arg_map[name] = sorted(after - before)
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
        print("\n".join(sorted(ModuleRegistry.available())))
        return 0

    project_root = Path("projects") / "kaggle" / args.project
    project_root.mkdir(parents=True, exist_ok=True)

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

    # Simple console reporting
    for mod_name, result in results.items():
        status = "ok" if result.success else "fail"
        print(f"[{status}] {mod_name}")
        if result.payload:
            print(f"  payload: {result.payload}")
        if result.error:
            print(f"  error: {result.error}")

    last = results.get(args.command)
    return 0 if (last and last.success) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
