"""Submissions listing module (compat with scripts/submissions_tracker.py)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

from mlarena.core.module import BaseModule, ModuleResult
from mlarena.core.registry import ModuleRegistry


def _extract_module_argv(argv: List[str], module_name: str) -> List[str]:
    if module_name in argv:
        idx = argv.index(module_name)
        return argv[idx + 1 :]
    return list(argv)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--project")
    subparsers = parser.add_subparsers(dest="command")

    add_parser = subparsers.add_parser("add")
    add_parser.add_argument("filename")
    add_parser.add_argument("model_name")
    add_parser.add_argument("--local-cv", type=float)
    add_parser.add_argument("--cv-std", type=float)
    add_parser.add_argument("--public", type=float)
    add_parser.add_argument("--private", type=float)
    add_parser.add_argument("--notes", default="")

    update_parser = subparsers.add_parser("update")
    update_parser.add_argument("id", type=int)
    update_parser.add_argument("--public", type=float)
    update_parser.add_argument("--private", type=float)

    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--limit", type=int)
    list_parser.add_argument(
        "--sort-by",
        default="id",
        choices=["id", "local_cv_score", "public_score", "private_score", "timestamp"],
    )

    subparsers.add_parser("export")

    return parser


def _parse_args(argv: List[str]) -> Tuple[argparse.Namespace, List[str]]:
    parser = _build_parser()
    return parser.parse_known_args(argv)


@ModuleRegistry.register
class SubmissionsModule(BaseModule):
    name = "submissions"
    description = "List tracked submissions"

    def execute(self) -> ModuleResult:
        raw_argv = []
        if self.context and self.context.state:
            raw_argv = self.context.state.run.get("argv", [])

        module_argv = _extract_module_argv(raw_argv, "submissions")
        args, _ = _parse_args(module_argv)
        command = args.command or "list"

        repo_root = Path(__file__).resolve().parents[3]
        scripts_dir = repo_root / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from submissions_tracker import SubmissionsTracker  # type: ignore

        tracker = SubmissionsTracker(self.context.project_root)

        if command == "list":
            tracker.display_submissions(limit=args.limit, sort_by=args.sort_by)
            return ModuleResult(success=True, payload={"command": "list"})

        if command == "add":
            tracker.add_submission(
                filename=args.filename,
                model_name=args.model_name,
                local_cv_score=args.local_cv,
                cv_std=args.cv_std,
                public_score=args.public,
                private_score=args.private,
                notes=args.notes,
            )
            tracker.display_submissions(limit=10, sort_by="id")
            return ModuleResult(success=True, payload={"command": "add"})

        if command == "update":
            tracker.update_scores(
                submission_id=args.id,
                public_score=args.public,
                private_score=args.private,
            )
            tracker.display_submissions(limit=10, sort_by="public_score")
            return ModuleResult(success=True, payload={"command": "update"})

        if command == "export":
            tracker.export_to_csv()
            return ModuleResult(success=True, payload={"command": "export"})

        return ModuleResult(success=False, error=f"Unknown command: {command}")
