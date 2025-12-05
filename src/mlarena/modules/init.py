"""Initialization module (parity with legacy init-project)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List

from mlarena.core.module import BaseModule, ModuleResult
from mlarena.core.registry import ModuleRegistry


# Repository root (absolute path to kaggle repo root)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


@ModuleRegistry.register
class InitModule(BaseModule):
    name = "init"
    description = "Initialize Kaggle project (legacy init-project parity)"
    dependencies = set()

    @classmethod
    def register_cli_args(cls, parser) -> None:
        parser.add_argument("--competition", "-c", help="Kaggle competition slug (defaults to --project)")
        parser.add_argument("--skip-download", action="store_true", help="Skip Kaggle data download")
        parser.add_argument("--keep-zip", action="store_true", help="Keep zip after extraction")
        parser.add_argument("--migrate", action="store_true", help="Migrate existing project by moving contents to .old")
        parser.add_argument("--target-column", help="Target column override")
        parser.add_argument("--problem-type", choices=["binary", "regression", "multiclass"], help="Problem type override")
        parser.add_argument("--metric", help="Evaluation metric override")
        parser.add_argument("--id-column", help="ID column override")
        parser.add_argument("--ignore-columns", nargs="*", help="Columns to ignore during training")
        group = parser.add_mutually_exclusive_group()
        group.add_argument("--submit-probas", action="store_true", help="Force probability submissions")
        group.add_argument("--submit-labels", action="store_true", help="Force label submissions")
        parser.add_argument("--cdp-url", help="CDP endpoint for Kaggle page scrape (optional)")

    def execute(self) -> ModuleResult:
        project = self.context.project_name
        competition = self.invocation_params.get("competition") or project

        cmd = [
            "python",
            str(REPO_ROOT / "scripts" / "experiment_manager.py"),
            "init-project",
            "--project",
            competition,
        ]

        flag_map: Dict[str, str] = {
            "skip_download": "--skip-download",
            "keep_zip": "--keep-zip",
            "migrate": "--migrate",
            "submit_probas": "--submit-probas",
            "submit_labels": "--submit-labels",
        }

        for param, flag in flag_map.items():
            if self.invocation_params.get(param):
                cmd.append(flag)

        str_map = {
            "target_column": "--target-column",
            "problem_type": "--problem-type",
            "metric": "--metric",
            "id_column": "--id-column",
            "cdp_url": "--cdp-url",
        }
        for param, flag in str_map.items():
            val = self.invocation_params.get(param)
            if val:
                cmd.extend([flag, str(val)])

        ignore_cols = self.invocation_params.get("ignore_columns")
        if ignore_cols:
            cmd.extend(["--ignore-columns", *ignore_cols])

        # Snapshot existing top-level entries to track created items after init
        pre_items: List[str] = []
        project_root = self.context.project_root
        if project_root.exists():
            pre_items = [p.name for p in project_root.iterdir()]

        # Run legacy init-project (kept for parity) but capture output for state logging
        result = subprocess.run(cmd, capture_output=True, text=True)
        success = result.returncode == 0

        post_items: List[str] = []
        if project_root.exists():
            post_items = [p.name for p in project_root.iterdir()]
        created = sorted(list(set(post_items) - set(pre_items)))

        payload = {
            "cmd": " ".join(cmd),
            "returncode": result.returncode,
            "created": created,
            # Truncate logs to keep state.json small
            "stdout_tail": result.stdout[-2000:] if result.stdout else "",
            "stderr_tail": result.stderr[-2000:] if result.stderr else "",
        }

        return ModuleResult(success=success, payload=payload, error=None if success else "init failed")
