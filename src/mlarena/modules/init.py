"""Initialization module."""

from __future__ import annotations

from typing import Dict

from mlarena.core.module import BaseModule, ModuleResult
from mlarena.core.registry import ModuleRegistry
from mlarena.utils.init import init_project


@ModuleRegistry.register
class InitModule(BaseModule):
    name = "init"
    description = "Initialize Kaggle project"
    dependencies = set()

    @classmethod
    def register_cli_args(cls, parser) -> None:
        parser.add_argument("--competition", "-c", help="Kaggle competition slug (defaults to --project)")
        parser.add_argument("--skip-download", action="store_true", help="Skip Kaggle data download")
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
        """Execute init using utils.init.init_project()."""
        competition = self.invocation_params.get("competition") or self.context.project_name

        # Call init_project directly
        result = init_project(
            project_root=self.context.project_root,
            competition_slug=competition,
            skip_download=self.invocation_params.get("skip_download", False),
            force=self.invocation_params.get("force", False),
            target_column=self.invocation_params.get("target_column"),
            problem_type=self.invocation_params.get("problem_type"),
            metric=self.invocation_params.get("metric"),
            id_column=self.invocation_params.get("id_column"),
            ignore_columns=self.invocation_params.get("ignore_columns"),
            submit_probas=self.invocation_params.get("submit_probas"),
            submit_labels=self.invocation_params.get("submit_labels", False),
            cdp_url=self.invocation_params.get("cdp_url"),
        )

        success = result.get("success", False)
        stats = result.get("stats", {})
        error = result.get("error")

        return ModuleResult(
            success=success,
            payload=stats,
            error=error if not success else None,
        )
