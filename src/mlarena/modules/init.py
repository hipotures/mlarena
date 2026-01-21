"""Initialization module.

Creates Kaggle project structure, downloads data, and records experiment metadata.
"""

from __future__ import annotations


from mlarena.core.module import BaseModule, ModuleResult
from mlarena.core.registry import ModuleRegistry
from mlarena.utils.init import init_project


@ModuleRegistry.register
class InitModule(BaseModule):
    """Initialize a Kaggle competition project directory."""

    name = "init"
    description = "Initialize Kaggle project"
    dependencies = set()

    def execute(self) -> ModuleResult:
        """
        Provision project files and download Kaggle data.

        Returns:
            ModuleResult with initialization statistics and artifacts.

        Raises:
            RuntimeError: When the project cannot be initialized.
        """
        competition = (
            self.invocation_params.get("competition") or self.context.project_name
        )

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
