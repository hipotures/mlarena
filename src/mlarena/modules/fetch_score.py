"""Score fetching module.

Uses Kaggle CLI to retrieve the latest public leaderboard score.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from mlarena.core.module import BaseModule, ModuleResult
from mlarena.core.registry import ModuleRegistry
from mlarena.utils.project import load_project_config
from mlarena.utils.kaggle_api import fetch_submissions


@ModuleRegistry.register
class FetchScoreModule(BaseModule):
    """Fetch the latest public leaderboard score for a submission."""

    name = "fetch-score"
    description = "Fetch public leaderboard score"
    dependencies = {"submit"}

    def _extract_submission_id(self, row: dict) -> Optional[str]:
        for key in ("id", "submissionId", "submission_id", "ref"):
            value = row.get(key)
            if value:
                return str(value)
        return None

    def _resolve_submission_file(self) -> Optional[str]:
        # 1) explicit invocation parameter
        submission_file = self.invocation_params.get("submission_file")
        if submission_file:
            return str(submission_file)

        # 2) state.json (submit payload preferred)
        state_path = self.context.experiment_dir / "state.json"
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text())
                for module_name in ("submit", "predict"):
                    payload = state.get("modules", {}).get(module_name, {}).get("payload", {})
                    candidate = payload.get("submission_file") or payload.get("predictions")
                    if candidate:
                        return str(candidate)
            except Exception:
                pass

        return None

    def _match_submission(
        self,
        rows: list[dict],
        submission_id: Optional[str],
        submission_file: Optional[str],
    ) -> Optional[dict]:
        if submission_id:
            target_id = str(submission_id)
            for row in rows:
                if self._extract_submission_id(row) == target_id:
                    return row

        if submission_file:
            target_name = Path(submission_file).name
            for row in rows:
                if row.get("fileName") == target_name or row.get("file_name") == target_name:
                    return row

        return None

    def execute(self) -> ModuleResult:
        """
        Fetch the most recent submission score from Kaggle.

        Returns:
            ModuleResult containing the public score and submission metadata.
        """
        artifact_dir: Path = self.context.artifact_dir
        artifact_dir.mkdir(parents=True, exist_ok=True)

        placeholder_score = self.invocation_params.get("score_placeholder")
        config = self.context.config_module or load_project_config(self.context.project_root)
        competition = getattr(config, "COMPETITION_NAME", self.context.project_name)

        submission_id = self.invocation_params.get("submission_id")
        submission_file = self._resolve_submission_file()

        rows = fetch_submissions(competition)
        latest = rows[0] if rows else None
        matched = self._match_submission(rows, submission_id, submission_file)

        # Prefer the matched submission; fall back to latest when no target is provided
        target = matched if (submission_id or submission_file) else latest
        score = placeholder_score
        if target:
            # Kaggle CLI uses 'publicScore' (lowercase); keep fallback for 'PublicScore'
            for key in ("publicScore", "PublicScore"):
                if target.get(key):
                    try:
                        score = float(target[key])
                    except Exception:
                        score = placeholder_score
                    break

        marker = artifact_dir / "fetch_score.txt"
        marker.write_text(
            f"Competition: {competition}\n"
            f"Score: {score}\n"
            f"Target: {submission_id or submission_file}\n"
            f"Matched: {matched}\n"
            f"Latest: {latest}"
        )

        # Print score summary
        from rich.console import Console
        from mlarena.core.module import print_next_steps

        console = Console()

        # If a specific submission was requested but not found, return failure
        if (submission_id or submission_file) and matched is None:
            console.print(f"\n[yellow]⚠[/yellow] Submission not found on Kaggle")
            console.print(
                f"[dim]Expected {submission_id or submission_file}. "
                f"Wait a moment or verify the submission name/id.[/dim]\n"
            )
            return ModuleResult(
                success=False,
                error="Target submission not found on Kaggle",
                payload={
                    "score": None,
                    "latest_submission": latest,
                    "matched_submission": None,
                    "target_submission": submission_id or submission_file,
                },
                artifacts=[marker],
            )

        # Return failure if score not available - allows retry without --force
        if score is None:
            console.print(f"\n[yellow]⚠[/yellow] Score not available yet")
            console.print(f"[dim]Kaggle may still be processing your submission. Wait a few minutes and try again.[/dim]\n")
            return ModuleResult(
                success=False,
                error="Score not available yet - retry in a few minutes",
                payload={
                    "score": None,
                    "latest_submission": latest,
                    "matched_submission": matched,
                    "target_submission": submission_id or submission_file,
                },
                artifacts=[marker],
            )

        # Score successfully fetched (footer will display it)
        return ModuleResult(
            success=True,
            payload={
                "score": score,
                "latest_submission": latest,
                "matched_submission": matched,
                "target_submission": submission_id or submission_file,
            },
            artifacts=[marker],
        )
