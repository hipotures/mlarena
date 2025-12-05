"""Score fetching module."""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from typing import Optional

from mlarena.core.module import BaseModule, ModuleResult
from mlarena.core.registry import ModuleRegistry
from mlarena.utils.project import load_project_config


@ModuleRegistry.register
class FetchScoreModule(BaseModule):
    name = "fetch-score"
    description = "Fetch public leaderboard score"
    dependencies = {"submit"}

    @classmethod
    def register_cli_args(cls, parser) -> None:
        parser.add_argument("--score-placeholder", type=float, default=None, help="Optional score value to store.")

    def _fetch_latest_submission(self, competition: str) -> Optional[dict]:
        try:
            out = subprocess.check_output(
                ["kaggle", "competitions", "submissions", "-c", competition, "--csv"],
                text=True,
            )
        except Exception:
            return None

        reader = csv.DictReader(out.splitlines())
        rows = list(reader)
        return rows[0] if rows else None

    def execute(self) -> ModuleResult:
        artifact_dir: Path = self.context.artifact_dir
        artifact_dir.mkdir(parents=True, exist_ok=True)

        placeholder_score = self.invocation_params.get("score_placeholder")
        config = self.context.config_module or load_project_config(self.context.project_root)
        competition = getattr(config, "COMPETITION_NAME", self.context.project_name)

        latest = self._fetch_latest_submission(competition)
        score = placeholder_score
        if latest and latest.get("PublicScore"):
            try:
                score = float(latest["PublicScore"])
            except Exception:
                score = placeholder_score

        marker = artifact_dir / "fetch_score.txt"
        marker.write_text(f"Competition: {competition}\nScore: {score}\nLatest: {latest}")

        return ModuleResult(
            success=True,
            payload={"score": score, "latest_submission": latest},
            artifacts=[marker],
        )
