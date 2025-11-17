"""
Unified submission workflow for Kaggle competitions.

Handles creating a Kaggle submission, waiting for processing,
fetching public leaderboard score via Playwright (CDP),
updating the submissions tracker, and creating a git commit
that ties code + local CV + public score together.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel

from kaggle_scraper import KaggleScraper
from submissions_tracker import SubmissionsTracker

console = Console()


@dataclass
class SubmissionArtifact:
    """Container returned by `create_submission` with tracking metadata."""

    path: Path
    filename: str
    project_root: Path
    competition: str
    tracker_entry: Optional[Dict[str, Any]] = None
    experiment: Optional[Dict[str, Any]] = None
    model_name: Optional[str] = None
    local_cv_score: Optional[float] = None
    notes: str = ""
    config: Optional[Dict[str, Any]] = None

    def tracker_id(self) -> Optional[int]:
        if self.tracker_entry:
            return self.tracker_entry.get("id")
        return None


class SubmissionRunner:
    """End-to-end Kaggle submission pipeline."""

    def __init__(
        self,
        artifact: SubmissionArtifact,
        kaggle_message: Optional[str] = None,
        *,
        wait_seconds: int = 30,
        cdp_url: str = "http://localhost:9222",
        auto_submit: bool = False,
        prompt: bool = True,
        skip_submit: bool = False,
        skip_browser: bool = False,
        skip_git: bool = False,
        extra_stage_paths: Optional[List[Path]] = None,
    ):
        self.artifact = artifact
        self.kaggle_message = kaggle_message or self._default_message()
        self.wait_seconds = wait_seconds
        self.cdp_url = cdp_url
        self.auto_submit = auto_submit
        self.prompt = prompt
        self.skip_submit = skip_submit
        self.skip_browser = skip_browser
        self.skip_git = skip_git
        self.extra_stage_paths = extra_stage_paths or []
        self.repo_root = artifact.project_root.parent

    def _default_message(self) -> str:
        parts = [self.artifact.model_name or "submission"]
        if self.artifact.local_cv_score is not None:
            parts.append(f"local {self.artifact.local_cv_score:.5f}")
        return " | ".join(parts)

    def execute(self) -> Optional[Dict[str, Any]]:
        """Run the submission workflow end-to-end."""
        if self.skip_submit:
            console.print("[yellow]Skipping Kaggle submission (flag enabled).[/yellow]")
            return None

        if self.prompt and not self.auto_submit:
            if not self._confirm():
                console.print("[yellow]Submission aborted by user.[/yellow]")
                return None

        self._submit_to_kaggle()
        if self.wait_seconds > 0:
            console.print(f"[cyan]Waiting {self.wait_seconds}s for Kaggle processing...[/cyan]")
            time.sleep(self.wait_seconds)

        score_data: Optional[Dict[str, Any]] = None
        if not self.skip_browser:
            score_data = asyncio.run(self._fetch_scores())
            if score_data and score_data.get("public_score") is not None:
                self._update_tracker(score_data["public_score"])
            else:
                console.print("[yellow]Could not fetch public score via Playwright.[/yellow]")

        if not self.skip_git:
            self._git_commit(score_data)
        else:
            console.print("[yellow]Skipping git commit (flag enabled).[/yellow]")

        if score_data:
            console.print(
                Panel.fit(
                    f"Public Score: {score_data.get('public_score', 'N/A')}\n"
                    f"Private Score: {score_data.get('private_score', 'N/A')}",
                    title="Kaggle Results",
                    border_style="green",
                )
            )

        return score_data

    def _confirm(self) -> bool:
        answer = input(
            f"\nSubmit {self.artifact.filename} to Kaggle competition "
            f"'{self.artifact.competition}'? [y/N]: "
        ).strip().lower()
        return answer in {"y", "yes"}

    def _submit_to_kaggle(self):
        console.print("[bold blue]Submitting file via Kaggle CLI...[/bold blue]")
        cmd = [
            "kaggle",
            "competitions",
            "submit",
            "-c",
            self.artifact.competition,
            "-f",
            str(self.artifact.path.name),
            "-m",
            self.kaggle_message,
        ]
        subprocess.run(
            cmd,
            cwd=self.artifact.path.parent,
            check=True,
        )
        console.print("[green]✓ Kaggle submission command completed[/green]")

    async def _fetch_scores(self) -> Optional[Dict[str, Any]]:
        console.print("[bold blue]Fetching public score via Playwright...[/bold blue]")
        scraper = KaggleScraper(self.cdp_url)
        try:
            await scraper.connect()
            result = await scraper.get_latest_submission_score(
                competition=self.artifact.competition,
                filename=self.artifact.filename,
            )
            return result
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Playwright error: {exc}[/red]")
            return None
        finally:
            await scraper.close()

    def _update_tracker(self, public_score: float):
        tracker_id = self.artifact.tracker_id()
        if tracker_id is None:
            console.print("[yellow]Tracker entry missing; skipping tracker update.[/yellow]")
            return

        tracker = SubmissionsTracker(self.artifact.project_root)
        tracker.update_scores(submission_id=tracker_id, public_score=public_score)

    def _git_commit(self, score_data: Optional[Dict[str, Any]]):
        console.print("[bold blue]Staging files for git commit...[/bold blue]")
        paths = [self.artifact.project_root] + self.extra_stage_paths
        for path in paths:
            if path and path.exists():
                subprocess.run(
                    ["git", "add", str(path)],
                    cwd=self.repo_root,
                    check=True,
                )

        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=self.repo_root,
        )

        if result.returncode == 0:
            console.print("[yellow]No staged changes detected; skipping git commit.[/yellow]")
            return

        message = self._build_commit_message(score_data)
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=self.repo_root,
            check=True,
        )
        console.print(f"[green]✓ Git commit created: {message}[/green]")

    def _build_commit_message(self, score_data: Optional[Dict[str, Any]]) -> str:
        parts = [
            f"submission({self.artifact.competition})",
            self.artifact.model_name or "model",
        ]
        if self.artifact.local_cv_score is not None:
            parts.append(f"local {self.artifact.local_cv_score:.5f}")
        public_score = None
        if score_data:
            public_score = score_data.get("public_score")
        if public_score is not None:
            parts.append(f"public {public_score:.5f}")
        else:
            parts.append("public pending")
        return ": ".join(parts[:1]) + " | " + " | ".join(parts[1:])
