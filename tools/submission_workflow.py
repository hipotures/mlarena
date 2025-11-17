"""
Unified submission workflow for Kaggle competitions.

Handles creating a Kaggle submission, waiting for processing,
fetching public leaderboard score via Playwright (CDP),
updating the submissions tracker, and creating a git commit
that ties code + local CV + public score together.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

from rich.console import Console
from rich.panel import Panel

from kaggle_scraper import KaggleScraper
from submissions_tracker import SubmissionsTracker
from experiment_manager import ExperimentManager

console = Console()
REPO_ROOT = Path(__file__).resolve().parent.parent


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
        resume_mode: bool = False,
        experiment_id: Optional[str] = None,
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
        self.resume_mode = resume_mode
        self.experiment_id = experiment_id
        self._experiment_manager = None
        if experiment_id:
            project_name = artifact.project_root.name
            self._experiment_manager = ExperimentManager.load_or_create(project_name, experiment_id)
            self._experiment_manager.start_module(
                "submit",
                {
                    "kaggle_message": self.kaggle_message,
                    "resume_mode": resume_mode,
                },
            )

    def _default_message(self) -> str:
        parts = [self.artifact.model_name or "submission"]
        if self.artifact.local_cv_score is not None:
            parts.append(f"local {self.artifact.local_cv_score:.5f}")
        return " | ".join(parts)

    def execute(self) -> Optional[Dict[str, Any]]:
        """Run the submission workflow end-to-end."""
        if self.skip_submit:
            console.print("[yellow]Skipping Kaggle submission (flag enabled).[/yellow]")
            if not self.resume_mode:
                return None
        else:
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
                self._update_tracker(score_data["public_score"], score_data=score_data)
            else:
                console.print("[yellow]Could not fetch public score via Playwright.[/yellow]")

        if not self.skip_git:
            self._git_commit(score_data)
        else:
            console.print("[yellow]Skipping git commit (flag enabled).[/yellow]")

        if score_data:
            public_score = score_data.get("public_score", "N/A")
            private_score = score_data.get("private_score")
            lines = [f"Public Score: {public_score}"]
            if private_score is not None:
                lines.append(f"Private Score: {private_score}")
            console.print(
                Panel.fit(
                    "\n".join(lines),
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

    def _update_tracker(self, public_score: float, score_data: Optional[Dict[str, Any]] = None):
        tracker = SubmissionsTracker(self.artifact.project_root)
        tracker_id = self.artifact.tracker_id()
        local_cv = self.artifact.local_cv_score
        if local_cv is None and score_data:
            local_cv = score_data.get("local_cv")
        if tracker_id is None:
            console.print("[yellow]Tracker entry missing; creating new submission entry.[/yellow]")
            entry = tracker.add_submission(
                filename=self.artifact.filename,
                model_name=self.artifact.model_name or self.artifact.filename,
                local_cv_score=local_cv,
                notes=self.artifact.notes or "Auto-added via resume",
                config=self.artifact.config,
                public_score=public_score,
            )
            self.artifact.tracker_entry = entry
            tracker_id = entry["id"]
        else:
            tracker.update_scores(submission_id=tracker_id, public_score=public_score)
            if local_cv is not None and (self.artifact.tracker_entry is None or not self.artifact.tracker_entry.get("local_cv_score")):
                for sub in tracker.submissions:
                    if sub["id"] == tracker_id:
                        sub["local_cv_score"] = local_cv
                        tracker._save_submissions()
                        self.artifact.local_cv_score = local_cv
                        if self.artifact.tracker_entry:
                            self.artifact.tracker_entry["local_cv_score"] = local_cv
                        break
        if self._experiment_manager:
            payload = {
                "public_score": public_score,
                "local_cv": local_cv,
                "tracker_id": tracker_id,
                "submission_file": str(self.artifact.path.relative_to(self.artifact.project_root)),
            }
            if score_data and score_data.get("row_text"):
                payload["snapshot"] = score_data["row_text"]
            self._experiment_manager.complete_module("submit", payload)

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


def _load_project_context(project_name: str):
    project_root = (REPO_ROOT / project_name).resolve()
    if not project_root.exists():
        raise FileNotFoundError(f"Project directory '{project_name}' not found at {project_root}")

    code_dir = project_root / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    config = importlib.import_module("utils.config")
    return project_root, config


def _build_artifact_from_filename(project_name: str, filename: str) -> SubmissionArtifact:
    project_root, config = _load_project_context(project_name)
    submission_path = project_root / "submissions" / filename
    if not submission_path.exists():
        raise FileNotFoundError(f"Submission file '{filename}' not found under {submission_path.parent}")

    tracker = SubmissionsTracker(project_root)
    tracker_entry = next((s for s in tracker.submissions if s["filename"] == filename), None)
    model_name = (tracker_entry or {}).get("model_name", filename)
    local_cv = (tracker_entry or {}).get("local_cv_score")
    notes = (tracker_entry or {}).get("notes", "")
    config_dict = (tracker_entry or {}).get("config")

    return SubmissionArtifact(
        path=submission_path,
        filename=filename,
        project_root=project_root,
        competition=getattr(config, "COMPETITION_NAME", project_name),
        tracker_entry=tracker_entry,
        experiment=None,
        model_name=model_name,
        local_cv_score=local_cv,
        notes=notes,
        config=config_dict,
    )


def _run_resume(args):
    artifact = _build_artifact_from_filename(args.project, args.filename)
    runner = SubmissionRunner(
        artifact=artifact,
        kaggle_message=args.kaggle_message or f"resume {args.filename}",
        wait_seconds=args.wait_seconds,
        cdp_url=args.cdp_url,
        auto_submit=True,
        prompt=False,
        skip_submit=True,
        skip_browser=args.skip_score_fetch,
        skip_git=args.skip_git,
        resume_mode=True,
        experiment_id=args.experiment_id,
    )
    runner.execute()


def main():
    parser = argparse.ArgumentParser(description="Submission workflow helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    resume_parser = subparsers.add_parser("resume", help="Fetch public score for an existing submission")
    resume_parser.add_argument("--project", required=True, help="Competition directory (e.g., playground-series-s5e11)")
    resume_parser.add_argument("--filename", required=True, help="Submission filename (e.g., submission-YYYYMMDDHHMMSS.csv)")
    resume_parser.add_argument("--cdp-url", default="http://localhost:9222", help="Playwright CDP endpoint")
    resume_parser.add_argument("--wait-seconds", type=int, default=0, help="Seconds to wait before scraping")
    resume_parser.add_argument("--skip-git", action="store_true", help="Do not stage/commit git changes")
    resume_parser.add_argument("--skip-score-fetch", action="store_true", help="Skip Playwright scraping (debug only)")
    resume_parser.add_argument("--kaggle-message", help="Override log message for resume action")
    resume_parser.add_argument("--experiment-id", help="Experiment identifier to update (optional)")

    args = parser.parse_args()

    if args.command == "resume":
        _run_resume(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
