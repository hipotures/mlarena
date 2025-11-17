"""
Unified submission workflow for Kaggle competitions.

Handles creating a Kaggle submission, waiting for processing,
fetching public/leaderboard scores via Kaggle CLI,
updating the submissions tracker, and creating a git commit
that ties code + local CV + public score + rank together.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import io
import subprocess
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

from rich.console import Console
from rich.panel import Panel

from submissions_tracker import SubmissionsTracker
from experiment_manager import ExperimentManager, ModuleStateError

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
                allow_restart=True,
            )

    def _default_message(self) -> str:
        parts = [self.artifact.model_name or "submission"]
        if self.artifact.local_cv_score is not None:
            parts.append(f"local {self.artifact.local_cv_score:.5f}")
        return " | ".join(parts)

    def execute(self) -> Optional[Dict[str, Any]]:
        """Run the submission workflow end-to-end."""
        score_data: Optional[Dict[str, Any]] = None
        try:
            skip_only = False
            if self.skip_submit:
                console.print("[yellow]Skipping Kaggle submission (flag enabled).[/yellow]")
                skip_only = not self.resume_mode
            else:
                if self.prompt and not self.auto_submit:
                    if not self._confirm():
                        console.print("[yellow]Submission aborted by user.[/yellow]")
                        return None
                self._submit_to_kaggle()
                if self.wait_seconds > 0:
                    console.print(f"[cyan]Waiting {self.wait_seconds}s for Kaggle processing...[/cyan]")
                    time.sleep(self.wait_seconds)

            if not self.skip_browser:
                score_data = self._fetch_scores()
                if score_data and score_data.get("public_score") is not None:
                    # Fetch leaderboard position
                    leaderboard_data = self._fetch_leaderboard_position()
                    if leaderboard_data:
                        score_data['leaderboard'] = leaderboard_data

                    self._update_tracker(score_data["public_score"], score_data=score_data)
                else:
                    console.print("[yellow]Could not fetch public score via Kaggle CLI.[/yellow]")

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

            self._finalize_experiment(score_data)
            if skip_only:
                return None
            return score_data
        except Exception as exc:
            self._fail_experiment(str(exc))
            raise

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

    def _fetch_scores(self) -> Optional[Dict[str, Any]]:
        """Fetch scores via Kaggle CLI."""
        console.print("[bold blue]Fetching scores via Kaggle CLI...[/bold blue]")
        try:
            result = subprocess.run(
                ["kaggle", "competitions", "submissions", "-c", self.artifact.competition, "--csv"],
                capture_output=True,
                text=True,
                check=True
            )

            # Parse CSV output
            for row in csv.DictReader(io.StringIO(result.stdout)):
                if row['fileName'] == self.artifact.filename:
                    public_score = float(row['publicScore']) if row['publicScore'] else None
                    private_score = float(row['privateScore']) if row['privateScore'] else None

                    if public_score is None:
                        console.print("[yellow]Public score not yet available[/yellow]")
                        return None

                    return {
                        'public_score': public_score,
                        'private_score': private_score,
                        'status': row['status'],
                        'date': row['date']
                    }

            console.print(f"[yellow]Submission {self.artifact.filename} not found in Kaggle submissions list[/yellow]")
            return None

        except subprocess.CalledProcessError as exc:
            console.print(f"[red]Kaggle CLI error: {exc.stderr}[/red]")
            return None
        except Exception as exc:
            console.print(f"[red]Error parsing scores: {exc}[/red]")
            return None

    def _fetch_leaderboard_position(self, username: str = "hipotures") -> Optional[Dict[str, Any]]:
        """Fetch leaderboard position via Kaggle CLI."""
        console.print(f"[bold blue]Fetching leaderboard position for {username}...[/bold blue]")
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Download leaderboard
                result = subprocess.run(
                    ["kaggle", "competitions", "leaderboard", "-c", self.artifact.competition, "--download"],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    check=True
                )

                # Find and extract zip file
                tmppath = Path(tmpdir)
                zip_files = list(tmppath.glob("*.zip"))
                if not zip_files:
                    console.print("[yellow]No leaderboard zip file downloaded[/yellow]")
                    return None

                zip_path = zip_files[0]
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(tmppath)

                # Find CSV file
                csv_files = list(tmppath.glob("*publicleaderboard*.csv"))
                if not csv_files:
                    console.print("[yellow]No leaderboard CSV found in zip[/yellow]")
                    return None

                csv_path = csv_files[0]

                # Parse leaderboard
                with open(csv_path, 'r', encoding='utf-8-sig') as f:
                    reader = csv.DictReader(f)
                    total_rows = 0
                    for row in reader:
                        total_rows += 1
                        # Check if username matches (case-insensitive)
                        if username.lower() in row.get('TeamMemberUserNames', '').lower():
                            rank = int(row['Rank'])
                            score = float(row['Score'])
                            percentile = (rank / total_rows) * 100 if total_rows > 0 else 0

                            console.print(f"[green]Found: Rank {rank}/{total_rows} (top {percentile:.1f}%)[/green]")
                            return {
                                'rank': rank,
                                'total': total_rows,
                                'percentile': percentile,
                                'score': score,
                                'team_name': row.get('TeamName', username)
                            }

                console.print(f"[yellow]Username '{username}' not found in leaderboard[/yellow]")
                return None

        except subprocess.CalledProcessError as exc:
            console.print(f"[red]Kaggle CLI error downloading leaderboard: {exc.stderr}[/red]")
            return None
        except Exception as exc:
            console.print(f"[red]Error processing leaderboard: {exc}[/red]")
            return None

    def _update_tracker(self, public_score: float, score_data: Optional[Dict[str, Any]] = None) -> Optional[int]:
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
            self.artifact.local_cv_score = local_cv
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
        return tracker_id

    def _finalize_experiment(self, score_data: Optional[Dict[str, Any]]):
        if not self._experiment_manager:
            return
        submission_rel = str(self.artifact.path.relative_to(self.artifact.project_root))
        payload = {
            "submission_file": submission_rel,
            "tracker_id": self.artifact.tracker_id(),
            "local_cv": self.artifact.local_cv_score,
        }
        if score_data:
            payload["public_score"] = score_data.get("public_score")
            if score_data.get("row_text"):
                payload["snapshot"] = score_data["row_text"]
            if 'leaderboard' in score_data:
                payload["leaderboard"] = score_data["leaderboard"]
        self._experiment_manager.complete_module("submit", payload)

    def _fail_experiment(self, reason: str):
        if self._experiment_manager:
            self._experiment_manager.fail_module("submit", reason)

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
        leaderboard_info = None
        if score_data:
            public_score = score_data.get("public_score")
            if 'leaderboard' in score_data:
                lb = score_data['leaderboard']
                rank = lb.get('rank')
                total = lb.get('total')
                if rank and total:
                    leaderboard_info = f"rank {rank}/{total}"
        if public_score is not None:
            parts.append(f"public {public_score:.5f}")
        else:
            parts.append("public pending")
        if leaderboard_info:
            parts.append(leaderboard_info)
        return ": ".join(parts[:1]) + " | " + " | ".join(parts[1:])


def _load_project_context(project_name: str):
    project_root = (REPO_ROOT / "projects" / "kaggle" / project_name).resolve()
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


def _run_pull_score(args):
    """Pull scores from Kaggle for an existing submission."""
    artifact = _build_artifact_from_filename(args.project, args.filename)
    runner = SubmissionRunner(
        artifact=artifact,
        kaggle_message=args.kaggle_message or f"pull-score {args.filename}",
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


def _run_pull_score_deprecated(args):
    """Deprecated wrapper for backward compatibility."""
    console.print("[yellow]Warning: 'resume' is deprecated. Use 'pull-score' instead.[/yellow]")
    _run_pull_score(args)


def _run_submit(args):
    artifact = _build_artifact_from_filename(args.project, args.filename)
    runner = SubmissionRunner(
        artifact=artifact,
        kaggle_message=args.kaggle_message or f"submit {args.filename}",
        wait_seconds=args.wait_seconds,
        cdp_url=args.cdp_url,
        auto_submit=True,
        prompt=False,
        skip_submit=False,
        skip_browser=args.skip_score_fetch,
        skip_git=args.skip_git,
        resume_mode=False,
        experiment_id=args.experiment_id,
    )
    runner.execute()


def main():
    parser = argparse.ArgumentParser(description="Submission workflow helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit", help="Submit an existing file and fetch the score")
    submit_parser.add_argument("--project", required=True, help="Competition directory (e.g., playground-series-s5e11)")
    submit_parser.add_argument("--filename", required=True, help="Submission filename (e.g., submission-YYYYMMDDHHMMSS.csv)")
    submit_parser.add_argument("--cdp-url", default="http://localhost:9222", help="Playwright CDP endpoint")
    submit_parser.add_argument("--wait-seconds", type=int, default=30, help="Seconds to wait before scraping")
    submit_parser.add_argument("--skip-git", action="store_true", help="Do not stage/commit git changes")
    submit_parser.add_argument("--skip-score-fetch", action="store_true", help="Skip Playwright scraping")
    submit_parser.add_argument("--kaggle-message", help="Override Kaggle message")
    submit_parser.add_argument("--experiment-id", help="Experiment identifier to update")
    submit_parser.set_defaults(func=_run_submit)

    # Main command: pull-score
    pull_score_parser = subparsers.add_parser("pull-score", help="Pull scores from Kaggle for existing submission")
    pull_score_parser.add_argument("--project", required=True, help="Competition directory (e.g., playground-series-s5e11)")
    pull_score_parser.add_argument("--filename", required=True, help="Submission filename (e.g., submission-YYYYMMDDHHMMSS.csv)")
    pull_score_parser.add_argument("--cdp-url", default="http://localhost:9222", help="[Deprecated] CDP endpoint (unused, kept for compatibility)")
    pull_score_parser.add_argument("--wait-seconds", type=int, default=0, help="Seconds to wait before fetching")
    pull_score_parser.add_argument("--skip-git", action="store_true", help="Do not stage/commit git changes")
    pull_score_parser.add_argument("--skip-score-fetch", action="store_true", help="Skip score fetching")
    pull_score_parser.add_argument("--kaggle-message", help="Override log message for commit")
    pull_score_parser.add_argument("--experiment-id", help="Experiment identifier to update (optional)")
    pull_score_parser.set_defaults(func=_run_pull_score)

    # Deprecated alias: resume (backward compatibility)
    resume_parser = subparsers.add_parser("resume", help="[DEPRECATED] Use 'pull-score' instead")
    resume_parser.add_argument("--project", required=True, help="Competition directory")
    resume_parser.add_argument("--filename", required=True, help="Submission filename")
    resume_parser.add_argument("--cdp-url", default="http://localhost:9222")
    resume_parser.add_argument("--wait-seconds", type=int, default=0)
    resume_parser.add_argument("--skip-git", action="store_true")
    resume_parser.add_argument("--skip-score-fetch", action="store_true")
    resume_parser.add_argument("--kaggle-message")
    resume_parser.add_argument("--experiment-id")
    resume_parser.set_defaults(func=_run_pull_score_deprecated)

    args = parser.parse_args()
    try:
        args.func(args)
    except ModuleStateError as exc:
        console.print(f"[Experiment] {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
