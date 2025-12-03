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
import json

import pandas as pd
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
    feature_count: Optional[int] = None

    def tracker_id(self) -> Optional[int]:
        if self.tracker_entry:
            return self.tracker_entry.get("id")
        return None


class SubmissionRunner:
    """End-to-end Kaggle submission pipeline."""

    @staticmethod
    def _compute_feature_count(artifact: SubmissionArtifact) -> Optional[int]:
        """Determine feature count from artifact metadata or project config."""
        count = getattr(artifact, "feature_count", None)
        if count is None and artifact.tracker_entry:
            count = artifact.tracker_entry.get("feature_count")

        train_path = None
        target_col = None
        ignored_cols: List[str] = []
        dataset_cfg = (artifact.config or {}).get("dataset") if isinstance(artifact.config, dict) else None
        if dataset_cfg:
            train_path = dataset_cfg.get("train_path")
            target_col = dataset_cfg.get("target")
            ignored_cols = list(dataset_cfg.get("ignored_columns") or [])
            id_col = dataset_cfg.get("id_column")
            if id_col and id_col not in ignored_cols:
                ignored_cols.append(id_col)

        if train_path is None or target_col is None:
            try:
                project_root, config = _load_project_context(artifact.project_root.name)
                train_path = train_path or getattr(config, "TRAIN_PATH", None)
                target_col = target_col or getattr(config, "TARGET_COLUMN", None)
                ignored_cols = ignored_cols or list(getattr(config, "IGNORED_COLUMNS", []))
                id_col = getattr(config, "ID_COLUMN", None)
                if id_col and id_col not in ignored_cols:
                    ignored_cols.append(id_col)
                # Prefer absolute paths
                if train_path and not Path(train_path).is_absolute():
                    train_path = (project_root / train_path).resolve()
            except Exception:
                pass

        if count is None and train_path and Path(train_path).exists():
            try:
                df = pd.read_csv(train_path, nrows=0)
                columns = list(df.columns)
                feature_cols = [
                    col for col in columns
                    if col != target_col and col not in set(ignored_cols or [])
                ]
                count = len(feature_cols)
            except Exception:
                count = None

        return count

    @classmethod
    def build_message(
        cls,
        artifact: SubmissionArtifact,
        *,
        experiment_id: Optional[str] = None,
        local_score: Optional[float] = None,
        model_label: Optional[str] = None,
        feature_count: Optional[int] = None,
        smoke: bool = False,
        stack_suffix: Optional[str] = None,
    ) -> str:
        fc = feature_count if feature_count is not None else cls._compute_feature_count(artifact)
        exp_id = experiment_id or (artifact.tracker_entry or {}).get("experiment_id")
        local = local_score if local_score is not None else artifact.local_cv_score
        model_token = model_label or artifact.model_name
        filename_token = artifact.filename
        if model_token and filename_token and str(model_token).strip() == str(filename_token).strip():
            model_token = None  # avoid duplicate tokens

        parts: List[str] = []
        seen: set[str] = set()

        def _add(token: Optional[str]):
            if token is None:
                return
            tok = str(token).strip()
            if not tok or tok in seen:
                return
            parts.append(tok)
            seen.add(tok)

        _add(exp_id)
        _add(f"local {local:.5f}" if local is not None else None)
        _add(model_token)
        _add(filename_token)
        _add(f"features: {fc}" if fc is not None else None)
        _add("smoke" if smoke else None)
        _add(stack_suffix)

        return " | ".join(parts)

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
        self.experiment_id = experiment_id
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
        self._feature_count_cache: Optional[int] = None
        self._experiment_manager = None
        self._submit_already_completed = False
        if experiment_id:
            project_name = artifact.project_root.name
            self._experiment_manager = ExperimentManager.load_or_create(project_name, experiment_id)
            # Check if submit module is already completed
            submit_entry = self._experiment_manager.modules().get("submit", {})
            if submit_entry.get("status") == "completed":
                console.print("[yellow]Module 'submit' already completed, updating scores only...[/yellow]")
                self._submit_already_completed = True

        # Build message after ExperimentManager init so stack info can be picked up
        self.kaggle_message = kaggle_message or self._default_message()

        if experiment_id and not self._submit_already_completed:
            self._experiment_manager.start_module(
                "submit",
                {
                    "kaggle_message": self.kaggle_message,
                    "resume_mode": resume_mode,
                },
                allow_restart=True,
            )

    def _print_submission_message(self):
        """Display the exact Kaggle message and submission details before prompting."""
        fc = self._resolve_feature_count()
        lines = [
            f"[bold]Kaggle message:[/bold] {self.kaggle_message}",
            f"File: {self.artifact.filename}",
            f"Competition: {self.artifact.competition}",
        ]
        if fc is not None:
            lines.append(f"Features: {fc}")
        if self.experiment_id:
            lines.append(f"Experiment: {self.experiment_id}")
        console.print(Panel.fit("\n".join(lines), title="Submission Preview", border_style="cyan"))

    def _default_message(self) -> str:
        exp_id = self.experiment_id or (self.artifact.tracker_entry or {}).get("experiment_id")
        stack_suffix = None
        if exp_id:
            try:
                mgr = ExperimentManager.load_existing(self.artifact.project_root.name, exp_id)
                stack_mod = mgr.get_module("stack")
                if stack_mod:
                    strategy = stack_mod.get("strategy") or "stack"
                    method = stack_mod.get("blend_method") or stack_mod.get("meta_model")
                    n_models = stack_mod.get("n_models")
                    stack_suffix = f"{strategy}{':' + method if method else ''}"
                    if n_models:
                        stack_suffix += f" models {n_models}"
            except Exception:
                pass

        return self.build_message(
            self.artifact,
            experiment_id=exp_id,
            local_score=self.artifact.local_cv_score,
            model_label=self.artifact.model_name,
            feature_count=self._resolve_feature_count(),
            stack_suffix=stack_suffix,
        )

    def _resolve_feature_count(self) -> Optional[int]:
        """Determine feature count from artifact metadata or project config."""
        if self._feature_count_cache is not None:
            return self._feature_count_cache
        self._feature_count_cache = self._compute_feature_count(self.artifact)
        return self._feature_count_cache

    def execute(self) -> Optional[Dict[str, Any]]:
        """Run the submission workflow end-to-end."""
        score_data: Optional[Dict[str, Any]] = None
        try:
            skip_only = False
            if self.skip_submit:
                console.print("[yellow]Skipping Kaggle submission (flag enabled).[/yellow]")
                skip_only = not self.resume_mode
            else:
                self._print_submission_message()
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
            str(self.artifact.path),  # absolute path to avoid cwd/path issues
            "-m",
            self.kaggle_message,
        ]
        result = subprocess.run(
            cmd,
            cwd=self.artifact.path.parent,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            # Surface Kaggle CLI error details instead of silent 400
            raise RuntimeError(
                f"Kaggle submit failed (exit {result.returncode}):\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
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
                            submission_count = int(row.get('SubmissionCount', 0))
                            percentile = (rank / total_rows) * 100 if total_rows > 0 else 0

                            console.print(f"[green]Found: Rank {rank}/{total_rows} (top {percentile:.1f}%) - {submission_count} submissions[/green]")
                            return {
                                'rank': rank,
                                'total': total_rows,
                                'percentile': percentile,
                                'score': score,
                                'submission_count': submission_count,
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
        feature_count = getattr(self.artifact, "feature_count", None) or self._resolve_feature_count()
        if tracker_id is None:
            console.print("[yellow]Tracker entry missing; creating new submission entry.[/yellow]")
            entry = tracker.add_submission(
                filename=self.artifact.filename,
                model_name=self.artifact.model_name or self.artifact.filename,
                local_cv_score=local_cv,
                notes=self.artifact.notes or "Auto-added via resume",
                config=self.artifact.config,
                public_score=public_score,
                feature_count=feature_count,
            )
            self.artifact.tracker_entry = entry
            tracker_id = entry["id"]
            self.artifact.local_cv_score = local_cv
            self.artifact.feature_count = feature_count
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
            if feature_count is not None and (self.artifact.tracker_entry is None or not self.artifact.tracker_entry.get("feature_count")):
                for sub in tracker.submissions:
                    if sub["id"] == tracker_id:
                        sub["feature_count"] = feature_count
                        tracker._save_submissions()
                        self.artifact.feature_count = feature_count
                        if self.artifact.tracker_entry:
                            self.artifact.tracker_entry["feature_count"] = feature_count
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
        feature_count = getattr(self.artifact, "feature_count", None) or self._resolve_feature_count()
        if feature_count is not None:
            payload["feature_count"] = feature_count
        if score_data:
            payload["public_score"] = score_data.get("public_score")
            if score_data.get("row_text"):
                payload["snapshot"] = score_data["row_text"]
            if 'leaderboard' in score_data:
                payload["leaderboard"] = score_data["leaderboard"]

        # If module already completed, just update data; otherwise mark as completed
        if self._submit_already_completed:
            self._experiment_manager.update_module("submit", payload)
        else:
            self._experiment_manager.complete_module("submit", payload)

    def _fail_experiment(self, reason: str):
        if self._experiment_manager and not self._submit_already_completed:
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
                sub_count = lb.get('submission_count')
                if rank and total:
                    if sub_count:
                        leaderboard_info = f"rank {rank}/{total} ({sub_count} subs)"
                    else:
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


def _infer_experiment_from_submission(project_root: Path, filename: str) -> Dict[str, Any]:
    """Attempt to infer experiment metadata from state files referencing the submission."""
    experiments_dir = project_root / "experiments"
    if not experiments_dir.exists():
        return {}

    for state_path in experiments_dir.glob("exp-*/state.json"):
        try:
            payload = json.loads(state_path.read_text())
        except Exception:
            continue
        predict_mod = (payload.get("modules") or {}).get("predict") or {}
        submission_file = predict_mod.get("submission_file")
        if not submission_file:
            continue
        sub_name = Path(submission_file).name
        if sub_name != filename:
            continue
        return {
            "experiment_id": payload.get("experiment_id"),
            "local_cv": predict_mod.get("local_cv"),
            "model_label": predict_mod.get("template") or predict_mod.get("model"),
            "feature_count": predict_mod.get("feature_count"),
        }
    return {}


def _build_artifact_from_filename(project_name: str, filename: str) -> SubmissionArtifact:
    project_root, config = _load_project_context(project_name)
    submission_path = project_root / "submissions" / filename
    if not submission_path.exists():
        raise FileNotFoundError(f"Submission file '{filename}' not found under {submission_path.parent}")

    tracker = SubmissionsTracker(project_root)
    tracker_entry = next((s for s in tracker.submissions if s["filename"] == filename), None)
    inferred = _infer_experiment_from_submission(project_root, filename)

    tracker_entry = tracker_entry or {}
    if inferred.get("experiment_id") and not tracker_entry.get("experiment_id"):
        tracker_entry["experiment_id"] = inferred["experiment_id"]
    model_name = tracker_entry.get("model_name") or inferred.get("model_label") or filename
    local_cv = tracker_entry.get("local_cv_score", inferred.get("local_cv"))
    notes = tracker_entry.get("notes", "")
    config_dict = tracker_entry.get("config")
    feature_count = tracker_entry.get("feature_count", inferred.get("feature_count"))

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
        feature_count=feature_count,
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
        kaggle_message=args.kaggle_message,
        wait_seconds=args.wait_seconds,
        cdp_url=args.cdp_url,
        auto_submit=False,
        prompt=True,
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
