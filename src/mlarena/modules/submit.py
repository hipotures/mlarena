"""Submission upload module."""

from __future__ import annotations

import select
import subprocess
import sys
import termios
import time
import tty
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from mlarena.core.module import BaseModule, ModuleResult
from mlarena.core.registry import ModuleRegistry
from mlarena.utils.project import data_paths, load_project_config


def _compute_feature_count(config) -> Optional[int]:
    """Best-effort feature count for submission description."""
    try:
        import pandas as pd  # local import to avoid hard dependency elsewhere

        train_path, _ = data_paths(config)
        path = Path(train_path)
        if not path.exists():
            return None

        df = pd.read_csv(path, nrows=0)
        cols = list(df.columns)

        target = getattr(config, "TARGET_COLUMN", None)
        ignored = list(getattr(config, "IGNORED_COLUMNS", []))
        id_col = getattr(config, "ID_COLUMN", None)
        if id_col:
            ignored.append(id_col)

        feature_cols = [c for c in cols if c != target and c not in set(ignored)]
        return len(feature_cols)
    except Exception:
        return None


def _validate_submission(submission_file: Path, config) -> tuple[bool, Optional[str]]:
    """
    Validate submission format before upload.
    Returns (is_valid, error_message)
    """
    try:
        import pandas as pd

        # Load submission
        sub_df = pd.read_csv(submission_file)

        # Load sample submission for format reference
        sample_sub_path = getattr(config, "SAMPLE_SUBMISSION_PATH", None)
        if not sample_sub_path or not Path(sample_sub_path).exists():
            # Try common names
            data_dir = Path(getattr(config, "DATA_DIR", submission_file.parent.parent.parent / "data"))
            for name in ["sample_submission.csv", "gender_submission.csv", "sample_solution.csv"]:
                candidate = data_dir / name
                if candidate.exists():
                    sample_sub_path = candidate
                    break

        if not sample_sub_path or not Path(sample_sub_path).exists():
            return True, None  # Can't validate without sample

        sample_df = pd.read_csv(sample_sub_path)

        # Check column names match
        if list(sub_df.columns) != list(sample_df.columns):
            return False, (
                f"Column mismatch!\n"
                f"  Expected: {list(sample_df.columns)}\n"
                f"  Got:      {list(sub_df.columns)}"
            )

        # Check row count matches
        if len(sub_df) != len(sample_df):
            return False, (
                f"Row count mismatch!\n"
                f"  Expected: {len(sample_df)} rows\n"
                f"  Got:      {len(sub_df)} rows"
            )

        # Check for missing values
        if sub_df.isnull().any().any():
            null_cols = sub_df.columns[sub_df.isnull().any()].tolist()
            return False, f"Missing values detected in columns: {null_cols}"

        # Check ID column matches (if present)
        id_col = sub_df.columns[0]  # First column is usually ID
        if not sub_df[id_col].equals(sample_df[id_col]):
            # Show first mismatch
            diff_idx = (sub_df[id_col] != sample_df[id_col]).idxmax()
            return False, (
                f"ID mismatch detected!\n"
                f"  Row {diff_idx}: expected {sample_df[id_col].iloc[diff_idx]}, "
                f"got {sub_df[id_col].iloc[diff_idx]}\n"
                f"  First 3 IDs - Expected: {sample_df[id_col].head(3).tolist()}\n"
                f"  First 3 IDs - Got:      {sub_df[id_col].head(3).tolist()}"
            )

        return True, None

    except Exception as e:
        return False, f"Validation error: {str(e)}"


def _build_kaggle_message(context, submission_file: Path, model_payload, feature_count: Optional[int]) -> str:
    """Build submission message: local_cv | features | exp | model_template | preprocess_template | filename."""
    parts = []
    exp_id = context.experiment_id

    local_cv = None
    model_template = None
    preprocess_template = None

    if model_payload:
        # Get local_cv from payload
        if getattr(model_payload, "payload", None):
            local_cv = model_payload.payload.get("local_cv")

        # Get templates from invocation (actual parameters used)
        if getattr(model_payload, "invocation", None):
            model_template = model_payload.invocation.get("model_template")
            preprocess_exp_dir = model_payload.invocation.get("preprocess_exp_dir")

            # Extract chain name from preprocess_exp_dir
            # e.g., "...experiments/pre-top0_te_scale_ext_m/5-top0_av_weights" -> "top0_te_scale_ext_m"
            if preprocess_exp_dir:
                from pathlib import Path
                exp_dir_path = Path(preprocess_exp_dir)
                # Go up to experiments/pre-{name} level
                while exp_dir_path.name and not exp_dir_path.name.startswith("pre-"):
                    exp_dir_path = exp_dir_path.parent
                if exp_dir_path.name.startswith("pre-"):
                    preprocess_template = exp_dir_path.name[4:]  # Remove "pre-" prefix

    if local_cv is not None:
        parts.append(f"{float(local_cv):.5f}")
    if feature_count is not None:
        parts.append(f"feat: {feature_count}")
    if exp_id:
        parts.append(exp_id)
    if model_template:
        parts.append(str(model_template))
    if preprocess_template:
        parts.append(str(preprocess_template))

    parts.append(submission_file.name)

    return " | ".join(parts) if parts else "MLArena submission"


@ModuleRegistry.register
class SubmitModule(BaseModule):
    """Upload predictions to Kaggle with optional validation and prompts."""

    name = "submit"
    description = "Submit predictions to Kaggle"
    dependencies = {"predict"}

    def execute(self) -> ModuleResult:
        """
        Validate and optionally upload a submission to Kaggle.

        Returns:
            ModuleResult indicating whether submission was performed and any metadata.

        Raises:
            RuntimeError: When submission upload fails unexpectedly.
        """
        artifact_dir: Path = self.context.artifact_dir
        artifact_dir.mkdir(parents=True, exist_ok=True)
        skip = bool(self.invocation_params.get("skip_submit", False))

        predict_payload = self.context.state.modules.get("predict")
        if not predict_payload or not getattr(predict_payload, "payload", None):
            marker = artifact_dir / "submit_failed.txt"
            marker.write_text("Predict step missing payload.")
            return ModuleResult(success=False, error="predict not run", artifacts=[marker])

        submission_file = Path(predict_payload.payload["submission_file"])  # type: ignore

        if skip:
            marker = artifact_dir / "submit_skipped.txt"
            marker.write_text("Skipped Kaggle submission.")
            return ModuleResult(success=True, payload={"submitted": False}, artifacts=[marker])

        config = self.context.config_module or load_project_config(self.context.project_root)
        competition = getattr(config, "COMPETITION_NAME", self.context.project_name)

        # Get feature_count from predict payload (real processed data), fallback to original train.csv
        feature_count = predict_payload.payload.get("feature_count") if predict_payload.payload else None  # type: ignore
        if feature_count is None:
            feature_count = _compute_feature_count(config)

        user_message = self.invocation_params.get("message")
        model_payload = self.context.state.modules.get("model")
        if user_message and user_message != "MLArena submission":
            message = user_message
        else:
            message = _build_kaggle_message(self.context, submission_file, model_payload, feature_count)

        # Validate submission format BEFORE upload
        console = Console()
        valid, error_msg = _validate_submission(submission_file, config)
        if not valid:
            console.print(f"\n[bold red]✗ Submission validation failed![/bold red]")
            console.print(f"[red]{error_msg}[/red]\n")
            marker = artifact_dir / "submit_failed.txt"
            marker.write_text(f"Validation failed: {error_msg}")
            return ModuleResult(
                success=False,
                error=f"Submission validation failed: {error_msg}",
                artifacts=[marker]
            )

        console.print(f"[green]✓[/green] Submission validation passed")

        # Preview + 60s countdown with interactive confirmation
        skip_submit = bool(self.invocation_params.get("skip_submit", False))

        console.print(f"\n[bold]Kaggle message:[/bold] {message}")

        if skip_submit:
            console.print("\n[yellow]⊘ Skipping submission (--skip-submit)[/yellow]")
            marker = artifact_dir / "submit_skipped.txt"
            marker.write_text("Submission skipped by user flag.")
            return ModuleResult(success=True, payload={"submitted": False, "skipped": True}, artifacts=[marker])

        # 60-second countdown with y/n input (no Enter needed)
        console.print("\n[bold cyan]Submit to Kaggle?[/bold cyan]")
        console.print("[dim]Press 'y' to submit now, 'n' to cancel, or wait 60s for auto-submit[/dim]\n")

        # Audio beep to alert user
        for _ in range(3):
            print("\a", end="", flush=True)
            time.sleep(0.1)

        # Try system beep as fallback
        try:
            subprocess.run(["paplay", "/usr/share/sounds/freedesktop/stereo/bell.oga"],
                         stderr=subprocess.DEVNULL, timeout=0.5)
        except:
            pass

        countdown_seconds = 60
        start_time = time.time()
        submitted = False
        cancelled = False

        # Save terminal settings
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            # Set terminal to raw mode for single-key input
            tty.setraw(fd)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=False,
            ) as progress:
                task = progress.add_task(
                    f"[cyan]Auto-submitting in {countdown_seconds}s...", total=countdown_seconds
                )

                while True:
                    elapsed = time.time() - start_time
                    remaining = max(0, countdown_seconds - int(elapsed))

                    # Check for key press (non-blocking)
                    if select.select([sys.stdin], [], [], 0)[0]:
                        ch = sys.stdin.read(1).lower()
                        if ch == 'y':
                            progress.update(task, description="[green]✓ Confirmed - submitting!")
                            submitted = True
                            break
                        elif ch == 'n':
                            progress.update(task, description="[red]✗ Cancelled by user")
                            cancelled = True
                            break

                    if remaining == 0:
                        progress.update(task, description="[green]⏱ Timeout - submitting!")
                        submitted = True
                        break

                    # Update countdown display
                    progress.update(
                        task,
                        description=f"[cyan]Auto-submitting in {remaining}s... [dim](y/n)[/dim]",
                        completed=int(elapsed)
                    )

                    time.sleep(0.1)

        finally:
            # Restore terminal settings
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        if cancelled:
            console.print("\n[yellow]⊘ Submission cancelled[/yellow]")
            marker = artifact_dir / "submit_aborted.txt"
            marker.write_text("Submission aborted by user.")
            return ModuleResult(success=False, error="aborted", artifacts=[marker])

        console.print()  # Empty line before upload progress

        try:
            subprocess.check_call(
                [
                    "kaggle",
                    "competitions",
                    "submit",
                    "-c",
                    competition,
                    "-f",
                    str(submission_file),
                    "-m",
                    message,
                ]
            )
            marker = artifact_dir / "submit_success.txt"
            marker.write_text(f"Submitted {submission_file} to {competition}")

            console.print(f"\n[bold green]✓[/bold green] Submitted to Kaggle: [cyan]{competition}[/cyan]")

            return ModuleResult(
                success=True,
                payload={"submitted": True, "competition": competition, "submission_file": str(submission_file)},
                artifacts=[marker],
            )
        except Exception as exc:
            marker = artifact_dir / "submit_failed.txt"
            marker.write_text(f"Submission failed: {exc}")
            return ModuleResult(success=False, error=str(exc), artifacts=[marker])
