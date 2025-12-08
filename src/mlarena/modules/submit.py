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


def _build_kaggle_message(context, submission_file: Path, model_payload, feature_count: Optional[int]) -> str:
    """Replicates legacy descriptive message: exp | local | model | features | filename."""
    parts = []
    exp_id = context.experiment_id
    if exp_id:
        parts.append(exp_id)

    local_cv = None
    model_label = None
    if model_payload and getattr(model_payload, "payload", None):
        local_cv = model_payload.payload.get("local_cv")
        model_label = (
            model_payload.payload.get("model_implementation")
            or model_payload.payload.get("template")
        )

    if local_cv is not None:
        parts.append(f"local {float(local_cv):.5f}")
    if model_label:
        parts.append(str(model_label))
    if feature_count is not None:
        parts.append(f"features: {feature_count}")

    # Legacy message sometimes added smoke/stack; omitted here when unavailable.
    parts.append(submission_file.name)

    return " | ".join(parts) if parts else "MLArena submission"


@ModuleRegistry.register
class SubmitModule(BaseModule):
    name = "submit"
    description = "Submit predictions to Kaggle"
    dependencies = {"predict"}

    @classmethod
    def register_cli_args(cls, parser) -> None:
        parser.add_argument("--skip-submit", action="store_true", help="Skip Kaggle submission (placeholder).")
        parser.add_argument("--message", type=str, default="MLArena submission", help="Submission message.")
        parser.add_argument("--auto-submit", action="store_true", help="Skip confirmation prompt and submit immediately.")

    def execute(self) -> ModuleResult:
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
        feature_count = _compute_feature_count(config)

        user_message = self.invocation_params.get("message")
        model_payload = self.context.state.modules.get("model")
        if user_message and user_message != "MLArena submission":
            message = user_message
        else:
            message = _build_kaggle_message(self.context, submission_file, model_payload, feature_count)

        # Preview + 60s countdown with interactive confirmation
        console = Console()
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
