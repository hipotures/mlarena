"""Submission queue management CLI."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from filelock import FileLock
from rich.console import Console
from rich.table import Table

console = Console()


class SubmissionQueue:
    """Manages submission queue operations."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.queue_file = self.project_root / "submissions" / "queue.json"
        self.lock_file = self.queue_file.with_suffix(".lock")

    def _load_queue(self) -> dict:
        """Load queue.json with lock."""
        if not self.queue_file.exists():
            return {"queue": []}

        with FileLock(str(self.lock_file), timeout=10):
            return json.loads(self.queue_file.read_text())

    def _save_queue(self, queue_data: dict) -> None:
        """Save queue.json with lock."""
        self.queue_file.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.lock_file), timeout=10):
            self.queue_file.write_text(json.dumps(queue_data, indent=2))

    def get_entry(self, identifier: Union[int, str]) -> Optional[dict]:
        """
        Lookup entry by:
        - int: queue number (ID)
        - str starting with 'exp-': experiment_id
        - str ending with '.csv': filename
        """
        queue_data = self._load_queue()

        if isinstance(identifier, int):
            # Queue number
            for entry in queue_data["queue"]:
                if entry["id"] == identifier:
                    return entry
        else:
            identifier = str(identifier)
            if identifier.startswith("exp-"):
                # Experiment ID
                for entry in queue_data["queue"]:
                    if entry["experiment_id"] == identifier:
                        return entry
            elif identifier.endswith((".csv", ".csv.gz")):
                # Filename (supports both .csv and .csv.gz)
                for entry in queue_data["queue"]:
                    if Path(entry["submission_file"]).name == identifier:
                        return entry
        return None

    def list_queue(self, show_submission_name: bool = False) -> None:
        """Display queue as Rich table."""
        queue_data = self._load_queue()
        entries = queue_data.get("queue", [])

        if not entries:
            console.print("[yellow]No queued submissions[/yellow]")
            return

        table = Table(
            title="Submission Queue", 
            show_header=True, 
            expand=True,
            row_styles=["", "reverse"]
        )
        table.add_column("#", style="cyan", width=4)
        table.add_column("Experiment ID", style="dim", width=20)
        table.add_column("Status", width=12)
        table.add_column("Submission", justify="center", width=12)
        table.add_column("Message Preview", width=50)
        table.add_column("Error", style="red", width=20)

        for entry in entries:
            # Format status with color
            status = entry.get("status", "pending")
            if status == "pending":
                status_str = "[yellow]pending[/yellow]"
            elif status == "submitted":
                status_str = "[green]submitted[/green]"
            elif status == "completed":
                status_str = "[bold green]completed[/bold green]"
            else:
                status_str = "[red]failed[/red]"

            # Submission display
            sub_file = entry.get("submission_file")
            submission_display = "-"
            if sub_file:
                if show_submission_name:
                    submission_display = Path(sub_file).name
                else:
                    submission_display = "✓"

            # Truncate message and error
            message = entry.get("kaggle_message", "")[:50]
            error = (entry.get("last_error") or "-")[:20]
            exp_id = entry.get("experiment_id", "")

            table.add_row(
                str(entry["id"]),
                exp_id,
                status_str,
                submission_display,
                message,
                error
            )

        console.print(table)
        console.print(f"\n[dim]Total: {len(entries)} entries[/dim]")

    def remove_entry(self, identifier: Union[int, str]) -> bool:
        """Remove entry from queue."""
        entry = self.get_entry(identifier)
        if not entry:
            console.print(f"[red]✗ Entry not found: {identifier}[/red]")
            return False

        queue_data = self._load_queue()
        queue_data["queue"] = [
            e for e in queue_data["queue"]
            if e["id"] != entry["id"]
        ]
        self._save_queue(queue_data)

        console.print(f"[green]✓ Removed entry #{entry['id']}[/green]")
        return True

    def _fetch_kaggle_submissions(self, competition: str) -> list[dict]:
        """Fetch existing submissions via Kaggle API."""
        # Import from mlarena utils
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from mlarena.utils.kaggle_api import fetch_submissions
        return fetch_submissions(competition)

    def _is_already_submitted(
        self,
        experiment_id: str,
        filename: str,
        kaggle_subs: list[dict]
    ) -> bool:
        """Check if submission exists on Kaggle."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from mlarena.utils.kaggle_api import (
            find_submission_by_message,
            find_submission_by_filename
        )

        # Check by experiment ID in message
        if find_submission_by_message(kaggle_subs, experiment_id):
            return True

        # Check by filename
        if find_submission_by_filename(kaggle_subs, filename):
            return True

        return False

    def submit_entry(
        self,
        identifier: Union[int, str],
        continue_flow: bool = False
    ) -> bool:
        """Submit entry from queue with duplicate detection."""
        # 1. Lookup entry
        entry = self.get_entry(identifier)
        if not entry:
            console.print(f"[red]✗ Entry not found: {identifier}[/red]")
            return False

        console.print(f"\n[bold]Submitting entry #{entry['id']}[/bold]")
        console.print(f"[dim]Experiment: {entry['experiment_id']}[/dim]")
        console.print(f"[dim]Message: {entry['kaggle_message']}[/dim]\n")

        # 2. Duplicate detection
        console.print("[dim]Checking for duplicates on Kaggle...[/dim]")
        kaggle_subs = self._fetch_kaggle_submissions(entry["competition"])

        if self._is_already_submitted(
            entry["experiment_id"],
            Path(entry["submission_file"]).name,
            kaggle_subs
        ):
            console.print(f"[yellow]⚠ Submission already exists on Kaggle (exp: {entry['experiment_id']})[/yellow]")
            console.print("[yellow]Skipping upload. Remove from queue manually if needed.[/yellow]")

            # Update status to submitted (user can verify manually)
            queue_data = self._load_queue()
            for e in queue_data["queue"]:
                if e["id"] == entry["id"]:
                    e["status"] = "submitted"
                    break
            self._save_queue(queue_data)

            return False

        # 3. Submit via mla.py
        console.print("[dim]Submitting to Kaggle...[/dim]")

        repo_root = self.project_root.parent.parent.parent
        result = subprocess.run(
            [
                "uv", "run", "python", "scripts/mla.py",
                "submit",
                "--project", entry["project"],
                "--exp-id", entry["experiment_id"],
                "submit.confirm_timeout=0",
                "--force"
            ],
            cwd=repo_root,
            capture_output=False
        )

        # 4. Log attempt
        queue_data = self._load_queue()
        for e in queue_data["queue"]:
            if e["id"] == entry["id"]:
                attempt = {
                    "timestamp": datetime.now().strftime("%Y%m%d %H%M%S"),
                    "success": result.returncode == 0,
                    "error": None if result.returncode == 0 else "Submit command failed"
                }
                e["submission_attempts"].append(attempt)

                if result.returncode != 0:
                    e["status"] = "failed"
                    e["last_error"] = "Submit command failed"
                    console.print(f"\n[red]✗ Submission failed[/red]")
                else:
                    e["status"] = "submitted"
                    e["last_error"] = None
                    console.print(f"\n[green]✓ Submitted to Kaggle[/green]")

                break

        self._save_queue(queue_data)

        if result.returncode != 0:
            return False

        # 5. Continue flow (if requested)
        if not continue_flow:
            console.print("[dim]Use --continue-flow to automatically fetch score[/dim]")
            return True

        # 6. Wait + fetch-score
        console.print(f"\n[dim]Waiting 30s for Kaggle processing...[/dim]")
        time.sleep(30)

        console.print("[dim]Fetching score...[/dim]")
        fetch_result = subprocess.run(
            [
                "uv", "run", "python", "scripts/mla.py",
                "fetch-score",
                "--project", entry["project"],
                "--exp-id", entry["experiment_id"]
            ],
            cwd=repo_root,
            capture_output=False
        )

        queue_data = self._load_queue()

        if fetch_result.returncode == 0:
            # Success: remove from queue
            console.print(f"\n[bold green]✓ Score fetched! Removing from queue[/bold green]")
            queue_data["queue"] = [
                e for e in queue_data["queue"]
                if e["id"] != entry["id"]
            ]
            self._save_queue(queue_data)
            return True
        else:
            # Failed to fetch: keep in queue
            console.print(f"\n[yellow]⚠ Failed to fetch score, keeping in queue[/yellow]")
            for e in queue_data["queue"]:
                if e["id"] == entry["id"]:
                    e["last_error"] = "Failed to fetch score"
                    break
            self._save_queue(queue_data)
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Submission queue management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List queue
  python scripts/submission_queue.py --project Titanic list

  # Submit by queue number
  python scripts/submission_queue.py --project Titanic submit 1

  # Submit by experiment-id
  python scripts/submission_queue.py --project Titanic submit exp-20251225-155517

  # Submit by filename
  python scripts/submission_queue.py --project Titanic submit submission.csv

  # Submit with auto fetch-score
  python scripts/submission_queue.py --project Titanic submit 1 --continue-flow

  # Remove from queue
  python scripts/submission_queue.py --project Titanic remove 1
        """
    )

    parser.add_argument(
        "--project", "-p",
        required=True,
        help="Project name (e.g., Titanic)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # List command
    list_parser = subparsers.add_parser("list", help="List queued submissions")
    list_parser.add_argument(
        "--show-submission-name",
        action="store_true",
        help="Show full submission filename instead of checkmark"
    )

    # Submit command
    submit_parser = subparsers.add_parser("submit", help="Submit from queue")
    submit_parser.add_argument(
        "identifier",
        help="Queue #, experiment-id, or filename"
    )
    submit_parser.add_argument(
        "--continue-flow",
        action="store_true",
        help="Wait 30s and fetch score after submit"
    )

    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove from queue")
    remove_parser.add_argument(
        "identifier",
        help="Queue #, experiment-id, or filename"
    )

    args = parser.parse_args()

    # Determine project root
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent
    project_root = repo_root / "projects" / "kaggle" / args.project

    if not project_root.exists():
        console.print(f"[red]Error: Project '{args.project}' not found[/red]")
        return 1

    queue = SubmissionQueue(project_root)

    if args.command == "list":
        queue.list_queue(show_submission_name=args.show_submission_name)
        return 0

    elif args.command == "submit":
        # Try parsing as int (queue number)
        try:
            identifier = int(args.identifier)
        except ValueError:
            identifier = args.identifier

        success = queue.submit_entry(identifier, args.continue_flow)
        return 0 if success else 1

    elif args.command == "remove":
        # Try parsing as int (queue number)
        try:
            identifier = int(args.identifier)
        except ValueError:
            identifier = args.identifier

        success = queue.remove_entry(identifier)
        return 0 if success else 1

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
