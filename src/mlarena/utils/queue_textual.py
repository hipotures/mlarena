"""
Textual-compatible Task Queue implementation.
Overrides execution to ensure clean logs (no Rich extras/colors).
"""

from __future__ import annotations

import os
import subprocess
import re
from datetime import datetime
from pathlib import Path

from rich.console import Console
from mlarena.utils.queue import TaskQueue

class TaskQueueTextual(TaskQueue):
    """TaskQueue specialized for TUI with plain-text logging."""

    def _execute_task(self, task: dict, console: Console, total_pending: int | None = None) -> bool:
        """Execute task with NO_COLOR=1 to avoid 'smieci' in logs."""
        task_id = task["id"]
        log_file = self.project_root / task["log_file"]
        log_file.parent.mkdir(parents=True, exist_ok=True)

        if total_pending is not None:
            console.print(f"\n[bold]Task #{task_id}/{total_pending}[/bold] (priority: {task['priority']})")
        else:
            console.print(f"\n[bold]Task #{task_id}[/bold] (priority: {task['priority']})")

        # Mark as running
        queue_data = self._load_queue()
        for t in queue_data["queue"]:
            if t["id"] == task_id:
                t["status"] = "running"
                t["started_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                break
        self._save_queue(queue_data)

        # Build command
        cmd_parts = task["command"].split()
        if "--project" not in cmd_parts and "-p" not in cmd_parts:
            project_name = self.project_root.name
            cmd_parts.insert(1, "--project")
            cmd_parts.insert(2, project_name)

        # Execute via subprocess with NO_COLOR=1
        repo_root = self.project_root.parent.parent.parent
        full_cmd = ["uv", "run", "python", "scripts/mla.py"] + cmd_parts

        # Disable Rich formatting for the subprocess
        env = os.environ.copy()
        env["NO_COLOR"] = "1"
        env["TERM"] = "dumb"
        env["PYTHONUNBUFFERED"] = "1"

        console.print(f"[dim]Command: {' '.join(full_cmd)}[/dim]")
        console.print(f"[dim]Log: {task['log_file']}[/dim]\n")

        try:
            with open(log_file, "w") as log:
                result = subprocess.run(
                    full_cmd,
                    cwd=repo_root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env
                )

            success = result.returncode == 0

            # Read log to extract experiment_id
            experiment_id = None
            if log_file.exists():
                log_content = log_file.read_text()
                match = re.search(r'exp-\d{8}-\d{6}', log_content)
                if match:
                    experiment_id = match.group(0)

            # Update queue state
            queue_data = self._load_queue()
            for t in queue_data["queue"]:
                if t["id"] == task_id:
                    t["status"] = "completed" if success else "failed"
                    t["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    attempt = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "success": success,
                        "error": None if success else f"Exit code: {result.returncode}",
                        "exit_code": result.returncode
                    }
                    t["attempts"].append(attempt)

                    if not success:
                        t["last_error"] = f"Exit code: {result.returncode}"

                    if experiment_id:
                        t["experiment_id"] = experiment_id

                    break

            self._save_queue(queue_data)

            if success:
                console.print(f"[green]✓ Task #{task_id} completed[/green]")
            else:
                console.print(f"[red]✗ Task #{task_id} failed (exit code: {result.returncode})[/red]")

            return success

        except Exception as e:
            console.print(f"[red]✗ Task #{task_id} error: {e}[/red]")
            self._mark_task_failed(task_id, str(e))
            return False
