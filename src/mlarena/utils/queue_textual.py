"""
Textual-compatible Task Queue implementation.
Overrides execution to ensure clean logs (no Rich extras/colors).
"""

from __future__ import annotations

import os
import subprocess
import re
import json
import requests
from datetime import datetime
from pathlib import Path

from rich.console import Console
from mlarena.utils.queue import TaskQueue

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Telegram notification setup
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
API_BASE = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}" if TELEGRAM_TOKEN else None

def send_telegram_notification(message: str) -> None:
    """Send message to Telegram."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram notification skipped: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set.")
        return
    try:
        url = f"{API_BASE}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        resp = requests.post(url, data=data, timeout=10)
        if resp.status_code != 200:
            print(f"Telegram Error: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"Telegram Exception: {e}")
        pass

from typing import Optional

class TaskQueueTextual(TaskQueue):
    """TaskQueue specialized for TUI with plain-text logging."""

    def __init__(self, project_root: Path):
        super().__init__(project_root)
        self.best_score: Optional[float] = None

    def run_queue(
        self,
        console: Console,
        max_tasks: Optional[int] = None,
        continue_on_error: bool = True
    ) -> dict[int, bool]:
        """Override to initialize best score cache before running."""
        # Initialize best score cache once at startup
        if self.best_score is None:
            console.print("[dim]Initializing best score cache...[/dim]")
            self.best_score = self._get_best_score()
            if self.best_score is not None:
                console.print(f"[dim]Current best score: {self.best_score:.6f}[/dim]")
            else:
                console.print("[dim]No previous scores found (first run?)[/dim]")

        return super().run_queue(console, max_tasks, continue_on_error)

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
        if not any(part.startswith("project=") for part in cmd_parts):
            project_name = self.project_root.name
            cmd_parts.insert(1, f"project={project_name}")

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
                
                # Check for NEW BEST SCORE and send Telegram
                if experiment_id:
                    self._check_and_notify_best(task, experiment_id, console)
            else:
                console.print(f"[red]✗ Task #{task_id} failed (exit code: {result.returncode})[/red]")

            return success

        except Exception as e:
            console.print(f"[red]✗ Task #{task_id} error: {e}[/red]")
            self._mark_task_failed(task_id, str(e))
            return False

    def _get_best_score(self) -> float | None:
        """Find the best CV score across all experiments in the project."""
        best_score = None
        exp_dir = self.project_root / "experiments"
        if not exp_dir.exists():
            return None
            
        for state_file in exp_dir.glob("*/state.json"):
            try:
                with open(state_file) as f:
                    data = json.load(f)
                
                for mod in data.get("modules", {}).values():
                    p = mod.get("payload", {})
                    score = p.get("local_cv_score") or p.get("local_cv")
                    if score is not None:
                        score = abs(float(score))
                        if best_score is None or score < best_score:
                            best_score = score
            except: continue
        return best_score

    def _check_and_notify_best(self, task: dict, exp_id: str, console: Console) -> None:
        """Check if current experiment has a new best score and notify."""
        try:
            # 1. Get current score
            state_path = self.project_root / "experiments" / exp_id / "state.json"
            if not state_path.exists(): return
            
            current_score = None
            with open(state_path) as f:
                data = json.load(f)
            
            for mod in data.get("modules", {}).values():
                p = mod.get("payload", {})
                score = p.get("local_cv_score") or p.get("local_cv")
                if score is not None:
                    current_score = abs(float(score))
                    # Break after finding first (furthest) score
                    break
            
            if current_score is None: return

            # 2. Compare with cached best score
            # Initialize if somehow None (though run_queue handles it)
            if self.best_score is None:
                 self.best_score = float('inf')

            prev_best = self.best_score
            
            if current_score < prev_best:
                # NEW BEST! Update cache immediately
                self.best_score = current_score

                task_id = task["id"]
                template = "N/A"
                cmd = task["command"]
                match = re.search(r'\bmodel_template=([^\s]+)', cmd) or re.search(r'\bpreprocess_template=([^\s]+)', cmd)
                if match: template = match.group(1)
                
                duration = "N/A"
                if task.get("started_at"):
                    try:
                        start = datetime.strptime(task["started_at"], "%Y-%m-%d %H:%M:%S")
                        dur_sec = (datetime.now() - start).total_seconds()
                        if dur_sec < 60: duration = f"{dur_sec:.1f}s"
                        elif dur_sec < 3600: duration = f"{dur_sec/60:.1f}m"
                        else: duration = f"{dur_sec/3600:.1f}h"
                    except: pass

                project = self.project_root.name
                msg = (
                    f"🚀 <b>New Best Score!</b>\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"<b>Project:</b> {project}\n"
                    f"<b>Score:</b> {current_score:.6f} (prev: {prev_best:.6f})\n"
                    f"<b>Task ID:</b> #{task_id}\n"
                    f"<b>Template:</b> <code>{template}</code>\n"
                    f"<b>Duration:</b> {duration}\n"
                    f"<b>Exp ID:</b> <code>{exp_id}</code>"
                )
                send_telegram_notification(msg)
                console.print(f"[bold gold1]🚀 NEW BEST SCORE: {current_score:.6f}![/bold gold1]")
        except Exception as e:
            console.print(f"[dim red]Notification error: {e}[/dim red]")
