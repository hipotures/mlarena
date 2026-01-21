"""
Textual-compatible Task Queue implementation.
Overrides execution to ensure clean logs (no Rich extras/colors).
"""

from __future__ import annotations

import os
import subprocess
import re
from datetime import datetime

from rich.console import Console
from mlarena.utils.queue import TaskQueue

from concurrent.futures import ThreadPoolExecutor

class TaskQueueParallel(TaskQueue):
    """TaskQueue specialized for TUI with parallel execution support."""

    def run_queue(
        self,
        console: Console,
        max_tasks: Optional[int] = None,
        continue_on_error: bool = True,
        max_jobs: int = 4
    ) -> dict[int, bool]:
        """Execute pending tasks in parallel."""
        results = {}
        executed = 0
        running_futures = {}

        console.print(f"[bold cyan]Starting parallel queue runner (jobs: {max_jobs})...[/bold cyan]\n")

        with ThreadPoolExecutor(max_workers=max_jobs) as executor:
            while True:
                # Check stop signal
                if self.stop_file.exists():
                    console.print("\n[yellow]Stop signal detected (.stop file)[/yellow]")
                    console.print("[yellow]Graceful shutdown...[/yellow]")
                    self.stop_file.unlink()
                    break

                # Check max_tasks limit
                if max_tasks and executed >= max_tasks:
                    if not running_futures:
                        break
                
                # Check if we can submit more tasks
                if (not max_tasks or executed < max_tasks) and len(running_futures) < max_jobs:
                    # Get next pending task safely
                    task = self._claim_next_task()
                    if task:
                        future = executor.submit(self._execute_task, task, console)
                        running_futures[future] = task["id"]
                        executed += 1
                        continue # Try to submit more if capacity allows

                # If no tasks to submit, wait for at least one to finish
                if running_futures:
                    # We use a short timeout to allow checking stop_file periodically
                    done_futures = []
                    try:
                        # Wait for any to complete with timeout
                        from concurrent.futures import wait, FIRST_COMPLETED
                        done, _ = wait(running_futures.keys(), timeout=1.0, return_when=FIRST_COMPLETED)
                        done_futures = done
                    except Exception:
                        pass

                    for future in done_futures:
                        task_id = running_futures.pop(future)
                        try:
                            success = future.result()
                            results[task_id] = success
                            if not success and not continue_on_error:
                                console.print(f"\n[red]Task #{task_id} failed, stopping queue submission[/red]")
                                # Cancel all pending if possible (though we don't have many pending in executor usually)
                                max_tasks = executed # Prevent more submissions
                        except Exception as e:
                            console.print(f"[red]Task #{task_id} raised exception: {e}[/red]")
                            results[task_id] = False
                else:
                    # No tasks running and no more to submit
                    break

        console.print("\n[bold]Parallel queue runner finished[/bold]")
        console.print(
            f"[dim]Executed: {executed}, "
            f"Success: {sum(results.values())}, "
            f"Failed: {executed - sum(results.values())}[/dim]"
        )
        return results

    def _claim_next_task(self) -> Optional[dict]:
        """Atomically find and claim (mark as running) the next pending task."""
        with FileLock(str(self.lock_file), timeout=10):
            if not self.queue_file.exists():
                return None
            
            queue_data = json.loads(self.queue_file.read_text())
            pending = [
                t for t in queue_data["queue"]
                if t["status"] == "pending"
            ]
            
            if not pending:
                return None
                
            # Sort by priority, then ID
            pending.sort(key=lambda t: (t["priority"], t["id"]))
            task = pending[0]
            
            # Mark as running immediately to reserve it
            for t in queue_data["queue"]:
                if t["id"] == task["id"]:
                    t["status"] = "running"
                    t["started_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    break
            
            self.queue_file.write_text(json.dumps(queue_data, indent=2))
            return task

    def _execute_task(self, task: dict, console: Console, total_pending: int | None = None) -> bool:
        """Execute single task. Note: status is already set to 'running' by _claim_next_task."""
        task_id = task["id"]
        log_file = self.project_root / task["log_file"]
        log_file.parent.mkdir(parents=True, exist_ok=True)

        console.print(f"[bold]Task #{task_id}[/bold] (priority: {task['priority']}) started")

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
