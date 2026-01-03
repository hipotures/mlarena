"""Task queue implementation with FileLock and priority management."""

from __future__ import annotations

import json
import re
import subprocess
import yaml
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from filelock import FileLock
from rich.console import Console
from rich.table import Table


class TaskQueue:
    """Per-project task queue with priority execution."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.queue_dir = self.project_root / "queue"
        self.queue_file = self.queue_dir / "queue.json"
        self.lock_file = self.queue_file.with_suffix(".lock")
        self.stop_file = self.queue_dir / ".stop"
        self.log_dir = self.queue_dir / "logs"

        # Ensure directories exist
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _load_queue(self) -> dict:
        """Load queue.json with FileLock."""
        if not self.queue_file.exists():
            return {"next_id": 1, "queue": []}

        with FileLock(str(self.lock_file), timeout=10):
            return json.loads(self.queue_file.read_text())

    def _save_queue(self, queue_data: dict) -> None:
        """Save queue.json with FileLock."""
        with FileLock(str(self.lock_file), timeout=10):
            self.queue_file.write_text(json.dumps(queue_data, indent=2))

    def _load_template_mla_settings(
        self,
        template_type: str,
        template_name: str
    ) -> dict:
        """
        Load mla: section from template YAML.

        Args:
            template_type: "model" or "preprocess"
            template_name: Template name (without .yaml extension)

        Returns:
            dict with keys: skip_submit, skip_git, priority (if present in YAML)
            Empty dict if template not found or no mla section
        """
        # Try project-specific first, then global
        template_paths = [
            self.project_root / "templates" / template_type / f"{template_name}.yaml",
            # Global templates (go up from utils/ to repo root)
            Path(__file__).parent.parent.parent / "templates" / template_type / f"{template_name}.yaml"
        ]

        for path in template_paths:
            if path.exists():
                try:
                    config = yaml.safe_load(path.read_text())
                    return config.get("mla", {}) if config else {}
                except Exception:
                    # YAML parse error - return empty dict
                    return {}

        return {}  # Template not found

    def add_task(self, command: str, priority: int = 10) -> int:
        """
        Add task to queue.

        Args:
            command: Full command string (e.g., "model --project X --template Y")
            priority: Task priority (1=highest, default=10)

        Returns:
            Task ID
        """
        queue_data = self._load_queue()
        task_id = queue_data["next_id"]

        task = {
            "id": task_id,
            "priority": priority,
            "status": "pending",
            "command": command,
            "added_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "started_at": None,
            "finished_at": None,
            "attempts": [],
            "last_error": None,
            "experiment_id": None,
            "log_file": f"queue/logs/task-{task_id}.log"
        }

        queue_data["queue"].append(task)
        queue_data["next_id"] = task_id + 1
        self._save_queue(queue_data)

        return task_id

    def add_task_from_template(
        self,
        model_template: Optional[str] = None,
        preprocess_template: Optional[str] = None,
        priority: Optional[int] = None,
        submit: Optional[bool] = None,
        git: Optional[bool] = None,
        overrides: Optional[List[str]] = None
    ) -> int:
        """
        Add task from template(s) with auto-loaded MLA settings.

        Priority order for settings:
        1. CLI flags (priority, submit, git)
        2. Template mla: section
        3. System defaults (priority=10, skip_submit=True, skip_git=True)

        Args:
            model_template: Model template name (triggers auto-flow)
            preprocess_template: Preprocess template (standalone or override model's)
            priority: Task priority (overrides template mla.priority)
            submit: Enable/disable submit (overrides template mla.skip_submit)
            git: Enable/disable git (overrides template mla.skip_git)
            overrides: Config overrides (key=value strings)

        Returns:
            Task ID

        Raises:
            ValueError: If neither model_template nor preprocess_template provided
        """
        # Determine command type
        if model_template:
            # Model template → auto-flow
            cmd_parts = ["model", "--model-template", model_template]

            # Override preprocess if specified
            if preprocess_template:
                cmd_parts.extend(["--preprocess-template", preprocess_template])

            # Load mla settings from MODEL template
            mla = self._load_template_mla_settings("model", model_template)

        elif preprocess_template:
            # Standalone preprocess (no model)
            cmd_parts = ["preprocess", "--preprocess-template", preprocess_template]

            # Load mla settings from PREPROCESS template
            mla = self._load_template_mla_settings("preprocess", preprocess_template)

        else:
            raise ValueError("Either model_template or preprocess_template required")

        # Resolve priority (CLI > template > default)
        resolved_priority = priority if priority is not None else mla.get("priority", 10)

        # Resolve submit flag (CLI > template > default)
        if submit is None:
            skip_submit = mla.get("skip_submit", True)  # Default TRUE for queue
        else:
            skip_submit = not submit

        if skip_submit:
            cmd_parts.append("skip_submit=true")

        # Resolve git flag (CLI > template > default)
        if git is None:
            skip_git = mla.get("skip_git", True)  # Default TRUE for queue
        else:
            skip_git = not git

        if skip_git:
            cmd_parts.append("skip_git=true")

        # Add config overrides
        if overrides:
            cmd_parts.extend(overrides)

        # Build command string
        command = " ".join(cmd_parts)

        # Use existing add_task
        return self.add_task(command, resolved_priority)

    def remove_task(self, task_id: int) -> bool:
        """Remove task from queue."""
        queue_data = self._load_queue()
        original_len = len(queue_data["queue"])

        queue_data["queue"] = [
            t for t in queue_data["queue"]
            if t["id"] != task_id
        ]

        if len(queue_data["queue"]) == original_len:
            raise ValueError(f"Task #{task_id} not found")

        self._save_queue(queue_data)
        return True

    def list_queue(self, console: Console) -> None:
        """Display queue as Rich table."""
        queue_data = self._load_queue()
        tasks = queue_data.get("queue", [])

        if not tasks:
            console.print("[yellow]Queue is empty[/yellow]")
            return

        table = Table(title="Task Queue", show_header=True, expand=True)
        table.add_column("#", style="cyan", width=4)
        table.add_column("Priority", width=8)
        table.add_column("Status", width=12)
        table.add_column("Module", width=10)
        table.add_column("Template", width=20)
        table.add_column("Options", width=30)
        table.add_column("Experiment", style="dim", width=20)

        # Sort by priority (ascending), then by ID (ascending)
        sorted_tasks = sorted(tasks, key=lambda t: (t["priority"], t["id"]))

        for task in sorted_tasks:
            status = task["status"]
            if status == "pending":
                status_str = "[yellow]pending[/yellow]"
            elif status == "running":
                status_str = "[blue]running[/blue]"
            elif status == "completed":
                status_str = "[green]completed[/green]"
            else:
                status_str = "[red]failed[/red]"

            # Parse command into module/template/options
            cmd_parts = task["command"].split()

            # Extract module (first element)
            module = cmd_parts[0] if cmd_parts else "-"

            # Extract template name
            template = "-"
            if "--model-template" in cmd_parts:
                idx = cmd_parts.index("--model-template")
                template = cmd_parts[idx + 1] if idx + 1 < len(cmd_parts) else "-"
            elif "--preprocess-template" in cmd_parts:
                idx = cmd_parts.index("--preprocess-template")
                template = cmd_parts[idx + 1] if idx + 1 < len(cmd_parts) else "-"

            # Extract options (everything except module and template flags)
            options = []
            skip_next = False
            for i, part in enumerate(cmd_parts[1:], 1):  # Skip module name
                if skip_next:
                    skip_next = False
                    continue
                if part in ["--model-template", "--preprocess-template", "--project", "-p"]:
                    skip_next = True
                    continue
                options.append(part)

            # Format options - each on new line if multiple
            options_str = "\n".join(options[:3]) if options else "-"
            if len(options) > 3:
                options_str += "\n..."

            # Show experiment ID if available
            exp_id = task.get("experiment_id") or "-"

            table.add_row(
                str(task["id"]),
                str(task["priority"]),
                status_str,
                module,
                template,
                options_str,
                exp_id
            )

        console.print(table)
        console.print(f"\n[dim]Total: {len(tasks)} tasks[/dim]")

    def clean_queue(self, status: str = "completed") -> int:
        """Remove tasks with specified status."""
        queue_data = self._load_queue()
        original_len = len(queue_data["queue"])

        queue_data["queue"] = [
            t for t in queue_data["queue"]
            if t["status"] != status
        ]

        removed = original_len - len(queue_data["queue"])
        self._save_queue(queue_data)

        return removed

    def update_priority(self, task_id: int, new_priority: int) -> None:
        """Update task priority."""
        queue_data = self._load_queue()

        found = False
        for t in queue_data["queue"]:
            if t["id"] == task_id:
                t["priority"] = new_priority
                found = True
                break

        if not found:
            raise ValueError(f"Task #{task_id} not found")

        self._save_queue(queue_data)

    def run_queue(
        self,
        console: Console,
        max_tasks: Optional[int] = None,
        continue_on_error: bool = True
    ) -> dict[int, bool]:
        """
        Execute pending tasks in priority order.

        Args:
            console: Rich console for output
            max_tasks: Maximum tasks to execute (None=all)
            continue_on_error: Continue after failures

        Returns:
            Dict mapping task_id to success status
        """
        results = {}
        executed = 0

        console.print("[bold cyan]Starting queue runner...[/bold cyan]\n")

        while True:
            # Check stop signal
            if self.stop_file.exists():
                console.print("\n[yellow]Stop signal detected (.stop file)[/yellow]")
                console.print("[yellow]Graceful shutdown...[/yellow]")
                self.stop_file.unlink()  # Remove stop file
                break

            # Check max_tasks limit
            if max_tasks and executed >= max_tasks:
                console.print(f"\n[yellow]Max tasks limit reached ({max_tasks})[/yellow]")
                break

            # Get next pending task
            queue_data = self._load_queue()
            pending = [
                t for t in queue_data["queue"]
                if t["status"] == "pending"
            ]

            if not pending:
                console.print("\n[green]All pending tasks completed[/green]")
                break

            # Sort by priority, then ID
            pending.sort(key=lambda t: (t["priority"], t["id"]))
            task = pending[0]

            # Execute task
            success = self._execute_task(task, console)
            results[task["id"]] = success
            executed += 1

            if not success and not continue_on_error:
                console.print(f"\n[red]Task #{task['id']} failed, stopping queue[/red]")
                break

        console.print(f"\n[bold]Queue runner finished[/bold]")
        console.print(
            f"[dim]Executed: {executed}, "
            f"Success: {sum(results.values())}, "
            f"Failed: {executed - sum(results.values())}[/dim]"
        )

        return results

    def _execute_task(self, task: dict, console: Console) -> bool:
        """
        Execute single task and update state.

        Returns:
            True if successful, False otherwise
        """
        task_id = task["id"]
        log_file = self.project_root / task["log_file"]
        log_file.parent.mkdir(parents=True, exist_ok=True)

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

        # Add --project if not already present
        if "--project" not in cmd_parts and "-p" not in cmd_parts:
            # Insert --project after the module name (first element)
            project_name = self.project_root.name
            cmd_parts.insert(1, "--project")
            cmd_parts.insert(2, project_name)

        # Execute via subprocess
        repo_root = self.project_root.parent.parent.parent
        full_cmd = ["uv", "run", "python", "scripts/mla.py"] + cmd_parts

        console.print(f"[dim]Command: {' '.join(full_cmd)}[/dim]")
        console.print(f"[dim]Log: {task['log_file']}[/dim]\n")

        try:
            with open(log_file, "w") as log:
                result = subprocess.run(
                    full_cmd,
                    cwd=repo_root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True
                )

            success = result.returncode == 0

            # Read log to extract experiment_id
            experiment_id = None
            if log_file.exists():
                log_content = log_file.read_text()
                # Look for patterns like "exp-YYYYMMDD-HHMMSS"
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

    def _mark_task_failed(self, task_id: int, error: str) -> None:
        """Mark task as failed with error message."""
        queue_data = self._load_queue()
        for t in queue_data["queue"]:
            if t["id"] == task_id:
                t["status"] = "failed"
                t["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                t["last_error"] = error

                attempt = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "success": False,
                    "error": error,
                    "exit_code": None
                }
                t["attempts"].append(attempt)
                break

        self._save_queue(queue_data)
