#!/usr/bin/env python
"""
Textual-based Task Queue Manager.

Displays queue dashboard, task list, and live logs.
"""
import sys
import json
import io
from pathlib import Path
from datetime import datetime
import asyncio

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, Grid
from textual.widgets import (
    Header, Footer, Static, DataTable, TabbedContent, TabPane, 
    Button, Label, RichLog, Select, Markdown, ProgressBar
)
from textual.screen import Screen
from textual.coordinate import Coordinate
from textual.reactive import reactive
from textual.worker import Worker
from rich.text import Text
from rich.console import Console

# Add src to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from mlarena.utils.queue_textual import TaskQueueTextual

class LogModal(Screen):
    """Modal screen to display log content."""
    
    BINDINGS = [("escape", "app.pop_screen", "Close Log")]
    
    def __init__(self, title: str, content: str):
        super().__init__()
        self.title_text = title
        self.log_content = content
        
    def compose(self) -> ComposeResult:
        yield Header()
        with Container(classes="modal-container"):
            yield Label(self.title_text, classes="modal-header")
            yield RichLog(id="modal-log", highlight=True, markup=True)
        yield Footer()
    
    def on_mount(self) -> None:
        self.query_one(RichLog).write(self.log_content)

class QueueDashboard(Vertical):
    def compose(self) -> ComposeResult:
        yield Label("Queue Statistics", classes="panel-header")
        with Container(classes="stats-container"):
            with Container(classes="dashboard-stats-grid"):
                yield Label("Pending:", classes="stat-label")
                yield Static("0", id="stat-pending", classes="stat-value")
                yield Label("Running:", classes="stat-label")
                yield Static("0", id="stat-running", classes="stat-value")
                yield Label("Completed:", classes="stat-label")
                yield Static("0", id="stat-completed", classes="stat-value")
                yield Label("Failed:", classes="stat-label")
                yield Static("0", id="stat-failed", classes="stat-value")
        
        yield Label("Current Status", classes="panel-header")
        with Container(classes="status-container"):
            yield Static("Idle", id="status-message")
            
        yield Label("Queue Progress", classes="panel-header")
        with Container(classes="progress-container"):
             yield ProgressBar(total=100, show_percentage=True, id="queue-progress")

        with Container(classes="controls-container"):
            yield Button("Run Queue", id="btn-run-queue", variant="success")
            yield Button("Stop Queue", id="btn-stop-queue", variant="error")

class TasksView(Container):
    status_filter = reactive("all")
    current_page = reactive(1)
    total_pages = reactive(1)
    tasks_data = {}

    def compose(self) -> ComposeResult:
        with Horizontal(classes="toolbar"):
            yield Label("Filter Status:", classes="label")
            yield Select.from_values(
                ["all", "pending", "running", "completed", "failed"], 
                value="all", 
                id="status-select",
                allow_blank=False
            )
            yield Button("Previous", id="btn-prev")
            yield Label("Page 1/1", id="page-label")
            yield Button("Next", id="btn-next")
            
        yield DataTable(cursor_type="row")

class LogView(Container):
    def compose(self) -> ComposeResult:
        yield Label("Waiting for logs...", id="log-status")
        yield RichLog(highlight=True, markup=True, wrap=True)

class TaskQueueApp(App):
    CSS = """
    /* Modal Styles */
    .modal-container {
        width: 100%;
        height: 100%;
        background: $surface;
        padding: 1;
    }
    
    .modal-header {
        width: 100%;
        text-align: center;
        text-style: bold;
        background: $primary;
        color: $text;
        padding: 1;
    }
    
    #modal-log {
        height: 1fr;
        border: none;
        margin: 0;
    }

    /* Dashboard Styles */
    QueueDashboard {
        width: 100%;
    }

    .panel-header {
        margin-top: 1;
        margin-left: 1;
        color: $secondary;
        text-style: bold;
    }

    .stats-container {
        width: 100%;
        height: auto;
        background: $boost;
        border: solid $secondary;
        margin: 0 1 1 1;
        padding: 1;
    }
    
    .dashboard-stats-grid {
        layout: grid;
        grid-size: 4 2;
        grid-columns: 1fr 1fr 1fr 1fr;
        height: auto;
    }
    
    .stat-label {
        text-align: left;
        color: $text-muted;
    }
    
    .stat-value {
        text-align: left;
        color: $accent;
        text-style: bold;
    }
    
    .status-container {
        width: 100%;
        height: auto;
        background: $surface;
        border: solid $accent;
        margin: 0 1 1 1;
        padding: 1;
    }
    
    #status-message {
        color: $text;
    }

    .progress-container {
        width: 100%;
        height: auto;
        background: $surface;
        border: solid $accent;
        margin: 0 1 1 1;
        padding: 1 1;
    }
    
    ProgressBar {
        width: 100%;
    }
    
    #queue-progress {
        width: 100%;
    }

    #queue-progress > Bar {
        width: 1fr;
    }

    .controls-container {
        height: auto;
        margin: 2 1;
        layout: horizontal;
        align: center middle;
    }
    
    .controls-container Button {
        height: 3;
        width: 16;
        margin: 0 1;
    }

    /* Tasks View Styles */
    .toolbar {
        height: auto;
        margin: 1;
        align-vertical: middle;
    }
    
    #status-select {
        width: 20;
        height: 3;
        margin-right: 2;
    }
    
    #page-label {
        margin: 0 2;
        height: 3;
        content-align: center middle;
    }
    
    DataTable {
        height: 1fr;
    }
    
    /* Log View Styles */
    RichLog {
        background: $surface;
        color: $text;
        border: solid $primary;
        height: 1fr;
    }
    
    #log-status {
        background: $primary;
        color: $text;
        padding: 1;
        width: 100%;
        text-align: center;
    }
    """
    
    BINDINGS = [
        ("q", "quit", "Quit"), 
        ("r", "refresh", "Refresh"), 
        ("s", "stop_queue", "Stop Queue"),
        ("x", "run_queue", "Run Queue")
    ]
    
    def __init__(self, project: str, auto_run: bool = False):
        super().__init__()
        self.project = project
        self.project_root = REPO_ROOT / "projects" / "kaggle" / project
        self.queue = TaskQueueTextual(self.project_root)
        self.current_log_file = None
        self.log_file_handle = None
        self.log_offset = 0
        self.auto_run = auto_run
        self.is_running_queue = False

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("Dashboard", id="tab-dashboard"):
                yield QueueDashboard()
            with TabPane("Tasks", id="tab-tasks"):
                yield TasksView()
            with TabPane("Live Log", id="tab-log"):
                yield LogView()
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(DataTable).add_columns(
            "ID", "Priority", "Status", "Module", "Template", "Added", "Duration", "Log File"
        )
        self.set_interval(2.0, self.refresh_data)
        self.set_interval(1.0, self.update_log)
        self.refresh_data()
        
        if self.auto_run:
            self.action_run_queue()

    def refresh_data(self) -> None:
        """Load queue data and update widgets."""
        try:
            # Access internal method to get raw data
            queue_data = self.queue._load_queue()
        except Exception as e:
            self.notify(f"Error loading queue: {e}", severity="error")
            return

        tasks = queue_data.get("queue", [])
        
        # 1. Update Dashboard
        counts = {"pending": 0, "running": 0, "completed": 0, "failed": 0}
        running_task = None
        
        for task in tasks:
            status = task.get("status", "pending")
            counts[status] = counts.get(status, 0) + 1
            if status == "running":
                running_task = task

        self.query_one("#stat-pending", Static).update(str(counts["pending"]))
        self.query_one("#stat-running", Static).update(str(counts["running"]))
        self.query_one("#stat-completed", Static).update(str(counts["completed"]))
        self.query_one("#stat-failed", Static).update(str(counts["failed"]))
        
        # Update Progress Bar
        total_tasks = len(tasks)
        completed_tasks = counts["completed"] + counts["failed"]
        if total_tasks > 0:
            pb = self.query_one("#queue-progress", ProgressBar)
            pb.update(total=total_tasks, progress=completed_tasks)
            # Only show ETA and elapsed time if a task is actually running
            is_running = (running_task is not None)
            pb.show_eta = is_running
            pb.show_elapsed = is_running

        status_msg = f"Running Task #{running_task['id']}: {running_task['command']}" if running_task else "Idle"
        if self.is_running_queue and not running_task:
            status_msg = "Queue Runner Active - Waiting for task..."
            
        self.query_one("#status-message", Static).update(status_msg)
        
        # Update button states
        self.query_one("#btn-run-queue", Button).disabled = self.is_running_queue
        
        # 2. Update Tasks Table
        self.update_tasks_table(tasks)
        
        # 3. Update Log File Target
        if running_task:
            log_path = self.project_root / running_task["log_file"]
            if str(log_path) != self.current_log_file:
                # New task running, switch log file
                self.current_log_file = str(log_path)
                self.log_offset = 0
                self.query_one("#log-status", Label).update(f"Monitoring: Task #{running_task['id']} ({log_path.name})")
                self.query_one(RichLog).clear()
        elif not running_task:
             if self.current_log_file:
                 self.query_one("#log-status", Label).update("Task finished. Waiting for next...")

    def update_tasks_table(self, tasks: list) -> None:
        table = self.query_one(DataTable)
        view = self.query_one(TasksView)
        
        # Store tasks data for lookup
        view.tasks_data = {str(t["id"]): t for t in tasks}
        
        # Filter
        filter_val = view.status_filter
        if filter_val != "all":
            filtered = [t for t in tasks if t["status"] == filter_val]
        else:
            filtered = tasks
            
        # Sort: Running first, then Pending (by priority), then Completed/Failed (newest first)
        def sort_key(t):
            s = t["status"]
            if s == "running": return (0, 0)
            if s == "pending": return (1, t["priority"])
            return (2, -t["id"]) # Descending ID for others
            
        filtered.sort(key=sort_key)
        
        # Pagination
        page_size = 10
        total = len(filtered)
        pages = (total + page_size - 1) // page_size or 1
        view.total_pages = pages
        
        if view.current_page > pages:
            view.current_page = pages
            
        start = (view.current_page - 1) * page_size
        end = start + page_size
        page_tasks = filtered[start:end]
        
        # Update Table
        table.clear()
        for t in page_tasks:
            # Format columns
            module = "-"
            template = "-"
            cmd_parts = t["command"].split()
            if cmd_parts:
                # Handle `uv run python scripts/mla.py` prefix if present
                if "mla.py" in t["command"]:
                   # Try to find module after mla.py
                   try:
                       mla_idx = cmd_parts.index("scripts/mla.py")
                       cmd_parts = cmd_parts[mla_idx+1:]
                   except ValueError:
                       pass
                
                if cmd_parts:
                    module = cmd_parts[0]
                    if "--model-template" in cmd_parts:
                         try:
                            template = cmd_parts[cmd_parts.index("--model-template") + 1]
                         except IndexError:
                            pass
                    elif "--preprocess-template" in cmd_parts:
                         try:
                            template = cmd_parts[cmd_parts.index("--preprocess-template") + 1]
                         except IndexError:
                             pass
            
            duration = "-"
            if t["started_at"] and t["finished_at"]:
                try:
                    start_dt = datetime.strptime(t["started_at"], "%Y-%m-%d %H:%M:%S")
                    end_dt = datetime.strptime(t["finished_at"], "%Y-%m-%d %H:%M:%S")
                    duration = str(end_dt - start_dt)
                except ValueError:
                    pass
            
            status_style = {
                "running": "bold blue",
                "completed": "green",
                "failed": "bold red",
                "pending": "yellow"
            }.get(t["status"], "white")
            
            # Format log file cell - hide for pending tasks
            if t["status"] == "pending":
                log_cell = "-"
            else:
                log_cell = t.get("log_file", "-")
            
            table.add_row(
                str(t["id"]),
                str(t["priority"]),
                Text(t["status"], style=status_style),
                module,
                template,
                t["added_at"],
                duration,
                log_cell
            )
            
        # Update Controls
        view.query_one("#page-label", Label).update(f"Page {view.current_page}/{view.total_pages}")
        view.query_one("#btn-prev", Button).disabled = (view.current_page == 1)
        view.query_one("#btn-next", Button).disabled = (view.current_page == view.total_pages)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection to open log modal."""
        table = self.query_one(DataTable)
        view = self.query_one(TasksView)
        
        # Get task ID from the first column (index 0) of the selected row
        try:
            # get_cell_at expects a Coordinate(row, col)
            task_id_cell = table.get_cell_at(Coordinate(event.cursor_row, 0))
            task_id = str(task_id_cell)
            
            task = view.tasks_data.get(task_id)
            
            if not task:
                return

            if task.get("status") == "pending":
                self.notify(f"Task #{task_id} is pending. No log available yet.", severity="warning")
                return
            
            if task.get("log_file"):
                log_path = self.project_root / task["log_file"]
                if log_path.exists():
                    try:
                        content = log_path.read_text()
                        self.push_screen(LogModal(f"Log: Task #{task_id}", content))
                    except Exception as e:
                        self.notify(f"Failed to read log: {e}", severity="error")
                else:
                    self.notify(f"Log file not found (task not started?): {task['log_file']}", severity="warning")
        except Exception as e:
            self.notify(f"Error opening log: {e}", severity="error")

    def update_log(self) -> None:
        """Read new lines from log file."""
        if not self.current_log_file:
            return
            
        try:
            # Simple polling read
            if Path(self.current_log_file).exists():
                with open(self.current_log_file, "r") as f:
                    f.seek(self.log_offset)
                    new_text = f.read()
                    if new_text:
                        self.log_offset = f.tell()
                        # Use Text.from_ansi to handle the "smieci" (ANSI codes)
                        self.query_one(RichLog).write(Text.from_ansi(new_text))
        except Exception:
            pass

    def on_select_changed(self, event: Select.Changed) -> None:
        self.query_one(TasksView).status_filter = str(event.value)
        self.query_one(TasksView).current_page = 1
        self.refresh_data()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-close":
            # Handled in LogModal
            return
            
        view = self.query_one(TasksView)
        if event.button.id == "btn-prev":
            if view.current_page > 1:
                view.current_page -= 1
            self.refresh_data()
        elif event.button.id == "btn-next":
            if view.current_page < view.total_pages:
                view.current_page += 1
            self.refresh_data()
        elif event.button.id == "btn-run-queue":
            self.action_run_queue()
        elif event.button.id == "btn-stop-queue":
            self.action_stop_queue()

    def action_run_queue(self) -> None:
        if self.is_running_queue:
            return
        self.is_running_queue = True
        self.run_worker(self.worker_run_queue, thread=True)
        self.notify("Queue runner started")

    def action_stop_queue(self) -> None:
        # Create .stop file for graceful shutdown of TaskQueue.run_queue
        self.queue.stop_file.touch()
        self.is_running_queue = False
        self.notify("Stop signal sent to queue")

    def worker_run_queue(self) -> None:
        """Background thread running the actual queue."""
        # Capture console output from TaskQueue
        capture_io = io.StringIO()
        capture_console = Console(file=capture_io, force_terminal=True)
        
        try:
            self.queue.run_queue(console=capture_console)
        except Exception as e:
            # self.notify is thread safe in Textual 0.20+
            # We schedule it on the main thread via call_from_thread if needed, 
            # but newer Textual handles it. To be safe:
            self.app.call_from_thread(self.notify, f"Runner error: {e}", severity="error")
        finally:
            self.is_running_queue = False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="Project name")
    parser.add_argument("command", nargs="?", choices=["run"], help="Optional command")
    args = parser.parse_args()
    
    auto_run = (args.command == "run")
    
    app = TaskQueueApp(project=args.project, auto_run=auto_run)
    app.run()
