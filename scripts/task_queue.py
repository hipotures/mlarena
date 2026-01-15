"""Task queue management CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlarena.utils.queue import TaskQueue

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Task queue management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List queue
  python scripts/task_queue.py --project Titanic list

  # Add task with priority
  python scripts/task_queue.py --project Titanic add "model model_template=lgbm skip_submit=true" --priority 5

  # Add task with default priority (10)
  python scripts/task_queue.py --project Titanic add "model model_template=xgb preset=high"

  # Run queue
  python scripts/task_queue.py --project Titanic run

  # Run first 3 tasks
  python scripts/task_queue.py --project Titanic run --max-tasks 3

  # Remove task by ID
  python scripts/task_queue.py --project Titanic remove 1

  # Clean completed tasks
  python scripts/task_queue.py --project Titanic clean

  # Clean failed tasks
  python scripts/task_queue.py --project Titanic clean --status failed

  # Update task priority
  python scripts/task_queue.py --project Titanic priority 2 --priority 1
        """
    )

    parser.add_argument(
        "--project", "-p",
        required=True,
        help="Project name (e.g., Titanic)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # List command
    list_parser = subparsers.add_parser("list", help="List queued tasks")
    list_parser.add_argument(
        "--status",
        choices=["completed", "failed", "pending", "running", "all"],
        help="Filter by status"
    )

    # Add command - hybrid mode (template OR command string)
    add_parser = subparsers.add_parser("add", help="Add task to queue")

    # Template mode (NEW)
    add_parser.add_argument(
        "--model-template", "-m",
        help="Model template name (triggers auto-flow)"
    )
    add_parser.add_argument(
        "--preprocess-template",
        help="Preprocess template (standalone or override model's preprocess)"
    )

    # MLA flags (NEW)
    add_parser.add_argument(
        "--enable-submit",
        dest="enable_submit",
        action="store_true",
        default=None,
        help="Enable submit (override template mla.skip_submit)"
    )
    add_parser.add_argument(
        "--no-submit",
        dest="enable_submit",
        action="store_false",
        help="Disable submit (override template mla.skip_submit)"
    )
    add_parser.add_argument(
        "--enable-git",
        dest="enable_git",
        action="store_true",
        default=None,
        help="Enable git commit (override template mla.skip_git)"
    )
    add_parser.add_argument(
        "--no-git",
        dest="enable_git",
        action="store_false",
        help="Disable git commit (override template mla.skip_git)"
    )

    # Priority
    add_parser.add_argument(
        "--priority",
        type=int,
        default=None,
        help="Task priority (1=highest, overrides template mla.priority)"
    )

    # Command string mode (BACKWARD COMPAT)
    add_parser.add_argument(
        "--command", "-c",
        dest="command_string",
        help="Full command string (alternative to --template, e.g., 'model model_template=lgbm')"
    )

    # Config overrides (positional to capture all remaining args)
    add_parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides (key=value)"
    )

    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove task from queue")
    remove_parser.add_argument(
        "task_id",
        type=int,
        help="Task ID to remove"
    )

    # Run command
    run_parser = subparsers.add_parser("run", help="Execute pending tasks")
    run_parser.add_argument(
        "--max-tasks",
        type=int,
        help="Maximum number of tasks to execute"
    )
    run_parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop on first error (default: continue on error)"
    )

    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Remove tasks by status")
    clean_parser.add_argument(
        "--status",
        default="completed",
        choices=["completed", "failed", "pending", "all"],
        help="Status to clean (default: completed)"
    )

    # Priority command
    priority_parser = subparsers.add_parser("priority", help="Update task priority")
    priority_parser.add_argument(
        "task_id",
        type=int,
        help="Task ID to update"
    )
    priority_parser.add_argument(
        "--priority",
        type=int,
        required=True,
        help="New priority value (1=highest)"
    )

    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset task(s) to pending")
    reset_parser.add_argument(
        "--status",
        choices=["completed", "failed", "pending", "running", "all"],
        help="Reset all tasks with this status"
    )
    reset_parser.add_argument(
        "--id",
        type=int,
        help="Reset a single task by ID"
    )

    args = parser.parse_args()

    # Determine project root
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent
    project_root = repo_root / "projects" / "kaggle" / args.project

    if not project_root.exists():
        console.print(f"[red]Error: Project '{args.project}' not found[/red]")
        return 1

    queue = TaskQueue(project_root)

    try:
        if args.command == "list":
            queue.list_queue(console, status=args.status)
            return 0

        elif args.command == "add":
            if args.model_template or args.preprocess_template:
                # Template mode (NEW)
                task_id = queue.add_task_from_template(
                    model_template=args.model_template,
                    preprocess_template=args.preprocess_template,
                    priority=args.priority,
                    submit=args.enable_submit,
                    git=args.enable_git,
                    overrides=args.overrides
                )

                console.print(f"[green]✓ Added task #{task_id}[/green]")
                if args.model_template:
                    console.print(f"[dim]Model template: {args.model_template}[/dim]")
                if args.preprocess_template:
                    console.print(f"[dim]Preprocess template: {args.preprocess_template}[/dim]")
                console.print(f"[dim]Priority: {args.priority or 'from template'}[/dim]")
                return 0

            elif args.command_string:
                # Command string mode (BACKWARD COMPAT)
                task_id = queue.add_task(args.command_string, args.priority or 10)
                console.print(f"[green]✓ Added task #{task_id}[/green]")
                console.print(f"[dim]Command: {args.command_string}[/dim]")
                console.print(f"[dim]Priority: {args.priority or 10}[/dim]")
                return 0

            else:
                console.print("[red]Error: Either --model-template or --command required[/red]")
                return 1

        elif args.command == "remove":
            queue.remove_task(args.task_id)
            console.print(f"[green]✓ Removed task #{args.task_id}[/green]")
            return 0

        elif args.command == "run":
            results = queue.run_queue(
                console=console,
                max_tasks=args.max_tasks,
                continue_on_error=not args.stop_on_error
            )
            return 0 if all(results.values()) else 1

        elif args.command == "clean":
            count = queue.clean_queue(status=args.status)
            console.print(f"[green]✓ Removed {count} {args.status} tasks[/green]")
            return 0

        elif args.command == "priority":
            queue.update_priority(args.task_id, args.priority)
            console.print(f"[green]✓ Updated task #{args.task_id} priority to {args.priority}[/green]")
            return 0

        elif args.command == "reset":
            if args.id is None and args.status is None:
                console.print("[red]Error: Provide --id or --status[/red]")
                return 1

            count = queue.reset_tasks(status=args.status, task_id=args.id)
            if args.id is not None:
                console.print(f"[green]✓ Task #{args.id} reset to pending[/green]")
            else:
                console.print(f"[green]✓ Reset {count} task(s) with status '{args.status}'[/green]")
            return 0

        else:
            parser.print_help()
            return 1

    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
