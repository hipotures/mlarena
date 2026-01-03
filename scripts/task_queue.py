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
  python scripts/task_queue.py --project Titanic add "model --model-template lgbm skip_submit=true" --priority 5

  # Add task with default priority (10)
  python scripts/task_queue.py --project Titanic add "model --model-template xgb preset=high"

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
    subparsers.add_parser("list", help="List queued tasks")

    # Add command
    add_parser = subparsers.add_parser("add", help="Add task to queue")
    add_parser.add_argument(
        "task_command",
        help="Full command string (e.g., 'model --model-template lgbm')"
    )
    add_parser.add_argument(
        "--priority",
        type=int,
        default=10,
        help="Task priority (1=highest, default=10)"
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
        choices=["completed", "failed", "pending"],
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
            queue.list_queue(console)
            return 0

        elif args.command == "add":
            task_id = queue.add_task(args.task_command, args.priority)
            console.print(f"[green]✓ Added task #{task_id}[/green]")
            console.print(f"[dim]Command: {args.task_command}[/dim]")
            console.print(f"[dim]Priority: {args.priority}[/dim]")
            return 0

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
