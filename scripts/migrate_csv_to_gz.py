#!/usr/bin/env python3
"""
Migrate CSV files to compressed .csv.gz format.

This script compresses all .csv files in a project to .csv.gz format, enabling
automatic compression/decompression via pandas compression='infer'.

Usage:
    # Dry run (preview without changes)
    python scripts/migrate_csv_to_gz.py --project Titanic --dry-run

    # Compress data and submissions
    python scripts/migrate_csv_to_gz.py --project Titanic

    # Include experiments directory
    python scripts/migrate_csv_to_gz.py --project Titanic --include-experiments

    # All projects at once
    python scripts/migrate_csv_to_gz.py --all-projects
"""

import argparse
import gzip
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel

console = Console()


def compress_csv(csv_path: Path, remove_original: bool = True, dry_run: bool = False) -> Dict:
    """
    Compress a single CSV file to .csv.gz format.

    Args:
        csv_path: Path to the CSV file
        remove_original: Whether to remove the original .csv after compression
        dry_run: If True, only simulate (don't actually compress)

    Returns:
        dict with compression results: {
            "status": "compressed" | "skipped" | "error",
            "original_size": int (bytes),
            "compressed_size": int (bytes),
            "ratio": float (percentage saved),
            "error": str (optional)
        }
    """
    gz_path = csv_path.with_suffix(csv_path.suffix + '.gz')

    # Skip if .gz already exists
    if gz_path.exists():
        return {
            "status": "skipped",
            "reason": "already exists",
            "original_size": csv_path.stat().st_size if csv_path.exists() else 0,
            "compressed_size": gz_path.stat().st_size,
        }

    original_size = csv_path.stat().st_size

    if dry_run:
        # Estimate compression (usually 70-90%)
        estimated_size = int(original_size * 0.2)  # Assume 80% compression
        estimated_ratio = 80.0
        return {
            "status": "would_compress",
            "original_size": original_size,
            "compressed_size": estimated_size,
            "ratio": estimated_ratio
        }

    try:
        # Compress file
        with open(csv_path, 'rb') as f_in:
            with gzip.open(gz_path, 'wb', compresslevel=9) as f_out:
                shutil.copyfileobj(f_in, f_out)

        compressed_size = gz_path.stat().st_size

        # Verify compressed file can be read
        try:
            pd.read_csv(gz_path, nrows=5, compression='infer')
        except Exception as e:
            # Verification failed - remove corrupted file
            gz_path.unlink()
            raise RuntimeError(f"Verification failed: {e}")

        # Remove original
        if remove_original:
            csv_path.unlink()

        ratio = (1 - compressed_size / original_size) * 100

        return {
            "status": "compressed",
            "original_size": original_size,
            "compressed_size": compressed_size,
            "ratio": ratio
        }

    except Exception as e:
        return {
            "status": "error",
            "original_size": original_size,
            "error": str(e)
        }


def find_csv_files(project_root: Path, include_experiments: bool = False) -> List[Path]:
    """
    Find all CSV files in a project.

    Args:
        project_root: Root directory of the project
        include_experiments: Whether to include experiments/**/*.csv

    Returns:
        List of CSV file paths
    """
    patterns = [
        "data/*.csv",
        "submissions/*.csv"
    ]

    if include_experiments:
        patterns.append("experiments/**/*.csv")

    files = []
    for pattern in patterns:
        files.extend(project_root.glob(pattern))

    # Sort by size (largest first) for better progress visibility
    files.sort(key=lambda f: f.stat().st_size, reverse=True)

    return files


def format_bytes(bytes_val: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f} TB"


def migrate_project(
    project_root: Path,
    include_experiments: bool = False,
    dry_run: bool = False,
    remove_original: bool = True
) -> Tuple[int, int, int, int, int]:
    """
    Migrate all CSV files in a project to .csv.gz.

    Returns:
        (compressed_count, skipped_count, error_count, total_saved_bytes, total_original_bytes)
    """
    files = find_csv_files(project_root, include_experiments)

    if not files:
        console.print(f"[yellow]No CSV files found in {project_root}[/yellow]")
        return (0, 0, 0, 0, 0)

    compressed_count = 0
    skipped_count = 0
    error_count = 0
    total_saved_bytes = 0
    total_original_bytes = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task(
            f"[cyan]{'[DRY RUN] ' if dry_run else ''}Compressing CSV files...",
            total=len(files)
        )

        for csv_file in files:
            rel_path = csv_file.relative_to(project_root)
            progress.update(task, description=f"[cyan]Processing {rel_path}...")

            result = compress_csv(csv_file, remove_original, dry_run)

            total_original_bytes += result.get("original_size", 0)

            if result["status"] in ("compressed", "would_compress"):
                compressed_count += 1
                saved = result["original_size"] - result["compressed_size"]
                total_saved_bytes += saved

                status_icon = "[green]✓[/green]" if result["status"] == "compressed" else "[cyan]→[/cyan]"
                console.print(
                    f"{status_icon} {rel_path}: "
                    f"{format_bytes(result['original_size'])} → {format_bytes(result['compressed_size'])} "
                    f"({result['ratio']:.1f}% saved)"
                )

            elif result["status"] == "skipped":
                skipped_count += 1
                console.print(f"[yellow]⊘[/yellow] {rel_path}: {result.get('reason', 'skipped')}")

            elif result["status"] == "error":
                error_count += 1
                console.print(f"[red]✗[/red] {rel_path}: {result.get('error', 'unknown error')}")

            progress.advance(task)

    return (compressed_count, skipped_count, error_count, total_saved_bytes, total_original_bytes)


def main():
    parser = argparse.ArgumentParser(
        description="Migrate CSV files to compressed .csv.gz format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--project",
        type=str,
        help="Project name (e.g., Titanic, playground-series-s5e12)"
    )
    parser.add_argument(
        "--all-projects",
        action="store_true",
        help="Migrate all projects in projects/kaggle/"
    )
    parser.add_argument(
        "--include-experiments",
        action="store_true",
        help="Include experiments/**/*.csv (can be large)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without actually compressing files"
    )
    parser.add_argument(
        "--keep-original",
        action="store_true",
        help="Keep original .csv files after compression (default: remove)"
    )

    args = parser.parse_args()

    if not args.project and not args.all_projects:
        parser.error("Must specify either --project or --all-projects")

    # Find repo root
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    projects_dir = repo_root / "projects" / "kaggle"

    if not projects_dir.exists():
        console.print(f"[red]Error: Projects directory not found: {projects_dir}[/red]")
        sys.exit(1)

    # Determine which projects to migrate
    if args.all_projects:
        projects = [p for p in projects_dir.iterdir() if p.is_dir() and not p.name.startswith('.')]
    else:
        project_root = projects_dir / args.project
        if not project_root.exists():
            console.print(f"[red]Error: Project not found: {project_root}[/red]")
            sys.exit(1)
        projects = [project_root]

    # Print header
    if args.dry_run:
        console.print(Panel(
            "[bold yellow]DRY RUN MODE[/bold yellow]\n"
            "No files will be modified. This is a preview only.",
            title="Migration Preview"
        ))
    else:
        console.print(Panel(
            "[bold cyan]CSV Compression Migration[/bold cyan]\n"
            f"Projects: {len(projects)}\n"
            f"Include experiments: {args.include_experiments}\n"
            f"Keep originals: {args.keep_original}",
            title="Migration Settings"
        ))

    # Migrate each project
    total_compressed = 0
    total_skipped = 0
    total_errors = 0
    total_saved = 0
    total_original = 0

    for project_root in projects:
        console.print(f"\n[bold]Project: {project_root.name}[/bold]")
        console.print("─" * 80)

        compressed, skipped, errors, saved, original = migrate_project(
            project_root,
            args.include_experiments,
            args.dry_run,
            not args.keep_original
        )

        total_compressed += compressed
        total_skipped += skipped
        total_errors += errors
        total_saved += saved
        total_original += original

    # Print summary
    console.print("\n" + "═" * 80)

    summary_table = Table(title="Migration Summary", show_header=True, header_style="bold magenta")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", justify="right", style="green")

    summary_table.add_row("Projects processed", str(len(projects)))
    summary_table.add_row(
        "Files compressed" if not args.dry_run else "Files to compress",
        str(total_compressed)
    )
    summary_table.add_row("Files skipped", str(total_skipped))
    if total_errors > 0:
        summary_table.add_row("Errors", str(total_errors), style="red")

    if total_original > 0:
        overall_ratio = (total_saved / total_original) * 100
        summary_table.add_row("Original size", format_bytes(total_original))
        summary_table.add_row(
            "Compressed size" if not args.dry_run else "Estimated size",
            format_bytes(total_original - total_saved)
        )
        summary_table.add_row(
            "Space saved" if not args.dry_run else "Estimated savings",
            f"{format_bytes(total_saved)} ({overall_ratio:.1f}%)"
        )

    console.print(summary_table)

    if args.dry_run:
        console.print("\n[bold yellow]This was a dry run. No files were modified.[/bold yellow]")
        console.print("Run without --dry-run to actually compress the files.")

    if total_errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
