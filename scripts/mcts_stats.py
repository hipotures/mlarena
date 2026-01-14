#!/usr/bin/env python3
"""
MCTS Tree Statistics Viewer (Rich Edition).
Displays most visited nodes and their scores for a specific study.
Usage: python scripts/mcts_stats.py --project playground-series-s6e1 --study s6e1_001
"""

import argparse
import sqlite3
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

# Add src to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

def main():
    parser = argparse.ArgumentParser(description="View MCTS Tree Stats")
    parser.add_argument("--project", "-p", required=True, help="Project name")
    parser.add_argument("--study", "-s", required=True, help="Study name")
    parser.add_argument("--limit", "-l", type=int, default=30, help="Number of nodes to show")
    
    args = parser.parse_args()
    console = Console()
    
    # Resolve project path
    base_projects_dir = REPO_ROOT / "projects" / "kaggle"
    project_root = (base_projects_dir / args.project).resolve()
    if not project_root.exists():
        for p in base_projects_dir.iterdir():
            if p.is_dir() and p.name.lower() == args.project.lower():
                project_root = p.resolve()
                break
    
    db_path = project_root / "experiments" / "db" / "mcts.db"
    if not db_path.exists():
        db_path = Path("/mnt/mlarena") / "projects" / "kaggle" / args.project / "experiments" / "db" / "mcts.db"
        if not db_path.exists():
            console.print(f"[bold red]Error:[/bold red] Database not found at {db_path}")
            sys.exit(1)

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    study = conn.execute("SELECT study_id FROM studies WHERE study_name=?", (args.study,)).fetchone()
    if not study:
        console.print(f"[bold red]Error:[/bold red] Study '{args.study}' not found.")
        sys.exit(1)
    study_id = study[0]

    query = """
        SELECT t.number, n.n_visits, n.value_sum, n.value_best, n.depth, n.pipeline_signature
        FROM trials t
        JOIN mcts_nodes n ON t.trial_id = n.trial_id
        WHERE t.study_id = ?
        ORDER BY n.n_visits DESC, n.value_best DESC
        LIMIT ?
    """
    
    rows = conn.execute(query, (study_id, args.limit)).fetchall()

    if not rows:
        console.print(f"[yellow]No statistics found for study '{args.study}'.[/yellow]")
        return

    table = Table(title=f"MCTS Statistics: {args.study}", box=box.ROUNDED, show_footer=True)
    table.add_column("Trial #", justify="right", style="cyan", no_wrap=True)
    table.add_column("Depth", justify="center", style="magenta")
    table.add_column("Visits", justify="right", style="green")
    table.add_column("Avg Score", justify="right", style="yellow")
    table.add_column("Best Score", justify="right", style="bold green")
    table.add_column("Pipeline (preview)", style="white")

    for r in rows:
        n_visits = r["n_visits"]
        v_sum = r["value_sum"] if r["value_sum"] is not None else 0.0
        v_best = r["value_best"] if r["value_best"] is not None else 0.0
        avg_score = (v_sum / n_visits) if n_visits > 0 else 0.0
        
        # Colorize scores
        avg_str = f"{avg_score:.4f}"
        best_str = f"{v_best:.4f}"
        
        # Get a preview of steps from trial_params
        params = conn.execute(
            "SELECT param_value FROM trial_params WHERE trial_id = (SELECT trial_id FROM trials WHERE number=? AND study_id=?) AND param_name LIKE 'step_%' ORDER BY param_name",
            (r["number"], study_id)
        ).fetchall()
        
        steps = []
        for p in params:
            try:
                d = json.loads(p[0])
                steps.append(f"{d['name']}")
            except: continue
        
        pipeline_desc = " -> ".join(steps) if steps else "Baseline"
        
        table.add_row(
            str(r["number"]),
            str(r["depth"]),
            str(n_visits),
            avg_str,
            best_str,
            pipeline_desc
        )

    console.print(table)

    # Summary Panel
    total_trials = conn.execute("SELECT COUNT(*) FROM trials WHERE study_id=?", (study_id,)).fetchone()[0]
    visited_nodes = conn.execute("SELECT COUNT(*) FROM mcts_nodes n JOIN trials t ON t.trial_id=n.trial_id WHERE t.study_id=? AND n.n_visits > 0", (study_id,)).fetchone()[0]
    total_visits = conn.execute("SELECT SUM(n_visits) FROM mcts_nodes n JOIN trials t ON t.trial_id=n.trial_id WHERE t.study_id=?", (study_id,)).fetchone()[0]
    
    summary_text = (
        f"Total Trials: [bold cyan]{total_trials}[/bold cyan] | "
        f"Visited Nodes: [bold green]{visited_nodes}[/bold green] | "
        f"Total Simulation Visits: [bold yellow]{total_visits or 0}[/bold yellow]"
    )
    console.print(Panel(summary_text, title="Study Summary", border_style="blue"))

if __name__ == "__main__":
    main()
