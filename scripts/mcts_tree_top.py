#!/usr/bin/env python3
"""
MCTS Top Trees Viewer.
Displays the tree paths for the N-th best results in a study.
Usage: python scripts/mcts_tree_top.py --project <proj> --study <study> -N 1
"""

import argparse
import sqlite3
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from rich.console import Console
from rich.tree import Tree
from rich.panel import Panel
from rich.text import Text

# Add src to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

def resolve_db_path(project_name: str) -> Path:
    base_projects_dir = REPO_ROOT / "projects" / "kaggle"
    project_root = (base_projects_dir / project_name).resolve()
    
    if not project_root.exists():
        for p in base_projects_dir.iterdir():
            if p.is_dir() and p.name.lower() == project_name.lower():
                project_root = p.resolve()
                break
    
    db_path = project_root / "experiments" / "db" / "mcts.db"
    if not db_path.exists():
        db_path = Path("/mnt/mlarena") / "projects" / "kaggle" / project_name / "experiments" / "db" / "mcts.db"
    
    return db_path

def get_study_id(conn: sqlite3.Connection, study_name: str) -> Optional[int]:
    row = conn.execute("SELECT study_id FROM studies WHERE study_name=?", (study_name,)).fetchone()
    return row[0] if row else None

def get_direction(conn: sqlite3.Connection, study_id: int) -> int:
    row = conn.execute("SELECT direction FROM study_directions WHERE study_id=? AND objective=0", (study_id,)).fetchone()
    return row[0] if row else 1 # Default MINIMIZE

def get_nth_best_score(conn: sqlite3.Connection, study_id: int, direction: int, n: int) -> Optional[float]:
    order = "DESC" if direction == 2 else "ASC" # 2 is MAXIMIZE
    query = f"""
        SELECT DISTINCT tv.value
        FROM trials t
        JOIN trial_values tv ON tv.trial_id = t.trial_id
        WHERE t.study_id = ? AND t.state = 1
        ORDER BY tv.value {order}
        LIMIT 1 OFFSET ?
    """
    row = conn.execute(query, (study_id, n - 1)).fetchone()
    return row[0] if row else None

def get_trials_with_score(conn: sqlite3.Connection, study_id: int, score: float) -> List[int]:
    query = """
        SELECT t.trial_id
        FROM trials t
        JOIN trial_values tv ON tv.trial_id = t.trial_id
        WHERE t.study_id = ? AND t.state = 1 AND ABS(tv.value - ?) < 1e-10
    """
    rows = conn.execute(query, (study_id, score)).fetchall()
    return [r[0] for r in rows]

def get_node_info(conn: sqlite3.Connection, trial_id: int) -> Dict[str, Any]:
    from mlarena.modules.mcts.node import Action as MCTSAction
    
    query = """
        SELECT n.trial_id, n.parent_trial_id, t.number, tv.value as original_value
        FROM mcts_nodes n
        JOIN trials t ON t.trial_id = n.trial_id
        LEFT JOIN trial_values tv ON tv.trial_id = t.trial_id AND tv.objective = 0
        WHERE n.trial_id = ?
    """
    row = conn.execute(query, (trial_id,)).fetchone()
    if not row:
        return {}
    
    # Get action info from edge
    action_info = {"step": "root", "var": "root", "sid": 0}
    if row["parent_trial_id"] is not None:
        edge = conn.execute(
            "SELECT action_json FROM mcts_edges WHERE child_trial_id = ?", 
            (trial_id,)
        ).fetchone()
        if edge:
            act_raw = json.loads(edge[0])
            act = MCTSAction.from_record(act_raw)
            action_info = {
                "step": act.step_name,
                "var": act.variant_name,
                "sid": act.param_sample_id
            }
    elif row["number"] == 0:
        action_info = {"step": "baseline", "var": "fixed", "sid": 0}

    return {
        "trial_id": row["trial_id"],
        "parent_id": row["parent_trial_id"],
        "score": row["original_value"],
        "number": row["number"],
        "action": action_info
    }

def get_path_to_root(conn: sqlite3.Connection, trial_id: int) -> List[Dict[str, Any]]:
    path = []
    current_id = trial_id
    while current_id is not None:
        info = get_node_info(conn, current_id)
        if not info:
            break
        path.append(info)
        current_id = info["parent_id"]
    return path[::-1] # Reverse to get root -> leaf

def format_label(info: Dict[str, Any]) -> str:
    act = info["action"]
    score = info["score"]
    number = info["number"]
    score_str = f"{score:.5f}" if score is not None else "N/A"
    return f"T{number}:{act['step']}/{act['var']}/{act['sid']}/{score_str}"

def main():
    parser = argparse.ArgumentParser(description="View N-th Best MCTS Tree Path")
    parser.add_argument("--project", "-p", required=True, help="Project name")
    parser.add_argument("--study", "-s", required=True, help="Study name")
    parser.add_argument("-N", type=int, default=1, help="Rank of result to show (1=best, 2=second best, ... or -1, -2, -3 to support user's preferred notation)")
    
    args = parser.parse_args()
    # Support negative indices as ranks (e.g., -1 -> 1)
    rank = abs(args.N)
    
    console = Console()
    
    db_path = resolve_db_path(args.project)
    if not db_path.exists():
        console.print(f"[bold red]Error:[/bold red] Database not found at {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    study_id = get_study_id(conn, args.study)
    if not study_id:
        console.print(f"[bold red]Error:[/bold red] Study '{args.study}' not found.")
        sys.exit(1)

    direction = get_direction(conn, study_id)
    score = get_nth_best_score(conn, study_id, direction, rank)
    
    if score is None:
        console.print(f"[yellow]No result found for rank {args.N}.[/yellow]")
        return

    trial_ids = get_trials_with_score(conn, study_id, score)
    
    console.print(Panel(
        f"Project: [bold cyan]{args.project}[/]\nStudy: [bold cyan]{args.study}[/]\nRank: [bold green]{rank}[/] | Score: [bold yellow]{score:.6f}[/]",
        title="MCTS Top Result Search"
    ))

    for tid in trial_ids:
        path = get_path_to_root(conn, tid)
        if not path:
            continue
            
        tree = Tree(f"[bold magenta]Trial #{path[-1]['number']}[/]")
        curr_tree = tree
        for i, node in enumerate(path):
            label = format_label(node)
            style = "bold yellow" if i == len(path) - 1 else "white"
            # Add node to tree
            if i == 0:
                # root node is the tree label itself or first branch
                curr_tree = tree.add(Text(label, style=style))
            else:
                curr_tree = curr_tree.add(Text(label, style=style))
        
        console.print(tree)
        console.print()

if __name__ == "__main__":
    main()
