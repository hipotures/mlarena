#!/usr/bin/env python3
"""
MCTS Stat Rehydration Script.
Recalculates n_visits, value_sum, and value_best from existing trial results and tree structure.
Usage: python scripts/mcts_rehydrate_stats.py --project playground-series-s6e1 --study s6e1_001
"""

import argparse
import sqlite3
import json
import sys
from pathlib import Path

# Add src to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

def main():
    parser = argparse.ArgumentParser(description="Rehydrate MCTS stats from history")
    parser.add_argument("--project", "-p", required=True, help="Project name")
    parser.add_argument("--study", "-s", required=True, help="Study name")
    parser.add_argument("--dry-run", action="store_true", help="Don't commit changes to DB")
    
    args = parser.parse_args()
    
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
            print(f"Error: Database not found at {db_path}")
            sys.exit(1)

    print(f"Opening Database: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # 1. Get Study and Direction
    study = conn.execute("SELECT study_id FROM studies WHERE study_name=?", (args.study,)).fetchone()
    if not study:
        print(f"Error: Study '{args.study}' not found.")
        sys.exit(1)
    study_id = study[0]
    
    direction_row = conn.execute("SELECT direction FROM study_directions WHERE study_id=?", (study_id,)).fetchone()
    maximize = (direction_row[0] == 2) # StudyDirection.MAXIMIZE = 2
    print(f"Study ID: {study_id}, Direction: {'MAXIMIZE' if maximize else 'MINIMIZE'}")

    # 2. Get all successful evaluations
    query = """
        SELECT e.trial_id, t.number, e.value
        FROM mcts_evaluations e
        JOIN trials t ON t.trial_id = e.trial_id
        WHERE t.study_id = ? AND e.status = 'COMPLETE' AND e.value IS NOT NULL
        ORDER BY t.number ASC
    """
    evals = conn.execute(query, (study_id,)).fetchall()
    print(f"Found {len(evals)} completed evaluations to process.")

    # 3. Map tree structure (child -> parent)
    edge_rows = conn.execute("""
        SELECT parent_trial_id, child_trial_id 
        FROM mcts_edges e
        JOIN trials t ON t.trial_id = e.child_trial_id
        WHERE t.study_id = ?
    """, (study_id,)).fetchall()
    
    child_to_parent = {row['child_trial_id']: row['parent_trial_id'] for row in edge_rows}
    print(f"Mapped {len(child_to_parent)} tree edges.")

    # 4. Reset current stats in memory for recalculation
    node_stats = {} # trial_id -> {'visits': 0, 'sum': 0.0, 'best': -inf/inf}
    
    # Pre-populate node_stats from all nodes in study
    all_nodes = conn.execute("SELECT n.trial_id FROM mcts_nodes n JOIN trials t ON t.trial_id=n.trial_id WHERE t.study_id=?", (study_id,)).fetchall()
    for n in all_nodes:
        node_stats[n[0]] = {
            'visits': 0, 
            'sum': 0.0, 
            'best': -float('inf') if maximize else float('inf')
        }

    # 5. Process each evaluation (Backpropagate)
    for ev in evals:
        tid = ev['trial_id']
        val = ev['value']
        
        # Traverse up the tree
        curr = tid
        while curr is not None:
            if curr not in node_stats:
                # Should not happen if DB is consistent, but let's be safe
                node_stats[curr] = {'visits': 0, 'sum': 0.0, 'best': -float('inf') if maximize else float('inf')}
            
            stats = node_stats[curr]
            stats['visits'] += 1
            stats['sum'] += val
            if maximize:
                if val > stats['best']: stats['best'] = val
            else:
                if val < stats['best']: stats['best'] = val
            
            # Move to parent
            curr = child_to_parent.get(curr)

    # 6. Update Database
    if args.dry_run:
        print("\nDRY RUN: No changes will be saved.")
    else:
        print("\nUpdating database...")
        for tid, stats in node_stats.items():
            if stats['visits'] > 0:
                conn.execute(
                    "UPDATE mcts_nodes SET n_visits=?, value_sum=?, value_best=? WHERE trial_id=?",
                    (stats['visits'], stats['sum'], stats['best'], tid)
                )
        conn.commit()
        print("Success! All stats rehydrated.")

    # Show a few top nodes as verification
    print("\nTop 5 nodes after rehydration:")
    sorted_nodes = sorted(node_stats.items(), key=lambda x: x[1]['visits'], reverse=True)
    for tid, s in sorted_nodes[:5]:
        print(f"Node {tid}: Visits={s['visits']}, Best={s['best']:.4f}, Sum={s['sum']:.4f}")

if __name__ == "__main__":
    main()
