#!/usr/bin/env python
import sqlite3
import pandas as pd
import argparse
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table

def parse_action(json_str):
    try:
        data = json.loads(json_str)
        return f"{data.get('group_name', '?')}:{data.get('variant', '?')}"
    except:
        return "unknown"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="projects/kaggle/playground-series-s6e1/experiments/db/mcts.db")
    args = parser.parse_args()

    console = Console()
    db_path = Path(args.db)
    
    if not db_path.exists():
        console.print(f"[red]DB not found: {db_path}[/red]")
        return

    conn = sqlite3.connect(db_path)
    
    # 1. Get Transitions with Scores (Parent -> Child)
    # We use mcts_evaluations to get the BEST COMPLETE score for each trial
    query = """
    WITH best_evals AS (
        SELECT trial_id, MAX(value) as val 
        FROM mcts_evaluations 
        WHERE status='COMPLETE' 
        GROUP BY trial_id
    )
    SELECT 
        e.action_json,
        val_child.val - val_parent.val as delta
    FROM mcts_edges e
    JOIN best_evals val_parent ON e.parent_trial_id = val_parent.trial_id
    JOIN best_evals val_child ON e.child_trial_id = val_child.trial_id
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        console.print("[yellow]No valid transitions found (yet).[/yellow]")
        return

    # Parse Actions
    df['action_desc'] = df['action_json'].apply(parse_action)
    
    # Categorize Delta
    df['type'] = 'ZERO'
    df.loc[df['delta'] > 0.00001, 'type'] = 'IMPROVE'
    df.loc[df['delta'] < -0.00001, 'type'] = 'WORSE'
    
    # Aggregate
    stats = df.groupby('action_desc').agg(
        count=('delta', 'count'),
        n_impr=('type', lambda x: (x == 'IMPROVE').sum()),
        n_zero=('type', lambda x: (x == 'ZERO').sum()),
        n_worse=('type', lambda x: (x == 'WORSE').sum()),
        mean_delta=('delta', 'mean')
    ).reset_index()
    
    # Add Ratios
    stats['pct_zero'] = (stats['n_zero'] / stats['count']) * 100
    stats['pct_impr'] = (stats['n_impr'] / stats['count']) * 100
    
    # Sort by uselessness (most Zeros)
    stats = stats.sort_values('pct_zero', ascending=False)
    
    # Display
    table = Table(title=f"Step Impact Analysis (N={len(df)} transitions)")
    table.add_column("Action (Group:Variant)", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Impr", justify="right", style="green")
    table.add_column("Zero", justify="right", style="yellow")
    table.add_column("Worse", justify="right", style="red")
    table.add_column("% Zero", justify="right")
    table.add_column("Mean Delta", justify="right")

    for _, row in stats.iterrows():
        pct_zero_style = "bold red" if row['pct_zero'] > 80 else "white"
        table.add_row(
            row['action_desc'],
            str(row['count']),
            str(row['n_impr']),
            str(row['n_zero']),
            str(row['n_worse']),
            f"[{pct_zero_style}]{row['pct_zero']:.1f}%[/{pct_zero_style}]",
            f"{row['mean_delta']:.5f}"
        )

    console.print(table)

if __name__ == "__main__":
    main()
