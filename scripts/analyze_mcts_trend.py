import sqlite3
import argparse
import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

def analyze_trend(project: str, window: int, step_size: int = None):
    console = Console()
    # Try local path first, then project-based path
    db_path = Path(f"projects/kaggle/{project}/experiments/db/mcts.db")
    if not db_path.exists():
        db_path = Path(f"experiments/db/mcts.db")
    
    if not db_path.exists():
        console.print(f"[bold red]Error:[/bold red] Database not found at projects/kaggle/{project}/experiments/db/mcts.db")
        return

    conn = sqlite3.connect(db_path)
    
    # Get study and direction
    study_info = conn.execute("SELECT study_id, study_name FROM studies LIMIT 1").fetchone()
    if not study_info:
        console.print("[bold red]Error:[/bold red] No studies found in database.")
        conn.close()
        return
    
    study_id, study_name = study_info
    direction_row = conn.execute("SELECT direction FROM study_directions WHERE study_id=?", (study_id,)).fetchone()
    is_maximize = direction_row[0] == 2 # StudyDirection.MAXIMIZE = 2

    # Fetch trials and scores
    query = """
        SELECT t.number, tv.value
        FROM trials t
        JOIN trial_values tv ON tv.trial_id = t.trial_id
        WHERE t.study_id = ? AND t.state = 1 -- COMPLETE
        ORDER BY t.number ASC
    """
    df = pd.read_sql_query(query, conn, params=(study_id,))
    conn.close()

    if df.empty:
        console.print("[yellow]No completed trials found.[/yellow]")
        return

    # Calculate moving average and best so far
    df['moving_avg'] = df['value'].rolling(window=window, min_periods=1).mean()
    
    if is_maximize:
        df['best_so_far'] = df['value'].cummax()
    else:
        df['best_so_far'] = df['value'].cummin()

    # Visualization
    table = Table(title=f"MCTS Trend Analysis: {study_name} (Window: {window}, Step: {step_size or 'Auto'})")
    table.add_column("Trial Range", justify="right", style="cyan")
    table.add_column("Avg Score", justify="right", style="magenta")
    table.add_column("Best Score", justify="right", style="green")
    table.add_column("Trend", justify="left")

    # If step_size is provided, use it. Otherwise, divide by 20 for readability.
    if step_size:
        step = step_size
    else:
        step = max(1, len(df) // 20)
    
    # Generate indices based on the requested step
    indices = []
    curr = step - 1
    while curr < len(df) - 1:
        indices.append(curr)
        curr += step
    indices.append(len(df) - 1) # Always include last trial

    for idx, i in enumerate(indices):
        chunk_start = indices[idx-1] + 1 if idx > 0 else 0
        chunk_end = i
        
        last_val = df['moving_avg'].iloc[i]
        best_val = df['best_so_far'].iloc[i]
        
        # Simple ASCII trend indicator
        if idx > 0:
            prev_val = df['moving_avg'].iloc[indices[idx-1]]
            if last_val > prev_val: trend_icon = "↗" if is_maximize else "↘"
            elif last_val < prev_val: trend_icon = "↘" if is_maximize else "↗"
            else: trend_icon = "→"
        else:
            trend_icon = "•"
            
        table.add_row(
            f"{chunk_start:03d}-{chunk_end:03d}",
            f"{last_val:.5f}",
            f"{best_val:.5f}",
            trend_icon
        )

    console.print(table)
    
    # Final Summary
    first_stable_idx = min(window - 1, len(df) - 1)
    first_avg = df['moving_avg'].iloc[first_stable_idx]
    last_avg = df['moving_avg'].iloc[-1]
    diff = last_avg - first_avg
    
    summary_txt = f"Analysis of {len(df)} trials.\n"
    summary_txt += f"First Window Avg (at trial {first_stable_idx}): {first_avg:.5f}\n"
    summary_txt += f"Latest Window Avg: {last_avg:.5f}\n"
    
    color = "green" if (diff > 0 and is_maximize) or (diff < 0 and not is_maximize) else "red"
    summary_txt += f"Overall Learning Trend: [{color}]{diff:+.5f}[/{color}]"
    
    console.print(Panel(summary_txt, title="Summary"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze MCTS learning trend from database.")
    parser.add_argument("-p", "--project", required=True, help="Project name (e.g., titanic)")
    parser.add_argument("-w", "--window", type=int, default=100, help="Moving average window size")
    parser.add_argument("-s", "--step", type=int, default=None, help="Table row step size (number of trials per row)")
    args = parser.parse_args()
    
    analyze_trend(args.project, args.window, args.step)
