import sqlite3
import argparse
import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

def analyze_trend(project: str, window: int, study_name: str = None):
    console = Console()
    # Try local path first, then project-based path
    db_path = Path(f"projects/kaggle/{project}/experiments/db/mcts.db")
    if not db_path.exists():
        db_path = Path(f"experiments/db/mcts.db")
    
    if not db_path.exists():
        console.print(f"[bold red]Error:[/bold red] Database not found at projects/kaggle/{project}/experiments/db/mcts.db")
        return

    conn = sqlite3.connect(db_path)
    
    # Get available studies
    studies = pd.read_sql_query("SELECT study_id, study_name FROM studies", conn)
    
    if studies.empty:
        console.print("[bold red]Error:[/bold red] No studies found in database.")
        conn.close()
        return

    if study_name:
        selected_study = studies[studies["study_name"] == study_name]
        if selected_study.empty:
            console.print(f"[bold red]Error:[/bold red] Study '{study_name}' not found.")
            console.print("[bold yellow]Available studies:[/bold yellow]")
            for name in studies["study_name"]:
                console.print(f" - {name}")
            conn.close()
            return
        study_id = selected_study.iloc[0]["study_id"]
        study_display_name = study_name
    else:
        # Default to the latest study if not specified
        latest_study = studies.iloc[-1]
        study_id = latest_study["study_id"]
        study_display_name = latest_study["study_name"]
        if len(studies) > 1:
            console.print(f"[yellow]Multiple studies found. Defaulting to the latest: {study_display_name}[/yellow]")

    direction_row = conn.execute("SELECT direction FROM study_directions WHERE study_id=?", (int(study_id),)).fetchone()
    is_maximize = direction_row[0] == 2 # StudyDirection.MAXIMIZE = 2

    # Fetch trials and values
    query = """
        SELECT t.number, tv.value
        FROM trials t
        JOIN trial_values tv ON tv.trial_id = t.trial_id
        WHERE t.study_id = ? AND t.state = 1 -- COMPLETE
        ORDER BY t.number ASC
    """
    df = pd.read_sql_query(query, conn, params=(int(study_id),))
    conn.close()

    if df.empty:
        console.print(f"[yellow]No completed trials found for study: {study_display_name}.[/yellow]")
        return

    # Calculate moving average and best so far
    # The 'window' parameter controls both the smoothing and the table step
    df['moving_avg'] = df['value'].rolling(window=window, min_periods=1).mean()
    
    if is_maximize:
        df['best_so_far'] = df['value'].cummax()
    else:
        df['best_so_far'] = df['value'].cummin()

    # Visualization
    table = Table(title=f"MCTS Trend: {study_display_name} (Window/Step: {window})")
    table.add_column("Trial Range", justify="right", style="cyan")
    table.add_column("Avg Val", justify="right", style="magenta")
    table.add_column("Best Val", justify="right", style="green")
    table.add_column("Trend", justify="left")

    # Generate indices for the table rows based on the window
    for i in range(window - 1, len(df), window):
        chunk_start = max(0, i - window + 1)
        avg_val = df['moving_avg'].iloc[i]
        best_val = df['best_so_far'].iloc[i]
        
        # Trend indicator based on moving average change
        if i >= window:
            prev_avg = df['moving_avg'].iloc[i - window]
            if avg_val > prev_avg: trend_icon = "↗" if is_maximize else "↘"
            elif avg_val < prev_avg: trend_icon = "↘" if is_maximize else "↗"
            else: trend_icon = "→"
        else:
            trend_icon = "•"
            
        table.add_row(
            f"{chunk_start:03d}-{i:03d}",
            f"{avg_val:.5f}",
            f"{best_val:.5f}",
            trend_icon
        )

    # Always include last trial if not already included
    last_idx = len(df) - 1
    if last_idx % window != (window - 1):
        chunk_start = (last_idx // window) * window
        avg_val = df['moving_avg'].iloc[last_idx]
        best_val = df['best_so_far'].iloc[last_idx]
        
        # Trend indicator based on moving average change
        prev_idx = max(0, chunk_start - 1)
        prev_avg = df['moving_avg'].iloc[prev_idx]
        if avg_val > prev_avg: trend_icon = "↗" if is_maximize else "↘"
        elif avg_val < prev_avg: trend_icon = "↘" if is_maximize else "↗"
        else: trend_icon = "→"
            
        table.add_row(
            f"{chunk_start:03d}-{last_idx:03d}",
            f"{avg_val:.5f}",
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
    summary_txt += f"Overall Learning Trend: [{color}]{diff:+.5f}[/{color}]\n"
    summary_txt += f"Best Val Achieved: [bold green]{df['best_so_far'].iloc[-1]:.5f}[/bold green]"
    
    console.print(Panel(summary_txt, title="Summary"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze MCTS learning trend from database.")
    parser.add_argument("-p", "--project", required=True, help="Project name (e.g., titanic)")
    parser.add_argument("-s", "--study-name", help="Optuna study name")
    parser.add_argument("-w", "--window", type=int, default=100, help="Window size for moving average and table step")
    args = parser.parse_args()
    
    analyze_trend(args.project, args.window, args.study_name)
