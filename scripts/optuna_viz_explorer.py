#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich.live import Live

# Ensure we have the plotting logic available (importing from our previous script)
# Or just copy the core logic for self-containment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import optuna
from optuna.visualization.matplotlib import (
    plot_optimization_history, 
    plot_param_importances, 
    plot_parallel_coordinate,
    plot_slice,
    plot_timeline,
    plot_edf,
    plot_contour
)
from optuna.importance import MeanDecreaseImpurityImportanceEvaluator

console = Console()

PLOT_TYPES = {
    "1": ("History", "history", "Optimization progress over time"),
    "2": ("Importance", "importance", "Which parameters matter most (MDI)"),
    "3": ("TPE Density", "tpe", "Garbiaste wykresy (umysł samplera)"),
    "4": ("Box Plots", "box", "Distribution by categorical variants"),
    "5": ("Parallel", "parallel", "High-dimensional coordinate paths"),
    "6": ("Slice", "slice", "Individual parameter impact dots"),
    "7": ("Contour", "contour", "2D interaction map (islands)"),
    "8": ("Rank", "rank", "Best score progression step-chart"),
    "9": ("Timeline", "timeline", "Trial durations and overlaps"),
    "0": ("EDF", "edf", "Empirical Distribution Function (success probability)"),
    "q": ("Quit", "quit", "Exit the explorer")
}

def get_top_params(study, n=5):
    df = study.trials_dataframe()
    df = df[df.state == "COMPLETE"]
    param_cols = [c for c in df.columns if c.startswith('params_')]
    corrs = []
    for col in param_cols:
        try:
            c = pd.to_numeric(df[col], errors='coerce').corr(df['value'])
            if not pd.isna(c):
                corrs.append((col.replace('params_', ''), abs(c)))
        except: continue
    return [name for name, _ in sorted(corrs, key=lambda x: x[1], reverse=True)[:n]]

def show_image(path):
    # Clear any existing images first
    try:
        subprocess.run(["kitty", "+kitten", "icat", "--clear"], check=False)
    except:
        pass

    console.print(f"\n[dim]Rendering image via kitty icat...[/dim]")
    try:
        # Using --hold or proper alignment to ensure it's visible but clearable
        subprocess.run(["kitty", "+kitten", "icat", "--align", "left", str(path)], check=True)
    except:
        console.print(f"[red]Error: Could not display image via kitty icat.[/red]")

def generate_plot(study, plot_type, db_path):
    tmp_path = Path("/tmp/optuna_viz_tui.png")
    plt.figure(figsize=(18, 10))
    
    try:
        if plot_type == "history":
            plot_optimization_history(study)
        elif plot_type == "importance":
            plot_param_importances(study, evaluator=MeanDecreaseImpurityImportanceEvaluator())
        elif plot_type == "rank":
            df = study.trials_dataframe()
            df = df[df.state == "COMPLETE"].copy()
            df['best_so_far'] = df['value'].cummax()
            plt.step(df['number'], df['best_so_far'], where='post', color='red', linewidth=3)
            plt.scatter(df['number'], df['value'], alpha=0.3, color='blue')
            plt.title("Best Score Progression")
        elif plot_type == "parallel":
            top = get_top_params(study, 8)
            plot_parallel_coordinate(study, params=top)
        elif plot_type == "slice":
            top = get_top_params(study, 4)
            plot_slice(study, params=top)
        elif plot_type == "timeline":
            plot_timeline(study)
        elif plot_type == "edf":
            plot_edf(study)
        elif plot_type == "box":
            # Simple box plot for top cat param
            df = study.trials_dataframe()
            cat_cols = [c for c in df.columns if c.startswith('params_') and df[c].nunique() < 10 and df[c].nunique() > 1]
            if cat_cols:
                import seaborn as sns
                sns.boxplot(data=df, x=cat_cols[0], y='value')
                plt.title(f"Box Plot: {cat_cols[0]}")
            else:
                plt.text(0.5, 0.5, "No categorical params found")
        elif plot_type == "tpe":
            import seaborn as sns
            df = study.trials_dataframe()
            df = df[df.state == "COMPLETE"].copy()
            top_p = get_top_params(study, 1)[0]
            threshold = df['value'].quantile(0.75)
            df['tpe_class'] = df['value'].apply(lambda x: 'Good' if x >= threshold else 'Bad')
            sns.kdeplot(data=df, x=f"params_{top_p}", hue='tpe_class', fill=True, common_norm=False)
            plt.title(f"TPE Density: {top_p}")
        elif plot_type == "contour":
            top = get_top_params(study, 2)
            plot_contour(study, params=top)

        plt.tight_layout()
        plt.savefig(tmp_path)
        plt.close()
        show_image(tmp_path)
    except Exception as e:
        console.print(f"[bold red]Plotting Error:[/bold red] {e}")

def make_menu():
    table = Table(title="Optuna Navigator", box=None, show_header=False)
    table.add_column("Key", style="cyan", justify="right")
    table.add_column("Name", style="bold white")
    table.add_column("Description", style="dim")
    
    for key, (name, _, desc) in PLOT_TYPES.items():
        table.add_row(key, name, desc)
    
    return Panel(table, title="[bold magenta]Choose Visualization[/bold magenta]", border_style="bright_blue")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to Optuna SQLite")
    args = parser.parse_args()
    
    db_path = Path(args.db).resolve()
    storage = f"sqlite:///{db_path}"
    
    try:
        study = optuna.load_study(study_name=None, storage=storage)
    except Exception as e:
        console.print(f"[red]Error loading study: {e}[/red]")
        return

    while True:
        os.system('clear')
        console.print(Panel(f"DB: [green]{db_path}[/green] | Study: [bold cyan]{study.study_name}[/bold cyan] | Trials: [yellow]{len(study.trials)}[/yellow]"))
        console.print(make_menu())
        
        choice = Prompt.ask("Select option", choices=list(PLOT_TYPES.keys()), default="1")
        
        if choice == 'q':
            try:
                subprocess.run(["kitty", "+kitten", "icat", "--clear"], check=False)
            except:
                pass
            console.print("[yellow]Happy tuning! Goodbye.[/yellow]")
            break
            
        _, p_type, _ = PLOT_TYPES[choice]
        
        # Action
        generate_plot(study, p_type, db_path)
        
        input("\nPress Enter to return to menu...")

if __name__ == "__main__":
    main()
