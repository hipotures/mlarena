#!/usr/bin/env python
import pandas as pd
import logging
from pathlib import Path
from autogluon.tabular import TabularPredictor
from rich.console import Console
from rich.table import Table

# Paths
PROJECT_DIR = Path("projects/kaggle/playground-series-s6e1")
ORACLE_DIR = PROJECT_DIR / "experiments/oracle"
CSV_PATH = ORACLE_DIR / "mcts_oracle.csv"

def main():
    console = Console()
    
    if not ORACLE_DIR.exists():
        console.print(f"[red]Oracle directory not found: {ORACLE_DIR}[/red]")
        return

    # 1. Load Schema to ensure correct columns
    console.print(f"[dim]Loading schema from {CSV_PATH}...[/dim]")
    if CSV_PATH.exists():
        schema_cols = pd.read_csv(CSV_PATH, nrows=0).columns.tolist()
        ignore_cols = {'is_improvement', 'child_score', 'delta_score', 'child_id', 'parent_id'}
        feature_cols = [c for c in schema_cols if c not in ignore_cols]
    else:
        console.print("[red]Schema CSV not found! Cannot ensure correct features.[/red]")
        return

    # 2. Define Test Cases
    # Format: (Description, Parent_Score, Prev_Group, Prev_Variant, Curr_Group, Curr_Variant)
    test_cases = [
        # Case 1: The Zero Impact (User Request)
        ("Imputer (Zero)", -8.77935, "baseline", "fixed", "imputer", "most_frequent"),
        
        # Case 2: The Disaster (-11.43)
        # Context: Depth=8 -> 9. Prev was likely clustering or rank_post based on logs
        ("Random Under (Fail)", -8.78895, "rank_features_post", "group_percentile", "imbalance_handler", "random_under"),
        
        # Case 3: The False Positive (-0.014)
        # Context: Oracle loved this (51%), but it failed.
        ("K-Means (False Pos)", -8.80000, "scaler", "robust", "clustering_features", "kmeans_distances"),
        
        # Case 4: A Good Step (Control Group)
        ("Quantile Normal (Good)", -8.77935, "baseline", "fixed", "scaler", "quantile_normal"),
    ]

    rows = []
    for desc, p_score, p_grp, p_var, c_grp, c_var in test_cases:
        row = {col: 0 for col in feature_cols} # Initialize with 0
        
        # Fill knowns
        row['parent_score'] = p_score
        row['depth'] = 1 if p_grp == 'baseline' else 5 # Approximate depth
        row['parent_visits'] = 10
        
        # Actions
        row['action_group'] = c_grp
        row['action_variant'] = c_var
        row['prev_action_group'] = p_grp
        row['prev_action_variant'] = p_var
        
        # Configs (dummies, as model mainly learns from group/variant names in tree models)
        # In a real scenario, we'd parse the full JSON config
        
        row['_desc'] = desc # Metadata for display
        rows.append(row)

    df = pd.DataFrame(rows)
    
    # Reindex to enforce schema
    X = df.reindex(columns=feature_cols, fill_value=0)

    # 3. Load Model & Predict
    console.print(f"[dim]Loading Oracle from {ORACLE_DIR}...[/dim]")
    try:
        predictor = TabularPredictor.load(str(ORACLE_DIR))
    except Exception as e:
        console.print(f"[red]Failed to load predictor: {e}[/red]")
        return

    console.print("[bold]Running Inference...[/bold]")
    probs = predictor.predict_proba(X)
    
    # Handle positive class label (usually 1)
    pos_label = 1
    if 1 not in probs.columns:
        pos_label = predictor.positive_class
    
    p_values = probs[pos_label]

    # 4. Display Results
    table = Table(title="Oracle Predictions on Specific Cases")
    table.add_column("Scenario", style="cyan")
    table.add_column("Transition", style="dim")
    table.add_column("Prob (Improve)", justify="right", style="bold magenta")
    table.add_column("Oracle Decision", justify="right")

    threshold = 0.01 # Our current config
    
    for i, p in enumerate(p_values):
        desc = df.loc[i, '_desc']
        trans = f"{df.loc[i, 'prev_action_group']} -> {df.loc[i, 'action_group']}:{df.loc[i, 'action_variant']}"
        
        decision = "[green]KEEP[/green]" if p >= threshold else "[red]PRUNE[/red]"
        
        table.add_row(
            desc,
            trans,
            f"{p:.4f}",
            decision
        )

    console.print(table)
    console.print(f"[dim]Current Threshold: {threshold}[/dim]")

if __name__ == "__main__":
    main()
