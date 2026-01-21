#!/usr/bin/env python
import sqlite3
import pandas as pd
import json
import argparse
import random
import warnings
from pathlib import Path
from autogluon.tabular import TabularPredictor
from rich.console import Console
from rich.table import Table

console = Console()
warnings.filterwarnings("ignore")

def info(message: str) -> None:
    console.print(message)

def load_best_parent(db_path: Path, study_name: str):
    info(f"Connecting to DB: {db_path} (study: {study_name})")
    conn = sqlite3.connect(db_path)
    
    cur = conn.execute("SELECT study_id FROM studies WHERE study_name = ?", (study_name,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return None
    study_id = row[0]

    query = """
    SELECT e.trial_id, e.value, n.depth
    FROM mcts_evaluations e
    JOIN mcts_nodes n ON e.trial_id = n.trial_id
    WHERE n.study_id = ? AND e.status = 'COMPLETE' AND e.value IS NOT NULL
    ORDER BY e.value DESC
    LIMIT 1
    """
    row = conn.execute(query, (study_id,)).fetchone()
    
    if not row:
        conn.close()
        return None

    trial_id, value, depth = row
    
    edge_query = "SELECT action_json FROM mcts_edges WHERE child_trial_id = ?"
    edge_row = conn.execute(edge_query, (trial_id,)).fetchone()
    conn.close()
    
    prev_action_json = edge_row[0] if edge_row else None
    
    info(f"Best Parent Found: Trial {trial_id} | Score: {value:.6f} | Depth: {depth}")
    return trial_id, value, depth, prev_action_json

def parse_action(action_json_str, prefix=""):
    if not action_json_str or pd.isna(action_json_str):
        return {}
    try:
        data = json.loads(action_json_str)
        flat = {}
        group = data.get("group_name", "unknown")
        variant = data.get("variant", "unknown")
        flat[f"{prefix}action_group"] = group
        flat[f"{prefix}action_variant"] = variant
        
        # Flatten config
        config = data.get("config", {})
        for k, v in config.items():
            # Only flat params usually matter for oracle
            if isinstance(v, (list, dict)): 
                 pass # Oracle csv might have flat columns already?
            else:
                 pass
        return flat
    except:
        return {}
        
# Re-use the parser logic from oracle training script to ensure 'prev_' cols match
def parse_action_full(action_json_str, prefix=""):
    if not action_json_str or pd.isna(action_json_str):
        return {}
    try:
        data = json.loads(action_json_str)

        flat = {}
        group = data.get("group_name", "unknown")
        variant = data.get("variant", "unknown")

        flat[f"{prefix}action_group"] = group
        flat[f"{prefix}action_variant"] = variant

        config = data.get("config", {})
        for k, v in config.items():
            key = f"{prefix}{group}_{k}"
            if isinstance(v, (list, dict)):
                flat[key] = json.dumps(v, sort_keys=True)
            else:
                flat[key] = v

        return flat
    except Exception:
        return {}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="playground-series-s6e1", help="Project slug")
    parser.add_argument("--count", type=int, default=1000, help="Number of historical samples to remix")
    parser.add_argument("--out", default="remix_candidates.csv", help="Output CSV file")
    parser.add_argument("--study", default="s6e1_008_xgb_gpu", help="Study name")
    args = parser.parse_args()

    # Paths
    project_dir = Path(f"projects/kaggle/{args.project}")
    exp_dir = project_dir / "experiments"
    db_path = exp_dir / "db" / "mcts.db"
    model_dir = exp_dir / "oracle" / "model"
    oracle_csv = model_dir / "mcts_oracle.csv"
    
    if not model_dir.exists() or not oracle_csv.exists():
        print("Model or data missing.")
        return

    # 1. Load Training Data (Source of "Actions")
    info(f"Loading history from {oracle_csv}...")
    hist_df = pd.read_csv(oracle_csv)
    
    if len(hist_df) == 0:
        print("Empty history.")
        return

    # 2. Load Best Parent (Target Context)
    res = load_best_parent(db_path, args.study)
    if not res:
        print("No parent found.")
        return
    parent_id, parent_score, depth, prev_action_json = res

    # 3. Prepare "Remixed" DataFrame using Conditional Shuffle (Within-Variant)
    info(f"Generating {args.count} new candidates via Conditional Shuffle...")
    
    # Identify parameter columns (exclude context/meta)
    exclude_cols = {"parent_score", "depth", "delta_score", "is_improvement", "prev_duration", "prob_improvement", "action_group", "action_variant"}
    param_cols = [c for c in hist_df.columns if c not in exclude_cols and not c.startswith("prev_")]
    
    # Group history by (group, variant) to preserve structure
    grouped = hist_df.groupby(["action_group", "action_variant"])
    variants_keys = list(grouped.groups.keys())
    # Weights for sampling based on frequency
    weights = [len(grouped.get_group(k)) for k in variants_keys]
    
    # Pre-calculate value pools for each variant and column
    # Structure: pools[(group, var)][col_name] = [val1, val2, ...]
    pools = {}
    for key in variants_keys:
        subset = grouped.get_group(key)
        pools[key] = {}
        for col in param_cols:
            vals = subset[col].dropna().tolist()
            if vals:
                pools[key][col] = vals

    # Initialize result dictionary
    new_data = {c: [None] * args.count for c in hist_df.columns if c not in exclude_cols and not c.startswith("prev_")}
    new_data["action_group"] = []
    new_data["action_variant"] = []
    
    # Generate batch of variant choices
    choices = random.choices(variants_keys, weights=weights, k=args.count)
    
    # Fill data
    # To do this efficiently, we iterate by variant choice count
    from collections import Counter
    counts = Counter(choices)
    
    # Lists to build columns
    # Re-initialize to simple lists for speed
    col_builders = {c: [] for c in param_cols}
    group_list = []
    variant_list = []
    
    for (grp, var), count in counts.items():
        key = (grp, var)
        group_list.extend([grp] * count)
        variant_list.extend([var] * count)
        
        variant_pool = pools[key]
        for col in param_cols:
            if col in variant_pool:
                # Sample 'count' values from the pool for this column
                sampled = random.choices(variant_pool[col], k=count)
                col_builders[col].extend(sampled)
            else:
                col_builders[col].extend([None] * count)

    # Assemble DataFrame
    new_data = pd.DataFrame(col_builders)
    new_data["action_group"] = group_list
    new_data["action_variant"] = variant_list
    
    # Shuffle the resulting dataframe to mix the blocks
    new_data = new_data.sample(frac=1).reset_index(drop=True)

    # Add Context columns (fixed)
    new_data["parent_score"] = parent_score
    new_data["depth"] = depth + 1
    new_data["prev_duration"] = 0.0
    
    # Prev context (fixed)
    # 1. Initialize ALL prev_ columns from history with None
    prev_hist_cols = [c for c in hist_df.columns if c.startswith("prev_")]
    for c in prev_hist_cols:
        new_data[c] = None

    # 2. Overwrite with actual parent values
    prev_cols = parse_action_full(prev_action_json, prefix="prev_")
    for k, v in prev_cols.items():
        if k in new_data.columns:
             new_data[k] = v
        else:
             # If column didn't exist in history (rare), we can add it or ignore
             new_data[k] = v
             
    # Deduplicate strictly on parameter columns + action info
    # (ignoring context cols which are identical for all)
    # We convert to tuple to check uniqueness quickly if needed, but pandas drop_duplicates is fine.
    # Note: we should deduplicate BEFORE prediction to save time, or AFTER to filter results?
    # Doing it BEFORE is more efficient.
    
    candidates = new_data.drop_duplicates(subset=param_cols + ["action_group", "action_variant"])
    info(f"Unique candidates after shuffle: {len(candidates)}")
        
    # 4. Predict
    info("Loading Predictor...")
    predictor = TabularPredictor.load(str(model_dir))
    
    info("Predicting...")
    # Ensure columns match (candidates already has correct columns since it came from training data)
    # Just need to handle any discrepancies if parsing logic changed slightly, but here we just updated values.
    
    if predictor.problem_type == "binary":
        probs = predictor.predict_proba(candidates)
        # Find positive class
        pos_label = 1
        if pos_label in probs.columns:
            candidates["prob_improvement"] = probs[pos_label]
        else:
            candidates["prob_improvement"] = probs.iloc[:, -1]
    else:
        candidates["prob_improvement"] = predictor.predict(candidates)
        
    # 5. Sort and Show
    candidates = candidates.sort_values("prob_improvement", ascending=False)
    
    table = Table(title=f"Top 10 Remixed Candidates (Parent Score: {parent_score:.6f})")
    cols = ["prob_improvement", "action_group", "action_variant"] 
    # Add a few interesting config cols
    extra = [c for c in candidates.columns if c not in cols and not c.startswith("prev_") and c not in ["parent_score", "depth", "delta_score", "is_improvement"]][:4]
    cols.extend(extra)
    
    for c in cols:
        table.add_column(c)
        
    for _, row in candidates.head(10).iterrows():
        vals = [str(row[c])[:20] for c in cols]
        table.add_row(*vals)
        
    console.print(table)
    candidates.to_csv(args.out, index=False)
    info(f"Saved to {args.out}")

if __name__ == "__main__":
    main()
