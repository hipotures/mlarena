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

# ML Arena imports
from mlarena.modules.mcts.space import SuperChainActionSpace
from mlarena.modules.mcts.sampler import ParameterSampler
from mlarena.modules.mcts.node import PipelineState, Action

console = Console()
warnings.filterwarnings("ignore")

def info(message: str) -> None:
    console.print(message)

def warn(message: str) -> None:
    console.print(f"[yellow]Warning:[/yellow] {message}")

def err(message: str) -> None:
    console.print(f"[red]Error:[/red] {message}")

# --- Helpers ---

def parse_action_full(action_json_str, prefix=""):
    """Parses action JSON into flat dictionary for DataFrame."""
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
            # Convert lists/dicts to string representation if needed, 
            # though usually params are scalars.
            if isinstance(v, (list, dict)):
                flat[key] = json.dumps(v, sort_keys=True)
            else:
                flat[key] = v
        return flat
    except Exception:
        return {}

def flatten_config(action_dict, prefix=""):
    """Flattens a generated action dictionary (not JSON string) for DataFrame."""
    flat = {}
    group = action_dict.get("group_name", "unknown")
    variant = action_dict.get("variant", "unknown")
    
    flat[f"{prefix}action_group"] = group
    flat[f"{prefix}action_variant"] = variant
    
    config = action_dict.get("config", {})
    for k, v in config.items():
        key = f"{prefix}{group}_{k}"
        if isinstance(v, (list, dict)):
            flat[key] = json.dumps(v, sort_keys=True)
        else:
            flat[key] = v
    return flat

def load_best_parent(db_path: Path, study_name: str):
    """Finds the best COMPLETED trial and reconstructs its FULL history."""
    info(f"Connecting to DB: {db_path} (study: {study_name})")
    conn = sqlite3.connect(db_path)
    
    # 1. Get study_id
    cur = conn.execute("SELECT study_id FROM studies WHERE study_name = ?", (study_name,))
    row = cur.fetchone()
    if not row:
        conn.close()
        warn(f"Study '{study_name}' not found in DB.")
        return None
    study_id = row[0]

    # 2. Find best trial
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
    info(f"Best Parent Found: Trial {trial_id} | Score: {value:.6f} | Depth: {depth}")

    # 3. Reconstruct history
    used_groups = {}
    curr_id = trial_id
    last_step_index = -1
    prev_action_json = None

    while True:
        edge_query = "SELECT parent_trial_id, action_json FROM mcts_edges WHERE child_trial_id = ?"
        edge_row = conn.execute(edge_query, (curr_id,)).fetchone()
        
        if not edge_row:
            break # Root
            
        parent_id_node, action_json = edge_row
        
        if curr_id == trial_id:
            prev_action_json = action_json
        
        try:
            act = json.loads(action_json)
            group = act.get("group_name") or act.get("group")
            step = act.get("step_name") or act.get("step")
            
            # Record group usage
            if group and group not in used_groups:
                used_groups[group] = step
            
            # Record max searched_index found in path (to resume search correctly)
            s_idx = int(act.get("searched_index", -1))
            if s_idx > last_step_index:
                last_step_index = s_idx
                
        except Exception:
            pass
            
        curr_id = parent_id_node

    conn.close()
    return trial_id, value, depth, prev_action_json, used_groups, last_step_index

def generate_candidates(space: SuperChainActionSpace, sampler: ParameterSampler, 
                       parent_state: PipelineState, count: int = 1000):
    """Generates FRESH random candidate actions using ActionSpace & Sampler."""
    candidates = []
    
    # Lookahead=10 ensures we skip over many incompatible/disabled steps 
    # and find all reachable next moves.
    discrete_actions = space.next_actions(parent_state, lookahead=10)
    
    if not discrete_actions:
        warn(f"No valid next actions found (last_index={parent_state.last_step_index}).")
        return []

    info(f"Found {len(discrete_actions)} valid action templates (groups/variants).")
    info(f"Sampling {count} fresh configurations...")

    for _ in range(count):
        # 1. Pick a random template
        action_template = random.choice(discrete_actions)
        
        # 2. Generate NEW random config
        config = sampler.sample_variant(
            action_template.template_name,
            action_template.variant_name,
            space.search_spaces
        )
        
        # 3. Construct action dict
        action_dict = {
            "step": action_template.step_name,
            "group_name": action_template.group_name,
            "template": action_template.template_name,
            "variant": action_template.variant_name,
            "config": config,
            "searched_index": action_template.searched_index,
            "original_index": action_template.original_index
        }
        candidates.append(action_dict)
        
    return candidates

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="playground-series-s6e1", help="Project slug")
    parser.add_argument("--count", type=int, default=1000, help="Number of candidates to generate")
    parser.add_argument("--out", default="oracle_generated.csv", help="Output CSV file")
    parser.add_argument("--study", default="s6e1_008_xgb_gpu", help="Study name")
    args = parser.parse_args()

    # Paths
    project_dir = Path(f"projects/kaggle/{args.project}")
    exp_dir = project_dir / "experiments"
    db_path = exp_dir / "db" / "mcts.db"
    model_dir = exp_dir / "oracle" / "model"
    oracle_csv = model_dir / "mcts_oracle.csv"
    
    if not model_dir.exists():
        err("Oracle model not found.")
        return

    # 1. Load Parent
    res = load_best_parent(db_path, args.study)
    if not res:
        err("No parent found.")
        return
    parent_id, parent_score, depth, prev_action_json, used_groups, last_step_index = res
    
    # 2. Setup Generator
    conf_dir = Path("conf/preprocess")
    super_chain_path = conf_dir / "mla_super_chain.yaml"
    space = SuperChainActionSpace(super_chain_path)
    sampler = ParameterSampler()

    # Mock State
    # used_groups requires Dict[group, step_name]
    state = PipelineState(
        steps=[], 
        depth=depth,
        used_groups=used_groups,
        last_step_index=last_step_index
    )
    
    # 3. Generate FRESH Candidates
    candidates_dicts = generate_candidates(space, sampler, state, count=args.count)
    if not candidates_dicts:
        return

    # 4. Prepare DataFrame
    info("Preparing DataFrame and aligning columns...")
    
    # Get expected columns from training data header
    if not oracle_csv.exists():
        err("mcts_oracle.csv not found (needed for column signature).")
        return
    expected_cols = pd.read_csv(oracle_csv, nrows=0).columns.tolist()
    
    # Parse prev_action context (fixed for all)
    prev_flat = parse_action_full(prev_action_json, prefix="prev_")
    
    rows = []
    for cand in candidates_dicts:
        # Flatten the candidate config
        curr_flat = flatten_config(cand, prefix="")
        
        # Combine context + prev + curr
        row = {
            "parent_score": parent_score,
            "depth": depth + 1,
            "prev_duration": 0.0,
            # Add prev_ columns
            **prev_flat,
            # Add curr columns
            **curr_flat,
            # Keep raw json for reference/queueing
            "curr_action_json": json.dumps(cand)
        }
        rows.append(row)
        
    df = pd.DataFrame(rows)
    
    # 5. ALIGN COLUMNS (The Fix)
    # Ensure every column in expected_cols exists in df
    # Ignore target column 'is_improvement'
    for col in expected_cols:
        if col == "is_improvement": 
            continue
        if col not in df.columns:
            df[col] = None
            
    # Also handle the reverse: keep columns in df that are not in expected (extra features?)
    # AutoGluon is usually fine with extra columns, but strict about missing ones.
    
    # 6. Predict
    info("Loading Predictor...")
    predictor = TabularPredictor.load(str(model_dir))
    
    info("Predicting...")
    if predictor.problem_type == "binary":
        probs = predictor.predict_proba(df)
        pos_label = 1
        if pos_label in probs.columns:
            df["prob_improvement"] = probs[pos_label]
        else:
            df["prob_improvement"] = probs.iloc[:, -1]
    else:
        df["prob_improvement"] = predictor.predict(df)
        
    # 7. Sort & Show
    df = df.sort_values("prob_improvement", ascending=False)
    
    # Display logic
    table = Table(title=f"Top 10 Generated Candidates (Base: {parent_score:.4f})")
    
    # Pick relevant columns to show (action + a few params)
    show_cols = ["prob_improvement", "action_group", "action_variant"]
    # Find dynamic param cols that are not null in the top results
    for c in df.columns:
        if c not in show_cols and c in expected_cols and not c.startswith("prev_") and "json" not in c and "score" not in c and "depth" not in c:
            if df.head(10)[c].notna().any():
                show_cols.append(c)
    
    show_cols = show_cols[:6] # Limit width
    
    for c in show_cols:
        table.add_column(c)
        
    for _, row in df.head(10).iterrows():
        vals = [str(row[c])[:20] for c in show_cols]
        table.add_row(*vals)
        
    console.print(table)
    
    # Save full results
    df.to_csv(args.out, index=False)
    info(f"Saved {len(df)} candidates to {args.out}")

if __name__ == "__main__":
    main()