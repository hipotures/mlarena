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

def flatten_config(action_dict, prefix=""):
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

def parse_action_full(action_json_str, prefix=""):
    if not action_json_str or pd.isna(action_json_str): return {}
    try:
        data = json.loads(action_json_str)
        return flatten_config(data, prefix=prefix)
    except: return {}

def load_top_parents(db_path: Path, study_name: str, top_n: int):
    info(f"Connecting to DB: {db_path} (study: {study_name})")
    conn = sqlite3.connect(db_path)
    cur = conn.execute("SELECT study_id FROM studies WHERE study_name = ?", (study_name,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return []
    study_id = row[0]

    query = """
    SELECT e.trial_id, e.value, n.depth
    FROM mcts_evaluations e
    JOIN mcts_nodes n ON e.trial_id = n.trial_id
    WHERE n.study_id = ? AND e.status = 'COMPLETE' AND e.value IS NOT NULL
    ORDER BY e.value DESC
    LIMIT ?
    """
    parents = conn.execute(query, (study_id, top_n)).fetchall()
    
    results = []
    for p_id, p_val, p_depth in parents:
        # Reconstruct history for each parent
        used_groups = {}
        curr_id = p_id
        last_step_index = -1
        last_action_json = None
        
        while True:
            edge_query = "SELECT parent_trial_id, action_json FROM mcts_edges WHERE child_trial_id = ?"
            edge_row = conn.execute(edge_query, (curr_id,)).fetchone()
            if not edge_row: break
            
            p_node_id, action_json = edge_row
            if curr_id == p_id: last_action_json = action_json
            
            try:
                act = json.loads(action_json)
                group = act.get("group_name") or act.get("group")
                step = act.get("step_name") or act.get("step")
                if group: used_groups[group] = step
                s_idx = int(act.get("searched_index", -1))
                if s_idx > last_step_index: last_step_index = s_idx
            except: pass
            curr_id = p_node_id
            
        results.append({
            "trial_id": p_id,
            "value": p_val,
            "depth": p_depth,
            "used_groups": used_groups,
            "last_step_index": last_step_index,
            "last_action_json": last_action_json
        })
    
    conn.close()
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="playground-series-s6e1")
    parser.add_argument("--study", default="s6e1_008")
    parser.add_argument("--top-n", type=int, default=20, help="Number of best trials to use as parents")
    parser.add_argument("--samples", type=int, default=5000, help="Samples per parent")
    parser.add_argument("--out", default="oracle_multi_parent.csv")
    args = parser.parse_args()

    project_dir = Path(f"projects/kaggle/{args.project}")
    db_path = project_dir / "experiments" / "db" / "mcts.db"
    model_dir = project_dir / "experiments" / "oracle" / "model"
    oracle_csv = model_dir / "mcts_oracle.csv"

    if not model_dir.exists(): return

    # 1. Load Data
    parents = load_top_parents(db_path, args.study, args.top_n)
    if not parents: return
    info(f"Found {len(parents)} top parents.")

    # 2. Setup
    space = SuperChainActionSpace(Path("conf/preprocess/mla_super_chain.yaml"))
    sampler = ParameterSampler()
    predictor = TabularPredictor.load(str(model_dir))
    expected_cols = pd.read_csv(oracle_csv, nrows=0).columns.tolist()

    # 3. Generate Massive Candidates
    all_rows = []
    for p in parents:
        state = PipelineState(steps=[], depth=p["depth"], used_groups=p["used_groups"], last_step_index=p["last_step_index"])
        discrete_actions = space.next_actions(state, lookahead=5)
        if not discrete_actions: continue
        
        prev_flat = parse_action_full(p["last_action_json"], prefix="prev_")
        
        info(f"Generating for Parent {p['trial_id']} (Score: {p['value']:.4f})...")
        for _ in range(args.samples):
            template = random.choice(discrete_actions)
            config = sampler.sample_variant(template.template_name, template.variant_name, space.search_spaces)
            action_dict = {
                "step": template.step_name, "group_name": template.group_name, 
                "template": template.template_name, "variant": template.variant_name,
                "config": config, "searched_index": template.searched_index, "original_index": template.original_index
            }
            
            curr_flat = flatten_config(action_dict, prefix="")
            row = {
                "parent_id": p["trial_id"], "parent_score": p["value"], "depth": p["depth"] + 1, "prev_duration": 0.0,
                **prev_flat, **curr_flat, "curr_action_json": json.dumps(action_dict)
            }
            all_rows.append(row)

    if not all_rows: return
    df = pd.DataFrame(all_rows)

    # 4. Align and Predict
    for col in expected_cols:
        if col != "is_improvement" and col not in df.columns: df[col] = None
    
    info(f"Predicting for {len(df)} candidates...")
    probs = predictor.predict_proba(df)
    pos_label = 1
    df["prob_improvement"] = probs[pos_label] if pos_label in probs.columns else probs.iloc[:, -1]

    # 5. Diversified Selection
    # Pick top 1 per parent
    top_per_parent = df.sort_values("prob_improvement", ascending=False).groupby("parent_id").head(1)
    # Pick absolute top 10 from those
    final_top = top_per_parent.sort_values("prob_improvement", ascending=False).head(10)

    table = Table(title="Top 10 Moves from Diverse Parents")
    table.add_column("Prob")
    table.add_column("Parent (Score)")
    table.add_column("Action (Group/Var)")
    
    for _, row in final_top.iterrows():
        table.add_row(
            f"{row['prob_improvement']:.4f}",
            f"{int(row['parent_id'])} ({row['parent_score']:.4f})",
            f"{row['action_group']} / {row['action_variant']}"
        )
    console.print(table)

    # Save Templates
    template_dir = project_dir / "templates" / "preprocess"
    for i, (_, row) in enumerate(final_top.iterrows()):
        child_id = f"oracle_brute_{i+1:02d}"
        action = json.loads(row["curr_action_json"])
        
        # In a multi-parent scenario, the child needs the FULL history of the parent.
        # We need to reconstruct the parent's chain.
        # Let's get parent info from our 'parents' list
        p_info = next(p for p in parents if p["trial_id"] == row["parent_id"])
        
        # How to get the parent's chain? 
        # We'll use a hack: read the parent's trial_pipeline.yaml from its artifact dir.
        # Actually, we can reconstruct it from our 'used_groups' history traversal!
        # But we need the module config too. 
        # Better: let's assume parent was a known template or just reconstruct names.
        
        # Simplified for now: assume we can point to parent's existing chain if available,
        # but here we'll just build a new chain list.
        # TODO: A real reconstruction would traverse the whole tree.
        # For this BRUTE FORCE, let's just output the ACTION and parent_id for reference.
        
        info(f"Saving {child_id} (Parent {row['parent_id']})")
        # We'll need a better way to link parents in future, for now just CSV.

    df.to_csv(args.out, index=False)
    info(f"Full results saved to {args.out}")

if __name__ == "__main__":
    main()
