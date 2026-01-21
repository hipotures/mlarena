#!/usr/bin/env python
import sqlite3
import pandas as pd
import json
import argparse
import random
import yaml
import subprocess
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

def info(msg): console.print(f"[blue]INFO:[/blue] {msg}")

def get_module_alias(template_name):
    global_tmpl_path = Path("src/mlarena/templates/preprocess") / f"{template_name}.yaml"
    if global_tmpl_path.exists():
        try:
            data = yaml.safe_load(global_tmpl_path.read_text())
            return data.get("module", template_name)
        except: pass
    return template_name

def reconstruct_chain(conn, trial_id):
    actions = []
    curr_id = trial_id
    while True:
        row = conn.execute("SELECT parent_trial_id, action_json FROM mcts_edges WHERE child_trial_id = ?", (curr_id,)).fetchone()
        if not row: break
        actions.insert(0, json.loads(row[1]))
        curr_id = row[0]
    return actions

def flatten_config(action_dict, prefix=""):
    flat = {}
    group = action_dict.get("group_name", "unknown")
    variant = action_dict.get("variant", "unknown")
    flat[f"{prefix}action_group"] = group
    flat[f"{prefix}action_variant"] = variant
    config = action_dict.get("config", {})
    for k, v in config.items():
        key = f"{prefix}{group}_{k}"
        if isinstance(v, (list, dict)): flat[key] = json.dumps(v, sort_keys=True)
        else: flat[key] = v
    return flat

def parse_action_full(action_json_str, prefix=""):
    if not action_json_str or pd.isna(action_json_str): return {}
    try:
        data = json.loads(action_json_str)
        return flatten_config(data, prefix=prefix)
    except: return {}

def load_top_per_level(db_path: Path, study_name: str, top_per_level: int):
    conn = sqlite3.connect(db_path)
    cur = conn.execute("SELECT study_id FROM studies WHERE study_name = ?", (study_name,))
    row = cur.fetchone()
    if not row: return []
    study_id = row[0]

    # Query top trials per depth (GLOBAL - all studies)
    query = """
    SELECT e.trial_id, e.value, n.depth
    FROM mcts_evaluations e
    JOIN mcts_nodes n ON e.trial_id = n.trial_id
    WHERE e.status = 'COMPLETE' AND e.value IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    
    # Sort and pick top N per depth
    top_trials = df.sort_values("value", ascending=False).groupby("depth").head(top_per_level)
    
    results = []
    for _, row in top_trials.iterrows():
        p_id = int(row["trial_id"])
        
        # Reconstruct state
        used_groups = {}
        curr_id = p_id
        last_step_index = -1
        last_action_json = None
        
        while True:
            edge_row = conn.execute("SELECT parent_trial_id, action_json FROM mcts_edges WHERE child_trial_id = ?", (curr_id,)).fetchone()
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
            "trial_id": p_id, "value": row["value"], "depth": int(row["depth"]),
            "used_groups": used_groups, "last_step_index": last_step_index, "last_action_json": last_action_json
        })
    
    conn.close()
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="playground-series-s6e1")
    parser.add_argument("--study", default="s6e1_008")
    parser.add_argument("--samples", type=int, default=2000)
    args = parser.parse_args()

    project_dir = Path(f"projects/kaggle/{args.project}")
    db_path = project_dir / "experiments" / "db" / "mcts.db"
    model_dir = project_dir / "experiments" / "oracle" / "model"
    oracle_csv = model_dir / "mcts_oracle.csv"
    preprocess_dir = project_dir / "templates" / "preprocess"

    # 1. Load Data
    parents = load_top_per_level(db_path, args.study, top_per_level=5)
    info(f"Found {len(parents)} parents across levels.")

    # 2. Setup
    space = SuperChainActionSpace(Path("conf/preprocess/mla_super_chain.yaml"))
    sampler = ParameterSampler()
    predictor = TabularPredictor.load(str(model_dir))
    expected_cols = pd.read_csv(oracle_csv, nrows=0).columns.tolist()

    # 3. Generate
    all_rows = []
    for p in parents:
        state = PipelineState(steps=[], depth=p["depth"], used_groups=p["used_groups"], last_step_index=p["last_step_index"])
        discrete_actions = space.next_actions(state, lookahead=10) # High lookahead to jump over dead steps
        if not discrete_actions: continue
        
        prev_flat = parse_action_full(p["last_action_json"], prefix="prev_")
        
        for _ in range(args.samples):
            template = random.choice(discrete_actions)
            config = sampler.sample_variant(template.template_name, template.variant_name, space.search_spaces)
            action_dict = {
                "step": template.step_name, "group_name": template.group_name, 
                "template": template.template_name, "variant": template.variant_name,
                "config": config, "searched_index": template.searched_index, "original_index": template.original_index
            }
            curr_flat = flatten_config(action_dict, prefix="")
            all_rows.append({
                "parent_id": p["trial_id"], "depth": p["depth"] + 1, "parent_score": p["value"],
                **prev_flat, **curr_flat, "curr_action_json": json.dumps(action_dict)
            })

    df = pd.DataFrame(all_rows)
    for col in expected_cols:
        if col != "is_improvement" and col not in df.columns: df[col] = None
    
    # 4. Predict
    info(f"Predicting for {len(df)} candidates...")
    probs = predictor.predict_proba(df)
    df["prob_improvement"] = probs[1] if 1 in probs.columns else probs.iloc[:, -1]

    # 5. Selection (Top 1 per depth level)
    top_per_level = df.sort_values("prob_improvement", ascending=False).groupby("depth").head(1)
    
    table = Table(title="Best Move per Level (Generated)")
    table.add_column("New Depth")
    table.add_column("Prob")
    table.add_column("Parent Score")
    table.add_column("Action")
    
    generated_ids = []
    conn = sqlite3.connect(db_path)
    for _, row in top_per_level.sort_values("depth").iterrows():
        table.add_row(str(int(row['depth'])), f"{row['prob_improvement']:.4f}", f"{row['parent_score']:.4f}", f"{row['action_group']}/{row['action_variant']}")
        
        # Reconstruct and Save
        child_id = f"oracle_lvl_{int(row['depth']):02d}"
        history = reconstruct_chain(conn, int(row["parent_id"]))
        history.append(json.loads(row["curr_action_json"]))
        
        chain_names = ["mcts"]
        for idx, act in enumerate(history):
            module_alias = get_module_alias(act.get("template", act.get("template_name")))
            module_filename = f"{child_id}-s{idx:02d}-{act.get('step', act.get('step_name'))}"
            with open(preprocess_dir / f"{module_filename}.yaml", "w") as f:
                yaml.dump({"module": module_alias, "config": act["config"]}, f)
            chain_names.append(module_filename)
            
        with open(preprocess_dir / f"{child_id}.yaml", "w") as f:
            yaml.dump({"chain": chain_names}, f)
        generated_ids.append(child_id)
    
    conn.close()
    console.print(table)

    # 6. Queue
    info("Clearing queue...")
    subprocess.run(["uv", "run", "python", "scripts/task_queue.py", "-p", args.project, "clean", "--status", "all"])
    for gid in generated_ids:
        cmd = f"model model_template=oracle_verify_medium preprocess_template={gid}"
        subprocess.run(["uv", "run", "python", "scripts/task_queue.py", "-p", args.project, "add", "--command", cmd])
    info(f"Added {len(generated_ids)} tasks to queue.")

if __name__ == "__main__":
    main()
