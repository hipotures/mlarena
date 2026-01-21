#!/usr/bin/env python
import sqlite3
import pandas as pd
import json
import yaml
import argparse
from pathlib import Path

def info(msg): print(f"INFO: {msg}")

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="playground-series-s6e1")
    parser.add_argument("--csv", default="oracle_multi_parent.csv")
    args = parser.parse_args()

    project_dir = Path(f"projects/kaggle/{args.project}")
    db_path = project_dir / "experiments" / "db" / "mcts.db"
    preprocess_dir = project_dir / "templates" / "preprocess"
    preprocess_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    # Pick top 1 per parent to ensure diversity as in previous step
    top_moves = df.sort_values("prob_improvement", ascending=False).groupby("parent_id").head(1).head(10)

    conn = sqlite3.connect(db_path)
    
    generated_ids = []
    for i, (_, row) in enumerate(top_moves.iterrows()):
        child_id = f"oracle_brute_v2_{i+1:02d}"
        parent_id = int(row["parent_id"])
        new_action = json.loads(row["curr_action_json"])
        
        info(f"Reconstructing for {child_id} (Parent {parent_id}, Prob {row['prob_improvement']:.4f})")
        
        # 1. Get history
        history = reconstruct_chain(conn, parent_id)
        history.append(new_action)
        
        # 2. Save modules and build chain list
        chain_names = ["mcts"]
        for idx, act in enumerate(history):
            mod_name = act["step_name"] if "step_name" in act else act.get("step")
            tmpl_name = act["template_name"] if "template_name" in act else act.get("template")
            
            module_alias = get_module_alias(tmpl_name)
            module_filename = f"{child_id}-step{idx:02d}-{mod_name}"
            
            payload = {
                "module": module_alias,
                "config": act["config"]
            }
            
            with open(preprocess_dir / f"{module_filename}.yaml", "w") as f:
                yaml.dump(payload, f, sort_keys=False)
            
            chain_names.append(module_filename)
            
        # 3. Save chain
        with open(preprocess_dir / f"{child_id}.yaml", "w") as f:
            yaml.dump({"chain": chain_names}, f, sort_keys=False)
            
        generated_ids.append(child_id)
        
    conn.close()

    # Clear queue
    info("Clearing queue...")
    import subprocess
    subprocess.run(["uv", "run", "python", "scripts/task_queue.py", "-p", args.project, "clean", "--status", "all"])

    # Add to queue
    for gid in generated_ids:
        cmd = f"model model_template=oracle_verify_medium preprocess_template={gid}"
        info(f"Adding to queue: {cmd}")
        subprocess.run(["uv", "run", "python", "scripts/task_queue.py", "-p", args.project, "add", "--command", cmd])

if __name__ == "__main__":
    main()
