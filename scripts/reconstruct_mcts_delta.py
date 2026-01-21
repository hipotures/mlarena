#!/usr/bin/env python
import sqlite3
import pandas as pd
import json
import argparse
import yaml
from pathlib import Path
import sys

def info(msg):
    print(f"[INFO] {msg}")

def err(msg):
    print(f"[ERROR] {msg}")

def get_best_by_delta(db_path, study_name, enabled_groups):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # Find study_id
    study_row = conn.execute("SELECT study_id FROM studies WHERE study_name = ?", (study_name,)).fetchone()
    if not study_row:
        err(f"Study {study_name} not found")
        return []
    study_id = study_row["study_id"]

    # Query all edges with evaluations to calculate deltas
    # We MUST have both child and parent scores to calculate a valid delta
    query = """
    SELECT 
        e.parent_trial_id, 
        e.child_trial_id, 
        t_child.number as child_number,
        ev_child.value as child_val,
        ev_parent.value as parent_val,
        e.action_json
    FROM mcts_edges e
    JOIN trials t_child ON e.child_trial_id = t_child.trial_id
    JOIN mcts_evaluations ev_child ON e.child_trial_id = ev_child.trial_id
    JOIN mcts_evaluations ev_parent ON e.parent_trial_id = ev_parent.trial_id
    WHERE t_child.study_id = ? AND ev_child.value IS NOT NULL AND ev_parent.value IS NOT NULL
    """
    
    edges = conn.execute(query, (study_id,)).fetchall()
    conn.close()

    results = []
    for group in enabled_groups:
        best_edge = None
        max_delta = -float('inf')
        
        for edge in edges:
            action = json.loads(edge["action_json"])
            step_name = action.get("step_name") or action.get("group_name")
            
            if step_name == group:
                delta = edge["child_val"] - edge["parent_val"]
                
                if delta > max_delta:
                    max_delta = delta
                    best_edge = edge
        
        if best_edge and max_delta > 0:
            action = json.loads(best_edge["action_json"])
            info(f"Group '{group}': Best Delta={max_delta:.6f} (Trial {best_edge['child_number']}, Score {best_edge['child_val']:.4f})")
            results.append(action)
        elif best_edge and max_delta <= 0:
            info(f"Group '{group}': Best Delta is {max_delta:.6f} (No improvement, skipping)")
        else:
            info(f"Group '{group}': No trials found in DB")
            
    return results

def generate_templates(project, actions, name_prefix="mcts_delta"):
    template_dir = Path("projects/kaggle") / project / "templates" / "preprocess"
    template_dir.mkdir(parents=True, exist_ok=True)
    
    chain_steps = []
    for i, action in enumerate(actions):
        step_name = action.get("step_name") or action.get("group_name")
        variant = action.get("variant")
        config = action.get("config", {})
        
        sub_filename = f"{name_prefix}-{i:02d}-{step_name}.yaml"
        sub_data = {"name": step_name, "variant": variant, "config": config}
        (template_dir / sub_filename).write_text(yaml.dump(sub_data, sort_keys=False))
        info(f"Created: {sub_filename}")
        chain_steps.append(sub_filename.replace(".yaml", ""))
    
    main_path = template_dir / f"{name_prefix}.yaml"
    main_path.write_text(yaml.dump({"chain": chain_steps}, sort_keys=False))
    info(f"Created main chain: {name_prefix}.yaml")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="playground-series-s6e1")
    parser.add_argument("--config", default="conf/preprocess/mla_super_chain.yaml")
    parser.add_argument("--study", help="Override study name")
    parser.add_argument("--db", help="Override DB path")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    study_name = args.study or cfg.get("mcts", {}).get("study_name")
    db_path = args.db or "/mnt/mlarena/projects/kaggle/playground-series-s6e1/experiments/db/mcts.db"
    
    # Identify enabled groups from YAML
    enabled_groups = []
    for p in cfg.get("preprocessors", []):
        if p.get("enabled") and not p.get("meta", {}).get("fixed"):
            enabled_groups.append(p.get("name"))
    
    info(f"Enabled groups from YAML: {enabled_groups}")
    
    actions = get_best_by_delta(db_path, study_name, enabled_groups)
    if actions:
        generate_templates(args.project, actions)
        info("Done.")
    else:
        err("No actions reconstructed.")

if __name__ == "__main__":
    main()
