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

def get_best_path(db_path, study_name):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # Find study_id
    study_query = "SELECT study_id FROM studies WHERE study_name = ?"
    study_row = conn.execute(study_query, (study_name,)).fetchone()
    if not study_row:
        err(f"Study {study_name} not found in DB")
        return []
    study_id = study_row["study_id"]

    # Find absolute best trial
    best_trial_query = """
    SELECT t.trial_id, t.number, e.value 
    FROM trials t
    JOIN mcts_evaluations e ON t.trial_id = e.trial_id
    WHERE t.study_id = ? AND e.value IS NOT NULL
    ORDER BY e.value DESC LIMIT 1
    """
    best_trial = conn.execute(best_trial_query, (study_id,)).fetchone()
    if not best_trial:
        err(f"No completed evaluations found for study {study_name}")
        return []
    
    target_trial_id = best_trial["trial_id"]
    info(f"Best trial found: ID={target_trial_id} (Number {best_trial['number']}), Score={best_trial['value']}")

    # Reconstruct path from child to parent
    path = []
    curr_id = target_trial_id
    
    while True:
        edge_query = "SELECT parent_trial_id, action_json FROM mcts_edges WHERE child_trial_id = ?"
        res = conn.execute(edge_query, (curr_id,)).fetchone()
        if not res:
            break
        parent_id, action_json = res["parent_trial_id"], res["action_json"]
        path.append(json.loads(action_json))
        curr_id = parent_id
        if curr_id is None: break

    conn.close()
    return path[::-1] # Reverse to get root -> leaf order

def get_best_by_level(db_path, study_name):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # Find study_id
    study_query = "SELECT study_id FROM studies WHERE study_name = ?"
    study_row = conn.execute(study_query, (study_name,)).fetchone()
    if not study_row:
        err(f"Study {study_name} not found in DB")
        return []
    study_id = study_row["study_id"]

    # Find best trial for each depth
    query = """
    SELECT n.depth, e.trial_id, t.number, e.value, ed.action_json
    FROM mcts_nodes n
    JOIN mcts_evaluations e ON n.trial_id = e.trial_id
    JOIN mcts_edges ed ON n.trial_id = ed.child_trial_id
    JOIN trials t ON n.trial_id = t.trial_id
    WHERE t.study_id = ?
    GROUP BY n.depth
    HAVING e.value = MAX(e.value)
    ORDER BY n.depth ASC
    """
    results = conn.execute(query, (study_id,)).fetchall()
    conn.close()
    
    actions = []
    for row in results:
        info(f"Depth {row['depth']}: Best Trial={row['trial_id']} (Number {row['number']}), Score={row['value']}")
        actions.append(json.loads(row["action_json"]))
    return actions

def generate_templates(project, actions, name_prefix="mcts_top01"):
    template_dir = Path("projects/kaggle") / project / "templates" / "preprocess"
    template_dir.mkdir(parents=True, exist_ok=True)
    
    chain_steps = []
    
    for i, action in enumerate(actions):
        step_name = action.get("step_name") or action.get("group_name")
        variant = action.get("variant")
        config = action.get("config", {})
        
        # Sub-template file name
        sub_filename = f"{name_prefix}-{step_name}.yaml"
        sub_path = template_dir / sub_filename
        
        # Write sub-template
        sub_data = {
            "name": step_name,
            "variant": variant,
            "config": config
        }
        sub_path.write_text(yaml.dump(sub_data, sort_keys=False))
        info(f"Created sub-template: {sub_filename}")
        
        # Add to chain
        chain_steps.append(sub_filename.replace(".yaml", ""))
    
    # Write main chain template
    main_chain_data = {
        "chain": chain_steps
    }
    main_path = template_dir / f"{name_prefix}.yaml"
    main_path.write_text(yaml.dump(main_chain_data, sort_keys=False))
    info(f"Created main chain template: {name_prefix}.yaml")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="playground-series-s6e1")
    parser.add_argument("--config", default="conf/preprocess/mla_super_chain.yaml")
    parser.add_argument("--study", help="Override study name from config")
    parser.add_argument("--db", help="Override database path")
    parser.add_argument("--mode", choices=["path_to_best", "best_per_level"], default="path_to_best")
    parser.add_argument("--prefix", help="Override template prefix (default depends on mode)")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        err(f"Config not found: {config_path}")
        return

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    study_name = args.study or cfg.get("mcts", {}).get("study_name")
    
    if args.db:
        db_path = Path(args.db)
    else:
        db_rel_path = cfg.get("mcts", {}).get("storage_url", "").replace("sqlite:///", "")
        if not db_rel_path:
            err("Missing mcts.storage_url in config")
            return
        
        db_path = Path(db_rel_path)
        if not db_path.exists():
            # Try relative to project if not found
            db_path = Path("projects/kaggle") / args.project / db_rel_path
            if not db_path.exists():
                err(f"Database not found at {db_rel_path} or projects/kaggle/{args.project}/{db_rel_path}")
                return

    prefix = args.prefix
    if not prefix:
        prefix = "mcts_path" if args.mode == "path_to_best" else "mcts_levels"

    info(f"Using database: {db_path}")
    info(f"Using study: {study_name}")
    info(f"Mode: {args.mode}")
    info(f"Prefix: {prefix}")
    
    if args.mode == "path_to_best":
        actions = get_best_path(db_path, study_name)
    else:
        actions = get_best_by_level(db_path, study_name)
    
    if actions:
        generate_templates(args.project, actions, prefix)
        info("Done.")
    else:
        err("Could not reconstruct actions.")

if __name__ == "__main__":
    main()
