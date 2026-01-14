#!/usr/bin/env python3
"""
Fix MCTS database indices by retrofitting 'searched_index' into 'action_json' in 'mcts_edges'.
This fixes the 'Action index 0 violates order' error after upgrading MCTS logic.
"""

import sqlite3
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any

def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    return yaml.safe_load(path.read_text()) or {}

def build_step_index_map(super_chain_path: Path) -> Dict[str, int]:
    """Build a map of step_name -> searched_index based on the super chain definition."""
    print(f"Loading super chain from: {super_chain_path}")
    data = load_yaml(super_chain_path)
    
    # Logic similar to SuperChainActionSpace
    steps = []
    
    # Handle both format with 'mcts' key and direct list
    mcts_config = data.get("mcts", {})
    if "searched_steps" in mcts_config:
        steps = mcts_config["searched_steps"]
    elif "preprocessors" in data:
        # Fallback to preprocessors list if mcts section missing (less likely for MCTS)
        steps = data["preprocessors"]
        
    if not steps:
        print("Warning: No steps found in super chain config!")
        
    index_map = {}
    for i, step in enumerate(steps):
        name = step.get("name")
        if name:
            index_map[name] = i
            
    print(f"Mapped {len(index_map)} steps.")
    return index_map

def fix_database(db_path: Path, index_map: Dict[str, int]):
    print(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    try:
        cur.execute("SELECT parent_trial_id, child_trial_id, action_json FROM mcts_edges")
        rows = cur.fetchall()
        print(f"Found {len(rows)} edges. Scanning for fixes...")
        
        updates = []
        fixed_count = 0
        
        for row in rows:
            try:
                action = json.loads(row["action_json"])
                step_name = action.get("step_name")
                
                if not step_name:
                    continue
                    
                current_idx = action.get("searched_index")
                correct_idx = index_map.get(step_name)
                
                if correct_idx is not None:
                    # Update if missing or 0 (when it shouldn't be 0, unless step is actually 0)
                    # Actually, if it's 0 but the correct index is e.g. 5, we must update.
                    # If correct index IS 0, and current is 0, we do nothing.
                    # But if current is missing, we update.
                    
                    if current_idx is None or current_idx != correct_idx:
                        action["searched_index"] = correct_idx
                        # Also fix original_index/step_index if present/messy
                        action["step_index"] = correct_idx 
                        action["original_index"] = correct_idx # Assuming simple mapping for now
                        
                        updates.append((json.dumps(action), row["parent_trial_id"], row["child_trial_id"]))
                        fixed_count += 1
                else:
                    print(f"Warning: Step '{step_name}' not found in super chain map. Skipping.")
                    
            except json.JSONDecodeError:
                print(f"Error decoding JSON for edge {row['parent_trial_id']}->{row['child_trial_id']}")
                
        if updates:
            print(f"Applying {len(updates)} fixes...")
            cur.executemany(
                "UPDATE mcts_edges SET action_json = ? WHERE parent_trial_id = ? AND child_trial_id = ?",
                updates
            )
            conn.commit()
            print("Database updated successfully.")
        else:
            print("No updates needed.")
            
    except Exception as e:
        print(f"Database error: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix MCTS database indices.")
    parser.add_argument("db_path", type=Path, help="Path to mcts.db")
    parser.add_argument("--super-chain", type=Path, default=Path("conf/preprocess/mla_super_chain.yaml"), help="Path to super chain yaml")
    
    args = parser.parse_args()
    
    if not args.db_path.exists():
        print(f"Database file not found: {args.db_path}")
        exit(1)
        
    idx_map = build_step_index_map(args.super_chain)
    fix_database(args.db_path, idx_map)
