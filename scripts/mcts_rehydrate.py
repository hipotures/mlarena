#!/usr/bin/env python3
"""
Pancerny skrypt do rehydratacji triali MCTS.
Odtwarza pliki YAML na podstawie danych z SQLite.
"""

import argparse
import json
import sys
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add src to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from mlarena.modules.mcts.storage import MCTSStorage
from mlarena.modules.mcts.node import PipelineState
from mlarena.modules.mcts.materializer import TemplateMaterializer

def main():
    parser = argparse.ArgumentParser(description="Rehydrate MCTS templates from DB")
    parser.add_argument("--project", "-p", required=True, help="Project name")
    parser.add_argument("--study", "-s", required=True, help="Study name")
    parser.add_argument("--trial", "-t", type=int, required=True, help="Trial number")
    
    args = parser.parse_args()
    
    # Resolve project path
    base_projects_dir = REPO_ROOT / "projects" / "kaggle"
    project_root = (base_projects_dir / args.project).resolve()
    if not project_root.exists():
        # Case-insensitive fallback
        for p in base_projects_dir.iterdir():
            if p.is_dir() and p.name.lower() == args.project.lower():
                project_root = p.resolve()
                break
    
    db_path = project_root / "experiments" / "db" / "mcts.db"
    if not db_path.exists():
        # NFS fallback
        db_path = Path("/mnt/mlarena") / "projects" / "kaggle" / args.project / "experiments" / "db" / "mcts.db"
        if not db_path.exists():
            print(f"Error: Database not found at {db_path}")
            sys.exit(1)

    storage = MCTSStorage(f"sqlite:///{db_path}")
    
    # 1. Find Study and Trial
    with storage._connect() as conn:
        conn.row_factory = sqlite3.Row
        study = conn.execute("SELECT study_id FROM studies WHERE study_name=?", (args.study,)).fetchone()
        if not study:
            print(f"Error: Study '{args.study}' not found.")
            sys.exit(1)
        study_id = study[0]
        
        trial = conn.execute(
            "SELECT trial_id, state FROM trials WHERE study_id=? AND number=?", 
            (study_id, args.trial)
        ).fetchone()
        if not trial:
            print(f"Error: Trial {args.trial} not found.")
            sys.exit(1)
        trial_id = trial[0]

        node = conn.execute("SELECT depth, pipeline_signature FROM mcts_nodes WHERE trial_id=?", (trial_id,)).fetchone()
        depth = node[0]
        db_sig = node[1]

    # 2. Reconstruct Steps
    steps = []
    with storage._connect() as conn:
        rows = conn.execute(
            "SELECT param_name, param_value FROM trial_params WHERE trial_id=? AND param_name LIKE 'step_%'", 
            (trial_id,)
        ).fetchall()
        sorted_rows = sorted(rows, key=lambda x: int(x[0].split("_")[1]))
        for _, val_json in sorted_rows:
            steps.append(json.loads(val_json))

    # 3. Reconstruct YAMLs
    print(f"Rehydrating Trial {args.trial} (Depth: {depth}, Signature: {db_sig[:8]})")
    
    run_id = f"mcts_s{study_id:04d}"
    templates_dir = project_root / "templates" / "preprocess"
    templates_dir.mkdir(parents=True, exist_ok=True)
    
    # Build base name matching MCTS logic
    base_name = f"{run_id}_n{args.trial:06d}_d{depth:02d}_{db_sig[:8]}"
    
    # Fixed steps (harness) - we assume they are always the same for a given study
    # In a perfect world we'd store them in DB too, but for now we take them as sanity/fraction
    # To be safe, we reconstruct only what's in 'steps' from DB if it is not baseline
    
    import yaml
    chain_list = []
    
    # Handle Baseline (Trial 0)
    if args.trial == 0 or db_sig == "baseline":
        # Baseline usually has 0 steps in trial_params, but needs sanity/fraction
        # For now, let's assume standard harness
        steps = [
            {"name": "sanity_check", "module": "sanity_check", "config": {}},
            {"name": "train_fraction", "module": "train_fraction", "config": {"train_fraction": 0.6}}
        ]

    for i, step in enumerate(steps):
        step_name = step.get("name", f"step_{i}")
        tpl_name = f"{base_name}_{i:02d}_{step_name}"
        tpl_path = templates_dir / f"{tpl_name}.yaml"
        
        payload = {
            "module": step.get("module") or step.get("template"),
            "config": step.get("config", {})
        }
        tpl_path.write_text(yaml.safe_dump(payload, sort_keys=False))
        chain_list.append(tpl_name)
        print(f"  [+] Created step: {tpl_name}")

    # Create Chain
    chain_path = templates_dir / f"{base_name}.yaml"
    chain_path.write_text(yaml.safe_dump({"chain": chain_list}, sort_keys=False))
    
    # Create Fidelity Copy (F2)
    fid_path = templates_dir / f"{base_name}_F2.yaml"
    fid_path.write_text(yaml.safe_dump({"chain": chain_list}, sort_keys=False))
    
    print(f"  [+] Created chain: {chain_path.name}")
    print(f"  [+] Created fidelity: {fid_path.name}")

    print(f"\nDone! Command to run trial {args.trial}:")
    print(f"uv run python scripts/mla.py model project={args.project} experiment_id=exp-{base_name}_F2 force=true preprocess_template={base_name}_F2")

if __name__ == "__main__":
    import sqlite3 # Import here to be sure
    main()
