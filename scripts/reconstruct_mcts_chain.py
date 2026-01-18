#!/usr/bin/env python
import sqlite3
import pandas as pd
import json
import argparse
import yaml
import re
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
try:
    from mlarena.utils.queue import TaskQueue
    _QUEUE_AVAILABLE = True
except ImportError:
    _QUEUE_AVAILABLE = False

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

def extract_fixed_steps(config, fast_mode=False):
    """
    Extracts fixed preprocessing steps from the main configuration.
    """
    fixed_steps = []
    preprocessors = config.get("preprocessors", [])
    
    for p in preprocessors:
        # Must be enabled and marked as fixed
        if p.get("enabled") and p.get("meta", {}).get("fixed"):
            name = p.get("name")
            
            # Special handling for train_fraction
            if name == "train_fraction":
                if not fast_mode:
                    info("Skipping 'train_fraction' (Fixed Step) - Production mode (use full data)")
                    continue
                else:
                    info("Including 'train_fraction' (Fixed Step) - Fast/Eval mode enabled")
            else:
                 info(f"Including '{name}' (Fixed Step)")

            fixed_steps.append({
                "step_name": name,
                "module": p.get("template", name), # Default template to name if missing
                "config": p.get("fixed_config", {}),
                "is_fixed": True
            })
            
    return fixed_steps

def resolve_next_version(project, base_prefix, overwrite=False):
    """
    Finds the next available version number (e.g., _001, _002).
    Scans projects/kaggle/<project>/templates/preprocess/ for pattern {base_prefix}_NNN.yaml
    """
    template_dir = Path("projects/kaggle") / project / "templates" / "preprocess"
    
    if not template_dir.exists():
        return f"{base_prefix}_001"

    # Pattern: exact prefix + _ + 3 digits + .yaml
    pattern = re.compile(rf"^{re.escape(base_prefix)}_(\d{{3}})\.yaml$")
    
    max_ver = 0
    
    # Scan directory
    for item in template_dir.iterdir():
        if item.is_file():
            match = pattern.match(item.name)
            if match:
                ver = int(match.group(1))
                if ver > max_ver:
                    max_ver = ver
    
    if max_ver == 0:
        return f"{base_prefix}_001"
    
    if overwrite:
        next_ver = max_ver
        info(f"Overwrite enabled: reusing latest version {next_ver:03d}")
    else:
        next_ver = max_ver + 1
        info(f"Auto-increment: next version is {next_ver:03d}")
        
    return f"{base_prefix}_{next_ver:03d}"

def generate_preprocess_templates(project, fixed_actions, mcts_actions, final_name):
    template_dir = Path("projects/kaggle") / project / "templates" / "preprocess"
    template_dir.mkdir(parents=True, exist_ok=True)
    
    chain_steps = []
    
    # 1. Process Fixed Actions (Prefix: init)
    for i, action in enumerate(fixed_actions):
        step_name = action["step_name"]
        module = action["module"]
        config = action["config"]
        
        # Sub-template naming: final_name-init-00-stepname.yaml
        sub_filename = f"{final_name}-init-{i:02d}-{step_name}.yaml"
        sub_path = template_dir / sub_filename
        
        sub_data = {
            "module": module,
            "config": config
        }
        sub_path.write_text(yaml.dump(sub_data, sort_keys=False))
        chain_steps.append(sub_filename.replace(".yaml", ""))

    # 2. Process MCTS Actions (Prefix: numeric index)
    for i, action in enumerate(mcts_actions):
        step_name = action.get("step_name") or action.get("group_name")
        variant = action.get("variant")
        config = action.get("config", {})
        
        sub_filename = f"{final_name}-{i:02d}-{step_name}.yaml"
        sub_path = template_dir / sub_filename
        
        sub_data = {
            "name": step_name,
            "variant": variant,
            "config": config
        }
        sub_path.write_text(yaml.dump(sub_data, sort_keys=False))
        chain_steps.append(sub_filename.replace(".yaml", ""))
    
    # Write main chain template
    main_chain_data = {
        "chain": chain_steps
    }
    main_path = template_dir / f"{final_name}.yaml"
    main_path.write_text(yaml.dump(main_chain_data, sort_keys=False))
    info(f"Created main chain template: {final_name}.yaml")
    return final_name

def generate_model_template(project, config, fast_mode, final_name):
    template_dir = Path("projects/kaggle") / project / "templates" / "model"
    template_dir.mkdir(parents=True, exist_ok=True)

    # Determine source section
    section_key = "evaluation" if fast_mode else "production"
    section_data = config.get(section_key, {})
    
    # Fallback to defaults if missing in config
    base_model = section_data.get("model")
    if not base_model:
        base_model = "autogluon_thin_fast" if fast_mode else "cpu-best-1h-boost"
        info(f"Model config missing for '{section_key}'. Using default: {base_model}")
    
    seed = section_data.get("seed", 42)
    
    # Generate the model template file
    model_template_name = f"{final_name}"
    model_path = template_dir / f"{model_template_name}.yaml"
    
    model_data = {
        # "model": base_model, # REMOVED per instruction
        "preprocess_template": final_name, # CRITICAL: Link model to preprocess
        "config": {
            "random_state": seed,
        }
    }
    
    model_path.write_text(yaml.dump(model_data, sort_keys=False))
    info(f"Created model template: {model_template_name}.yaml (Base: {base_model}, Preprocess: {final_name})")
    
    return model_template_name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="playground-series-s6e1")
    parser.add_argument("--config", default="conf/preprocess/mla_super_chain.yaml")
    parser.add_argument("--study", help="Override study name from config")
    parser.add_argument("--db", help="Override database path")
    parser.add_argument("--mode", choices=["path_to_best", "best_per_level"], default="path_to_best")
    parser.add_argument("--prefix", help="Override template prefix (default depends on mode)")
    parser.add_argument("--fast", action="store_true", help="Use evaluation settings (fast model, subset data). Default is production (full data).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the latest existing version instead of incrementing.")
    parser.add_argument("--enqueue", action="store_true", help="Enqueue the generated experiment automatically.")
    parser.add_argument("--env", choices=["local", "remote"], default="local", help="Environment (local or remote/NFS). Defaults to local.")
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
        # Auto-resolve based on environment
        if args.env == "remote":
            db_path = Path("/mnt/mlarena") / "projects/kaggle" / args.project / "experiments/db/mcts.db"
        else:
            # Local default
            db_path = Path("projects/kaggle") / args.project / "experiments/db/mcts.db"
            
        if not db_path.exists():
            # Fallback to config path if auto-resolve fails
            db_rel_path = cfg.get("mcts", {}).get("storage_url", "").replace("sqlite:///", "")
            if db_rel_path:
                db_path = Path(db_rel_path)
                if not db_path.exists():
                     db_path = Path("projects/kaggle") / args.project / db_rel_path
            
            if not db_path.exists():
                err(f"Database not found at {db_path}. Please provide --db or check --env.")
                return

    prefix = args.prefix
    if not prefix:
        prefix = f"mcts_{study_name}"
        if args.fast:
            prefix += "_eval"
        else:
            prefix += "_prod"

    info(f"Using database: {db_path}")
    info(f"Using study: {study_name}")
    info(f"Mode: {args.mode}")
    info(f"Profile: {'FAST/EVAL' if args.fast else 'PRODUCTION'}")
    
    # 0. Resolve Version
    final_name = resolve_next_version(args.project, prefix, overwrite=args.overwrite)
    info(f"Target Template Name: {final_name}")

    # 1. Reconstruct MCTS Actions
    if args.mode == "path_to_best":
        mcts_actions = get_best_path(db_path, study_name)
    else:
        mcts_actions = get_best_by_level(db_path, study_name)
    
    if not mcts_actions and args.mode == "path_to_best":
        err("Could not reconstruct MCTS actions.")
        return

    # 2. Extract Fixed Actions from Config
    fixed_actions = extract_fixed_steps(cfg, fast_mode=args.fast)

    # 3. Generate Preprocess Templates
    preprocess_template_name = generate_preprocess_templates(args.project, fixed_actions, mcts_actions, final_name)
    
    # 4. Generate Model Template
    model_template_name = generate_model_template(args.project, cfg, fast_mode=args.fast, final_name=final_name)
    
    # Final info
    command_str = f"model model_template={model_template_name}"
    full_cli_cmd = f"uv run python scripts/mla.py model project={args.project} model_template={model_template_name}"

    print("\n" + "="*60)
    print(f"✅ Configuration generated successfully!")
    print(f"  Preprocess: {preprocess_template_name}")
    print(f"  Model:      {model_template_name}")

    if args.enqueue:
        if _QUEUE_AVAILABLE:
            try:
                # Use TaskQueue directly
                queue = TaskQueue(Path("projects/kaggle") / args.project)
                # Note: TaskQueue expects just the command string, NOT 'python scripts/mla.py ...'
                # It prefixes 'mla ' internally if needed or runs raw.
                # Looking at task_queue.py, it expects e.g. "model model_template=..."
                
                queue.add_task(command_str, priority=10)
                print(f"\n🚀 [ENQUEUED] Task added to queue: {command_str}")
            except Exception as e:
                err(f"Failed to enqueue task: {e}")
        else:
            err("Queue module not found (src/mlarena/utils/queue.py missing?)")
    else:
        print("\nRun this experiment with:")
        print(f"\n{full_cli_cmd}")
        print("\nOr enqueue it:")
        print(f"python scripts/mla.py queue --project {args.project} add \"{command_str}\"")

    print("="*60 + "\n")

if __name__ == "__main__":
    main()