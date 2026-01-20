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

def warn(msg):
    print(f"[WARNING] {msg}")

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
    preprocess_template_dir = Path("projects/kaggle") / project / "templates" / "preprocess"
    model_template_dir = Path("projects/kaggle") / project / "templates" / "model"
    
    search_dirs = [preprocess_template_dir, model_template_dir]

    # If using remote environment, assume templates might be on NFS (mounted at /mnt/mlarena)
    if Path("/mnt/mlarena").exists():
         search_dirs.append(Path("/mnt/mlarena") / "projects/kaggle" / project / "templates" / "preprocess")
         search_dirs.append(Path("/mnt/mlarena/projects/kaggle") / project / "templates" / "model")
    
    max_ver = 0
    pattern = re.compile(rf"^{re.escape(base_prefix)}_(\d{{3}})\.yaml$")
    
    # Scan directory
    for sp in search_dirs: # Iterate over all potential dirs
        if not sp.exists(): continue
        for item in sp.iterdir():
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

def generate_preprocess_templates(project, fixed_actions, mcts_actions, final_name, config):
    template_dir = Path("projects/kaggle") / project / "templates" / "preprocess"
    template_dir.mkdir(parents=True, exist_ok=True)
    
    # Build Order Map from Config
    order_map = {}
    if config and "preprocessors" in config:
        for idx, p in enumerate(config["preprocessors"]):
            p_name = p.get("name")
            p_group = p.get("group")
            if p_name:
                order_map[p_name] = idx
            if p_group and p_group not in order_map:
                order_map[p_group] = idx

    # Normalize and Combine Actions
    combined_actions = []
    
    # Add Fixed Actions
    for action in fixed_actions:
        combined_actions.append(action)

    # Add MCTS Actions
    for action in mcts_actions:
        step_name = action.get("step_name") or action.get("group_name")
        # Preserve original MCTS action data + normalized step_name
        mcts_item = action.copy()
        mcts_item["step_name"] = step_name
        mcts_item["is_fixed"] = False
        combined_actions.append(mcts_item)

    # Sort Actions
    def get_sort_index(action):
        name = action.get("step_name")
        return order_map.get(name, 999) # Default to end if not found

    combined_actions.sort(key=get_sort_index)
    
    chain_steps = []
    
    for i, action in enumerate(combined_actions):
        step_name = action["step_name"]
        is_fixed = action.get("is_fixed", False)
        
        if is_fixed:
            # Fixed Action Structure
            module = action["module"]
            config_data = action["config"]
            
            sub_data = {
                "module": module,
                "config": config_data
            }
        else:
            # MCTS Action Structure
            variant = action.get("variant")
            config_data = action.get("config", {})
            
            sub_data = {
                "name": step_name,
                "variant": variant,
                "config": config_data
            }

        # Unified Naming: final_name-00-stepname.yaml
        sub_filename = f"{final_name}-{i:02d}-{step_name}.yaml"
        sub_path = template_dir / sub_filename
        
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
    
    # Template resolution logic
    resolved_model = base_model
    additional_config = {}

    # Check if base_model is a template name (YAML)
    # Search paths: projects/kaggle/<project>/templates/model/, src/mlarena/templates/model/, templates/model/
    search_paths = [
        Path("projects/kaggle") / project / "templates" / "model",
        Path("src/mlarena/templates/model"),
        Path("templates/model")
    ]
    
    # If using remote environment, assume templates might be on NFS (mounted at /mnt/mlarena)
    # This is a heuristic, but covers the common case where you want to use the framework defaults from NFS
    # We check if /mnt/mlarena exists first
    if Path("/mnt/mlarena").exists():
         search_paths.append(Path("/mnt/mlarena/src/mlarena/templates/model"))
         search_paths.append(Path("/mnt/mlarena/templates/model"))
    
    template_found = False
    for sp in search_paths:
        yaml_path = sp / f"{base_model}.yaml"
        if yaml_path.exists():
            try:
                with open(yaml_path) as f:
                    tpl_data = yaml.safe_load(f)
                if tpl_data and "model" in tpl_data:
                    info(f"Resolved model template '{base_model}' from {yaml_path}")
                    resolved_model = tpl_data["model"] # The real .py implementation name
                    # Copy other keys (preset, time_limit, included_model_types, etc.)
                    for k, v in tpl_data.items():
                        if k != "model" and k != "preprocess_template": # Don't copy preprocess linkage
                            if k == "config":
                                if "config" not in additional_config:
                                    additional_config["config"] = {}
                                additional_config["config"].update(v)
                            else:
                                additional_config[k] = v
                    template_found = True
                    break
            except Exception as e:
                warn(f"Failed to load template {yaml_path}: {e}")

    if not template_found:
        # Fallback for known templates if file lookup fails (safety net)
        if base_model == "autogluon_thin_fast":
            resolved_model = "autogluon_baseline"
            additional_config["preset"] = "medium"
            additional_config["time_limit"] = 600
        elif base_model == "autogluon_best_quality":
            resolved_model = "autogluon_baseline"
            additional_config["preset"] = "best_quality"
            additional_config["time_limit"] = 3600
        elif base_model == "cpu-best-1h-boost":
            resolved_model = "autogluon_baseline" 
            additional_config["time_limit"] = 3600
            additional_config["preset"] = "best"
            additional_config["included_model_types"] = ["GBM", "CAT", "XGB"]

    
    # Generate the model template file
    model_template_name = f"{final_name}"
    model_path = template_dir / f"{model_template_name}.yaml"
    
    model_data = {
        "model": resolved_model,
        "preprocess_template": final_name, # CRITICAL: Link model to preprocess
        "config": {
            "random_state": seed,
        }
    }
    
    # Merge additional config from template resolution
    for k, v in additional_config.items():
        if k == "config":
            model_data["config"].update(v)
        else:
            model_data[k] = v
    
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

    # Auto-resolve config path for remote env if not overridden
    default_config = "conf/preprocess/mla_super_chain.yaml"
    if args.config == default_config and args.env == "remote":
        remote_config = Path("/mnt/mlarena/conf/preprocess/mla_super_chain.yaml")
        if remote_config.exists():
            args.config = str(remote_config)
            info(f"Using remote config: {args.config}")
        else:
            warn(f"Remote config not found at {remote_config}, using local default.")

    config_path = Path(args.config)
    if not config_path.exists():
        err(f"Config not found: {config_path}")
        return

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    study_name = args.study or cfg.get("mcts", {}).get("study_name")
    if not study_name:
        err("Study name not provided and not found in config.")
        return

    db_path = None
    if args.db:
        db_path = Path(args.db)
    else:
        storage_url = cfg.get("mcts", {}).get("storage_url", "")
        if storage_url.startswith("sqlite:///"):
            storage_path = Path(storage_url.replace("sqlite:///", "", 1))
            if storage_path == Path("experiments/db/mcts.db"):
                db_path = Path("projects/kaggle") / args.project / storage_path
            elif storage_path.is_absolute():
                db_path = storage_path
            else:
                db_path = Path("projects/kaggle") / args.project / storage_path
        else:
            db_path = Path("projects/kaggle") / args.project / "experiments" / "db" / "mcts.db"

    if args.env == "remote":
        if not db_path.is_absolute():
            db_path = Path("/mnt/mlarena") / db_path
        elif not db_path.exists():
            workspace_root = Path(__file__).resolve().parent.parent
            try:
                db_path = Path("/mnt/mlarena") / db_path.relative_to(workspace_root)
            except ValueError:
                pass

    if not db_path.exists():
        err(f"Database not found: {db_path}")
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
    preprocess_template_name = generate_preprocess_templates(args.project, fixed_actions, mcts_actions, final_name, cfg)
    
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
