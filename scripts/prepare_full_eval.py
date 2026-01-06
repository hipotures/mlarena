#!/usr/bin/env python3
import argparse
import json
import yaml
from pathlib import Path
import os
import sys

def load_yaml(path):
    if not path.exists(): return None
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)

def main():
    examples = """
Examples:
  # 1. Just generate _full templates for the top 5 experiments:
  python3 scripts/prepare_full_eval.py --project playground-series-s6e1 -n 5

  # 2. Filter top experiments from a specific series (mask) and generate templates:
  python3 scripts/prepare_full_eval.py --project playground-series-s6e1 --mask test_c_01_ -n 3

  # 3. Generate and automatically add to queue for model training (Model + Preproc):
  python3 scripts/prepare_full_eval.py --project playground-series-s6e1 -n 5 --enqueue

  # 4. Generate and automatically add to queue for the WHOLE FLOW (Preproc + Model + Predict + Submit + Score):
  python3 scripts/prepare_full_eval.py --project playground-series-s6e1 -n 5 --enqueue --module fetch-score

  # 5. Run on a server where experiments are in a custom directory:
  python3 scripts/prepare_full_eval.py --project playground-series-s6e1 --exp-dir /mnt/mlarena/my_custom_path
    """
    parser = argparse.ArgumentParser(
        description="Prepare full evaluation templates for top N experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=examples
    )
    parser.add_argument("--project", required=True, help="Project name")
    parser.add_argument("-n", type=int, default=5, help="Number of top experiments to process")
    parser.add_argument("--mask", help="Filter templates by prefix (e.g., test_c_01_)")
    parser.add_argument("--enqueue", action="store_true", help="Automatically add generated templates to queue")
    parser.add_argument("--module", default="model", help="Module to run in queue (default: model, use fetch-score for full flow)")
    args = parser.parse_args()

    # Path to current repo
    repo_root = Path(__file__).resolve().parents[1]
    project_rel_path = f"projects/kaggle/{args.project}"
    local_project_dir = repo_root / project_rel_path
    
    if not local_project_dir.exists():
        print(f"Error: Local project directory {local_project_dir} not found")
        return

    # Potential experiment locations
    exp_dirs = [
        local_project_dir / "experiments",
        Path("/mnt/mlarena") / project_rel_path / "experiments"
    ]
    
    results = []
    scanned_dirs = []
    
    for d in exp_dirs:
        if not d.exists():
            continue
        
        print(f"Scanning experiments in {d}...")
        scanned_dirs.append(str(d))
        
        for state_path in d.glob("**/state.json"):
            if "artifacts" in state_path.parts: continue
            try:
                with open(state_path, 'r') as f:
                    state = json.load(f)
                
                model_info = state.get("modules", {}).get("model", {})
                score = model_info.get("payload", {}).get("local_cv_score")
                
                if score is not None:
                    model_template = model_info.get("invocation", {}).get("model_template")
                    
                    # Apply mask filter if provided
                    if args.mask and model_template and not model_template.startswith(args.mask):
                        continue

                    results.append({
                        "score": score,
                        "model_template": model_template,
                        "exp_id": state.get("experiment_id")
                    })
            except Exception:
                continue

    if not results:
        print(f"Error: No experiments with scores found.")
        print(f"Checked directories: {', '.join(scanned_dirs) if scanned_dirs else 'None found'}")
        return

    # 2. Sort by score (descending)
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Deduplicate by model_template and skip existing _full templates
    seen_templates = set()
    unique_top = []
    for r in results:
        tmpl = r['model_template']
        if not tmpl: continue
        if tmpl.endswith("_full"): continue
        
        if tmpl not in seen_templates:
            unique_top.append(r)
            seen_templates.add(tmpl)
        if len(unique_top) >= args.n: break

    print(f"Top {len(unique_top)} templates to upgrade:")
    for res in unique_top:
        orig_model_name = res['model_template']
        full_model_name = f"{orig_model_name}_full"
        print(f"  {orig_model_name} ({res['score']:.5f}) -> {full_model_name}")

        # Upgrade Model Template
        model_tmpl_path = local_project_dir / "templates" / "model" / f"{orig_model_name}.yaml"
        model_tmpl = load_yaml(model_tmpl_path)
        if not model_tmpl: continue

        full_model_tmpl = model_tmpl.copy()
        orig_preproc = model_tmpl.get("preprocess_template")
        full_preproc = f"{orig_preproc}_full" if orig_preproc else None
        
        full_model_tmpl["preprocess_template"] = full_preproc
        full_model_tmpl["preset"] = "best"
        full_model_tmpl["time_limit"] = 3600
        full_model_tmpl["included_model_types"] = ["GBM", "XGB", "CAT"]
        
        # Remove fit_args to allow 'best' preset to use high-quality defaults
        if "fit_args" in full_model_tmpl:
            del full_model_tmpl["fit_args"]
            
        save_yaml(full_model_tmpl, local_project_dir / "templates" / "model" / f"{full_model_name}.yaml")

        # Upgrade Preprocess Chain/Modules
        if orig_preproc:
            preproc_tmpl_path = local_project_dir / "templates" / "preprocess" / f"{orig_preproc}.yaml"
            preproc_tmpl = load_yaml(preproc_tmpl_path)
            if not preproc_tmpl: continue

            if "chain" in preproc_tmpl:
                full_chain = []
                for step in preproc_tmpl["chain"]:
                    # Load step to see if it's a data splitter
                    step_path = local_project_dir / "templates" / "preprocess" / f"{step}.yaml"
                    step_data = load_yaml(step_path)
                    
                    if step_data and step_data.get("module") == "train_fraction":
                        # REMOVE data splitters for full evaluation (use 100% data)
                        print(f"  - Removing data splitter: {step}")
                        continue
                    
                    full_step = f"{step}_full"
                    full_chain.append(full_step)
                    
                    # Copy module config
                    if step_data:
                        save_yaml(step_data, local_project_dir / "templates" / "preprocess" / f"{full_step}.yaml")
                
                save_yaml({"chain": full_chain}, local_project_dir / "templates" / "preprocess" / f"{full_preproc}.yaml")
            else:
                # Single module
                save_yaml(preproc_tmpl, local_project_dir / "templates" / "preprocess" / f"{full_preproc}.yaml")

    print("\nSuccess. Full evaluation templates created in local project folder.")

    if args.enqueue:
        # Add to path to import TaskQueue
        sys.path.insert(0, str(repo_root / "src"))
        from mlarena.utils.queue import TaskQueue
        
        queue = TaskQueue(local_project_dir)
        print(f"\nAdding {len(unique_top)} tasks to queue...")
        
        for res in unique_top:
            full_model_name = f"{res['model_template']}_full"
            # Command format: [module] --model-template [name]
            # We also set submit.confirm_timeout=0 for non-interactive queue runs.
            cmd = f"{args.module} --model-template {full_model_name} submit.confirm_timeout=0"
            
            task_id = queue.add_task(
                command=cmd,
                priority=10
            )
            print(f"  + Added to Queue ({args.module}): {full_model_name} (Task #{task_id})")
        
        print("\nQueue updated. Run 'mla queue list' to verify.")

if __name__ == "__main__":
    main()