#!/usr/bin/env python3
import argparse
import json
import yaml
from pathlib import Path
import os

def load_yaml(path):
    if not path.exists(): return None
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)

def main():
    parser = argparse.ArgumentParser(description="Prepare full evaluation templates for top N experiments")
    parser.add_argument("--project", required=True, help="Project name")
    parser.add_argument("-n", type=int, default=5, help="Number of top experiments to process")
    args = parser.parse_args()

    # Path to current repo
    repo_root = Path(__file__).resolve().parents[1]
    project_rel_path = f"projects/kaggle/{args.project}"
    local_project_dir = repo_root / project_rel_path
    
    if not local_project_dir.exists():
        print(f"Error: Local project directory {local_project_dir} not found")
        return

    # NFS Experiments path (structure 1:1)
    mnt_exp_dir = Path("/mnt/mlarena") / project_rel_path / "experiments"
    
    if not mnt_exp_dir.exists():
        print(f"Error: NFS experiments directory {mnt_exp_dir} not found.")
        print("Check if /mnt/mlarena is mounted correctly.")
        return

    print(f"Scanning NFS experiments in {mnt_exp_dir}...")
    results = []
    
    for state_path in mnt_exp_dir.glob("**/state.json"):
        if "artifacts" in state_path.parts: continue
        try:
            with open(state_path, 'r') as f:
                state = json.load(f)
            
            # Use local_cv_score from model module
            model_info = state.get("modules", {}).get("model", {})
            score = model_info.get("payload", {}).get("local_cv_score")
            
            if score is not None:
                results.append({
                    "score": score,
                    "model_template": model_info.get("invocation", {}).get("model_template"),
                    "exp_id": state.get("experiment_id")
                })
        except Exception:
            continue

    if not results:
        print(f"No experiments with scores found.")
        return

    # 2. Sort by score (descending)
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Deduplicate by model_template
    seen_templates = set()
    unique_top = []
    for r in results:
        tmpl = r['model_template']
        if tmpl and tmpl not in seen_templates:
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
        save_yaml(full_model_tmpl, local_project_dir / "templates" / "model" / f"{full_model_name}.yaml")

        # Upgrade Preprocess Chain/Modules
        if orig_preproc:
            preproc_tmpl_path = local_project_dir / "templates" / "preprocess" / f"{orig_preproc}.yaml"
            preproc_tmpl = load_yaml(preproc_tmpl_path)
            if not preproc_tmpl: continue

            if "chain" in preproc_tmpl:
                full_chain = []
                for step in preproc_tmpl["chain"]:
                    full_step = f"{step}_full"
                    full_chain.append(full_step)
                    # Copy module config
                    step_path = local_project_dir / "templates" / "preprocess" / f"{step}.yaml"
                    step_data = load_yaml(step_path)
                    if step_data:
                        save_yaml(step_data, local_project_dir / "templates" / "preprocess" / f"{full_step}.yaml")
                
                save_yaml({"chain": full_chain}, local_project_dir / "templates" / "preprocess" / f"{full_preproc}.yaml")
            else:
                # Single module
                save_yaml(preproc_tmpl, local_project_dir / "templates" / "preprocess" / f"{full_preproc}.yaml")

    print("\nSuccess. Run 'mla queue add' for the new _full templates.")

if __name__ == "__main__":
    main()