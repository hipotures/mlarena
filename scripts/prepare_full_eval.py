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
    parser.add_argument("--exp-dir", help="Custom experiments directory")
    args = parser.parse_args()

    from rich.console import Console
    console = Console()

    # Path to current repo
    repo_root = Path(__file__).resolve().parents[1]
    project_rel_path = f"projects/kaggle/{args.project}"
    local_project_dir = repo_root / project_rel_path
    
    if not local_project_dir.exists():
        console.print(f"[bold red]Error:[/bold red] Local project directory {local_project_dir} not found")
        return

    # Potential experiment locations
    exp_dirs = []
    if args.exp_dir:
        exp_dirs.append(Path(args.exp_dir))
    
    exp_dirs.extend([
        local_project_dir / "experiments",
        Path("/mnt/mlarena") / project_rel_path / "experiments"
    ])
    
    raw_results = []
    submitted_exp_ids = set()
    submitted_templates = {} # tmpl -> score
    scanned_dirs = []
    
    for d in exp_dirs:
        if not d.exists():
            continue
        
        console.print(f"Scanning experiments in [dim]{d}[/dim]...")
        scanned_dirs.append(str(d))
        
        for state_path in d.glob("**/state.json"):
            if "artifacts" in state_path.parts: continue
            try:
                with open(state_path, 'r') as f:
                    state = json.load(f)
                
                exp_id = state.get("experiment_id")
                modules = state.get("modules", {})
                
                # Check if this experiment has a completed submission
                submit_info = modules.get("submit", {})
                is_submitted = submit_info.get("status") == "completed"
                
                fetch_info = modules.get("fetch-score", {}) or modules.get("fetch_score", {})
                public_score = fetch_info.get("payload", {}).get("score")
                
                model_info = modules.get("model", {})
                model_template = model_info.get("invocation", {}).get("model_template")
                preproc_template = model_info.get("invocation", {}).get("preprocess_template")
                if not preproc_template:
                    preproc_info = modules.get("preprocess", {}) or modules.get("pre-process", {}) or {}
                    preproc_template = preproc_info.get("invocation", {}).get("preprocess_template")
                
                # If submitted, track both ID and template name
                if is_submitted:
                    if exp_id: submitted_exp_ids.add(exp_id)
                    if model_template:
                        if model_template not in submitted_templates or public_score is not None:
                            submitted_templates[model_template] = public_score
                        
                        # Handle _full version tracking
                        if model_template.endswith("_full"):
                            base_tmpl = model_template[:-5]
                            if base_tmpl not in submitted_templates or submitted_templates[base_tmpl] is None:
                                submitted_templates[base_tmpl] = public_score
                
                score = model_info.get("payload", {}).get("local_cv_score")
                
                if score is not None:
                    # Apply mask filter if provided
                    if args.mask and model_template and not model_template.startswith(args.mask):
                        continue

                    raw_results.append({
                        "score": score,
                        "model_template": model_template,
                        "preprocess_template": preproc_template,
                        "exp_id": exp_id,
                        "is_submitted": is_submitted,
                        "public_score": public_score
                    })
            except Exception:
                continue

    if not raw_results:
        console.print(f"[bold red]Error:[/bold red] No experiments with scores found.")
        return

    # Sort all by score (descending)
    raw_results.sort(key=lambda x: x['score'], reverse=True)
    
    # Deduplicate by model_template and skip existing _full templates
    seen_templates = set()
    unique_top = []
    for r in raw_results:
        tmpl = r['model_template']
        if not tmpl: continue
        if tmpl.endswith("_full"): continue
        
        if tmpl not in seen_templates:
            # Re-check submission status from the aggregated maps
            r["is_submitted_tmpl"] = (tmpl in submitted_templates or f"{tmpl}_full" in submitted_templates)
            r["final_public_score"] = submitted_templates.get(tmpl) or submitted_templates.get(f"{tmpl}_full")
            
            unique_top.append(r)
            seen_templates.add(tmpl)
        if len(unique_top) >= args.n: break

    console.print(f"\n[bold]Top {len(unique_top)} templates to upgrade:[/bold]")
    for res in unique_top:
        orig_model_name = res['model_template']
        full_model_name = f"{orig_model_name}_full"
        
        is_done = res["is_submitted_tmpl"]
        style = "dim" if is_done else "bold green"
        
        # CV Score info
        score_info = f"CV: {res['score']:.5f}"
        
        # Status and Public Score info
        status_parts = []
        if is_done:
            status_parts.append("SUBMITTED")
        if res["final_public_score"] is not None:
            status_parts.append(f"Public: {res['final_public_score']:.5f}")
            
        status_tag = ""
        if status_parts:
            tag_content = " | ".join(status_parts)
            status_tag = f" [blue]({tag_content})[/blue]"
        
        console.print(f"  [{style}]{orig_model_name}[/{style}] ({score_info}) -> {full_model_name}{status_tag}")

        # Upgrade Model Template
        model_tmpl_path = local_project_dir / "templates" / "model" / f"{orig_model_name}.yaml"
        model_tmpl = load_yaml(model_tmpl_path)

        if model_tmpl:
            full_model_tmpl = model_tmpl.copy()
            orig_preproc = model_tmpl.get("preprocess_template")
        else:
            full_model_tmpl = {}
            orig_preproc = res.get("preprocess_template")
        full_preproc = f"{orig_preproc}_full" if orig_preproc else None
        
        if model_tmpl:
            full_model_tmpl["preprocess_template"] = full_preproc
            full_model_tmpl["preset"] = "best"
            full_model_tmpl["time_limit"] = 3600
            full_model_tmpl["included_model_types"] = ["GBM", "XGB", "CAT"]
            
            # Remove fit_args to allow 'best' preset to use high-quality defaults
            if "fit_args" in full_model_tmpl:
                del full_model_tmpl["fit_args"]
        else:
            # Fallback: create a default full model template
            full_model_tmpl = {
                "model": "autogluon_baseline",
                "preset": "high",
                "time_limit": 600,
                "included_model_types": ["GBM", "XGB", "CAT"],
            }
            if full_preproc:
                full_model_tmpl["preprocess_template"] = full_preproc
            console.print(
                f"  [yellow]i[/yellow] Missing model template {orig_model_name}.yaml -> "
                f"created default {full_model_name} (high, 10m, GBM+XGB+CAT)"
            )
        
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

    console.print("\n[bold green]Success.[/bold green] Full evaluation templates created in local project folder.")

    if args.enqueue:
        # Add to path to import TaskQueue
        sys.path.insert(0, str(repo_root / "src"))
        try:
            from mlarena.utils.queue import TaskQueue
            queue = TaskQueue(local_project_dir)
            
            to_enqueue = [r for r in unique_top if not r["is_submitted_tmpl"]]
            
            if not to_enqueue:
                console.print("\n[yellow]No new tasks to enqueue (all top experiments already submitted).[/yellow]")
                return

            console.print(f"\nAdding {len(to_enqueue)} tasks to queue...")
            
            for res in to_enqueue:
                full_model_name = f"{res['model_template']}_full"
                exp_id = f"exp-{full_model_name}"
                cmd = f"{args.module} model_template={full_model_name} experiment_id={exp_id} submit.confirm_timeout=0"
                
                task_id = queue.add_task(
                    command=cmd,
                    priority=10
                )
                console.print(f"  [green]+ Added to Queue ({args.module}):[/green] {full_model_name} (Task #{task_id})")
            
            console.print("\nQueue updated. Run 'mla queue list' to verify.")
        except ImportError:
            console.print("\n[red]Error:[/red] Could not import TaskQueue. Ensure MLArena source is in path.")

if __name__ == "__main__":
    main()
