#!/usr/bin/env python
import sqlite3
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

def get_best_pipeline(db_path, study_name):
    """Get the best pipeline from the study.

    Returns:
        dict: Pipeline data with 'pipeline' (dict), 'value' (float), 'trial_id' (int), 'phase' (str), 'round' (int)
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Find study_id
    study_query = "SELECT study_id FROM studies WHERE study_name = ?"
    study_row = conn.execute(study_query, (study_name,)).fetchone()
    if not study_row:
        err(f"Study {study_name} not found in DB")
        return None
    study_id = study_row["study_id"]

    # Find absolute best trial
    best_trial_query = """
    SELECT rt.trial_id, t.number, tv.value, rt.pipeline_json, rt.phase, rt.round, rt.parent_trial_id
    FROM rant_trials rt
    JOIN trials t ON rt.trial_id = t.trial_id
    JOIN trial_values tv ON rt.trial_id = tv.trial_id
    WHERE rt.study_id = ? AND t.state = 1
    ORDER BY tv.value DESC LIMIT 1
    """
    best_trial = conn.execute(best_trial_query, (study_id,)).fetchone()
    if not best_trial:
        err(f"No completed trials found for study {study_name}")
        return None

    result = {
        'trial_id': best_trial['trial_id'],
        'trial_number': best_trial['number'],
        'value': best_trial['value'],
        'pipeline': json.loads(best_trial['pipeline_json']),
        'phase': best_trial['phase'],
        'round': best_trial['round'],
        'parent_trial_id': best_trial['parent_trial_id']
    }

    info(f"Best trial found: ID={result['trial_id']} (Number {result['trial_number']}), "
         f"Score={result['value']:.5f}, Phase={result['phase']}, Round={result['round']}")

    conn.close()
    return result

def get_best_by_round(db_path, study_name):
    """Get best pipeline from each round.

    Returns:
        list: List of pipeline dicts for each round
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Find study_id
    study_query = "SELECT study_id FROM studies WHERE study_name = ?"
    study_row = conn.execute(study_query, (study_name,)).fetchone()
    if not study_row:
        err(f"Study {study_name} not found in DB")
        return []
    study_id = study_row["study_id"]

    # Find best trial for each round
    query = """
    SELECT rt.round, rt.trial_id, t.number, tv.value, rt.pipeline_json, rt.phase
    FROM rant_trials rt
    JOIN trials t ON rt.trial_id = t.trial_id
    JOIN trial_values tv ON rt.trial_id = tv.trial_id
    WHERE rt.study_id = ? AND t.state = 1
    GROUP BY rt.round
    HAVING tv.value = MAX(tv.value)
    ORDER BY rt.round ASC
    """
    results = conn.execute(query, (study_id,)).fetchall()
    conn.close()

    pipelines = []
    for row in results:
        info(f"Round {row['round']}: Best Trial={row['trial_id']} (Number {row['number']}), "
             f"Score={row['value']:.5f}, Phase={row['phase']}")
        pipelines.append({
            'trial_id': row['trial_id'],
            'trial_number': row['number'],
            'value': row['value'],
            'pipeline': json.loads(row['pipeline_json']),
            'phase': row['phase'],
            'round': row['round']
        })
    return pipelines

def get_best_top_n_diverse(db_path, study_name, top_n):
    """Get top N best pipelines from different trees (different parents).

    Args:
        db_path: Path to database
        study_name: Study name
        top_n: Number of diverse pipelines to retrieve

    Returns:
        list: List of pipeline dicts, each from a different parent tree
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Find study_id
    study_query = "SELECT study_id FROM studies WHERE study_name = ?"
    study_row = conn.execute(study_query, (study_name,)).fetchone()
    if not study_row:
        err(f"Study {study_name} not found in DB")
        return []
    study_id = study_row["study_id"]

    # Get all completed trials sorted by value (best first)
    query = """
    SELECT rt.trial_id, t.number, tv.value, rt.pipeline_json, rt.phase, rt.round, rt.parent_trial_id
    FROM rant_trials rt
    JOIN trials t ON rt.trial_id = t.trial_id
    JOIN trial_values tv ON rt.trial_id = tv.trial_id
    WHERE rt.study_id = ? AND t.state = 1
    ORDER BY tv.value DESC
    """
    all_trials = conn.execute(query, (study_id,)).fetchall()
    conn.close()

    if not all_trials:
        err(f"No completed trials found for study {study_name}")
        return []

    # Select diverse pipelines (different parents)
    selected = []
    seen_parents = set()

    for row in all_trials:
        parent_id = row['parent_trial_id']

        # For trials with parent_trial_id = NULL (random phase), treat each as unique
        # For trials with a parent, check if we already have a trial from this parent
        if parent_id is None:
            # Random phase trial - each is unique (no parent)
            parent_key = f"random_{row['trial_id']}"
        else:
            parent_key = f"parent_{parent_id}"

        if parent_key not in seen_parents:
            selected.append({
                'trial_id': row['trial_id'],
                'trial_number': row['number'],
                'value': row['value'],
                'pipeline': json.loads(row['pipeline_json']),
                'phase': row['phase'],
                'round': row['round'],
                'parent_trial_id': parent_id
            })
            seen_parents.add(parent_key)

            info(f"Selected #{len(selected)}: Trial={row['trial_id']} (Number {row['number']}), "
                 f"Score={row['value']:.5f}, Phase={row['phase']}, Round={row['round']}, "
                 f"Parent={parent_id if parent_id else 'None (random)'}")

            if len(selected) >= top_n:
                break

    if len(selected) < top_n:
        warn(f"Only found {len(selected)} diverse trials (requested {top_n})")

    return selected

def extract_pipeline_steps(pipeline_data):
    """Extract preprocessing steps from RANT pipeline JSON.

    Args:
        pipeline_data: Pipeline dict from RANT trial

    Returns:
        list: List of step dicts with 'step_name', 'module', 'variant', 'config'
    """
    # RANT pipeline has 'full_chain' which is a list of steps
    full_chain = pipeline_data.get('full_chain', [])

    steps = []
    for step in full_chain:
        if step.get("enabled") is False:
            disabled_name = step.get("step") or step.get("name") or step.get("group") or step.get("template")
            warn(f"Skipping disabled step: {disabled_name}")
            continue

        module_name = step.get('template') or step.get('name') or step.get('step') or step.get('group')
        step_name = step.get('step') or step.get('name') or step.get('group') or module_name
        variant = step.get('variant')
        # Prefer config, fallback to fixed_config for fixed steps
        config = step.get('config') or step.get('fixed_config', {})

        if step_name and module_name:
            steps.append({
                'step_name': step_name,
                'module': module_name,
                'variant': variant,
                'config': config
            })

    return steps

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
    for sp in search_dirs:
        if not sp.exists():
            continue
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

def generate_preprocess_templates(project, steps, final_name):
    """Generate preprocessing templates from RANT pipeline steps.

    Args:
        project: Project name
        steps: List of step dicts from extract_pipeline_steps
        final_name: Template name (e.g., 'rant_study_001')

    Returns:
        str: Template name
    """
    template_dir = Path("projects/kaggle") / project / "templates" / "preprocess"
    template_dir.mkdir(parents=True, exist_ok=True)

    chain_steps = []

    for i, step in enumerate(steps):
        step_name = step['step_name']
        module_name = step.get("module") or step_name
        variant = step.get('variant')
        config_data = step['config']

        # Generate step template
        sub_data = {"module": module_name, "config": config_data}
        if variant is not None:
            sub_data["variant"] = variant

        # Naming: final_name-00-stepname.yaml
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
    info(f"Created main chain template: {final_name}.yaml with {len(chain_steps)} steps")
    return final_name

def generate_model_template(project, config, fast_mode, final_name):
    """Generate model template that links to the preprocessing chain."""
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
    search_paths = [
        Path("projects/kaggle") / project / "templates" / "model",
        Path("src/mlarena/templates/model"),
        Path("templates/model")
    ]

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
                    resolved_model = tpl_data["model"]
                    for k, v in tpl_data.items():
                        if k != "model" and k != "preprocess_template":
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
        # Fallback for known templates
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
        "preprocess_template": final_name,
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
    parser = argparse.ArgumentParser(
        description="Reconstruct preprocessing chain from RANT study results"
    )
    parser.add_argument("--project", default="playground-series-s6e1")
    parser.add_argument("--config", default="conf/preprocess/mla_super_chain.yaml")
    parser.add_argument("--study", help="Override study name from config")
    parser.add_argument("--db", help="Override database path")
    parser.add_argument("--mode", choices=["best", "best_per_round", "best_top_n"], default="best",
                        help="Reconstruction mode: 'best' for single best pipeline, 'best_per_round' for best from each round, 'best_top_n' for top N diverse pipelines")
    parser.add_argument("--top-n", type=int, default=5, help="Number of diverse pipelines to generate (for best_top_n mode)")
    parser.add_argument("--prefix", help="Override template prefix (default: rant_<study>)")
    parser.add_argument("--fast", action="store_true", help="Use evaluation settings (fast model, subset data)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the latest existing version")
    parser.add_argument("--enqueue", action="store_true", help="Enqueue the generated experiment(s)")
    parser.add_argument("--env", choices=["local", "remote"], default="local", help="Environment (local or remote/NFS)")
    args = parser.parse_args()

    # Auto-resolve config path for remote env
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

    study_name = args.study or cfg.get("rant", {}).get("study_name")
    if not study_name:
        err("Study name not provided and not found in config.")
        return

    # Resolve database path
    db_path = None
    if args.db:
        db_path = Path(args.db)
    else:
        storage_url = cfg.get("rant", {}).get("storage_url", "")
        if storage_url.startswith("sqlite:///"):
            storage_path = Path(storage_url.replace("sqlite:///", ""))
            if storage_path == Path("experiments/db/rant.db"):
                db_path = Path("projects/kaggle") / args.project / storage_path
            elif storage_path.is_absolute():
                db_path = storage_path
            else:
                db_path = Path("projects/kaggle") / args.project / storage_path
        else:
            db_path = Path("projects/kaggle") / args.project / "experiments" / "db" / "rant.db"

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

    # Determine prefix
    prefix = args.prefix
    if not prefix:
        prefix = f"rant_{study_name}"
        if args.fast:
            prefix += "_eval"
        else:
            prefix += "_prod"

    info(f"Using database: {db_path}")
    info(f"Using study: {study_name}")
    info(f"Mode: {args.mode}")
    if args.mode == "best_top_n":
        info(f"Top N: {args.top_n}")
    info(f"Profile: {'FAST/EVAL' if args.fast else 'PRODUCTION'}")

    # Get pipeline(s) from RANT based on mode
    if args.mode == "best":
        best_pipeline = get_best_pipeline(db_path, study_name)
        if not best_pipeline:
            err("Could not retrieve best pipeline.")
            return

        # Extract steps
        steps = extract_pipeline_steps(best_pipeline['pipeline'])
        if not steps:
            err("No preprocessing steps found in pipeline.")
            return

        info(f"Extracted {len(steps)} preprocessing steps from best pipeline")

        # Resolve Version
        final_name = resolve_next_version(args.project, prefix, overwrite=args.overwrite)
        info(f"Target Template Name: {final_name}")

        # Generate templates
        preprocess_template_name = generate_preprocess_templates(args.project, steps, final_name)
        model_template_name = generate_model_template(args.project, cfg, fast_mode=args.fast, final_name=final_name)

        # Final info
        print("\n" + "="*60)
        print("✅ Configuration generated successfully!")
        print(f"  Preprocess: {preprocess_template_name}")
        print(f"  Model:      {model_template_name}")
        print(f"  Source:     Trial #{best_pipeline['trial_number']} (Score: {best_pipeline['value']:.5f})")

        if args.enqueue:
            if _QUEUE_AVAILABLE:
                try:
                    queue = TaskQueue(Path("projects/kaggle") / args.project)
                    command_str = f"model_template={model_template_name}"
                    queue.add_task(command_str, priority=10)
                    print(f"\n🚀 [ENQUEUED] Task added to queue: {command_str}")
                except Exception as e:
                    err(f"Failed to enqueue task: {e}")
            else:
                err("Queue module not found (src/mlarena/utils/queue.py missing?)")
        else:
            auto_flow_cmd = f"uv run python scripts/mla.py project={args.project} model_template={model_template_name}"
            only_model_cmd = f"uv run python scripts/mla.py model project={args.project} model_template={model_template_name}"

            print("\nRun full auto-flow (model -> predict -> submit -> fetch-score):")
            print(f"  {auto_flow_cmd}")
            print(f"  {auto_flow_cmd} common.use_gpu=true")

            print("\nRun only model training (no submit/fetch):")
            print(f"  {only_model_cmd}")
            print(f"  {only_model_cmd} common.use_gpu=true")

            print("\nOr enqueue it:")
            print(f"  python scripts/mla.py queue --project {args.project} add \"model_template={model_template_name}\"")
            print(f"  python scripts/mla.py queue --project {args.project} add \"model_template={model_template_name} common.use_gpu=true\"")

        print("="*60 + "\n")

    elif args.mode == "best_per_round":
        pipelines = get_best_by_round(db_path, study_name)
        if not pipelines:
            err("Could not retrieve pipelines by round.")
            return

        # Use the last round's best pipeline
        best_pipeline = pipelines[-1]
        steps = extract_pipeline_steps(best_pipeline['pipeline'])
        if not steps:
            err("No preprocessing steps found in pipeline.")
            return

        info(f"Using best pipeline from round {best_pipeline['round']} with {len(steps)} steps")

        # Resolve Version
        final_name = resolve_next_version(args.project, prefix, overwrite=args.overwrite)
        info(f"Target Template Name: {final_name}")

        # Generate templates
        preprocess_template_name = generate_preprocess_templates(args.project, steps, final_name)
        model_template_name = generate_model_template(args.project, cfg, fast_mode=args.fast, final_name=final_name)

        # Final info
        print("\n" + "="*60)
        print("✅ Configuration generated successfully!")
        print(f"  Preprocess: {preprocess_template_name}")
        print(f"  Model:      {model_template_name}")
        print(f"  Source:     Trial #{best_pipeline['trial_number']} (Score: {best_pipeline['value']:.5f})")

        if args.enqueue:
            if _QUEUE_AVAILABLE:
                try:
                    queue = TaskQueue(Path("projects/kaggle") / args.project)
                    command_str = f"model_template={model_template_name}"
                    queue.add_task(command_str, priority=10)
                    print(f"\n🚀 [ENQUEUED] Task added to queue: {command_str}")
                except Exception as e:
                    err(f"Failed to enqueue task: {e}")
            else:
                err("Queue module not found (src/mlarena/utils/queue.py missing?)")
        else:
            auto_flow_cmd = f"uv run python scripts/mla.py project={args.project} model_template={model_template_name}"
            only_model_cmd = f"uv run python scripts/mla.py model project={args.project} model_template={model_template_name}"

            print("\nRun full auto-flow (model -> predict -> submit -> fetch-score):")
            print(f"  {auto_flow_cmd}")
            print(f"  {auto_flow_cmd} common.use_gpu=true")

            print("\nRun only model training (no submit/fetch):")
            print(f"  {only_model_cmd}")
            print(f"  {only_model_cmd} common.use_gpu=true")

            print("\nOr enqueue it:")
            print(f"  python scripts/mla.py queue --project {args.project} add \"model_template={model_template_name}\"")
            print(f"  python scripts/mla.py queue --project {args.project} add \"model_template={model_template_name} common.use_gpu=true\"")

        print("="*60 + "\n")

    else:  # best_top_n
        pipelines = get_best_top_n_diverse(db_path, study_name, args.top_n)
        if not pipelines:
            err("Could not retrieve diverse pipelines.")
            return

        print("\n" + "="*60)
        print(f"✅ Generating {len(pipelines)} diverse configurations...")
        print("="*60 + "\n")

        generated_templates = []

        for idx, pipeline in enumerate(pipelines, start=1):
            steps = extract_pipeline_steps(pipeline['pipeline'])
            if not steps:
                warn(f"No preprocessing steps found in pipeline #{idx} (Trial {pipeline['trial_number']}), skipping")
                continue

            # Generate unique name for each template
            template_name = f"{prefix}_{idx:03d}"
            info(f"\n[{idx}/{len(pipelines)}] Generating template: {template_name}")
            info(f"  Source: Trial #{pipeline['trial_number']} (Score: {pipeline['value']:.5f})")

            # Generate templates
            preprocess_name = generate_preprocess_templates(args.project, steps, template_name)
            model_name = generate_model_template(args.project, cfg, fast_mode=args.fast, final_name=template_name)

            generated_templates.append({
                'model_template': model_name,
                'preprocess_template': preprocess_name,
                'trial_number': pipeline['trial_number'],
                'score': pipeline['value']
            })

        # Summary
        print("\n" + "="*60)
        print(f"✅ Generated {len(generated_templates)} configurations successfully!")
        print("="*60)
        for idx, tpl in enumerate(generated_templates, start=1):
            print(f"\n{idx}. {tpl['model_template']}")
            print(f"   Source: Trial #{tpl['trial_number']} | Score: {tpl['score']:.5f}")

        # Enqueue all if requested
        if args.enqueue:
            if _QUEUE_AVAILABLE:
                try:
                    queue = TaskQueue(Path("projects/kaggle") / args.project)
                    for tpl in generated_templates:
                        command_str = f"model_template={tpl['model_template']}"
                        queue.add_task(command_str, priority=10)
                    print(f"\n🚀 [ENQUEUED] {len(generated_templates)} tasks added to queue")
                except Exception as e:
                    err(f"Failed to enqueue tasks: {e}")
            else:
                err("Queue module not found (src/mlarena/utils/queue.py missing?)")
        else:
            print("\n" + "-"*60)
            print("Run commands:")
            print("-"*60)
            for idx, tpl in enumerate(generated_templates, start=1):
                print(f"\n# Template {idx} (Trial #{tpl['trial_number']}, Score: {tpl['score']:.5f})")
                print(f"uv run python scripts/mla.py project={args.project} model_template={tpl['model_template']}")

            print("\n" + "-"*60)
            print("Or enqueue all at once:")
            print("-"*60)
            for tpl in generated_templates:
                print(f"python scripts/mla.py queue --project {args.project} add \"model_template={tpl['model_template']}\"")

        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
