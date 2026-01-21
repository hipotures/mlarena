#!/usr/bin/env python
import argparse
import json
import random
import sqlite3
import warnings
from pathlib import Path

import pandas as pd
import yaml
from autogluon.tabular import TabularPredictor
from rich.console import Console
from rich.table import Table

# ML Arena imports
from mlarena.modules.mcts.space import SuperChainActionSpace
from mlarena.modules.mcts.sampler import ParameterSampler
from mlarena.modules.mcts.node import PipelineState, Action

console = Console()
warnings.filterwarnings("ignore")

def info(message: str) -> None:
    console.print(message)

def flatten_config(action_dict, prefix=""):
    """Flattens a generated action dictionary (not JSON string) for DataFrame."""
    flat = {}
    group = action_dict.get("group_name", "unknown")
    variant = action_dict.get("variant", "unknown")
    
    flat[f"{prefix}action_group"] = group
    flat[f"{prefix}action_variant"] = variant
    
    config = action_dict.get("config", {})
    for k, v in config.items():
        key = f"{prefix}{group}_{k}"
        if isinstance(v, (list, dict)):
            flat[key] = json.dumps(v, sort_keys=True)
        else:
            flat[key] = v
    return flat

def parse_action_full(action_dict, prefix=""):
    """Parses action DICT (not json) into flat dictionary for DataFrame context."""
    # Similar to flatten_config but assumes input is already a dict from state history
    return flatten_config(action_dict, prefix=prefix)

def warn(message: str) -> None:
    console.print(f"[yellow]Warning:[/yellow] {message}")

def err(message: str) -> None:
    console.print(f"[red]Error:[/red] {message}")

def load_best_parent(conn: sqlite3.Connection, study_name: str):
    row = conn.execute("SELECT study_id FROM studies WHERE study_name = ?", (study_name,)).fetchone()
    if not row:
        return None
    study_id = row[0]
    query = """
    SELECT e.trial_id, e.value, n.depth
    FROM mcts_evaluations e
    JOIN mcts_nodes n ON e.trial_id = n.trial_id
    WHERE n.study_id = ? AND e.status = 'COMPLETE' AND e.value IS NOT NULL
    ORDER BY e.value DESC
    LIMIT 1
    """
    return conn.execute(query, (study_id,)).fetchone()

def load_best_parent_at_depth(conn: sqlite3.Connection, study_name: str, depth: int):
    row = conn.execute("SELECT study_id FROM studies WHERE study_name = ?", (study_name,)).fetchone()
    if not row:
        return None
    study_id = row[0]
    query = """
    SELECT e.trial_id, e.value, n.depth
    FROM mcts_evaluations e
    JOIN mcts_nodes n ON e.trial_id = n.trial_id
    WHERE n.study_id = ? AND n.depth = ? AND e.status = 'COMPLETE' AND e.value IS NOT NULL
    ORDER BY e.value DESC
    LIMIT 1
    """
    return conn.execute(query, (study_id, depth)).fetchone()

def load_parent_by_id(conn: sqlite3.Connection, trial_id: int):
    query = """
    SELECT e.trial_id, e.value, n.depth
    FROM mcts_evaluations e
    JOIN mcts_nodes n ON e.trial_id = n.trial_id
    WHERE e.trial_id = ? AND e.status = 'COMPLETE' AND e.value IS NOT NULL
    LIMIT 1
    """
    return conn.execute(query, (trial_id,)).fetchone()

def reconstruct_chain(conn: sqlite3.Connection, trial_id: int):
    actions = []
    curr_id = trial_id
    while True:
        row = conn.execute(
            "SELECT parent_trial_id, action_json FROM mcts_edges WHERE child_trial_id = ?",
            (curr_id,),
        ).fetchone()
        if not row:
            break
        parent_id, action_json = row
        try:
            actions.insert(0, json.loads(action_json))
        except Exception:
            pass
        curr_id = parent_id
    return actions

def action_to_step(action_dict):
    return {
        "name": action_dict.get("step_name") or action_dict.get("step"),
        "template": action_dict.get("template_name") or action_dict.get("template"),
        "group": action_dict.get("group_name") or action_dict.get("group"),
        "variant": action_dict.get("variant") or action_dict.get("variant_name"),
        "config": action_dict.get("config") or {},
        "searched_index": int(action_dict.get("searched_index", action_dict.get("step_index", -1))),
        "original_index": int(action_dict.get("original_index", action_dict.get("step_index", -1))),
        "param_sample_id": int(action_dict.get("param_sample_id", 0)),
    }

class BeamNode:
    def __init__(self, state: PipelineState, score: float, history: list, parent_score: float):
        self.state = state
        self.cumulative_prob = score # Sum of probs or product? Oracle gives prob of improvement. 
        # Actually, Oracle predicts "prob of improvement over parent".
        # We can sum log probs, or just keep the prob of the last step if we assume greedy?
        # Let's track "estimated value". Since Oracle doesn't predict value, only delta prob...
        # We will use the prob as a proxy for "quality of move".
        # Let's maximize the sum of probabilities of improvement along the path.
        self.score = score 
        self.history = history # List of action dicts
        self.last_parent_score = parent_score # The score of the trial we are extending (mocked)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="playground-series-s6e1", help="Project slug")
    parser.add_argument("--beam-width", type=int, default=5, help="Number of paths to keep")
    parser.add_argument("--samples", type=int, default=50, help="Samples per action type")
    parser.add_argument("--study", default="s6e1_008", help="Study name (to get root context)")
    parser.add_argument("--parent-id", type=int, default=None, help="Trial id to extend")
    parser.add_argument("--parent-depth", type=int, default=None, help="Pick best parent at this depth")
    parser.add_argument("--max-depth", type=int, default=5, help="Number of steps to add")
    parser.add_argument("--lookahead", type=int, default=3, help="Lookahead for next_actions")
    parser.add_argument("--output-prefix", default="oracle_beam", help="Prefix for generated YAMLs")
    args = parser.parse_args()

    # Paths
    project_dir = Path(f"projects/kaggle/{args.project}")
    exp_dir = project_dir / "experiments"
    model_dir = exp_dir / "oracle" / "model"
    oracle_csv = model_dir / "mcts_oracle.csv"
    db_path = exp_dir / "db" / "mcts.db"
    
    if not model_dir.exists():
        console.print("[red]Oracle model not found.[/red]")
        return
    if not db_path.exists():
        err("MCTS DB not found.")
        return

    # 1. Setup
    conf_dir = Path("conf/preprocess")
    super_chain_path = conf_dir / "mla_super_chain.yaml"
    space = SuperChainActionSpace(super_chain_path)
    sampler = ParameterSampler()
    
    # Load Oracle
    info("Loading Oracle...")
    predictor = TabularPredictor.load(str(model_dir))
    
    # Get column signature
    if oracle_csv.exists():
        expected_cols = pd.read_csv(oracle_csv, nrows=0).columns.tolist()
    else:
        expected_cols = predictor.feature_metadata_in.get_features()

    # 2. Initial State (Parent from DB)
    conn = sqlite3.connect(db_path)
    if args.parent_id is not None:
        row = load_parent_by_id(conn, args.parent_id)
        if not row:
            err(f"Parent trial {args.parent_id} not found or not COMPLETE.")
            conn.close()
            return
    elif args.parent_depth is not None:
        row = load_best_parent_at_depth(conn, args.study, args.parent_depth)
        if not row:
            err(f"No COMPLETE parent found at depth {args.parent_depth}.")
            conn.close()
            return
    else:
        row = load_best_parent(conn, args.study)
        if not row:
            err(f"Study '{args.study}' not found or empty.")
            conn.close()
            return

    parent_id, parent_score, parent_depth = row
    parent_actions = reconstruct_chain(conn, int(parent_id))
    conn.close()

    parent_steps = [action_to_step(a) for a in parent_actions]
    root_state = PipelineState(steps=parent_steps)
    info(f"Parent: {parent_id} | score={parent_score:.6f} | depth={parent_depth}")
    info(f"Starting depth: {root_state.depth}")

    # Root node (history includes parent chain for correct prev_ context)
    root = BeamNode(
        state=root_state,
        score=0.0,
        history=list(parent_actions),
        parent_score=parent_score,
    )
    
    beam = [root]
    
    # 3. Beam Search Loop
    max_depth = int(args.max_depth)
    
    for depth in range(max_depth):
        info(f"--- Depth {root_state.depth + depth + 1} (Beam size: {len(beam)}) ---")
        
        candidates = [] # Tuples of (node, action_dict)
        
        # Expansion
        for node in beam:
            # Generate possible next actions
            discrete_actions = space.next_actions(node.state, lookahead=args.lookahead)
            if not discrete_actions:
                continue
                
            # Sample configs
            for template in discrete_actions:
                for _ in range(args.samples):
                    config = sampler.sample_variant(
                        template.template_name,
                        template.variant_name,
                        space.search_spaces
                    )
                    
                    action_dict = {
                        "step": template.step_name,
                        "group_name": template.group_name,
                        "template": template.template_name,
                        "variant": template.variant_name,
                        "config": config,
                        "searched_index": template.searched_index,
                        "original_index": template.original_index
                    }
                    candidates.append((node, action_dict))
        
        if not candidates:
            info("No more actions possible.")
            break
            
        info(f"Evaluating {len(candidates)} candidates...")
        
        # Prepare DataFrame for Oracle
        rows = []
        for node, action in candidates:
            # Context
            prev_action = node.history[-1] if node.history else {}
            prev_flat = flatten_config(prev_action, prefix="prev_")
            curr_flat = flatten_config(action, prefix="")
            
            row = {
                "parent_score": node.last_parent_score,
                "depth": node.state.depth + 1,
                "prev_duration": 0.0,
                **prev_flat,
                **curr_flat
            }
            rows.append(row)
            
        df = pd.DataFrame(rows)
        
        # Align columns
        for col in expected_cols:
            if col == "is_improvement": continue
            if col not in df.columns:
                df[col] = None
                
        # Predict
        if predictor.problem_type == "binary":
            probs = predictor.predict_proba(df)
            pos_label = 1
            scores = probs[pos_label] if pos_label in probs.columns else probs.iloc[:, -1]
        else:
            scores = predictor.predict(df)
            
        # Select Top K with Diversity Enforcement
        scored_candidates = []
        for i, (node, action) in enumerate(candidates):
            prob = scores.iloc[i]
            new_path_score = node.score + prob
            scored_candidates.append((new_path_score, prob, node, action))
            
        # Sort desc
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Keep Top K (Beam Width) with uniqueness check
        new_beam = []
        seen_signatures = set()
        
        for path_score, prob, parent_node, action in scored_candidates:
            if len(new_beam) >= args.beam_width:
                break
                
            # Reconstruct Action object
            act_obj = Action(
                step_name=action["step"],
                template_name=action["template"],
                group_name=action["group_name"],
                variant_name=action["variant"],
                config=action["config"],
                searched_index=action["searched_index"],
                original_index=action["original_index"]
            )
            
            try:
                # Preview the next state's signature
                temp_state = parent_node.state.add_action(act_obj)
                sig = temp_state.signature
                
                if sig in seen_signatures:
                    continue # Skip duplicates
                
                seen_signatures.add(sig)
                new_history = parent_node.history + [action]
                
                child = BeamNode(
                    state=temp_state,
                    score=path_score,
                    history=new_history,
                    parent_score=parent_node.last_parent_score
                )
                new_beam.append(child)
                
                info(f"  -> Added Unique: {action['step']} ({action['variant']}) | Prob: {prob:.4f} | Path: {path_score:.4f}")
            except Exception as e:
                # Silently skip invalid moves (e.g. group already used)
                pass
                
        beam = new_beam

    # 4. Save Final Pipelines (Correct Project Structure)
    info("-" * 40)
    info(f"Generation Complete. Saving {len(beam)} best pipelines...")
    
    preprocess_dir = project_dir / "templates" / "preprocess"
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    
    for i, node in enumerate(beam):
        child_id = f"{args.output_prefix}_{i+1:02d}"
        
        # In this project, chain usually starts with 'mcts' baseline
        chain_names = ["mcts"]
        
        # Save each action as a separate module file
        for idx, action in enumerate(node.history):
            module_name = action.get("step") or action.get("step_name")
            template_name = action.get("template") or action.get("template_name")
            
            # Resolve module alias/name
            # 1. Try to find it in global template config
            global_tmpl_path = Path("src/mlarena/templates/preprocess") / f"{template_name}.yaml"
            module_alias = module_name # Default
            
            if global_tmpl_path.exists():
                try:
                    global_tmpl = yaml.safe_load(global_tmpl_path.read_text())
                    if global_tmpl and "module" in global_tmpl:
                        module_alias = global_tmpl["module"]
                except:
                    pass
            
            module_filename = f"{child_id}-step{idx:02d}-{module_name}"
            module_payload = {
                "module": module_alias,
                "config": action.get("config") or {}
            }
            
            # Save module file
            with open(preprocess_dir / f"{module_filename}.yaml", "w") as f:
                yaml.dump(module_payload, f, sort_keys=False)
                
            chain_names.append(module_filename)
            
        # Save the chain file
        chain_payload = {"chain": chain_names}
        with open(preprocess_dir / f"{child_id}.yaml", "w") as f:
            yaml.dump(chain_payload, f, sort_keys=False)
            
        info(f"Saved Chain: {child_id}.yaml (Score: {node.score:.4f})")

if __name__ == "__main__":
    main()
