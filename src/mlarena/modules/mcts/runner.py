from __future__ import annotations
import time
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

from mlarena.core.module import ModuleContext, ModuleResult
from mlarena.modules.mcts.config import load_mcts_config, MCTSConfig
from mlarena.modules.mcts.storage import MCTSStorage, StudyDirection, TrialState
from mlarena.modules.mcts.space import SuperChainActionSpace
from mlarena.modules.mcts.sampler import ParameterSampler
from mlarena.modules.mcts.tree import MCTSTree, MCTSNode
from mlarena.modules.mcts.node import PipelineState
from mlarena.modules.mcts.materializer import TemplateMaterializer
from mlarena.modules.mcts.executor import MlaCliExecutor, ExperimentResult

from rich.tree import Tree
from rich.live import Live
from rich.console import Group
from rich.text import Text

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SUPER_CHAIN = REPO_ROOT / "conf/preprocess/mla_super_chain.yaml"

class MCTSRunner:
    def __init__(self, context: ModuleContext, params: Dict[str, Any]):
        self.context = context
        self.params = params
        
        super_chain_path = Path(params.get("super_chain", DEFAULT_SUPER_CHAIN))
        if not super_chain_path.is_absolute():
             p_path = context.project_root / super_chain_path
             if p_path.exists():
                 super_chain_path = p_path
             else:
                 r_path = REPO_ROOT / super_chain_path
                 if r_path.exists():
                     super_chain_path = r_path
                 else:
                     super_chain_path = p_path
            
        self.config = load_mcts_config(super_chain_path)
        
        if params.get("study_name"):
            self.config.study_name = params["study_name"]
            
        if "debug" in params:
            self.config.debug = bool(params["debug"])
            
        if not self.config.study_name:
            proj = context.project_name or "unknown"
            self.config.study_name = f"mcts_preprocess_{proj}"

        if self.config.storage_url == "sqlite:///experiments/db/mcts.db":
            db_dir = context.project_root / "experiments" / "db"
            db_dir.mkdir(parents=True, exist_ok=True)
            self.config.storage_url = f"sqlite:///{db_dir / 'mcts.db'}"

        self.storage = MCTSStorage(self.config.storage_url)
        self.space = SuperChainActionSpace(super_chain_path)
        self.sampler = ParameterSampler(seed=self.config.seed)
        self.tree = MCTSTree(self.config, self.space, self.sampler)
        
        self.direction = StudyDirection.MAXIMIZE if self.config.direction == "maximize" else StudyDirection.MINIMIZE
        self.study_id, self.is_new_study = self.storage.create_study(self.config.study_name, self.direction)
        
        # Rebuild tree from database if resuming
        if not self.is_new_study:
            nodes = self.storage.get_all_nodes(self.study_id)
            edges = self.storage.get_all_edges(self.study_id)
            self.tree.rebuild_tree(nodes, edges)
            
            # Print brief resume stats
            best = self.storage.get_best_trial(self.study_id)
            best_val_str = f"{best['value']:.4f}" if best else "N/A"
            
            # Find baseline score explicitly for display
            base_val_str = "N/A"
            if self.tree.root.n_visits > 0:
                base_val_str = f"{self.tree.root.value_best:.4f}"
            else:
                # Fallback: check storage directly if tree root is somehow not updated
                base_id = self.storage.get_trial_id_by_signature(self.study_id, "baseline")
                if base_id:
                    evals = self.storage.get_evaluations(base_id)
                    base_val = next((e["value"] for e in evals if e["fidelity"] == "F2" and e["status"] == "COMPLETE"), None)
                    if base_val is not None:
                        base_val_str = f"{base_val:.4f}"
                        self.tree.root.value_best = base_val
                        self.tree.root.n_visits = 1

            print(f"  -> Resume Stats: {len(nodes)} trials found, Baseline Score: {base_val_str}, Best Score: {best_val_str}", flush=True)
        
        self.run_id = f"mcts_s{self.study_id:04d}"
        
        self.materializer = TemplateMaterializer(
            run_id=self.run_id, 
            project_root=context.project_root
        )
        self.executor = MlaCliExecutor(REPO_ROOT, log_root=context.project_root)
        
        self._setup_logging()

    def _setup_logging(self):
        log_dir = self.context.project_root / "experiments" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(f"mcts.{self.run_id}")
        self.logger.setLevel(logging.DEBUG if self.config.debug else logging.INFO)
        self.logger.handlers = []
        
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        
        file_handler = logging.FileHandler(log_dir / "mcts.log")
        file_handler.setLevel(logging.DEBUG if self.config.debug else logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"--- MCTS Study Started (Run ID: {self.run_id}) ---")

    def run(self) -> ModuleResult:
        status_msg = "Starting NEW MCTS Study" if self.is_new_study else "Resuming EXISTING MCTS Study"
        print(f"{status_msg}: {self.config.study_name} (Budget: {self.config.budget})")
        self.logger.info(f"{status_msg}: {self.config.study_name} (Budget: {self.config.budget})")
        
        self._evaluate_baseline()
        
        best_so_far = -float('inf') if self.direction == StudyDirection.MAXIMIZE else float('inf')
        base_trial = self.storage.get_best_trial(self.study_id)
        if base_trial:
            best_so_far = base_trial["value"]
            self.logger.info(f"Baseline Score: {best_so_far}")

        mcts_live = self.params.get("mcts_live") or self.context.config.mcts_live

        if mcts_live:
            live = Live(self._render_tree(best_so_far), refresh_per_second=1, vertical_overflow="visible")
            live.start()
        else:
            live = None

        try:
            for i in range(self.config.budget):
                node = self.tree.select(self.tree.root)
                child = self.tree.expand(node)
                
                if child == node:
                    self.tree.backpropagate(node, node.value_best if node.value_best > -float('inf') else 0.0)
                    print(f"Iteration {i+1}/{self.config.budget} -> Skipped (terminal/limits)")
                    continue
                    
                trial_id = self.storage.create_trial(
                    study_id=self.study_id,
                    pipeline_signature=child.state.signature,
                    depth=child.state.depth,
                    params=self._state_to_params(child.state)
                )
                
                # Record the edge
                # Parent trial ID should already exist if the tree was correctly navigated or rebuilt
                parent_trial_id = self.storage.get_trial_id_by_signature(
                    self.study_id, 
                    node.state.signature if node != self.tree.root else "baseline"
                )
                
                if parent_trial_id:
                    action_data = {
                        "step_name": child.action_from_parent.step_name,
                        "template_name": child.action_from_parent.template_name,
                        "variant": child.action_from_parent.variant_name,
                        "config": child.action_from_parent.config
                    }
                    self.storage.add_edge(parent_trial_id, trial_id, action_data)

                fidelity = self._get_next_fidelity(child, trial_id)
                if not fidelity:
                    self.logger.info(f"  -> Trial {trial_id}: Pruned or done")
                    if live: live.update(self._render_tree(best_so_far))
                    continue
                
                self.logger.info(f"Iteration {i+1}/{self.config.budget} -> Trial {trial_id} ({fidelity})")
                
                # Log selected steps and their full config for debugging
                steps_desc = " -> ".join([f"{s['name']}:{s['variant']}" for s in child.state.steps]) or "No Preprocessing"
                self.logger.info(f"  -> Trial {trial_id} ({fidelity}): {steps_desc}")
                self.logger.debug(f"  -> Trial {trial_id} Full Config: {json.dumps(child.state.steps)}")
                
                result = self._execute_trial(child, trial_id, fidelity)
                
                if result.success and result.value is not None:
                    self.storage.add_evaluation(trial_id, fidelity, "COMPLETE", result.value, result.metric, result.duration, result.details)
                    self.tree.backpropagate(child, result.value)
                    self.storage.set_trial_value(trial_id, result.value)
                    
                    # Save raw JSON to file for debugging
                    raw = result.details.get("raw_json", {})
                    if raw:
                        json_path = self.context.project_root / "experiments" / "logs" / f"model_{trial_id}.json"
                        json_path.write_text(json.dumps(raw, indent=2))
                    
                    success_msg = f"Iteration {i+1}/{self.config.budget} -> Trial {trial_id} ({fidelity}) SUCCESS: {result.value:.4f} ({result.metric})"
                    
                    is_new_best = False
                    if self.direction == StudyDirection.MAXIMIZE:
                        if result.value > best_so_far: is_new_best = True
                    else:
                        if result.value < best_so_far: is_new_best = True
                            
                    if is_new_best:
                        best_so_far = result.value
                        success_msg += f" -> NEW BEST: {best_so_far:.4f}"
                        self.logger.info(f"*** NEW BEST SCORE: {best_so_far} (Trial {trial_id}, Fid: {fidelity}) ***")
                    
                    print(success_msg)
                    
                    best_model = raw.get("best_model", "N/A")
                    exp_id_res = raw.get("experiment_id", "N/A")
                    
                    self.logger.info(
                        f"  -> Trial {trial_id} Success: {result.value:.4f} | Model: {best_model} | ExpID: {exp_id_res}"
                    )
                    
                    target_fid = self.config.multi_fidelity.levels[-1]["name"] if self.config.multi_fidelity.levels else "F2"
                    if fidelity == target_fid:
                        self.storage.set_trial_state(trial_id, TrialState.COMPLETE)
                else:
                    print(f"Iteration {i+1}/{self.config.budget} -> Trial {trial_id} ({fidelity}) FAILED")
                    error_text = result.details.get("stderr", "") or result.details.get("stdout", "")
                    summary = "\n".join(error_text.strip().split("\n")[-5:])
                    self.logger.error(f"  -> Trial {trial_id} failed. Error Summary:\n{summary}")
                    
                    penalty = -1.0 if self.direction == StudyDirection.MAXIMIZE else 1.0
                    self.tree.backpropagate(child, penalty)
                    self.storage.set_trial_state(trial_id, TrialState.FAIL)
                    self.storage.add_evaluation(trial_id, fidelity, "FAIL", None, "", 0.0, result.details)
                
                if live: live.update(self._render_tree(best_so_far))
        finally:
            if live:
                live.stop()

        best = self.storage.get_best_trial(self.study_id)
        msg = f"MCTS completed. Best Score: {best['value'] if best else 'N/A'}"
        print(msg)
        self.logger.info(msg)
        return ModuleResult(success=True, payload={"best_trial": best})

    def _render_tree(self, best_score: float) -> Tree:
        root_label = f"MCTS: {self.config.study_name}"
        root_tree = Tree(f"[bold cyan]{root_label}[/bold cyan]")
        
        # We need to map signature to trial_id for display
        # Let's get them from storage or a local cache
        sig_to_trial = {}
        nodes = self.storage.get_all_nodes(self.study_id)
        for n in nodes:
            sig_to_trial[n["pipeline_signature"]] = n["trial_id"]

        def _add_nodes(mcts_node: MCTSNode, rich_tree: Tree):
            for child in mcts_node.children:
                trial_id = sig_to_trial.get(child.state.signature, "?")
                score = child.value_best
                
                # Format score for display
                score_str = f"{score:.4f}" if score > -float('inf') and score != 0.0 else "N/A"
                
                # Check if this is the best score
                is_best = False
                if score > -float('inf') and abs(score - best_score) < 1e-9:
                    is_best = True
                
                fidelity = "F2" # Default, could be improved if multi-fidelity is on
                label = f"{trial_id}/{fidelity}/{score_str}"
                
                if is_best:
                    node_text = Text(label, style="bold blue")
                else:
                    node_text = Text(label, style="grey70")
                
                # Add action info as tooltip or small text? 
                # For now just the requested format
                branch = rich_tree.add(node_text)
                _add_nodes(child, branch)

        # Add baseline (root) info to label if needed, or as first child
        _add_nodes(self.tree.root, root_tree)
        return root_tree

    def _evaluate_baseline(self):
        # If tree was rebuilt and root has visits, we already have the baseline
        if self.tree.root.n_visits > 0:
            self.logger.info(f"Baseline already exists (visits: {self.tree.root.n_visits}, best: {self.tree.root.value_best}).")
            return

        with self.storage._connect() as conn:
            cur = conn.cursor()
            # Check by signature in current study for maximum robustness
            query = """
                SELECT t.trial_id FROM trials t
                JOIN mcts_nodes n ON n.trial_id = t.trial_id
                WHERE t.study_id=? AND n.pipeline_signature='baseline' AND t.state=?
            """
            cur.execute(query, (self.study_id, TrialState.COMPLETE.value))
            if cur.fetchone():
                self.logger.info("Baseline already exists in database and is complete.")
                return

        print("Evaluating Baseline (Model Zero)")
        self.logger.info("Evaluating Baseline (Model Zero)")

        trial_id = self.storage.create_trial(
            self.study_id, 
            pipeline_signature="baseline", 
            depth=0, 
            number=0, 
            state=TrialState.RUNNING
        )
        
        state = PipelineState()
        templates = self.materializer.materialize(state, node_id=0, fixed_steps=self.space.fixed_steps)
        
        # Use consistent naming for baseline experiment
        preprocess_template = templates["base_name"]
        exp_id = f"exp-{preprocess_template}"
        
        model_template = self.params.get("model_template") or "baseline"
        
        cmd = self.executor.build_command(
            project=self.context.project_name,
            module="model",
            model_template=model_template,
            preprocess_template=preprocess_template,
            exp_id=exp_id
        )
        if self.config.model_verbosity is not None:
            cmd.append(f"model.verbosity={self.config.model_verbosity}")
            
        self.logger.debug(f"Executing Baseline MLA: {' '.join(cmd)}")
        result = self.executor.run(cmd)
        
        if result.success and result.value is not None:
            self.storage.set_trial_value(trial_id, result.value)
            self.storage.set_trial_state(trial_id, TrialState.COMPLETE)
            self.storage.add_evaluation(trial_id, "F2", "COMPLETE", result.value, result.metric, result.duration, result.details)
            self.tree.root.update(result.value)
            print(f"Baseline Score: {result.value:.4f} ({result.metric})", flush=True)
            self.logger.info(f"Baseline Score: {result.value}")
        else:
            error_msg = f"Baseline Failed! Check mcts.log for details."
            if result.details.get("error"):
                error_msg = f"Baseline Failed: {result.details['error']}"
            print(error_msg, flush=True)
            self.logger.error(f"Baseline failed: {result.details}")
            self.storage.set_trial_state(trial_id, TrialState.FAIL)

    def _execute_trial(self, node: MCTSNode, trial_id: int, fidelity: str) -> ExperimentResult:
        # 1. Materialize templates
        templates = self.materializer.materialize(node.state, node_id=trial_id, fixed_steps=self.space.fixed_steps)
        base_name = templates["base_name"]
        
        # 2. Preprocess template is the CHAIN name, not a step name
        preprocess_template = f"{base_name}_{fidelity}"
        
        # Create fidelity-specific chain YAML copy
        fid_path = templates["chain_path"].parent / f"{preprocess_template}.yaml"
        if not fid_path.exists():
            fid_path.write_text(templates["chain_path"].read_text())
            
        exp_id = f"exp-{preprocess_template}"
        
        model_template = self.params.get("model_template") or "baseline"
        
        cmd = self.executor.build_command(
            project=self.context.project_name,
            module="model",
            model_template=model_template,
            preprocess_template=preprocess_template,
            exp_id=exp_id
        )
        
        if self.config.model_verbosity is not None:
            cmd.append(f"model.verbosity={self.config.model_verbosity}")
        if self.config.model_cleanup:
            cmd.append("model.mla_retention=true")
            
        self.logger.debug(f"Executing MLA: {' '.join(cmd)}")
        return self.executor.run(cmd)

    def _get_next_fidelity(self, node: MCTSNode, trial_id: int) -> Optional[str]:
        levels = self.config.multi_fidelity.levels
        if not self.config.multi_fidelity.enable or not levels:
            evals = self.storage.get_evaluations(trial_id)
            if any(e["fidelity"] == "F2" for e in evals if e["status"] == "COMPLETE"): return None
            return "F2"
            
        evals = self.storage.get_evaluations(trial_id)
        completed_fids = {e["fidelity"] for e in evals if e["status"] == "COMPLETE"}
        
        for i, level in enumerate(levels):
            name = level["name"]
            if name in completed_fids: continue
            if i == 0: return name
            prev_name = levels[i-1]["name"]
            if prev_name not in completed_fids: return None 
            prev_score = next((e["value"] for e in evals if e["fidelity"] == prev_name), None)
            if prev_score is None: return None
            history = self.storage.get_fidelity_history(self.study_id, prev_name)
            top_frac = self.config.multi_fidelity.promotion.get("top_fraction", 0.25)
            if self._should_promote(prev_score, history, top_frac): return name
            else:
                self.storage.add_evaluation(trial_id, name, "PRUNED", None, "pruned", 0.0)
                return None
        return None

    def _should_promote(self, value: float, history: List[float], top_fraction: float) -> bool:
        valid_history = [h for h in history if h is not None]
        if not valid_history: return True
        valid_history.sort()
        if self.direction == StudyDirection.MAXIMIZE:
            cutoff_idx = int(len(valid_history) * (1 - top_fraction))
            if cutoff_idx >= len(valid_history): cutoff_idx = len(valid_history) - 1
            return value >= valid_history[cutoff_idx]
        else:
            cutoff_idx = int(len(valid_history) * top_fraction)
            if cutoff_idx >= len(valid_history): cutoff_idx = len(valid_history) - 1
            return value <= valid_history[cutoff_idx]

    def _state_to_params(self, state: PipelineState) -> Dict[str, Any]:
        params = {}
        for i, step in enumerate(state.steps):
            prefix = f"step_{i}"
            params[f"{prefix}.name"] = step["name"]
            params[f"{prefix}.variant"] = step["variant"]
            for k, v in step.get("config", {}).items():
                params[f"{prefix}.{k}"] = v
        return params
