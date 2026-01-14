from __future__ import annotations
import time
import logging
import json
import math
import random
import sys
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
from mlarena.utils.notification import TelegramNotifier

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
        self.mcts_live = bool(params.get("mcts_live") or context.config.mcts_live)
        
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

        # Apply overrides from params (e.g. mcts.budget, mcts.multi_fidelity.enable)
        self._apply_overrides(params)

        self.storage = MCTSStorage(self.config.storage_url)
        self.space = SuperChainActionSpace(super_chain_path)
        self.sampler = ParameterSampler(seed=self.config.seed)
        
        # Initialize root state with groups from fixed steps
        initial_groups = {}
        for fs in self.space.fixed_steps:
            cfg = fs["config"]
            group = cfg.get("group") or cfg.get("name")
            if group:
                initial_groups[group] = cfg.get("name")
        
        self.tree = MCTSTree(self.config, self.space, self.sampler)
        self.tree.initial_groups = initial_groups
        self.tree.root = MCTSNode(state=PipelineState(used_groups=initial_groups))
        
        self.direction = StudyDirection.MAXIMIZE if self.config.direction == "maximize" else StudyDirection.MINIMIZE
        self.study_id, self.is_new_study = self.storage.create_study(self.config.study_name, self.direction)
        
        # Rebuild tree from database if resuming
        if not self.is_new_study:
            nodes = self.storage.get_all_nodes(self.study_id)
            edges = self.storage.get_all_edges(self.study_id)
            self.tree.rebuild_tree(nodes, edges)
            
            # Print brief resume stats if not in live mode
            if not self.mcts_live:
                best = self.storage.get_best_trial(self.study_id)
                best_val_str = f"{best['value']:.4f}" if best else "N/A"
                
                # Fetch actual baseline score (Trial 0) instead of root record
                base_score = None
                with self.storage._connect() as conn:
                    row = conn.execute("SELECT value FROM trial_values WHERE trial_id = (SELECT trial_id FROM trials WHERE study_id=? AND number=0)", (self.study_id,)).fetchone()
                    if row: base_score = row[0]
                
                base_val_str = f"{base_score:.4f}" if base_score is not None else "N/A"
                print(f"  -> Resume Stats: {len(nodes)} trials found, Baseline Score: {base_val_str}, Best Score: {best_val_str}", flush=True)
        
        self.run_id = f"mcts_s{self.study_id:04d}"
        
        self.materializer = TemplateMaterializer(
            run_id=self.run_id, 
            project_root=context.project_root
        )
        self.executor = MlaCliExecutor(REPO_ROOT, log_root=context.project_root)
        
        self.notifier = TelegramNotifier()
        if bool(params.get("telegram_test") or context.config.telegram_test):
            self.notifier.send_test(source="MCTSRunner")
        
        self._setup_logging()

    def _apply_overrides(self, params: Dict[str, Any]):
        """Apply CLI parameters to config, supporting 'mcts.section.key' notation."""
        # 1. Explicit short args
        if "budget" in params:
            self.config.budget = int(params["budget"])
        if "seed" in params:
            self.config.seed = int(params["seed"])
            
        # 2. Dotted mcts.* args
        for key, value in params.items():
            if key.startswith("mcts."):
                path = key[5:] # remove "mcts."
                self._set_config_value(self.config, path, value)

    def _set_config_value(self, obj: Any, path: str, value: Any):
        """Recursively set value in Pydantic model or dict using dot notation."""
        parts = path.split(".")
        current = obj
        
        # Traverse to parent
        for part in parts[:-1]:
            if hasattr(current, part):
                current = getattr(current, part)
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return # Path not found
        
        # Set value on leaf
        last = parts[-1]
        if hasattr(current, last):
            # Attempt naive type casting based on existing value
            existing = getattr(current, last)
            if isinstance(existing, bool):
                # Handle bool strings from CLI
                if isinstance(value, str):
                    value = value.lower() in ("true", "1", "yes")
            elif isinstance(existing, int):
                try: value = int(value)
                except: pass
            elif isinstance(existing, float):
                try: value = float(value)
                except: pass
            
            setattr(current, last, value)
        elif isinstance(current, dict):
            current[last] = value

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
        
        if not self.mcts_live:
            print(f"{status_msg}: {self.config.study_name} (Budget: {self.config.budget})")
        self.logger.info(f"{status_msg}: {self.config.study_name} (Budget: {self.config.budget})")
        
        best_so_far = -float('inf') if self.direction == StudyDirection.MAXIMIZE else float('inf')
        base_trial = self.storage.get_best_trial(self.study_id)
        if base_trial:
            best_so_far = base_trial["value"]
            self.logger.info(f"Baseline Score: {best_so_far}")

        if self.mcts_live:
            self.live = Live(self._render_tree(best_so_far), refresh_per_second=1, vertical_overflow="visible")
            self.live.start()
        else:
            self.live = None

        try:
            self._evaluate_baseline()
            
            # Update best_so_far after baseline potentially ran
            if best_so_far == -float('inf') or best_so_far == float('inf'):
                best_trial = self.storage.get_best_trial(self.study_id)
                if best_trial:
                    best_so_far = best_trial["value"]

            for i in range(self.config.budget):
                node = self.tree.select(self.tree.root)
                child = self.tree.expand(node)
                
                if child == node:
                    # Skip path
                    self.tree.backpropagate(node, node.value_best if node.value_best > -float('inf') else 0.0)
                    with self.storage.atomic() as conn:
                        self._persist_node_stats_path(node, conn)
                    
                    if not self.mcts_live:
                        print(f"Iteration {i+1}/{self.config.budget} -> Skipped (terminal/limits)")
                    continue
                    
                # 1. Create trial and edge in one transaction
                with self.storage.atomic() as conn:
                    trial_id = self.storage.create_trial(
                        study_id=self.study_id,
                        pipeline_signature=child.state.signature,
                        depth=child.state.depth,
                        params=self._state_to_params(child.state),
                        conn=conn
                    )
                    child.trial_id = trial_id # For visualization
                    
                    # Record the edge
                    parent_trial_id = self.storage.get_trial_id_by_signature(
                        self.study_id, 
                        node.state.signature if node != self.tree.root else "baseline",
                        conn=conn
                    )
                    
                    if parent_trial_id:
                        self.storage.add_edge(parent_trial_id, trial_id, child.action_from_parent.to_record(), conn=conn)

                fidelity = self._get_next_fidelity(child, trial_id)
                if not fidelity:
                    # Pruned/Done path
                    self.logger.info(f"  -> Trial {trial_id}: Pruned or done")
                    
                    # If PRUNED, we should update stats to reflect the visit (even if no result value)
                    # to encourage UCT to explore other paths.
                    evals = self.storage.get_evaluations(trial_id)
                    if any(e["status"] == "PRUNED" for e in evals):
                         # Backpropagate current best value (or 0) to increment visit count
                         val = child.value_best if child.value_best > -float('inf') else 0.0
                         self.tree.backpropagate(child, val)
                         with self.storage.atomic() as conn:
                             self._persist_node_stats_path(child, conn)

                    if self.live: self.live.update(self._render_tree(best_so_far))
                    continue
                
                self.logger.info(f"Iteration {i+1}/{self.config.budget} -> Trial={trial_id} Depth={child.state.depth}")
                if self.logger.isEnabledFor(logging.DEBUG):
                    try:
                        self.logger.debug(f"  -> Trial {trial_id} Full Config: {json.dumps(child.state.steps)}")
                    except Exception:
                        self.logger.debug(f"  -> Trial {trial_id} Full Config (Raw): {child.state.steps}")
                
                # Materialize and execute
                trial_templates = self.materializer.materialize(child.state, node_id=trial_id, fixed_steps=self.space.fixed_steps)
                result = self._execute_trial_with_templates(trial_templates, trial_id, fidelity)
                
                if result.success and result.value is not None:
                    # Success path
                    raw = result.details.get("raw_json", {})
                    final_value = result.value
                    
                    # 1. Feature Penalty
                    feat_lambda = self.config.penalties.features_lambda
                    feat_penalty = 0.0
                    if feat_lambda > 0:
                        shapes = raw.get("shapes", {})
                        if not shapes:
                            shapes = (result.details.get("payload") or {}).get("shapes", {})
                        
                        in_cols = shapes.get("train_before", [0, 13])[1] if shapes.get("train_before") else 13
                        out_cols = shapes.get("train_after", [0, 13])[1] if shapes.get("train_after") else 13
                        
                        if out_cols > in_cols:
                            feat_penalty = feat_lambda * math.log10(out_cols / in_cols)
                            final_value -= feat_penalty
                    
                    # 2. Time Penalty
                    time_lambda = self.config.penalties.time_lambda
                    time_penalty = 0.0
                    if time_lambda > 0:
                        duration_min = result.duration / 60.0
                        time_penalty = time_lambda * duration_min
                        final_value -= time_penalty
                    
                    if feat_penalty > 0 or time_penalty > 0:
                        self.logger.debug(
                            f"[PENALTY] Node {trial_id}: Original={result.value:.4f}, "
                            f"FeatPenalty={feat_penalty:.4f}, TimePenalty={time_penalty:.4f}, Final={final_value:.4f}"
                        )

                    # 2. Persist results and backprop stats in one transaction
                    with self.storage.atomic() as conn:
                        self.storage.add_evaluation(trial_id, fidelity, "COMPLETE", result.value, result.metric, result.duration, result.details, conn=conn)
                        self.logger.debug(f"[BACKPROP] Propagating value {final_value:.4f} (original: {result.value:.4f}) from node {trial_id}")
                        
                        # Update in-memory tree
                        self.tree.backpropagate(child, final_value)
                        
                        # PERSIST to database
                        self._persist_node_stats_path(child, conn)

                        self.storage.set_trial_value(trial_id, result.value, conn=conn) # Keep original score in DB for reporting
                    
                    # Save raw JSON to file for debugging
                    raw = result.details.get("raw_json", {})
                    if raw:
                        json_path = self.context.project_root / "experiments" / "logs" / f"model_{trial_id}.json"
                        json_path.write_text(json.dumps(raw, indent=2))
                    
                    success_msg = f"Iteration {i+1}/{self.config.budget} -> Trial={trial_id} Depth={child.state.depth} SUCCESS: {result.value:.4f} ({result.metric})"
                    
                    is_new_best = False
                    if self.direction == StudyDirection.MAXIMIZE:
                        if result.value > best_so_far: is_new_best = True
                    else:
                        if result.value < best_so_far: is_new_best = True
                            
                    if is_new_best:
                        best_so_far = result.value
                        success_msg += f" -> NEW BEST: {best_so_far:.4f}"
                        self.logger.info(f"*** NEW BEST SCORE: {best_so_far} (Trial={trial_id}, Depth={child.state.depth}) ***")
                        
                        # Send Telegram Notification
                        proj_name = self.context.project_name or "Unknown Project"
                        msg = (
                            f"🚀 <b>New Best Score!</b>\n\n"
                            f"<b>Project:</b> {proj_name}\n"
                            f"<b>Study:</b> {self.config.study_name}\n"
                            f"<b>Trial:</b> {trial_id}\n"
                            f"<b>Score:</b> {best_so_far:.5f}\n"
                            f"<b>Metric:</b> {result.metric}"
                        )
                        self.notifier.send(msg)
                    
                    if not self.mcts_live:
                        print(success_msg)
                    
                    # Cleanup templates unless it's new best
                    self._cleanup_templates(trial_templates, fidelity, keep=is_new_best)
                    
                    best_model = raw.get("best_model", "N/A")
                    exp_id_res = raw.get("experiment_id", "N/A")
                    
                    self.logger.info(
                        f"  -> Trial {trial_id} Success: {result.value:.4f} | Model: {best_model} | ExpID: {exp_id_res}"
                    )
                    
                    target_fid = self.config.multi_fidelity.levels[-1]["name"] if self.config.multi_fidelity.levels else "F2"
                    if fidelity == target_fid:
                        self.storage.set_trial_state(trial_id, TrialState.COMPLETE)
                else:
                    if not self.mcts_live:
                        print(f"Iteration {i+1}/{self.config.budget} -> Trial={trial_id} Depth={child.state.depth} FAILED")
                    
                    # Extract error from JSON if available, otherwise fallback to stderr
                    error_msg = result.details.get("error")
                    if not error_msg:
                        error_text = result.details.get("stderr", "") or result.details.get("stdout", "")
                        error_msg = "\n".join(error_text.strip().split("\n")[-10:])
                    
                    self.logger.error(f"  -> Trial {trial_id} failed. Error: {error_msg}")
                    
                    with self.storage.atomic() as conn:
                        penalty = -1.0 if self.direction == StudyDirection.MAXIMIZE else 1.0
                        self.tree.backpropagate(child, penalty)
                        self.storage.set_trial_state(trial_id, TrialState.FAIL, conn=conn)
                        self.storage.add_evaluation(trial_id, fidelity, "FAIL", None, "", 0.0, result.details, conn=conn)
                        # PERSIST failure penalty
                        self._persist_node_stats_path(child, conn)
                    
                    # Cleanup templates on failure (respect configuration)
                    keep_on_fail = self.config.templates.retain_failures
                    self._cleanup_templates(trial_templates, fidelity, keep=keep_on_fail)
                
                if self.live: self.live.update(self._render_tree(best_so_far))
        finally:
            if self.live:
                self.live.stop()

        best = self.storage.get_best_trial(self.study_id)
        msg = f"MCTS completed. Best Score: {best['value'] if best else 'N/A'}"
        if not self.mcts_live:
            print(msg)
        self.logger.info(msg)
        return ModuleResult(success=True, payload={"best_trial": best})

    def _persist_node_stats_path(self, node: MCTSNode, conn: sqlite3.Connection):
        """Helper to sync all nodes from current back to root in the database."""
        seen = set()
        curr = node
        updates = []
        
        while curr is not None:
            if id(curr) in seen:
                self.logger.error(f"Cycle detected in MCTS parent chain (Node ID: {id(curr)}); aborting stats persist.")
                break
            seen.add(id(curr))
            
            tid = curr.trial_id
            if tid:
                # Collect update: (trial_id, n_visits, value_sum, value_best)
                updates.append((tid, curr.n_visits, curr.value_sum, curr.value_best))
            curr = curr.parent
            
        if updates:
            self.storage.update_node_stats_many(updates, conn=conn)

    def _render_tree(self, best_score: float) -> Tree:
        root_label = f"MCTS Study: {self.config.study_name}"
        root_tree = Tree(f"[bold cyan]{root_label}[/bold cyan]")
        
        # Add Baseline as the first virtual node
        base_score = self.tree.root.value_best
        base_score_str = f"{base_score:.4f}" if base_score > -float('inf') and base_score != 0.0 else "N/A"
        
        is_base_best = False
        if base_score > -float('inf') and abs(base_score - best_score) < 1e-7:
            is_base_best = True
            
        base_id = self.tree.root.trial_id or 1
        base_label = f"{base_id}/F2/{base_score_str} (Baseline)"
        
        base_node_text = Text(base_label, style="bold blue" if is_base_best else "grey70")
        base_branch = root_tree.add(base_node_text)

        def _add_nodes(mcts_node: MCTSNode, rich_tree: Tree):
            for child in mcts_node.children:
                trial_id = child.trial_id or "?"
                score = child.value_best
                score_str = f"{score:.4f}" if score > -float('inf') and score != 0.0 else "N/A"
                
                is_best = False
                if score > -float('inf') and abs(score - best_score) < 1e-7:
                    is_best = True
                
                label = f"{trial_id}/F2/{score_str}"
                node_text = Text(label, style="bold blue" if is_best else "grey70")
                branch = rich_tree.add(node_text)
                _add_nodes(child, branch)

        # Root in memory tree is the baseline, its children are Level 1 search trials
        _add_nodes(self.tree.root, base_branch)
        return root_tree

    def _evaluate_baseline(self):
        if self.tree.root.n_visits > 0:
            self.logger.info(f"Baseline already exists (visits: {self.tree.root.n_visits}, best: {self.tree.root.value_best}).")
            return

        with self.storage._connect() as conn:
            cur = conn.cursor()
            query = """
                SELECT t.trial_id FROM trials t
                JOIN mcts_nodes n ON n.trial_id = t.trial_id
                WHERE t.study_id=? AND n.pipeline_signature='baseline' AND t.state=?
            """
            cur.execute(query, (self.study_id, TrialState.COMPLETE.value))
            if cur.fetchone():
                self.logger.info("Baseline already exists in database and is complete.")
                return

        if not self.mcts_live:
            print("Evaluating Baseline (Model Zero)")
        self.logger.info("Evaluating Baseline (Model Zero)")

        trial_id = self.storage.create_trial(
            study_id=self.study_id, 
            pipeline_signature="baseline", 
            depth=0, 
            number=0, 
            state=TrialState.RUNNING
        )
        self.tree.root.trial_id = trial_id
        
        state = PipelineState()
        templates = self.materializer.materialize(state, node_id=0, fixed_steps=self.space.fixed_steps)
        
        # Consistent naming for baseline
        base_name = templates["base_name"]
        preprocess_template = f"{base_name}_F2"
        
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
            
        self.logger.debug(f"Executing Baseline MLA: {' '.join(cmd)}")
        result = self.executor.run(cmd)
        
        if result.success and result.value is not None:
            self.storage.set_trial_value(trial_id, result.value)
            self.storage.set_trial_state(trial_id, TrialState.COMPLETE)
            self.storage.add_evaluation(trial_id, "F2", "COMPLETE", result.value, result.metric, result.duration, result.details)
            self.logger.debug(f"[BACKPROP] Propagating baseline value {result.value:.4f}")
            self.tree.root.update(result.value)
            
            # PERSIST root stats
            self.storage.update_node_stats(trial_id, self.tree.root.n_visits, self.tree.root.value_sum, self.tree.root.value_best)
            
            if not self.mcts_live:
                print(f"Baseline Score: {result.value:.4f} ({result.metric})", flush=True)
            self.logger.info(f"Baseline Score: {result.value}")
            
            # Initial best score notification
            proj_name = self.context.project_name or "Unknown Project"
            msg = (
                f"📈 <b>Baseline Score Set</b>\n\n"
                f"<b>Project:</b> {proj_name}\n"
                f"<b>Study:</b> {self.config.study_name}\n"
                f"<b>Baseline Score:</b> {result.value:.5f}\n"
                f"<b>Metric:</b> {result.metric}"
            )
            self.notifier.send(msg)
        else:
            error_msg = f"Baseline Failed! Check mcts.log for details."
            if result.details.get("error"):
                error_msg = f"Baseline Failed: {result.details['error']}"
            if not self.mcts_live:
                print(error_msg, flush=True)
            self.logger.error(f"Baseline failed: {result.details}")
            self.storage.set_trial_state(trial_id, TrialState.FAIL)

    def _cleanup_templates(self, templates: Dict[str, Any], fidelity: str, keep: bool = False):
        """Delete template files if they are not the best."""
        if keep:
            return
            
        policy = self.config.templates.retention
        if policy == "all":
            return
            
        base_name = templates.get("base_name")
        if not base_name or not fidelity:
            self.logger.warning(f"Cleanup skipped: empty base_name or fidelity")
            return

        templates_dir = self.materializer.templates_dir.resolve()
            
        try:
            # 1. Delete chain YAML
            chain_path = Path(templates["chain_path"]).resolve()
            if chain_path.exists() and chain_path.is_relative_to(templates_dir):
                chain_path.unlink()
                self.logger.debug(f"[CLEANUP] Deleted chain template: {chain_path}")
            
            # 2. Delete step YAMLs
            for p in templates["step_paths"]:
                step_path = Path(p).resolve()
                if step_path.exists() and step_path.is_relative_to(templates_dir):
                    step_path.unlink()
                    self.logger.debug(f"[CLEANUP] Deleted step template: {step_path}")
        except Exception as e:
            self.logger.warning(f"Failed to cleanup templates: {e}")

    def _execute_trial_with_templates(self, templates: Dict[str, Any], trial_id: int, fidelity: str) -> ExperimentResult:
        base_name = templates["base_name"]
        
        # Use fidelity-specific template name for consistent pre- folder naming
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
                if prev_score is not None:
                    # Backpropagate previous score to update visits/means, acknowledging we explored this path
                    self.tree.backpropagate(node, prev_score)
                    with self.storage.atomic() as conn:
                        self._persist_node_stats_path(node, conn)
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
            params[f"step_{i}"] = {
                "name": step["name"],
                "variant": step["variant"],
                "config": step["config"]
            }
        return params