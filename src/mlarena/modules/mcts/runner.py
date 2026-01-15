from __future__ import annotations
import time
import logging
import json
import math
import random
import sys
import signal
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import defaultdict

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
from rich.console import Group, Console
from rich.table import Table
from rich.text import Text

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SUPER_CHAIN = REPO_ROOT / "conf/preprocess/mla_super_chain.yaml"

class MCTSRunner:
    def __init__(self, context: ModuleContext, params: Dict[str, Any]):
        self.context = context
        self.params = params
        self.mcts_live = bool(params.get("mcts_live") or context.config.mcts_live)
        self.console = Console()
        
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
                best_val_str = f"{best['value']:.5f}" if best else "N/A"
                
                # Fetch actual baseline score (Trial 0) instead of root record
                base_score = None
                with self.storage._connect() as conn:
                    row = conn.execute("SELECT value FROM trial_values WHERE trial_id = (SELECT trial_id FROM trials WHERE study_id=? AND number=0)", (self.study_id,)).fetchone()
                    if row: base_score = row[0]
                
                base_val_str = f"{base_score:.5f}" if base_score is not None else "N/A"
                
                # Fetch metric name from any evaluation
                metric_name = "Unknown"
                with self.storage._connect() as conn:
                    row = conn.execute("SELECT metric_name FROM mcts_evaluations WHERE metric_name IS NOT NULL LIMIT 1").fetchone()
                    if row and row[0]: metric_name = row[0]

                table = Table(title="MCTS Resume Statistics")
                table.add_column("Statistic", style="cyan")
                table.add_column("Value", style="magenta")
                
                table.add_row("Evaluation Metric", metric_name)
                table.add_row("Trials Found", str(len(nodes)))
                table.add_row("Baseline Score", base_val_str)
                table.add_row("Best Score", best_val_str)
                
                self.console.print(table)
        
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
        elif "n_trials" in params:
            # Map standard tune flag to budget for UX consistency
            self.config.budget = int(params["n_trials"])
            
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

        self.stop_requested = False
        def handler(signum, frame):
            # Use local console to ensure Rich formatting works inside signal handler
            from rich.console import Console
            c = Console()
            if not self.stop_requested:
                self.stop_requested = True
                c.print("\n⚠️ [bold yellow]Ctrl+C detected. Finishing current trial and stopping gracefully...[/bold yellow]")
                self.logger.info("Interrupt requested. Stopping after current iteration.")
            else:
                # Second Ctrl+C - force quit
                c.print("\n⚠️ [bold red]Force quitting...[/bold red]")
                if hasattr(self.executor, "current_proc") and self.executor.current_proc:
                    try:
                        c.print(f"⚠️ Killing current subprocess (PID: {self.executor.current_proc.pid})...")
                        self.executor.current_proc.kill()
                    except Exception:
                        pass
                sys.exit(1)
        
        old_handler = signal.signal(signal.SIGINT, handler)

        try:
            self._evaluate_baseline()
            
            # Update best_so_far after baseline potentially ran
            if best_so_far == -float('inf') or best_so_far == float('inf'):
                best_trial = self.storage.get_best_trial(self.study_id)
                if best_trial:
                    best_so_far = best_trial["value"]

            n_executed = 0
            iteration = 0
            max_iterations = self.config.budget * 10  # Safety limit
            
            while n_executed < self.config.budget and not self.stop_requested:
                if iteration >= max_iterations:
                    self.logger.warning(f"Reached max safety iterations ({max_iterations}) with only {n_executed} trials. Tree might be fully explored.")
                    break

                iteration += 1
                node = self.tree.select(self.tree.root)
                child = self.tree.expand(node)
                
                if child == node:
                    untried = self.tree._get_untried_actions(node)
                    # ---------- Diagnostics (pre-backprop) ----------
                    children_n = len(node.children)
                    last_idx = getattr(node.state, "last_step_index", -1)
                    total_steps = getattr(self.space, "total_steps_count", None)
                    if total_steps is None:
                        # fallback if property not present
                        total_steps = len(getattr(self.space, "steps", []) or [])
                    start_idx = (last_idx if last_idx is not None else -1) + 1
                    end_reached = start_idx >= int(total_steps)

                    # candidates in the underlying action space (operator candidates)
                    try:
                        candidates = len(self.space.next_actions(node.state))
                    except Exception:
                        candidates = -1

                    # 2-layer PW: operator + param stats derived from existing children
                    # (works even if you don't have 2-layer PW enabled; fields default)
                    def _op_key(a):
                        return (getattr(a, "searched_index", None), getattr(a, "step_name", None), getattr(a, "variant_name", None))

                    op_children_visits = defaultdict(int)
                    op_tried_sids = defaultdict(set)
                    active_ops = set()
                    for ch in node.children:
                        a = ch.action_from_parent
                        if not a:
                            continue
                        ok = _op_key(a)
                        active_ops.add(ok)
                        op_children_visits[ok] += int(getattr(ch, "n_visits", 0))
                        sid = int(getattr(a, "param_sample_id", 0))
                        op_tried_sids[ok].add(sid)

                    # operator PW limit (layer 1)
                    k_ops = float(getattr(self.config, "expansion_width", 2.0))
                    a_ops = float(getattr(self.config, "expansion_alpha", 0.5))
                    n_node = max(1, int(getattr(node, "n_visits", 0)))
                    m_ops = int(k_ops * (n_node ** a_ops))
                    if m_ops < 1:
                        m_ops = 1

                    # param PW limit (layer 2) — report max across active ops
                    k_p = float(getattr(self.config, "param_expansion_width", 1.0))
                    a_p = float(getattr(self.config, "param_expansion_alpha", 0.5))
                    max_p = int(getattr(self.config, "param_expansion_max_samples", 16))
                    max_m_params = 0
                    max_op_visits = 0
                    max_tried_sids = 0
                    for ok, v in op_children_visits.items():
                        n_op = max(1, int(v))
                        m_params = int(k_p * (n_op ** a_p))
                        if m_params < 1:
                            m_params = 1
                        if max_p > 0:
                            m_params = min(m_params, max_p)
                        max_m_params = max(max_m_params, m_params)
                        max_op_visits = max(max_op_visits, int(v))
                        max_tried_sids = max(max_tried_sids, len(op_tried_sids.get(ok, set())))

                    # classify skip reason
                    if children_n == 0 and len(untried) == 0:
                        reason = "terminal (leaf)"
                        if end_reached:
                            reason += " / end"
                    else:
                        reason = "limits/fully-expanded"

                    diag_info = (
                        f"{reason} [T={node.number if node.number is not None else 'Root'} "
                        f"N={node.n_visits} children={children_n} untried={len(untried)} "
                        f"D={getattr(node.state,'depth', '?')} last_idx={last_idx} "
                        f"End={end_reached} ({start_idx}/{total_steps}) "
                        f"Candidates={candidates} "
                        f"PW_ops={m_ops} active_ops={len(active_ops)} "
                        f"PW_params_max={max_m_params} op_visits_max={max_op_visits} tried_sids_max={max_tried_sids}]"
                    )

                    # ---------- Backprop (skip) ----------
                    # Use mean (not best) to avoid optimistic replay bias; fallback to root mean if no visits.
                    val = node.value_mean if node.n_visits > 0 else self.tree.root.value_mean
                    self.tree.backpropagate(node, val)
                    with self.storage.atomic() as conn:
                        self._persist_node_stats_path(node, conn)

                    if not self.mcts_live:
                        txt = Text(f"I={iteration} (Ts={n_executed}/{self.config.budget}) ", style="dim")
                        txt.append(" → ", style="bold white")
                        txt.append(" ", style="dim") # spacer
                        txt.append(diag_info, style="dim")
                        self.console.print(txt)
                        
                    self.logger.debug(f"Iter {iteration} skipped: {diag_info}")
                    continue

                    
                # 1. Create trial and edge in one transaction
                with self.storage.atomic() as conn:
                    if node.trial_id is None:
                        # Try to recover (resume case). If it fails, abort to avoid corrupting the tree.
                        if not self._sync_root_from_baseline_db():
                            raise RuntimeError(
                                "MCTS parent has no trial_id (root not initialized). "
                                "Refusing to create root-level nodes that would break resume."
                            )

                    trial_id, trial_number = self.storage.create_trial(
                        study_id=self.study_id,
                        pipeline_signature=child.state.signature,
                        depth=child.state.depth,
                        parent_trial_id=node.trial_id, # Link directly to the selected parent node
                        params=self._state_to_params(child.state),
                        conn=conn
                    )
                    child.trial_id = trial_id # For database/file operations
                    child.number = trial_number # For UI/Logs
                    
                    # Record the edge
                    if node.trial_id is not None:
                        self.storage.add_edge(node.trial_id, trial_id, child.action_from_parent.to_record(), conn=conn)

                fidelity = self._get_next_fidelity(child, trial_id)
                if not fidelity:
                    # Pruned/Done path
                    self.logger.info(f"  -> Trial {trial_number} (ID:{trial_id}): Pruned or done")
                    
                    # NOTE: Backpropagation for PRUNED is handled in _get_next_fidelity to ensure correctness
                    
                    if self.live: self.live.update(self._render_tree(best_so_far))
                    # Do NOT consume budget for skipped/done trials during resume traversal
                    # n_executed += 1 
                    continue
                
                self.logger.info(f"I={iteration} (Ts={n_executed + 1}/{self.config.budget}) -> T={trial_number} D={child.state.depth}")
                if self.logger.isEnabledFor(logging.DEBUG):
                    try:
                        self.logger.debug(f"  -> Trial {trial_number} Full Config: {json.dumps(child.state.steps)}")
                    except Exception:
                        self.logger.debug(f"  -> Trial {trial_number} Full Config (Raw): {child.state.steps}")
                
                # Materialize and execute
                trial_templates = self.materializer.materialize(child.state, node_id=trial_id, fixed_steps=self.space.fixed_steps)
                result = self._execute_trial_with_templates(trial_templates, trial_id, fidelity)
                
                n_executed += 1
                
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
                    
                    # Log action details
                    act = child.action_from_parent
                    act_info = f"Action=[step={act.step_name} var={act.variant_name} sid={act.param_sample_id}]"

                    is_new_best = False
                    if self.direction == StudyDirection.MAXIMIZE:
                        if result.value > best_so_far: is_new_best = True
                    else:
                        if result.value < best_so_far: is_new_best = True
                            
                    if is_new_best:
                        best_so_far = result.value
                        self.logger.info(f"*** NEW BEST SCORE: {best_so_far} (T={trial_number}, D={child.state.depth}) ***")
                        
                        # Send Telegram Notification
                        proj_name = self.context.project_name or "Unknown Project"
                        msg = (
                            f"↑ <b>New Best Score!</b>\n\n"
                            f"<b>Project:</b> {proj_name}\n"
                            f"<b>Study:</b> {self.config.study_name}\n"
                            f"<b>Trial:</b> {trial_number}\n"
                            f"<b>Score:</b> {best_so_far:.5f}\n"
                            f"<b>Metric:</b> {result.metric}"
                        )
                        self.notifier.send(msg)
                    
                    if not self.mcts_live:
                        # Rich styling for success message
                        txt = Text(f"I={iteration} (Ts={n_executed}/{self.config.budget}) ", style="dim")
                        txt.append(" ✓ ", style="bold green")
                        parent_no = node.number if node.number is not None else 0
                        txt.append(f" T={trial_number} P={parent_no} D={child.state.depth} ", style="dim")
                        txt.append(f"{act_info}: ", style="cyan")
                        
                        val_style = "bold yellow" if is_new_best else "bold white"
                        txt.append(f"{result.value:.5f}", style=val_style)
                        
                        if is_new_best:
                            txt.append(" ★", style="bold yellow")
                            
                        self.console.print(txt)
                    
                    # Cleanup templates unless it's new best
                    self._cleanup_templates(trial_templates, fidelity, keep=is_new_best)
                    
                    best_model = raw.get("best_model", "N/A")
                    exp_id_res = raw.get("experiment_id", "N/A")
                    parent_no = node.number if node.number is not None else 0
                    
                    self.logger.info(
                        f"  -> Trial {trial_number} (Parent: {parent_no}) Success: {result.value:.4f} | {act_info} | Model: {best_model} | ExpID: {exp_id_res}"
                    )
                    
                    target_fid = self.config.multi_fidelity.levels[-1]["name"] if self.config.multi_fidelity.levels else "F2"
                    if fidelity == target_fid:
                        self.storage.set_trial_state(trial_id, TrialState.COMPLETE)
                else:
                    if not self.mcts_live:
                        txt = Text(f"I={iteration} (Ts={n_executed}/{self.config.budget}) ", style="dim")
                        txt.append(" ✗ ", style="bold red")
                        parent_no = node.number if node.number is not None else 0
                        txt.append(f" T={trial_number} P={parent_no} D={child.state.depth}", style="bold red")
                        self.console.print(txt)
                    
                    # Extract error from JSON if available, otherwise fallback to stderr
                    error_msg = result.details.get("error")
                    if not error_msg:
                        error_text = result.details.get("stderr", "") or result.details.get("stdout", "")
                        error_msg = "\n".join(error_text.strip().split("\n")[-10:])
                    
                    self.logger.error(f"  -> Trial {trial_number} failed. Error: {error_msg}")
                    
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
            signal.signal(signal.SIGINT, old_handler)
            if self.live:
                self.live.stop()

        best = self.storage.get_best_trial(self.study_id)
        best_val_str = f"{best['value']:.5f}" if best else "N/A"
        msg = f"MCTS completed. Best Score: {best_val_str}"
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
        base_score_str = f"{base_score:.4f}" if base_score is not None else "N/A"
        
        is_base_best = False
        if base_score is not None and abs(base_score - best_score) < 1e-7:
            is_base_best = True
            
        base_no = self.tree.root.number if self.tree.root.number is not None else 0
        base_label = f"{base_no}/F2/{base_score_str} (Baseline)"
        
        base_node_text = Text(base_label, style="bold blue" if is_base_best else "grey70")
        base_branch = root_tree.add(base_node_text)

        def _add_nodes(mcts_node: MCTSNode, rich_tree: Tree):
            for child in mcts_node.children:
                trial_no = child.number if child.number is not None else "?"
                score = child.value_best
                score_str = f"{score:.4f}" if score is not None else "N/A"
                
                is_best = False
                if score is not None and abs(score - best_score) < 1e-7:
                    is_best = True
                
                label = f"{trial_no}/F2/{score_str}"
                node_text = Text(label, style="bold blue" if is_best else "grey70")
                branch = rich_tree.add(node_text)
                _add_nodes(child, branch)

        # Root in memory tree is the baseline, its children are Level 1 search trials
        _add_nodes(self.tree.root, base_branch)
        return root_tree

    def _evaluate_baseline(self):
        # Prefer DB as source of truth on resume.
        if self._sync_root_from_baseline_db():
            self.logger.info(
                f"Baseline loaded from DB (trial_id={self.tree.root.trial_id}, "
                f"visits={self.tree.root.n_visits}, best={self.tree.root.value_best})."
            )
            return
        # Fallback: if in-memory root was already populated somehow.
        if self.tree.root.n_visits > 0 and self.tree.root.trial_id is not None:
            self.logger.info(
                f"Baseline already exists in memory (trial_id={self.tree.root.trial_id}, "
                f"visits={self.tree.root.n_visits}, best={self.tree.root.value_best})."
            )
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
            pass
        self.logger.info("Evaluating Baseline (Model Zero)")

        trial_id, trial_number = self.storage.create_trial(
            study_id=self.study_id, 
            pipeline_signature="baseline", 
            depth=0, 
            parent_trial_id=None, # Root has no parent
            number=0, 
            state=TrialState.RUNNING
        )
        self.tree.root.trial_id = trial_id
        self.tree.root.number = trial_number
        
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
            # Critical: attach DB identity to root so children are linked to baseline after resume.
            self.tree.root.trial_id = trial_id
            self.tree.root.number = trial_number
            self.tree.root.update(result.value, direction=self.config.direction)
            
            # PERSIST root stats
            self.storage.update_node_stats(trial_id, self.tree.root.n_visits, self.tree.root.value_sum, self.tree.root.value_best)
            
            if not self.mcts_live:
                txt = Text(f"I=0 (Ts=0/{self.config.budget}) ", style="dim")
                txt.append(" ✓ ", style="bold green")
                txt.append(f" T={trial_number} P=~ D=0 ", style="dim")
                txt.append("Action=[step=baseline var=fixed sid=0]: ", style="cyan")
                txt.append(f"{result.value:.5f}", style="bold white")
                txt.append(" =", style="bold blue")
                self.console.print(txt)
            self.logger.info(f"Baseline Score: {result.value}")
            
            # Initial best score notification
            proj_name = self.context.project_name or "Unknown Project"
            msg = (
                f"≈ <b>Baseline Score Set</b>\n\n"
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
                # Ensure the trial is marked as PRUNED in the main table
                self.storage.set_trial_state(trial_id, TrialState.PRUNED)
                
                if prev_score is not None:
                     # Backpropagate mean to increment visit count without artificial boosting/penalizing.
                     # If the node has no visits yet, fall back to root mean (baseline scale).
                     val = node.value_mean if node.n_visits > 0 else self.tree.root.value_mean
                     self.tree.backpropagate(node, val)
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

    def _sync_root_from_baseline_db(self) -> bool:
        """
        Ensure self.tree.root has trial_id/number/stats loaded from DB baseline (trial number 0).
        Returns True if baseline exists in DB and was loaded.
        """
        try:
            baseline = self.storage.get_trial_by_number(self.study_id, 0)
        except Exception:
            baseline = None
        if not baseline or not baseline.get("trial_id"):
            return False

        self.tree.root.trial_id = int(baseline["trial_id"])
        self.tree.root.number = int(baseline.get("number") or 0)
        self.tree.root.n_visits = int(baseline.get("n_visits") or 0)
        self.tree.root.value_sum = float(baseline.get("value_sum") or 0.0)
        vb = baseline.get("value_best")
        self.tree.root.value_best = float(vb) if vb is not None else -float("inf")
        return True
