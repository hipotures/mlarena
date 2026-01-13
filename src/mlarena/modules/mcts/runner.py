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

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SUPER_CHAIN = REPO_ROOT / "conf" / "preprocess" / "mla_super_chain.yaml"

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
        
        self.run_id = f"{self.config.study_name}_{int(time.time())}"
        
        self.materializer = TemplateMaterializer(
            run_id=self.run_id, 
            project_root=context.project_root
        )
        self.executor = MlaCliExecutor(REPO_ROOT)
        
        self._setup_logging()
        
        self.direction = StudyDirection.MAXIMIZE if self.config.direction == "maximize" else StudyDirection.MINIMIZE
        self.study_id = self.storage.create_study(self.config.study_name, self.direction)

    def _setup_logging(self):
        log_dir = self.context.project_root / "experiments" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(f"mcts.{self.run_id}")
        self.logger.setLevel(logging.DEBUG if self.config.debug else logging.INFO)
        self.logger.handlers = []
        
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        
        info_handler = logging.FileHandler(log_dir / "mcts.log")
        info_handler.setLevel(logging.INFO)
        info_handler.setFormatter(formatter)
        self.logger.addHandler(info_handler)
        
        debug_handler = logging.FileHandler(log_dir / "mcts.debug.log")
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(formatter)
        self.logger.addHandler(debug_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"--- MCTS Logger Initialized (Run ID: {self.run_id}) ---")

    def run(self) -> ModuleResult:
        self.logger.info(f"Starting MCTS Study: {self.config.study_name} (Budget: {self.config.budget})")
        self._evaluate_baseline()
        
        best_so_far = -float('inf') if self.direction == StudyDirection.MAXIMIZE else float('inf')
        base_trial = self.storage.get_best_trial(self.study_id)
        if base_trial:
            best_so_far = base_trial["value"]
            self.logger.info(f"Baseline Score: {best_so_far}")
        
        for i in range(self.config.budget):
            self.logger.info(f"Iteration {i+1}/{self.config.budget}")
            node = self.tree.select(self.tree.root)
            child = self.tree.expand(node)
            
            if child == node:
                self.logger.warning("  -> Could not expand (terminal or limits)")
                self.tree.backpropagate(node, node.value_best if node.value_best > -float('inf') else 0.0)
                continue
                
            trial_id = self.storage.create_trial(
                self.study_id,
                number=self._get_next_trial_number(),
                pipeline_signature=child.state.signature,
                depth=child.state.depth,
                params=self._state_to_params(child.state)
            )
            
            fidelity = self._get_next_fidelity(child, trial_id)
            if not fidelity:
                self.logger.info(f"  -> Trial {trial_id}: Pruned or done")
                continue
                
            self.logger.info(f"  -> Trial {trial_id} (Fid: {fidelity}): Executing...")
            result = self._execute_trial(child, trial_id, fidelity)
            
            if result.success and result.value is not None:
                self.storage.add_evaluation(trial_id, fidelity, "COMPLETE", result.value, result.metric, result.duration, result.details)
                self.tree.backpropagate(child, result.value)
                self.storage.set_trial_value(trial_id, result.value)
                
                target_fid = self.config.multi_fidelity.levels[-1]["name"] if self.config.multi_fidelity.levels else "F2"
                if fidelity == target_fid:
                    self.storage.set_trial_state(trial_id, TrialState.COMPLETE)
                    
                is_new_best = False
                if self.direction == StudyDirection.MAXIMIZE:
                    if result.value > best_so_far: is_new_best = True
                else:
                    if result.value < best_so_far: is_new_best = True
                        
                if is_new_best:
                    best_so_far = result.value
                    self.logger.info(f"*** NEW BEST SCORE: {best_so_far} (Trial {trial_id}, Fid: {fidelity}) ***")
            else:
                penalty = -1.0 if self.direction == StudyDirection.MAXIMIZE else 1.0
                self.logger.error(f"  -> Trial {trial_id} failed. Details: {result.details}")
                self.tree.backpropagate(child, penalty)
                self.storage.set_trial_state(trial_id, TrialState.FAIL)
                self.storage.add_evaluation(trial_id, fidelity, "FAIL", None, "", 0.0, result.details)

        best = self.storage.get_best_trial(self.study_id)
        self.logger.info(f"MCTS completed. Best Score: {best['value'] if best else 'N/A'}")
        return ModuleResult(success=True, payload={"best_trial": best})

    def _evaluate_baseline(self):
        self.logger.info("Evaluating Baseline (Model Zero)")
        with self.storage._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT trial_id FROM trials WHERE study_id=? AND number=0", (self.study_id,))
            if cur.fetchone():
                self.logger.info("Baseline already exists, skipping.")
                return

        trial_id = self.storage.create_trial(
            self.study_id, 0, "baseline", 0, {}, state=TrialState.RUNNING
        )
        state = PipelineState()
        templates = self.materializer.materialize(state, node_id=0)
        result = self._run_mla(templates, "baseline_exp")
        
        if result.success and result.value is not None:
            self.storage.set_trial_value(trial_id, result.value)
            self.storage.set_trial_state(trial_id, TrialState.COMPLETE)
            self.storage.add_evaluation(trial_id, "F2", "COMPLETE", result.value, result.metric, result.duration, result.details)
            self.tree.root.update(result.value)
            self.logger.info(f"Baseline Score: {result.value}")
        else:
            self.logger.error(f"Baseline failed: {result.details}")
            self.storage.set_trial_state(trial_id, TrialState.FAIL)

    def _execute_trial(self, node: MCTSNode, trial_id: int, fidelity: str) -> ExperimentResult:
        templates = self.materializer.materialize(node.state, node_id=trial_id)
        base_name = templates["base_name"]
        exp_id = f"{base_name}_{fidelity}"
        
        fid_cfg = next((l for l in self.config.multi_fidelity.levels if l["name"] == fidelity), {})
        
        overrides = {}
        if "cv_folds" in fid_cfg: overrides["model.cv_folds"] = fid_cfg["cv_folds"]
        if "time_limit_sec" in fid_cfg: overrides["model.time_limit"] = fid_cfg["time_limit_sec"]
        return self._run_mla(templates, exp_id, overrides)

    def _run_mla(self, templates: Dict[str, Any], exp_id: str, overrides: Dict[str, Any] = {}) -> ExperimentResult:
        model_template = self.params.get("model_template") or "baseline"
        extra_args = [f"{k}={v}" for k, v in overrides.items()]
        cmd = self.executor.build_command(
            project=self.context.project_name,
            model_template=model_template,
            preprocess_template=templates["chain_path"].stem,
            exp_id=exp_id
        )
        cmd.extend(extra_args)
        self.logger.debug(f"Executing: {' '.join(cmd)}")
        return self.executor.run(cmd)

    def _get_next_fidelity(self, node: MCTSNode, trial_id: int) -> Optional[str]:
        levels = self.config.multi_fidelity.levels
        if not self.config.multi_fidelity.enable or not levels:
            evals = self.storage.get_evaluations(trial_id)
            if any(e["fidelity"] == "F2" for e in evals): return None
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
                self.logger.debug(f"Trial {trial_id}: {name} PRUNED")
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

    def _get_next_trial_number(self) -> int:
        with self.storage._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT MAX(number) FROM trials WHERE study_id=?", (self.study_id,))
            res = cur.fetchone()
            return (res[0] or 0) + 1

    def _state_to_params(self, state: PipelineState) -> Dict[str, Any]:
        params = {}
        for i, step in enumerate(state.steps):
            prefix = f"step_{i}"
            params[f"{prefix}.name"] = step["name"]
            params[f"{prefix}.variant"] = step["variant"]
            for k, v in step.get("config", {}).items():
                params[f"{prefix}.{k}"] = v
        return params