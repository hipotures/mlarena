from __future__ import annotations
import time
from pathlib import Path
from typing import Dict, Any, Optional

from mlarena.core.module import ModuleContext, ModuleResult
from mlarena.modules.mcts.config import load_mcts_config, MCTSConfig
from mlarena.modules.mcts.storage import MCTSStorage, StudyDirection, TrialState
from mlarena.modules.mcts.space import SuperChainActionSpace
from mlarena.modules.mcts.sampler import ParameterSampler
from mlarena.modules.mcts.tree import MCTSTree, MCTSNode
from mlarena.modules.mcts.node import PipelineState
from mlarena.modules.mcts.materializer import TemplateMaterializer
from mlarena.modules.mcts.executor import MlaCliExecutor, ExperimentResult

DEFAULT_SUPER_CHAIN = "conf/preprocess/mla_super_chain.yaml"

class MCTSRunner:
    def __init__(self, context: ModuleContext, params: Dict[str, Any]):
        self.context = context
        self.params = params
        
        # Load Config
        # If config is passed in params, use it, else load from yaml
        # Usually we load from yaml and override with params
        super_chain_path = Path(params.get("super_chain", DEFAULT_SUPER_CHAIN))
        if not super_chain_path.is_absolute():
            super_chain_path = context.project_root / super_chain_path
            
        self.config = load_mcts_config(super_chain_path)
        
        # Override study name if passed
        if params.get("study_name"):
            self.config.study_name = params["study_name"]
            
        # Init components
        self.storage = MCTSStorage(self.config.storage_url)
        self.space = SuperChainActionSpace(super_chain_path)
        self.sampler = ParameterSampler(seed=self.config.seed)
        self.tree = MCTSTree(self.config, self.space, self.sampler)
        self.materializer = TemplateMaterializer(
            run_id=f"{self.config.study_name}_{int(time.time())}", 
            project_root=context.project_root
        )
        self.executor = MlaCliExecutor(context.project_root)
        
        # Determine direction
        self.direction = StudyDirection.MAXIMIZE if self.config.direction == "maximize" else StudyDirection.MINIMIZE
        self.study_id = self.storage.create_study(self.config.study_name, self.direction)

    def run(self) -> ModuleResult:
        print(f"Starting MCTS Study: {self.config.study_name}")
        
        # 1. Baseline Evaluation (Model Zero)
        self._evaluate_baseline()
        
        # 2. Main Loop
        for i in range(self.config.budget):
            print(f"Iteration {i+1}/{self.config.budget}")
            
            # Selection
            node = self.tree.select(self.tree.root)
            
            # Expansion
            # If node is fully expanded, we might need to backpropagate directly or loop
            # But with PW, select() returns a node that is allowed to expand.
            # If select returns a leaf that can extend, we extend.
            
            child = self.tree.expand(node)
            
            # Evaluation
            if child == node:
                # Could not expand (no actions available)
                # Re-evaluate or just backprop current value
                # For now, treat as terminal, backprop best value seen or 0
                # Ideally, if it's terminal, we should have evaluated it when it was created.
                # If we just created it, we eval.
                # But here child == node means expand failed.
                print("  -> Could not expand (terminal or limits)")
                self.tree.backpropagate(node, node.value_best if node.value_best > -float('inf') else 0.0)
                continue
                
            # Materialize & Execute
            # We need a trial ID for this node.
            # We should probably persist the node to DB now.
            trial_id = self.storage.create_trial(
                self.study_id,
                number=self._get_next_trial_number(),
                pipeline_signature=child.state.signature,
                depth=child.state.depth,
                params=self._state_to_params(child.state)
            )
            
            result = self._execute_trial(child, trial_id)
            
            # Backpropagate
            if result.success and result.value is not None:
                self.tree.backpropagate(child, result.value)
                self.storage.set_trial_value(trial_id, result.value)
                self.storage.set_trial_state(trial_id, TrialState.COMPLETE)
            else:
                # Penalize failure?
                penalty = -1.0 if self.direction == StudyDirection.MAXIMIZE else 1.0
                self.tree.backpropagate(child, penalty) # Simple penalty
                self.storage.set_trial_state(trial_id, TrialState.FAIL)

        best = self.storage.get_best_trial(self.study_id)
        return ModuleResult(success=True, payload={"best_trial": best})

    def _evaluate_baseline(self):
        # Todo: check if baseline exists in DB
        # If not, create trial 0 and run
        # For now, simplistic run
        print("Evaluating Baseline (Model Zero)")
        
        # Check if exists
        # Simplification: just run it as trial 0
        trial_id = self.storage.create_trial(
            self.study_id, 0, "baseline", 0, {}, state=TrialState.RUNNING
        )
        
        # Baseline = empty chain
        # Materializer should handle empty state
        state = PipelineState()
        templates = self.materializer.materialize(state, node_id=0)
        
        # Execute
        result = self._run_mla(templates, "baseline_exp")
        
        if result.success and result.value is not None:
            self.storage.set_trial_value(trial_id, result.value)
            self.storage.set_trial_state(trial_id, TrialState.COMPLETE)
            # Update root stats?
            # Root corresponds to baseline state.
            self.tree.root.update(result.value)
        else:
            self.storage.set_trial_state(trial_id, TrialState.FAIL)

    def _execute_trial(self, node: MCTSNode, trial_id: int) -> ExperimentResult:
        # Materialize
        templates = self.materializer.materialize(node.state, node_id=trial_id)
        exp_id = f"mcts_{self.config.study_name}_{trial_id}"
        
        return self._run_mla(templates, exp_id)

    def _run_mla(self, templates: Dict[str, Any], exp_id: str) -> ExperimentResult:
        # We need model template from params or config
        model_template = self.params.get("model_template") or "baseline" # Fallback
        
        cmd = self.executor.build_command(
            project=self.context.project_name,
            model_template=model_template,
            preprocess_template=templates["chain_path"].stem, # pass name, not path? MLA usually takes name if in templates dir
            exp_id=exp_id
        )
        
        # Run
        return self.executor.run(cmd)

    def _get_next_trial_number(self) -> int:
        # Simple counter using DB could be better
        # For now, query max number
        with self.storage._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT MAX(number) FROM trials WHERE study_id=?", (self.study_id,))
            res = cur.fetchone()
            return (res[0] or 0) + 1

    def _state_to_params(self, state: PipelineState) -> Dict[str, Any]:
        # Flatten state for DB
        params = {}
        for i, step in enumerate(state.steps):
            prefix = f"step_{i}"
            params[f"{prefix}.name"] = step["name"]
            params[f"{prefix}.variant"] = step["variant"]
            for k, v in step.get("config", {}).items():
                params[f"{prefix}.{k}"] = v
        return params