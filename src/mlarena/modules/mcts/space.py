from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
from mlarena.modules.mcts.node import PipelineState, Action

# Hardcoded for now, mimicking preprocess_tune
REPO_ROOT = Path(__file__).resolve().parents[4]
SEARCH_SPACE_DIR = REPO_ROOT / "src" / "mlarena" / "search_spaces" / "preprocess"

def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a dict: {path}")
    return data

def _load_search_spaces() -> Dict[str, Dict[str, Any]]:
    spaces: Dict[str, Dict[str, Any]] = {}
    if not SEARCH_SPACE_DIR.exists():
        return spaces
    for path in SEARCH_SPACE_DIR.glob("*.yaml"):
        name = path.stem
        spaces[name] = _load_yaml(path)
    return spaces

class SuperChainActionSpace:
    def __init__(self, super_chain_path: Path):
        self.super_chain_path = super_chain_path
        self.super_chain_config = _load_yaml(super_chain_path)
        self.all_steps = self.super_chain_config.get("preprocessors", []) or []
        self.search_spaces = _load_search_spaces()
        
        # Split steps into fixed harness and searched transforms
        self.fixed_steps = []
        self.searched_steps = []
        
        for idx, step in enumerate(self.all_steps):
            if not step.get("enabled", True):
                continue
                
            meta = step.get("meta", {}) or {}
            if meta.get("fixed"):
                self.fixed_steps.append({"index": idx, "config": step})
            else:
                self.searched_steps.append({"index": idx, "config": step})
        
        # For MCTS logic, we only iterate over searched_steps
        self.steps = [s["config"] for s in self.searched_steps]
        # Map original index
        self.searched_index_map = {i: s["index"] for i, s in enumerate(self.searched_steps)}

    @property
    def total_steps_count(self) -> int:
        return len(self.steps)

    def next_actions(self, state: PipelineState) -> List[Action]:
        """Generate possible next actions from the current state."""
        actions: List[Action] = []
        
        # We can pick any searched step that comes AFTER the last used searched step index
        # state.last_step_index refers to the index in self.steps (searched_steps)
        start_index = state.last_step_index + 1
        
        for i in range(start_index, len(self.steps)):
            step_def = self.steps[i]
            step_name = step_def.get("name")
            group = step_def.get("group") or step_name
            
            if group in state.used_groups:
                continue
            
            template_name = step_def.get("template") or step_name
            space = self.search_spaces.get(template_name, {})
            variants = space.get("variants", [])
            
            if not variants:
                variants = [{"name": "fixed", "params": {}}]

            for variant in variants:
                vname = variant.get("name")
                action = Action(
                    step_name=step_name,
                    template_name=template_name,
                    group_name=group, # Use the actual group name from super-chain
                    variant_name=vname,
                    config={}, 
                    searched_index=i, # Index in self.steps
                    original_index=self.searched_index_map[i] # Original index in super-chain
                )
                actions.append(action)
                
        return actions
