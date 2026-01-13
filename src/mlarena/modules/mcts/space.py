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
        self.steps = self.super_chain_config.get("preprocessors", []) or []
        self.search_spaces = _load_search_spaces()

    @property
    def total_steps_count(self) -> int:
        return len(self.steps)

    def next_actions(self, state: PipelineState) -> List[Action]:
        """Generate possible next actions from the current state."""
        actions: List[Action] = []
        
        # We can pick any step that comes AFTER the last used step index
        start_index = state.last_step_index + 1
        
        for idx in range(start_index, len(self.steps)):
            step_def = self.steps[idx]
            step_name = step_def.get("name")
            group = step_def.get("group") or step_name
            
            # Constraint: One step per group
            if group in state.used_groups:
                continue
            
            # TODO: Add EDA gating, heavy step check, etc.
            
            # Get variants for this step
            space = self.search_spaces.get(step_def.get("template") or step_name, {})
            variants = space.get("variants", [])
            
            # If no variants defined, maybe it's fixed or single default?
            # For MCTS search, we usually want explicit choices.
            # If variants empty but step is valid, we might add a "default" variant.
            # For now, let's assume variants exist or we skip.
            
            if not variants:
                # If no variants, check if it's a fixed step we can just enable?
                # The prompt implies we pick variants.
                # Let's add a "fixed" variant if none exist but it's a valid step?
                # Or just skip. Let's iterate variants.
                pass

            for variant in variants:
                vname = variant.get("name")
                # Create action stub (params will be sampled later)
                action = Action(
                    step_name=step_name,
                    variant_name=vname,
                    config={}, # Config will be sampled during expansion
                    step_index=idx
                )
                actions.append(action)
                
        return actions
