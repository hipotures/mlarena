from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class Action:
    step_name: str
    template_name: str # Added to lookup search space
    group_name: str    # Added to respect exclusions
    variant_name: str
    config: Dict[str, Any]
    step_index: int  # Index in the super-chain

@dataclass
class PipelineState:
    steps: List[Dict[str, Any]] = field(default_factory=list)
    depth: int = 0
    # Mapping of group_name -> step_name used in this pipeline
    used_groups: Dict[str, str] = field(default_factory=dict)
    # Index of the last step added (from super-chain), to enforce order
    last_step_index: int = -1

    def __post_init__(self):
        # Auto-calculate depth and groups if not provided but steps are
        if self.steps and not self.used_groups:
            self.depth = len(self.steps)
            for step in self.steps:
                group = step.get("group") or step.get("name")
                if group:
                    self.used_groups[group] = step.get("name")

    @property
    def signature(self) -> str:
        """Canonical signature of the pipeline state."""
        # Use a stable JSON serialization of steps
        # We assume 'steps' contains only JSON-serializable data
        # To be safe, we sort keys in dictionaries
        serialized = json.dumps(self.steps, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def add_action(self, action: Action) -> PipelineState:
        """Return a new state with the action applied."""
        new_step = {
            "name": action.step_name,
            "template": action.template_name,
            "group": action.group_name, 
            "variant": action.variant_name,
            "config": action.config,
        }
        
        new_steps = list(self.steps) + [new_step]
        new_state = PipelineState(
            steps=new_steps,
            last_step_index=action.step_index
        )
        return new_state
