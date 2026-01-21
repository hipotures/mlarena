from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class Action:
    step_name: str
    template_name: str
    group_name: str
    variant_name: str
    config: Dict[str, Any]
    searched_index: int  # Index in the searched_steps list (for MCTS order)
    original_index: int  # Index in the full super-chain (for Materializer sorting)
    # 3rd dimension: multiple parameter samples for the same (step, variant)
    param_sample_id: int = 0
    # Heuristic prior probability from Oracle
    prior: float = 1.0

    def to_record(self) -> Dict[str, Any]:
        """Convert action to a stable dictionary format for database storage."""
        return {
            "step_name": self.step_name,
            "template_name": self.template_name,
            "group_name": self.group_name,
            "variant": self.variant_name,
            "config": self.config,
            "searched_index": self.searched_index,
            "original_index": self.original_index,
            "param_sample_id": int(self.param_sample_id),
            "prior": float(self.prior),
        }

    @staticmethod
    def from_record(d: Dict[str, Any]) -> Action:
        """Create an Action from a database record, handling legacy/variant keys."""
        return Action(
            step_name=d["step_name"],
            template_name=d.get("template_name") or d["step_name"],
            group_name=d.get("group_name") or d.get("group") or d["step_name"],
            variant_name=d.get("variant") or d.get("variant_name"),
            config=d.get("config") or {},
            # Robust index recovery
            searched_index=int(d.get("searched_index", d.get("step_index", 0))),
            original_index=int(d.get("original_index", d.get("step_index", 0))),
            param_sample_id=int(d.get("param_sample_id", 0)),
            prior=float(d.get("prior", 1.0)),
        )


@dataclass
class PipelineState:
    steps: List[Dict[str, Any]] = field(default_factory=list)
    depth: int = 0
    # Mapping of group_name -> step_name used in this pipeline
    used_groups: Dict[str, str] = field(default_factory=dict)
    # Index of the last searched step added, to enforce order
    last_step_index: int = -1

    def __post_init__(self):
        # Auto-calculate depth and groups if not provided but steps are
        if self.steps:
            self.depth = len(self.steps)
            if not self.used_groups:
                for step in self.steps:
                    group = step.get("group") or step.get("name")
                    if group:
                        self.used_groups[group] = step.get("name")

            # ALWAYS recover last_step_index from steps if it's unset (-1)
            if self.last_step_index == -1:
                for step in self.steps:
                    s_idx = step.get("searched_index", -1)
                    if s_idx > self.last_step_index:
                        self.last_step_index = s_idx

    @property
    def signature(self) -> str:
        """Canonical signature of the pipeline state.
        Includes searched_index/original_index/param_sample_id to ensure strict-tree uniqueness.
        """
        canonical_steps = []
        for s in self.steps:
            canonical_steps.append(
                {
                    "name": s["name"],
                    "template": s.get("template") or s["name"],
                    "group": s.get("group") or s.get("name"),
                    "variant": s["variant"],
                    "config": s["config"],
                    "searched_index": int(s.get("searched_index", -1)),
                    "original_index": int(s.get("original_index", -1)),
                    "param_sample_id": int(s.get("param_sample_id", 0)),
                }
            )

        serialized = json.dumps(canonical_steps, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def add_action(self, action: Action) -> PipelineState:
        """Return a new state with the action applied."""
        if action.group_name in self.used_groups:
            raise ValueError(
                f"Group '{action.group_name}' already used by '{self.used_groups[action.group_name]}'"
            )

        if action.searched_index <= self.last_step_index:
            raise ValueError(
                f"Action index {action.searched_index} violates order (last: {self.last_step_index})"
            )

        new_step = {
            "name": action.step_name,
            "template": action.template_name,
            "group": action.group_name,
            "variant": action.variant_name,
            "config": action.config,
            "searched_index": action.searched_index,
            "original_index": action.original_index,
            "param_sample_id": int(action.param_sample_id),
        }

        new_steps = list(self.steps) + [new_step]

        # Explicitly propagate and update used_groups to preserve fixed step constraints
        new_used = dict(self.used_groups)
        new_used[action.group_name] = action.step_name

        new_state = PipelineState(
            steps=new_steps,
            depth=len(new_steps),
            used_groups=new_used,
            last_step_index=action.searched_index,
        )
        return new_state
