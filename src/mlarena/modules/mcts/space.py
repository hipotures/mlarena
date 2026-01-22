from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
import logging
from mlarena.modules.mcts.node import PipelineState, Action

logger = logging.getLogger(__name__)

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
        self.searched_index_map = {
            i: s["index"] for i, s in enumerate(self.searched_steps)
        }

        # Validate that there are no circular 'after' dependencies
        self._validate_after_dependencies()

    def _validate_after_dependencies(self) -> None:
        """Detect circular 'after' dependencies in search spaces."""
        from collections import defaultdict

        after_graph = defaultdict(set)

        # Build dependency graph from all search spaces
        for space_name, space in self.search_spaces.items():
            variants = space.get("variants", [])
            for variant in variants:
                requirements = variant.get("requires_preproc", [])
                for req in requirements:
                    if req.get("step") == "after":
                        req_group = req.get("group")
                        if req_group:
                            after_graph[space_name].add(req_group)

        # DFS cycle detection
        visited = set()
        rec_stack = set()

        def has_cycle(node, path):
            if node not in after_graph:
                return False

            visited.add(node)
            rec_stack.add(node)

            for neighbor in after_graph[node]:
                if neighbor not in visited:
                    if has_cycle(neighbor, path + [neighbor]):
                        return True
                elif neighbor in rec_stack:
                    raise ValueError(
                        f"Circular 'after' dependency detected: {' -> '.join(path + [neighbor])}"
                    )

            rec_stack.remove(node)
            return False

        for node in after_graph:
            if node not in visited:
                has_cycle(node, [node])

    @property
    def total_steps_count(self) -> int:
        return len(self.steps)

    def next_actions(
        self, state: PipelineState, lookahead: Optional[int] = None
    ) -> List[Action]:
        """Generate possible next actions from the current state."""
        actions: List[Action] = []

        # We can pick any searched step that comes AFTER the last used searched step index
        # state.last_step_index refers to the index in self.steps (searched_steps)
        start_index = state.last_step_index + 1
        end_index = len(self.steps)

        if lookahead is not None and lookahead > 0:
            end_index = min(end_index, start_index + lookahead)

        for i in range(start_index, end_index):
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
                # Check for dependencies (requires_preproc)
                requirements = variant.get("requires_preproc", [])
                met = True
                pending_after = []
                activated_by_before = []

                for req in requirements:
                    req_group = req.get("group")
                    step_timing = req.get("step", "before")  # Default to "before"

                    if step_timing == "before":
                        # Current behavior: must already be in used_groups
                        if req_group and req_group not in state.used_groups:
                            met = False
                            break
                        if req_group:
                            activated_by_before.append(req_group)
                    elif step_timing == "after":
                        # New behavior: collect for later auto-injection
                        pending_after.append(req)
                    else:
                        # Validation: reject invalid values
                        raise ValueError(
                            f"Invalid 'step' value: {step_timing}. Must be 'before' or 'after'"
                        )

                if not met:
                    continue

                vname = variant.get("name")

                if activated_by_before:
                    logger.debug(
                        "Dependency step=before: %s:%s activated by %s",
                        step_name,
                        vname,
                        activated_by_before,
                    )
                if pending_after:
                    after_groups = [
                        r.get("group") for r in pending_after if r.get("group")
                    ]
                    if after_groups:
                        logger.debug(
                            "Dependency step=after: %s:%s activates %s",
                            step_name,
                            vname,
                            after_groups,
                        )

                action = Action(
                    step_name=step_name,
                    template_name=template_name,
                    group_name=group,  # Use the actual group name from super-chain
                    variant_name=vname,
                    config={},
                    searched_index=i,  # Index in self.steps
                    original_index=self.searched_index_map[
                        i
                    ],  # Original index in super-chain
                    param_sample_id=0,
                    pending_after=pending_after,  # NEW: Pass after-dependencies
                )
                actions.append(action)

        return actions

    def analyze_next_actions(self, state: PipelineState) -> Dict[str, int]:
        """Diagnostic method to understand why actions are filtered."""
        stats = {
            "total_candidates": 0,
            "filtered_by_group": 0,
            "filtered_by_no_variants": 0,
            "emitted_actions": 0,
            "steps_checked": 0,
        }

        start_index = state.last_step_index + 1
        stats["total_candidates"] = len(self.steps) - start_index

        for i in range(start_index, len(self.steps)):
            stats["steps_checked"] += 1
            step_def = self.steps[i]
            step_name = step_def.get("name")
            group = step_def.get("group") or step_name

            if group in state.used_groups:
                stats["filtered_by_group"] += 1
                continue

            template_name = step_def.get("template") or step_name
            space = self.search_spaces.get(template_name, {})
            variants = space.get("variants", [])

            if not variants:
                # In next_actions we default to 'fixed', so this isn't strictly a filter there
                # unless we change logic. Count it if empty but treated as 1 action.
                # Currently next_actions adds 1 action.
                pass

            count = len(variants) if variants else 1
            stats["emitted_actions"] += count

        return stats
