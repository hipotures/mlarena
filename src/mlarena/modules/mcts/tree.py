from __future__ import annotations
import math
import random
import json
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from mlarena.modules.mcts.config import MCTSConfig
from mlarena.modules.mcts.node import PipelineState, Action
from mlarena.modules.mcts.space import SuperChainActionSpace
from mlarena.modules.mcts.sampler import ParameterSampler

import logging

logger = logging.getLogger(__name__)

@dataclass
class MCTSNode:
    state: PipelineState
    parent: Optional[MCTSNode] = None
    action_from_parent: Optional[Action] = None
    children: List[MCTSNode] = field(default_factory=list)
    
    trial_id: Optional[int] = None
    n_visits: int = 0
    value_sum: float = 0.0
    value_best: float = -float('inf')
    
    # Untried actions for expansion (lazy generation)
    untried_actions: Optional[List[Action]] = None

    @property
    def value_mean(self) -> float:
        if self.n_visits == 0:
            return 0.0
        return self.value_sum / self.n_visits

    def update(self, value: float):
        self.n_visits += 1
        self.value_sum += value
        if value > self.value_best:
            self.value_best = value

class MCTSTree:
    def __init__(
        self, 
        config: MCTSConfig, 
        space: SuperChainActionSpace, 
        sampler: ParameterSampler
    ):
        self.config = config
        self.space = space
        self.sampler = sampler
        
        # Initialize root state with groups from fixed steps
        initial_groups = {}
        for fs in space.fixed_steps:
            cfg = fs["config"]
            group = cfg.get("group") or cfg.get("name")
            if group:
                initial_groups[group] = cfg.get("name")
        
        self.initial_groups = initial_groups # Store for rebuild_tree
        self.root = MCTSNode(state=PipelineState(used_groups=initial_groups))

    def rebuild_tree(self, nodes_data: List[Dict[str, Any]], edges_data: List[Dict[str, Any]]):
        """Reconstruct the tree structure from database records."""
        if not nodes_data:
            return

        # 1. Create all nodes first
        # Map: signature -> MCTSNode
        node_map: Dict[str, MCTSNode] = {}
        
        for n in nodes_data:
            sig = n["pipeline_signature"]
            # Important: Baseline needs initial_groups to preserve constraints after resume
            state_groups = self.initial_groups if sig == "baseline" else {}
            node = MCTSNode(state=PipelineState(used_groups=state_groups)) 
            node.trial_id = n["trial_id"]
            node.n_visits = n["n_visits"]
            node.value_sum = n["value_sum"]
            node.value_best = n["value_best"] if n["value_best"] is not None else -float('inf')
            
            node_map[sig] = node
            
            if sig == "baseline":
                # Ensure baseline preserves initial groups constraints
                node.state.used_groups = self.initial_groups.copy()
                self.root = node

        # 2. Link them using edges and reconstruct states
        # Map: trial_id -> signature
        id_to_sig = {n["trial_id"]: n["pipeline_signature"] for n in nodes_data}

        # We need to process edges in an order that builds the tree from top to bottom.
        # Sorting by child depth (which we have in nodes_data) is a good proxy.
        node_depths = {n["pipeline_signature"]: n["depth"] for n in nodes_data}
        sorted_edges = sorted(edges_data, key=lambda e: node_depths.get(id_to_sig.get(e["child_trial_id"], ""), 0))

        for edge in sorted_edges:
            parent_sig = id_to_sig.get(edge["parent_trial_id"])
            child_sig = id_to_sig.get(edge["child_trial_id"])
            
            if parent_sig and child_sig:
                parent = node_map[parent_sig]
                child = node_map[child_sig]
                
                action_dict = json.loads(edge["action_json"])
                action = Action.from_record(action_dict)
                
                child.parent = parent
                child.action_from_parent = action
                # Reconstruct child state from parent + action
                child.state = parent.state.add_action(action)
                
                if child not in parent.children:
                    parent.children.append(child)

    def select(self, node: MCTSNode) -> MCTSNode:
        """Select a leaf node to expand using UCT/PUCT."""
        current = node
        
        while True:
            # Check Progressive Widening limits
            # m(n) = k * N^alpha
            limit = self.config.expansion_width * (current.n_visits ** self.config.expansion_alpha)
            # Ensure at least 1 if visits > 0, but if visits=0 we still want to expand?
            # Usually PW logic applies when we COULD expand more but choose not to.
            # If current node is not fully expanded AND we are below limit -> Expand it
            # But "expand" is a separate step in MCTS loop.
            # "Select" usually descends until it hits a node that needs expansion.
            
            # If untried actions are None, generate them
            if current.untried_actions is None:
                actions = self.space.next_actions(current.state)
                
                # Filter out actions already present in children (crucial for resume)
                existing_keys = set()
                for child in current.children:
                    if child.action_from_parent:
                        act = child.action_from_parent
                        # Key: (searched_index, variant) - sufficient for uniqueness in this space
                        existing_keys.add((act.searched_index, act.variant_name))
                
                current.untried_actions = [
                    a for a in actions 
                    if (a.searched_index, a.variant_name) not in existing_keys
                ]
                # Shuffle to ensure random selection order
                random.shuffle(current.untried_actions)

            # Condition to stop selection and Expand:
            # 1. We have untried actions AND we are allowed to add more children (PW limit)
            # 2. Or node has no children (must expand)
            
            is_expandable = len(current.untried_actions) > 0
            current_children_count = len(current.children)
            
            # If we haven't reached the width limit yet, we stop here to Expand
            # Note: If visits=0, limit=0. We must ensure at least 1 child is tried.
            if limit < 1.0: limit = 1.0
            
            if is_expandable and current_children_count < limit:
                return current
            
            # If we can't expand (fully expanded OR limited by PW), we descend to best child
            if not current.children:
                # Terminal or no valid actions
                return current
                
            current = self._best_child(current)

    def _get_untried_actions(self, node: MCTSNode) -> List[Action]:
        """Return actions from the search space that haven't been tried yet for this node."""
        if node.untried_actions is None:
            # Lazy initialize all possible next actions
            node.untried_actions = self.space.next_actions(node.state)
            
        # Use a stable key to identify "tried" actions (step + variant)
        # instead of state signature which varies with sampled config.
        tried_keys = set()
        for child in node.children:
            act = child.action_from_parent
            if act:
                # Key: (searched_index, name, variant)
                key = (act.searched_index, act.step_name, act.variant_name)
                tried_keys.add(key)
        
        untried = [
            a for a in node.untried_actions 
            if (a.searched_index, a.step_name, a.variant_name) not in tried_keys
        ]
                
        return untried

    def expand(self, node: MCTSNode) -> MCTSNode:
        """Expand a node by adding a new child with a sampled action."""
        untried_actions = self._get_untried_actions(node)
        if not untried_actions:
            return node
            
        # Selection from untried: shuffle happened during lazy init in _get_untried_actions
        # or we just pick the first one from the list.
        action = untried_actions[0]
        
        # Determine a stable seed for this specific expansion
        action_seed_base = f"{self.config.seed}_{node.trial_id}_{action.step_name}_{action.variant_name}"
        import hashlib
        seed_hash = int(hashlib.md5(action_seed_base.encode()).hexdigest(), 16) % (2**32)
        local_rng = random.Random(seed_hash)
        
        # Pass RNG and search spaces
        sampled_config = self.sampler.sample(
            action.template_name, 
            action.variant_name, 
            search_spaces=self.space.search_spaces, 
            rng=local_rng
        )
        
        final_action = Action(
            step_name=action.step_name,
            template_name=action.template_name,
            group_name=action.group_name,
            variant_name=action.variant_name,
            config=sampled_config,
            searched_index=action.searched_index,
            original_index=action.original_index
        )
        
        new_state = node.state.add_action(final_action)
        child_node = MCTSNode(state=new_state, parent=node, action_from_parent=final_action)
        node.children.append(child_node)
        
        # Remove from untried to prevent immediate re-selection in same session
        if node.untried_actions:
            node.untried_actions = [a for a in node.untried_actions if a != action]
            
        return child_node

    def backpropagate(self, node: MCTSNode, value: float):
        """Update stats from node to root."""
        seen = set()
        current = node
        while current is not None:
            if id(current) in seen:
                # Use standard print or logger if available
                break
            seen.add(id(current))
            current.update(value)
            current = current.parent

    def _best_child(self, node: MCTSNode) -> MCTSNode:
        """Select best child using UCT/PUCT."""
        # UCT = Q + c * sqrt(ln(N_parent) / N_child)
        # PUCT logic adds prior, but for now simple UCT
        
        best_score = -float('inf')
        best_nodes = []
        
        c = self.config.exploration_weight
        ln_n = math.log(node.n_visits) if node.n_visits > 0 else 0
        
        logger.debug(f"[SELECTION] Evaluating {len(node.children)} children of node {node.trial_id or 'Root'}:")
        
        for child in node.children:
            exploit = child.value_mean
            if child.n_visits > 0:
                explore = c * math.sqrt(ln_n / child.n_visits)
            else:
                explore = float('inf') # Ensure unvisited children are picked? 
                # But PW usually ensures children added are visited immediately.
            
            score = exploit + explore
            
            # Action description
            act = child.action_from_parent
            act_desc = f"{act.step_name}:{act.variant_name}" if act else "unknown"
            logger.debug(f"  -> Child {child.trial_id or '?'}: {act_desc} | Q={exploit:.4f}, N={child.n_visits}, Explore={explore:.4f}, Total={score:.4f}")
            
            if score > best_score:
                best_score = score
                best_nodes = [child]
            elif score == best_score:
                best_nodes.append(child)
        
        if not best_nodes:
            return node.children[0] if node.children else None
            
        selected = random.choice(best_nodes)
        logger.debug(f"  -> Selected child {selected.trial_id or '?'}")
        return selected

def action_description(name: str) -> str:
    # Helper to clean up name if needed
    return name
