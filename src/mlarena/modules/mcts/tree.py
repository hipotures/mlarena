from __future__ import annotations
import math
import random
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from mlarena.modules.mcts.config import MCTSConfig
from mlarena.modules.mcts.node import PipelineState, Action
from mlarena.modules.mcts.space import SuperChainActionSpace
from mlarena.modules.mcts.sampler import ParameterSampler

@dataclass
class MCTSNode:
    state: PipelineState
    parent: Optional[MCTSNode] = None
    action_from_parent: Optional[Action] = None
    children: List[MCTSNode] = field(default_factory=list)
    
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
        self.root = MCTSNode(state=PipelineState())

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
                current.untried_actions = self.space.next_actions(current.state)
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

    def expand(self, node: MCTSNode) -> MCTSNode:
        """Add a new child to the node."""
        if not node.untried_actions:
            return node # Cannot expand
            
        action = node.untried_actions.pop()
        
        # Sample parameters for the action
        sampled_config = {} 
        
        # Look up search space for this template and variant
        space = self.space.search_spaces.get(action.template_name, {})
        variants = space.get("variants", [])
        variant_spec = next((v for v in variants if v.get("name") == action.variant_name), None)
        
        if variant_spec:
            params_spec = variant_spec.get("params", {})
            for p_name, p_spec in params_spec.items():
                if isinstance(p_spec, dict):
                    sampled_config[p_name] = self.sampler.sample(f"{action.step_name}.{p_name}", p_spec)
        
        # Update action with sampled config
        final_action = Action(
            step_name=action.step_name,
            template_name=action.template_name,
            variant_name=action.variant_name,
            config=sampled_config,
            step_index=action.step_index
        )
        
        new_state = node.state.add_action(final_action)
        child = MCTSNode(
            state=new_state,
            parent=node,
            action_from_parent=final_action
        )
        node.children.append(child)
        return child

    def backpropagate(self, node: MCTSNode, value: float):
        """Update stats from node to root."""
        current = node
        while current is not None:
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
        
        for child in node.children:
            exploit = child.value_mean
            if child.n_visits > 0:
                explore = c * math.sqrt(ln_n / child.n_visits)
            else:
                explore = float('inf') # Ensure unvisited children are picked? 
                # But PW usually ensures children added are visited immediately.
            
            score = exploit + explore
            
            if score > best_score:
                best_score = score
                best_nodes = [child]
            elif score == best_score:
                best_nodes.append(child)
        
        if not best_nodes:
            return node.children[0] if node.children else None
            
        return random.choice(best_nodes)
