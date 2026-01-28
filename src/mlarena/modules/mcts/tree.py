from __future__ import annotations
import math
import random
import json
from typing import List, Optional, Dict, Any, Literal
from dataclasses import dataclass, field

from mlarena.modules.mcts.config import MCTSConfig
from mlarena.modules.mcts.node import PipelineState, Action
from mlarena.modules.mcts.space import SuperChainActionSpace
from mlarena.modules.mcts.sampler import ParameterSampler
from mlarena.modules.mcts.storage import TrialState
from mlarena.core.oracle import ActionOracle

import logging

logger = logging.getLogger(__name__)


@dataclass
class MCTSNode:
    state: PipelineState
    parent: Optional[MCTSNode] = None
    action_from_parent: Optional[Action] = None
    children: List[MCTSNode] = field(default_factory=list)

    trial_id: Optional[int] = None
    number: Optional[int] = None
    n_visits: int = 0
    value_sum: float = 0.0
    value_best: Optional[float] = None

    # Base action pool (operator candidates: step+variant) cached per node.
    action_pool: Optional[List[Action]] = None

    # Terminal flag: if True, this node cannot be expanded (e.g., failed trials)
    is_terminal: bool = False

    @property
    def value_mean(self) -> float:
        if self.n_visits == 0:
            return 0.0
        return self.value_sum / self.n_visits

    def update(
        self, value: float, *, direction: Literal["minimize", "maximize"] = "maximize"
    ):
        self.n_visits += 1
        self.value_sum += value
        if self.value_best is None:
            self.value_best = value
        elif direction == "minimize":
            if value < self.value_best:
                self.value_best = value
        else:
            if value > self.value_best:
                self.value_best = value


class MCTSTree:
    def __init__(
        self,
        config: MCTSConfig,
        space: SuperChainActionSpace,
        sampler: ParameterSampler,
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

        self.initial_groups = initial_groups  # Store for rebuild_tree
        self.root = MCTSNode(state=PipelineState(used_groups=initial_groups))

        # Initialize Oracle
        self.oracle = ActionOracle(config.oracle.model_dump())

    def rebuild_tree(
        self, nodes_data: List[Dict[str, Any]], edges_data: List[Dict[str, Any]]
    ):
        """Reconstruct the tree structure from database records."""
        if not nodes_data:
            return

        # 1. Create all nodes first
        # Map: trial_id -> MCTSNode (tree-safe; signatures may repeat under different parents)
        node_map: Dict[int, MCTSNode] = {}

        for n in nodes_data:
            sig = n["pipeline_signature"]
            # Important: Baseline needs initial_groups to preserve constraints after resume
            state_groups = self.initial_groups if sig == "baseline" else {}
            node = MCTSNode(state=PipelineState(used_groups=state_groups))
            node.trial_id = n["trial_id"]
            node.number = n.get("number")
            node.n_visits = n["n_visits"]
            node.value_sum = n["value_sum"]
            node.value_best = n.get("value_best")
            trial_state = n.get("trial_state")
            if trial_state is not None:
                try:
                    ts_val = int(trial_state)
                except (TypeError, ValueError):
                    ts_val = None
                if ts_val in (TrialState.FAIL.value, TrialState.PRUNED.value):
                    node.is_terminal = True

            node_map[node.trial_id] = node

            if sig == "baseline" and node.trial_id is not None:
                # Ensure baseline preserves initial groups constraints
                node.state.used_groups = self.initial_groups.copy()
                self.root = node

        # 2. Link them using edges and reconstruct states
        # Map: trial_id -> depth for sorting (optional, but good for stability)
        node_depths = {n["trial_id"]: n["depth"] for n in nodes_data}

        # We need to process edges in an order that builds the tree from top to bottom.
        # Sorting by child depth (which we have in nodes_data) is a good proxy.
        sorted_edges = sorted(
            edges_data, key=lambda e: node_depths.get(e["child_trial_id"], 0)
        )

        for edge in sorted_edges:
            parent = node_map.get(edge["parent_trial_id"])
            child = node_map.get(edge["child_trial_id"])

            if parent and child:
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
            # Check if current node is terminal (e.g., failed trial)
            if current.is_terminal:
                return current

            # With 2-layer PW, _get_untried_actions() already enforces both:
            # - operator PW (how many operators can be active)
            # - param PW (how many param samples per active operator)
            untried = self._get_untried_actions(current)
            if untried:
                return current

            # If we can't expand (fully expanded OR limited by PW), we descend to best child
            if not current.children:
                # Terminal or no valid actions
                return current

            current = self._best_child(current)

    def _get_untried_actions(self, node: MCTSNode) -> List[Action]:
        """Return untried actions using 2-layer PW:
        (1) operator PW: how many distinct (searched_index, step_name, variant_name) are active
        (2) param PW: how many param_sample_id per active operator
        """
        if node.action_pool is None:
            import hashlib

            # Determine randomize_order and seed
            randomize = getattr(self.config, 'randomize_order', False)
            seed_for_order = None

            if randomize:
                # Stable seed per node state for reproducibility
                seed_base = (
                    f"{self.config.seed}|order|{node.state.signature}"
                    f"|last={node.state.last_step_index}"
                )
                seed_hash = int(hashlib.md5(seed_base.encode()).hexdigest(), 16) % (2**32)
                seed_for_order = seed_hash

            raw_pool = self.space.next_actions(
                node.state,
                lookahead=getattr(self.config, "lookahead", 3),
                randomize_order=randomize,
                seed=seed_for_order
            )

            # Deterministic RNG for pool ordering
            seed_base = f"{self.config.seed}|pool|{node.state.signature}|last={node.state.last_step_index}"
            seed_hash = int(hashlib.md5(seed_base.encode()).hexdigest(), 16) % (2**32)
            rng = random.Random(seed_hash)

            # Apply group weights to all actions
            for act in raw_pool:
                group_weight = self.config.group_weights.get(act.group_name, 1.0)
                act.group_weight = group_weight

            if self.config.oracle.enabled and raw_pool:
                # --- Oracle Logic ---
                # Evaluate actions: filters out low-prob ones (Pruning) and returns priors
                filtered_pool, priors = self.oracle.evaluate_actions(node, raw_pool)

                # Assign priors and Calculate Gumbel Score for Weighted Shuffle
                # Score = log(P * group_weight + eps) + Gumbel(0,1)
                # Gumbel(0,1) ~ -log(-log(U)) where U ~ Uniform(0,1)
                eps = 1e-9

                scored_pool = []
                for act, p in zip(filtered_pool, priors):
                    # Apply group weight to prior
                    weighted_p = p * getattr(act, "group_weight", 1.0)
                    act.prior = weighted_p  # Store weighted prior for PUCT

                    # Gumbel noise
                    u = rng.random()
                    u = max(1e-9, min(1.0 - 1e-9, u))  # Clip
                    gumbel = -math.log(-math.log(u))

                    # Log-prob + Gumbel
                    score = math.log(weighted_p + eps) + gumbel
                    scored_pool.append((score, act))

                # Sort by Gumbel score (equivalent to weighted sampling without replacement)
                scored_pool.sort(key=lambda x: x[0], reverse=True)
                node.action_pool = [x[1] for x in scored_pool]

                if any(getattr(act, "group_weight", 1.0) != 1.0 for act in node.action_pool[:5]):
                    logger.debug(
                        f"[GroupWeights] Applied weights to actions at node. "
                        f"Top 5: {[(act.group_name, getattr(act, 'group_weight', 1.0)) for act in node.action_pool[:5]]}"
                    )
            else:
                # --- Standard Logic with Group Weights ---
                # Weight-based shuffle: actions with lower group_weight appear later
                weighted_pool = []
                for act in raw_pool:
                    weight = getattr(act, "group_weight", 1.0)
                    # Score = log(weight) + Gumbel noise
                    u = rng.random()
                    u = max(1e-9, min(1.0 - 1e-9, u))
                    gumbel = -math.log(-math.log(u))
                    score = math.log(weight + 1e-9) + gumbel
                    weighted_pool.append((score, act))

                weighted_pool.sort(key=lambda x: x[0], reverse=True)
                node.action_pool = [x[1] for x in weighted_pool]

                if any(getattr(act, "group_weight", 1.0) != 1.0 for act in node.action_pool[:5]):
                    logger.debug(
                        f"[GroupWeights] Applied weights (no Oracle). "
                        f"Top 5: {[(act.group_name, getattr(act, 'group_weight', 1.0)) for act in node.action_pool[:5]]}"
                    )

        def op_key(a: Action):
            return (a.searched_index, a.step_name, a.variant_name)

        # Map operator -> base action (for template/group/original_index)
        op_to_base: Dict[tuple, Action] = {}
        for a in node.action_pool:
            k = op_key(a)
            if k not in op_to_base:
                op_to_base[k] = a

        # Existing operators already present in children
        existing_ops = set()
        for ch in node.children:
            act = ch.action_from_parent
            if act:
                existing_ops.add(op_key(act))
                if op_key(act) not in op_to_base:
                    # In case action_pool is missing it for any reason
                    op_to_base[op_key(act)] = act

        # -------- Layer 1: operator PW --------
        # m_ops = max(1, floor(k_ops * N^alpha))
        k_ops = self.config.expansion_width
        a_ops = self.config.expansion_alpha
        n = max(1, node.n_visits)
        m_ops = int(k_ops * (n**a_ops))
        if m_ops < 1:
            m_ops = 1
        # never less than existing operators
        m_ops = max(m_ops, len(existing_ops))
        m_ops = min(m_ops, len(op_to_base))

        active_ops = list(existing_ops)
        if len(active_ops) < m_ops:
            # add new operators from pool until we reach m_ops
            for a in node.action_pool:
                k = op_key(a)
                if k in existing_ops:
                    continue
                active_ops.append(k)
                existing_ops.add(k)
                if len(active_ops) >= m_ops:
                    break

        # -------- Layer 2: param PW per operator --------
        k_p = self.config.param_expansion_width
        a_p = self.config.param_expansion_alpha
        max_p = int(self.config.param_expansion_max_samples)

        untried: List[Action] = []
        for ok in active_ops:
            base = op_to_base.get(ok)
            if base is None:
                continue

            # op_visits = sum visits over children that belong to this operator
            op_children = []
            op_visits = 0
            tried_sids = set()
            for ch in node.children:
                act = ch.action_from_parent
                if not act:
                    continue
                if op_key(act) == ok:
                    op_children.append(ch)
                    op_visits += ch.n_visits
                    tried_sids.add(int(getattr(act, "param_sample_id", 0)))

            # m_params = max(1, floor(k_p * (max(1, op_visits)^a_p)))
            n_op = max(1, op_visits)
            m_params = int(k_p * (n_op**a_p))
            if m_params < 1:
                m_params = 1
            if max_p > 0:
                m_params = min(m_params, max_p)

            for sid in range(m_params):
                if sid in tried_sids:
                    continue
                untried.append(
                    Action(
                        step_name=base.step_name,
                        template_name=base.template_name,
                        group_name=base.group_name,
                        variant_name=base.variant_name,
                        config={},
                        searched_index=base.searched_index,
                        original_index=base.original_index,
                        param_sample_id=sid,
                        prior=base.prior,
                    )
                )

        return untried

    def _get_untried_actions_random_shuffle(
        self, node: MCTSNode, untried: List[Action]
    ) -> List[Action]:
        """Deterministic shuffle for untried actions list."""
        if not untried:
            return untried
        import hashlib

        seed_base = (
            f"{self.config.seed}|untried|{node.state.signature}|n={node.n_visits}"
        )
        seed_hash = int(hashlib.md5(seed_base.encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed_hash)
        rng.shuffle(untried)
        return untried

    def expand(self, node: MCTSNode) -> MCTSNode:
        """Expand a node by adding a new child with a sampled action."""
        # Cannot expand terminal nodes (e.g., failed trials)
        if node.is_terminal:
            return node

        untried_actions = self._get_untried_actions_random_shuffle(
            node, self._get_untried_actions(node)
        )
        if not untried_actions:
            return node

        # Selection from untried: shuffle happened during lazy init in _get_untried_actions
        # or we just pick the first one from the list.
        action = untried_actions[0]

        # Determine a stable seed for this specific expansion
        # Stable seed per (parent-state, operator, param_sample_id)
        parent_sig = node.state.signature if node != self.root else "baseline"
        action_seed_base = f"{self.config.seed}|{parent_sig}|{action.searched_index}|{action.step_name}|{action.variant_name}|sid={action.param_sample_id}"
        import hashlib

        seed_hash = int(hashlib.md5(action_seed_base.encode()).hexdigest(), 16) % (
            2**32
        )
        local_rng = random.Random(seed_hash)

        # Pass RNG and search spaces
        sampled_config = self.sampler.sample(
            action.template_name,
            action.variant_name,
            search_spaces=self.space.search_spaces,
            rng=local_rng,
        )

        final_action = Action(
            step_name=action.step_name,
            template_name=action.template_name,
            group_name=action.group_name,
            variant_name=action.variant_name,
            config=sampled_config,
            searched_index=action.searched_index,
            original_index=action.original_index,
            param_sample_id=action.param_sample_id,
            prior=action.prior,  # Preserve Oracle prior
        )

        new_state = node.state.add_action(final_action)
        child_node = MCTSNode(
            state=new_state, parent=node, action_from_parent=final_action
        )
        node.children.append(child_node)

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
            current.update(value, direction=self.config.direction)
            current = current.parent

    def _best_child(self, node: MCTSNode) -> MCTSNode:
        """Select best child using UCT/PUCT."""
        # UCT = Q + c * sqrt(ln(N_parent) / N_child)
        # PUCT logic adds prior, but for now simple UCT

        best_score = -float("inf")
        best_nodes = []

        c = self.config.exploration_weight
        ln_n = math.log(node.n_visits) if node.n_visits > 0 else 0
        minimize = self.config.direction == "minimize"

        logger.debug(
            f"[SELECTION] Evaluating {len(node.children)} children of node {node.trial_id or 'Root'}:"
        )

        for child in node.children:
            exploit = child.value_mean
            # Convert to "higher is better" for selection when minimizing.
            if minimize:
                exploit = -exploit

            prior = 1.0
            if child.action_from_parent:
                prior = child.action_from_parent.prior

            if child.n_visits > 0:
                if self.config.oracle.use_prior_in_puct:
                    # PUCT: Q + c * P * sqrt(N) / (1 + n)
                    explore = (
                        c * prior * math.sqrt(node.n_visits) / (1 + child.n_visits)
                    )
                else:
                    # Standard UCT
                    explore = c * math.sqrt(ln_n / child.n_visits)
            else:
                # Unvisited
                if self.config.oracle.use_prior_in_puct:
                    # First visit priority guided by Oracle
                    explore = c * prior * math.sqrt(node.n_visits + 1)  # +1 to avoid 0
                else:
                    explore = float("inf")

            score = exploit + explore

            # Action description
            act = child.action_from_parent
            act_desc = f"{act.step_name}:{act.variant_name}" if act else "unknown"
            logger.debug(
                f"  -> Child {child.trial_id or '?'}: {act_desc} | Q={exploit:.4f}, N={child.n_visits}, Explore={explore:.4f}, Total={score:.4f}"
            )

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
