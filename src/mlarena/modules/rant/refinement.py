from __future__ import annotations
import random
import numpy as np
from typing import Dict, Any, List, Optional


class NeighborhoodGenerator:
    """Generate parameter neighborhoods for refinement."""

    def __init__(self, grid_size: int = 20, log_grid_size: int = 20):
        """Initialize neighborhood generator.

        Args:
            grid_size: Grid size for float_range parameters
            log_grid_size: Grid size for log_uniform parameters
        """
        self.grid_size = grid_size
        self.log_grid_size = log_grid_size

    def neighbors_int_range(
        self, value: int, min_val: int, max_val: int, step: Optional[int] = None
    ) -> List[int]:
        """Generate neighbors for int_range parameter.

        Strategy: v±step, v±2×step, ... ordered by distance from v

        Args:
            value: Current value
            min_val: Minimum value
            max_val: Maximum value
            step: Step size (default 1)

        Returns:
            List of neighbor values
        """
        if step is None:
            step = 1

        neighbors = []
        for i in range(1, max(abs(max_val - value), abs(value - min_val)) // step + 1):
            delta = i * step
            if value + delta <= max_val:
                neighbors.append(value + delta)
            if value - delta >= min_val:
                neighbors.append(value - delta)

        return neighbors

    def neighbors_float_range(
        self,
        value: float,
        min_val: float,
        max_val: float,
        step: Optional[float] = None,
        log: bool = False
    ) -> List[float]:
        """Generate neighbors for float_range parameter.

        Strategy:
        - With step: v±step, v±2×step, ... ordered by distance
        - Without step: Use grid_size=(max-min)/20
        - Log scale: v×exp(±δ), v×exp(±2δ), ... ordered by distance

        Args:
            value: Current value
            min_val: Minimum value
            max_val: Maximum value
            step: Step size (optional)
            log: Log scale (log_uniform)

        Returns:
            List of neighbor values
        """
        if log:
            # Log scale: use multiplicative steps
            if min_val <= 0:
                min_val = 1e-10
            if value <= 0:
                value = max(value, 1e-10)

            log_min = np.log(min_val)
            log_max = np.log(max_val)
            log_value = np.log(value)

            # Compute delta for log_grid_size steps
            delta = (log_max - log_min) / self.log_grid_size

            neighbors = []
            for i in range(1, self.log_grid_size + 1):
                log_delta = i * delta

                # Positive direction
                new_log = log_value + log_delta
                if new_log <= log_max:
                    neighbors.append(float(np.exp(new_log)))

                # Negative direction
                new_log = log_value - log_delta
                if new_log >= log_min:
                    neighbors.append(float(np.exp(new_log)))

            return neighbors

        else:
            # Linear scale
            if step is None:
                # Use grid_size
                step = (max_val - min_val) / self.grid_size

            neighbors = []
            for i in range(1, int((max_val - min_val) / step) + 1):
                delta = i * step

                # Positive direction
                new_val = value + delta
                if new_val <= max_val:
                    neighbors.append(float(new_val))

                # Negative direction
                new_val = value - delta
                if new_val >= min_val:
                    neighbors.append(float(new_val))

            return neighbors

    def neighbors_choice(self, value: Any, values: List[Any], seed: int) -> List[Any]:
        """Generate neighbors for choice parameter.

        Strategy: Other values, seeded shuffle

        Args:
            value: Current value
            values: All possible values
            seed: Seed for shuffle

        Returns:
            List of neighbor values (all except current)
        """
        neighbors = [v for v in values if v != value]
        rng = random.Random(seed)
        rng.shuffle(neighbors)
        return neighbors

    def neighbors_bool(self, value: bool) -> List[bool]:
        """Generate neighbors for bool parameter.

        Strategy: Flip value (single neighbor)

        Args:
            value: Current value

        Returns:
            List with single flipped value
        """
        return [not value]

    def neighbors_subset(
        self, value: List[Any], values: List[Any], min_items: int, max_items: int, seed: int
    ) -> List[List[Any]]:
        """Generate neighbors for subset parameter.

        Strategy: Add/remove one item, seeded shuffle

        Args:
            value: Current subset
            values: All possible values
            min_items: Minimum items
            max_items: Maximum items
            seed: Seed for shuffle

        Returns:
            List of neighbor subsets
        """
        neighbors = []
        value_set = set(value)

        # Add one item (if not at max)
        if len(value) < max_items:
            for item in values:
                if item not in value_set:
                    new_subset = value + [item]
                    neighbors.append(new_subset)

        # Remove one item (if not at min)
        if len(value) > min_items:
            for item in value:
                new_subset = [v for v in value if v != item]
                neighbors.append(new_subset)

        # Shuffle
        rng = random.Random(seed)
        rng.shuffle(neighbors)

        return neighbors

    def neighbors_fixed(self) -> List:
        """Generate neighbors for fixed parameter.

        Strategy: No neighbors (empty list)

        Returns:
            Empty list
        """
        return []


class RefinementEngine:
    """Engine for generating refinement children using round-robin mutation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize refinement engine.

        Args:
            config: Refinement config dict (grid_size, log_grid_size, etc.)
        """
        self.config = config
        self.generator = NeighborhoodGenerator(
            grid_size=config.get('grid_size', 20),
            log_grid_size=config.get('log_grid_size', 20)
        )

    def generate_children(
        self,
        parent: Dict[str, Any],
        search_spaces: Dict[str, Any],
        n_children: int,
        seed: int
    ) -> List[Dict[str, Any]]:
        """Generate N children from parent using round-robin parameter mutation.

        Strategy:
        1. Identify mutable parameters (neighbors ≠ ∅)
        2. Seeded shuffle (seed = hash(parent_signature))
        3. Generate P children by rotating through params
        4. Child #i mutates param #(i mod |mutable_params|)

        Args:
            parent: Parent pipeline dict
            search_spaces: Search spaces dict
            n_children: Number of children to generate
            seed: Seed for deterministic generation

        Returns:
            List of child pipelines (may be < n_children if neighborhoods exhausted)
        """
        rng = random.Random(seed)

        # Extract mutable steps and parameters
        mutable_params = self._extract_mutable_params(parent, search_spaces)

        if not mutable_params:
            # No mutable parameters - return empty list
            return []

        # Seeded shuffle
        rng.shuffle(mutable_params)

        # Generate children
        children = []
        param_neighbors: Dict[int, List[Any]] = {}  # Cache neighborhoods

        for i in range(n_children):
            param_idx = i % len(mutable_params)
            param_info = mutable_params[param_idx]

            # Get or compute neighborhood
            if param_idx not in param_neighbors:
                neighbors = self._compute_neighborhood(param_info, seed)
                param_neighbors[param_idx] = neighbors
            else:
                neighbors = param_neighbors[param_idx]

            # Check if exhausted
            neighbor_offset = i // len(mutable_params)
            if neighbor_offset >= len(neighbors):
                # Exhausted this parameter - stop
                break

            # Mutate parameter
            new_value = neighbors[neighbor_offset]
            child = self._mutate_parameter(parent, param_info, new_value)
            children.append(child)

        return children

    def _extract_mutable_params(
        self, parent: Dict[str, Any], search_spaces: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract mutable parameters from parent pipeline.

        Returns:
            List of dicts with step_idx, param_name, current_value, spec
        """
        mutable = []

        for step_idx, step in enumerate(parent['full_chain']):
            template = step.get('template')
            variant = step.get('variant')
            config = step.get('config', {})

            if not template or not variant:
                continue

            # Get search space
            space = search_spaces.get(template)
            if not space:
                continue

            # Find variant spec
            variant_spec = next(
                (v for v in space.get('variants', []) if v['name'] == variant),
                None
            )
            if not variant_spec:
                continue

            # Check parameters
            params_spec = variant_spec.get('params', {})
            for param_name, param_spec in params_spec.items():
                if param_spec.get('type') == 'fixed':
                    continue

                current_value = config.get(param_name)
                if current_value is None:
                    continue

                mutable.append({
                    'step_idx': step_idx,
                    'step': step,
                    'param_name': param_name,
                    'current_value': current_value,
                    'spec': param_spec
                })

        return mutable

    def _compute_neighborhood(self, param_info: Dict[str, Any], seed: int) -> List[Any]:
        """Compute neighborhood for a parameter.

        Args:
            param_info: Parameter info dict
            seed: Seed for random operations

        Returns:
            List of neighbor values
        """
        spec = param_info['spec']
        current_value = param_info['current_value']
        ptype = spec.get('type')

        if ptype == 'int_range':
            min_val = spec.get('min', 0)
            max_val = spec.get('max', 10)
            step = spec.get('step')
            return self.generator.neighbors_int_range(current_value, min_val, max_val, step)

        elif ptype == 'float_range':
            min_val = spec.get('min', 0.0)
            max_val = spec.get('max', 1.0)
            step = spec.get('step')
            log = spec.get('log', False)
            return self.generator.neighbors_float_range(current_value, min_val, max_val, step, log)

        elif ptype == 'choice':
            values = spec.get('values', [])
            return self.generator.neighbors_choice(current_value, values, seed)

        elif ptype == 'bool':
            return self.generator.neighbors_bool(current_value)

        elif ptype == 'subset':
            values = spec.get('values', [])
            min_items = spec.get('min_items', 1)
            max_items = spec.get('max_items', len(values))
            return self.generator.neighbors_subset(current_value, values, min_items, max_items, seed)

        elif ptype == 'fixed':
            return self.generator.neighbors_fixed()

        else:
            return []

    def _mutate_parameter(
        self, parent: Dict[str, Any], param_info: Dict[str, Any], new_value: Any
    ) -> Dict[str, Any]:
        """Create child by mutating a single parameter.

        Args:
            parent: Parent pipeline
            param_info: Parameter info
            new_value: New parameter value

        Returns:
            Child pipeline dict
        """
        import copy
        child = copy.deepcopy(parent)

        step_idx = param_info['step_idx']
        param_name = param_info['param_name']

        # Update parameter
        child['full_chain'][step_idx]['config'][param_name] = new_value

        # Recompute signatures
        from mlarena.modules.rant.space import compute_signature
        child['pipeline_signature'] = compute_signature(child['full_chain'], include_params=True)
        child['architecture_signature'] = compute_signature(child['full_chain'], include_params=False)

        return child
