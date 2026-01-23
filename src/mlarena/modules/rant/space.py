from __future__ import annotations
import random
import hashlib
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set
from collections import defaultdict, deque

from mlarena.modules.mcts.sampler import ParameterSampler


class PipelineGenerator:
    """Generate random preprocessing pipelines with dependency resolution."""

    def __init__(
        self,
        super_chain_path: str | Path,
        search_spaces_dir: str | Path,
        seed: int = 42
    ):
        """Initialize generator with super-chain and search spaces.

        Args:
            super_chain_path: Path to mla_super_chain.yaml
            search_spaces_dir: Directory containing search space YAML files
        """
        self.super_chain_path = Path(super_chain_path)
        self.search_spaces_dir = Path(search_spaces_dir)
        self.seed = seed
        self.rng = random.Random(seed)
        self.sampler = ParameterSampler(seed)

        # Load configurations
        self.super_chain = self._load_super_chain()
        self.search_spaces = self._load_search_spaces()

        # Parse super-chain structure
        self.steps_by_name = {s['name']: s for s in self.super_chain['preprocessors']}
        self.steps_by_group = defaultdict(list)
        for step in self.super_chain['preprocessors']:
            self.steps_by_group[step['group']].append(step)

        # Extract fixed harness
        self.fixed_start, self.fixed_end = self._extract_fixed_harness()

    def _load_super_chain(self) -> Dict[str, Any]:
        """Load super-chain configuration."""
        with open(self.super_chain_path) as f:
            return yaml.safe_load(f)

    def _load_search_spaces(self) -> Dict[str, Any]:
        """Load all search space YAML files."""
        spaces = {}
        for yaml_file in self.search_spaces_dir.glob("*.yaml"):
            with open(yaml_file) as f:
                space = yaml.safe_load(f)
                template_name = space.get('template')
                if template_name:
                    spaces[template_name] = space
        return spaces

    def _extract_fixed_harness(self) -> Tuple[List[Dict], List[Dict]]:
        """Extract fixed steps from start and end of super-chain.

        Returns:
            (fixed_start, fixed_end): Lists of fixed step dicts
        """
        steps = self.super_chain['preprocessors']
        fixed_start = []
        fixed_end = []

        # Find first non-fixed
        idx = 0
        for i, step in enumerate(steps):
            if not step.get('meta', {}).get('fixed', False):
                idx = i
                break
            fixed_start.append(step)
        else:
            # All fixed
            return steps, []

        # Find last non-fixed
        for i in range(len(steps) - 1, idx - 1, -1):
            if not steps[i].get('meta', {}).get('fixed', False):
                break
            fixed_end.insert(0, steps[i])

        return fixed_start, fixed_end

    def generate_random_pipeline(
        self,
        min_len: int,
        max_len: int,
        allow_heavy_steps: bool = True,
        allow_heavy_variants: bool = True,
        local_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate a random preprocessing pipeline.

        Args:
            min_len: Minimum core length
            max_len: Maximum core length
            allow_heavy_steps: Whether to allow heavy steps
            allow_heavy_variants: Whether to allow heavy variants
            local_seed: Optional seed for this generation (for deterministic refinement)

        Returns:
            Pipeline dict with 'core', 'injected', 'signatures', 'full_chain'
        """
        # Use local RNG if seed provided
        rng = random.Random(local_seed) if local_seed is not None else self.rng

        # Sample core length
        core_len = rng.randint(min_len, max_len)

        # Get eligible groups (enabled, non-fixed)
        eligible_groups = []
        for step in self.super_chain['preprocessors']:
            meta = step.get('meta', {})
            if meta.get('fixed', False):
                continue
            if not step.get('enabled', False):
                continue
            if not allow_heavy_steps and meta.get('heavy_step', False):
                continue
            eligible_groups.append(step['group'])

        if len(eligible_groups) < core_len:
            # Not enough eligible groups
            core_len = len(eligible_groups)

        if core_len == 0:
            # No eligible groups - return empty core
            return self._build_pipeline_result([], [], local_seed or self.seed)

        # Sample unique groups
        core_groups = rng.sample(eligible_groups, core_len)

        # Generate core steps
        core = []
        for group in core_groups:
            step = self._generate_step(group, allow_heavy_variants, rng)
            if step:
                core.append(step)

        # Resolve dependencies recursively
        injected = self._resolve_dependencies(core, allow_heavy_variants, rng)

        # Build full pipeline
        return self._build_pipeline_result(core, injected, local_seed or self.seed)

    def _generate_step(
        self,
        group: str,
        allow_heavy_variants: bool,
        rng: random.Random
    ) -> Optional[Dict[str, Any]]:
        """Generate a single step with random variant and parameters.

        Args:
            group: Group name
            allow_heavy_variants: Whether to allow heavy variants
            rng: Random number generator

        Returns:
            Step dict or None
        """
        # Get super-chain step metadata
        steps_in_group = self.steps_by_group.get(group, [])
        if not steps_in_group:
            return None

        # Use first step in group (groups should have single step in super-chain)
        super_step = steps_in_group[0]
        template = super_step['template']

        # Get search space
        space = self.search_spaces.get(template)
        if not space:
            return None

        variants = space.get('variants', [])
        if not variants:
            return None

        # Filter by heavy_variants
        heavy_variants_list = super_step.get('meta', {}).get('heavy_variants', [])
        if not allow_heavy_variants and heavy_variants_list:
            variants = [v for v in variants if v['name'] not in heavy_variants_list]

        if not variants:
            return None

        # Select variant by weight
        variant = self._sample_variant_by_weight(variants, rng)

        # Sample parameters
        params = self.sampler.sample_variant(template, variant['name'], self.search_spaces, rng=rng)

        # Merge with base_config
        base_config = space.get('base_config', {})
        config = {**base_config, **params}

        return {
            'step': super_step['name'],
            'group': group,
            'template': template,
            'variant': variant['name'],
            'config': config,
            'requires_preproc': variant.get('requires_preproc', []),
            'original_index': self.super_chain['preprocessors'].index(super_step)
        }

    def _sample_variant_by_weight(self, variants: List[Dict], rng: random.Random) -> Dict:
        """Sample variant by weight (default 1.0)."""
        weights = [v.get('weight', 1.0) for v in variants]
        return rng.choices(variants, weights=weights)[0]

    def _resolve_dependencies(
        self,
        core: List[Dict[str, Any]],
        allow_heavy_variants: bool,
        rng: random.Random
    ) -> List[Dict[str, Any]]:
        """Recursively resolve dependencies for core steps.

        Args:
            core: Core steps
            allow_heavy_variants: Whether to allow heavy variants
            rng: Random number generator

        Returns:
            List of injected steps
        """
        injected = []
        seen_groups: Set[str] = set()

        # Add core groups to seen
        for step in core:
            seen_groups.add(step['group'])

        # Process core steps
        queue = deque(core[:])

        while queue:
            step = queue.popleft()
            requires = step.get('requires_preproc', [])

            for req in requires:
                req_group = req['group']
                req_step_type = req.get('step')  # 'before', 'after', or None

                if req_step_type == 'before':
                    # Prerequisite check: must already exist in seen or core
                    if req_group not in seen_groups:
                        # Reject this pipeline (prerequisite not met)
                        raise ValueError(f"Prerequisite not met: {req_group} required before {step['group']}")

                elif req_step_type == 'after':
                    # Auto-injection: inject deterministic step
                    if req_group not in seen_groups:
                        injected_step = self._inject_step(req, allow_heavy_variants, rng)
                        if injected_step:
                            injected.append(injected_step)
                            seen_groups.add(req_group)
                            # Recursively resolve injected step's dependencies
                            queue.append(injected_step)

                else:
                    # Default: require to exist (no injection)
                    if req_group not in seen_groups:
                        raise ValueError(f"Required group {req_group} not in pipeline for {step['group']}")

        return injected

    def _inject_step(
        self,
        req: Dict[str, Any],
        allow_heavy_variants: bool,
        rng: random.Random  # noqa: ARG002 - Not used for deterministic injection
    ) -> Optional[Dict[str, Any]]:
        """Inject a deterministic step based on requirement.

        Args:
            req: Requirement dict with group, variant, config/fixed_config
            allow_heavy_variants: Whether to allow heavy variants (not used, kept for API consistency)
            rng: Random number generator (not used for deterministic injection)

        Returns:
            Injected step dict or None
        """
        group = req['group']
        variant_name = req.get('variant')
        fixed_config = req.get('fixed_config', req.get('config', {}))

        # Get super-chain step
        steps_in_group = self.steps_by_group.get(group, [])
        if not steps_in_group:
            return None

        super_step = steps_in_group[0]
        template = super_step['template']

        # Get search space
        space = self.search_spaces.get(template)
        if not space:
            return None

        variants = space.get('variants', [])
        if not variants:
            return None

        # Select variant
        if variant_name:
            variant = next((v for v in variants if v['name'] == variant_name), None)
            if not variant:
                variant = variants[0]  # Fallback
        else:
            variant = variants[0]  # Default to first

        # Use fixed_config (no random sampling)
        base_config = space.get('base_config', {})
        config = {**base_config, **fixed_config}

        return {
            'step': super_step['name'],
            'group': group,
            'template': template,
            'variant': variant['name'],
            'config': config,
            'requires_preproc': variant.get('requires_preproc', []),
            'injected': True,
            'original_index': self.super_chain['preprocessors'].index(super_step)
        }

    def _build_pipeline_result(
        self,
        core: List[Dict[str, Any]],
        injected: List[Dict[str, Any]],
        seed: int
    ) -> Dict[str, Any]:
        """Build final pipeline result with topological sorting and signatures.

        Args:
            core: Core steps
            injected: Injected steps
            seed: Seed for random topological sort

        Returns:
            Pipeline dict with full_chain, signatures, etc.
        """
        # Detect cycles
        all_steps = core + injected
        if self._has_cycle(all_steps):
            raise ValueError("Circular dependency detected in pipeline")

        # Random topological sort
        sorted_steps = self._random_toposort(all_steps, seed)

        # Add fixed harness
        full_chain = self.fixed_start + sorted_steps + self.fixed_end

        # Compute signatures
        pipeline_sig = compute_signature(full_chain, include_params=True)
        arch_sig = compute_signature(full_chain, include_params=False)

        return {
            'core': core,
            'injected': injected,
            'full_chain': full_chain,
            'pipeline_signature': pipeline_sig,
            'architecture_signature': arch_sig,
            'seed': seed
        }

    def _has_cycle(self, steps: List[Dict[str, Any]]) -> bool:
        """Check for cycles in dependency graph using DFS.

        Args:
            steps: All steps (core + injected)

        Returns:
            True if cycle detected
        """
        # Build adjacency list
        graph = defaultdict(list)
        step_by_group = {s['group']: s for s in steps}

        for step in steps:
            for req in step.get('requires_preproc', []):
                req_group = req['group']
                if req_group in step_by_group:
                    # Edge: req_group -> step['group'] (req must come before step)
                    graph[req_group].append(step['group'])

        # DFS cycle detection
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {s['group']: WHITE for s in steps}

        def dfs(node):
            if color[node] == GRAY:
                return True  # Back edge = cycle
            if color[node] == BLACK:
                return False

            color[node] = GRAY
            for neighbor in graph[node]:
                if dfs(neighbor):
                    return True
            color[node] = BLACK
            return False

        for step in steps:
            if color[step['group']] == WHITE:
                if dfs(step['group']):
                    return True

        return False

    def _random_toposort(self, steps: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
        """Perform random topological sort respecting dependencies.

        Args:
            steps: All steps (core + injected)
            seed: Seed for random selection

        Returns:
            Sorted list of steps
        """
        rng = random.Random(seed)

        # Build adjacency list and indegree
        graph = defaultdict(list)
        indegree = defaultdict(int)
        step_by_group = {s['group']: s for s in steps}

        # Initialize indegree
        for step in steps:
            indegree[step['group']] = 0

        # Build graph
        for step in steps:
            for req in step.get('requires_preproc', []):
                req_group = req['group']
                if req_group in step_by_group and req.get('step') != 'before':
                    # Edge: req_group -> step['group']
                    graph[req_group].append(step['group'])
                    indegree[step['group']] += 1

        # Kahn's algorithm with random selection
        queue = [s['group'] for s in steps if indegree[s['group']] == 0]
        rng.shuffle(queue)

        result = []

        while queue:
            # Random select from nodes with indegree 0
            current = queue.pop(0)
            result.append(step_by_group[current])

            # Reduce indegree for neighbors
            for neighbor in graph[current]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)
                    rng.shuffle(queue)

        if len(result) != len(steps):
            # Cycle detected (should not happen if _has_cycle passed)
            raise ValueError("Topological sort failed - cycle detected")

        # Sort by original_index to maintain super-chain order where possible
        result.sort(key=lambda s: s['original_index'])

        return result


def compute_signature(obj: Any, include_params: bool = True) -> str:
    """Compute SHA256 signature of a pipeline.

    Args:
        obj: Pipeline object (dict or list)
        include_params: If True, include parameter values. If False, structure only.

    Returns:
        str: Hex digest of SHA256 hash
    """
    if not include_params:
        # Architecture signature: remove parameter values, keep structure
        if isinstance(obj, list):
            obj_filtered = []
            for item in obj:
                if isinstance(item, dict):
                    filtered_item = {}
                    for k in ('step', 'variant', 'group', 'template'):
                        if k in item:
                            filtered_item[k] = item[k]
                    obj_filtered.append(filtered_item)
            obj = obj_filtered
        elif isinstance(obj, dict):
            obj_filtered = {}
            for k in ('step', 'variant', 'group', 'template'):
                if k in obj:
                    obj_filtered[k] = obj[k]
            obj = obj_filtered

    # Canonical JSON: sorted keys, no whitespace
    canonical = json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()
