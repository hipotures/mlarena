"""Unit tests for RANT pipeline generation."""

import tempfile
from pathlib import Path

import pytest
import yaml

from mlarena.modules.rant.space import PipelineGenerator, compute_signature


@pytest.fixture
def temp_super_chain():
    """Create temporary super-chain for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        super_chain_path = Path(tmpdir) / "super_chain.yaml"
        search_spaces_dir = Path(tmpdir) / "search_spaces"
        search_spaces_dir.mkdir()

        # Create minimal super-chain
        super_chain = {
            "preprocessors": [
                {
                    "name": "fixed_start",
                    "group": "fixed_start",
                    "template": "fixed_start",
                    "enabled": True,
                    "meta": {"fixed": True}
                },
                {
                    "name": "imputer",
                    "group": "imputer",
                    "template": "imputer",
                    "enabled": True,
                    "meta": {"heavy_step": False}
                },
                {
                    "name": "scaler",
                    "group": "scaler",
                    "template": "scaler",
                    "enabled": True,
                    "meta": {"heavy_step": False}
                },
                {
                    "name": "encoder",
                    "group": "encoder",
                    "template": "encoder",
                    "enabled": True,
                    "meta": {"heavy_step": False}
                },
                {
                    "name": "polynomial",
                    "group": "polynomial",
                    "template": "polynomial",
                    "enabled": True,
                    "meta": {"heavy_step": True}
                },
                {
                    "name": "fixed_end",
                    "group": "fixed_end",
                    "template": "fixed_end",
                    "enabled": True,
                    "meta": {"fixed": True}
                }
            ]
        }
        super_chain_path.write_text(yaml.safe_dump(super_chain))

        # Create search spaces
        spaces = {
            "imputer": {
                "template": "imputer",
                "base_config": {"add_indicator": False},
                "variants": [
                    {
                        "name": "simple",
                        "weight": 2.0,
                        "params": {
                            "strategy": {"type": "choice", "values": ["mean", "median"]}
                        },
                        "requires_preproc": []
                    }
                ]
            },
            "scaler": {
                "template": "scaler",
                "base_config": {},
                "variants": [
                    {
                        "name": "standard",
                        "params": {
                            "with_mean": {"type": "bool"}
                        },
                        "requires_preproc": [{"group": "imputer"}]
                    }
                ]
            },
            "encoder": {
                "template": "encoder",
                "base_config": {},
                "variants": [
                    {
                        "name": "target",
                        "params": {
                            "smoothing": {"type": "float_range", "min": 0.0, "max": 1.0}
                        },
                        "requires_preproc": []
                    }
                ]
            },
            "polynomial": {
                "template": "polynomial",
                "base_config": {},
                "variants": [
                    {
                        "name": "degree_2",
                        "params": {
                            "degree": {"type": "fixed", "value": 2}
                        },
                        "requires_preproc": [
                            {"group": "imputer"},
                            {"group": "encoder", "step": "after"}
                        ]
                    }
                ]
            }
        }

        for name, space in spaces.items():
            space_path = search_spaces_dir / f"{name}.yaml"
            space_path.write_text(yaml.safe_dump(space))

        yield super_chain_path, search_spaces_dir


def test_generator_initialization(temp_super_chain):
    """Test generator initializes correctly."""
    super_chain_path, search_spaces_dir = temp_super_chain
    gen = PipelineGenerator(super_chain_path, search_spaces_dir, seed=42)

    assert len(gen.super_chain['preprocessors']) == 6
    assert len(gen.search_spaces) == 4
    assert len(gen.fixed_start) == 1
    assert len(gen.fixed_end) == 1


def test_generate_random_pipeline_basic(temp_super_chain):
    """Test basic random pipeline generation."""
    super_chain_path, search_spaces_dir = temp_super_chain
    gen = PipelineGenerator(super_chain_path, search_spaces_dir, seed=42)

    pipeline = gen.generate_random_pipeline(
        min_len=2,
        max_len=3,
        allow_heavy_steps=True,
        allow_heavy_variants=True
    )

    assert 'core' in pipeline
    assert 'injected' in pipeline
    assert 'full_chain' in pipeline
    assert 'pipeline_signature' in pipeline
    assert 'architecture_signature' in pipeline

    # Core should be 2-3 steps
    assert 2 <= len(pipeline['core']) <= 3

    # Full chain includes fixed + core + injected
    assert len(pipeline['full_chain']) >= len(pipeline['core']) + 2  # At least fixed start/end


def test_generate_pipeline_respects_length(temp_super_chain):
    """Test that generated pipeline respects min/max length."""
    super_chain_path, search_spaces_dir = temp_super_chain
    gen = PipelineGenerator(super_chain_path, search_spaces_dir, seed=42)

    for _ in range(10):
        pipeline = gen.generate_random_pipeline(
            min_len=2,
            max_len=2,
            allow_heavy_steps=False  # Exclude heavy steps
        )
        # With heavy_steps=False, polynomial is excluded, so we have 3 eligible groups
        # Core should be exactly 2 (if enough eligible groups)
        assert len(pipeline['core']) == 2


def test_dependency_injection_after(temp_super_chain):
    """Test automatic injection of 'after' dependencies."""
    super_chain_path, search_spaces_dir = temp_super_chain
    gen = PipelineGenerator(super_chain_path, search_spaces_dir, seed=42)

    # Generate pipeline with polynomial (requires encoder with step=after)
    # Force polynomial by retrying
    for _ in range(50):
        pipeline = gen.generate_random_pipeline(
            min_len=1,
            max_len=1,
            allow_heavy_steps=True
        )

        # Check if polynomial was generated
        has_poly = any(s['group'] == 'polynomial' for s in pipeline['core'])
        if has_poly:
            # Should have encoder injected (step=after means injected)
            has_encoder = any(s['group'] == 'encoder' for s in pipeline['full_chain'])
            assert has_encoder, "Encoder should be auto-injected for polynomial"

            # Encoder should be in injected list
            encoder_injected = any(s['group'] == 'encoder' for s in pipeline['injected'])
            assert encoder_injected, "Encoder should appear in injected list"
            break


def test_dependency_resolution_prerequisite(temp_super_chain):
    """Test prerequisite checking (step=before)."""
    super_chain_path, search_spaces_dir = temp_super_chain
    gen = PipelineGenerator(super_chain_path, search_spaces_dir, seed=42)

    # Scaler requires imputer (no step= means prerequisite)
    for _ in range(50):
        pipeline = gen.generate_random_pipeline(
            min_len=2,
            max_len=3,
            allow_heavy_steps=False
        )

        has_scaler = any(s['group'] == 'scaler' for s in pipeline['core'])
        if has_scaler:
            # Must have imputer in core or injected
            has_imputer = any(
                s['group'] == 'imputer'
                for s in pipeline['core'] + pipeline['injected']
            )
            assert has_imputer, "Scaler requires imputer as prerequisite"
            break


def test_signature_deterministic(temp_super_chain):
    """Test that signature is deterministic for same pipeline."""
    super_chain_path, search_spaces_dir = temp_super_chain
    gen = PipelineGenerator(super_chain_path, search_spaces_dir, seed=42)

    pipeline1 = gen.generate_random_pipeline(min_len=2, max_len=2, local_seed=100)
    pipeline2 = gen.generate_random_pipeline(min_len=2, max_len=2, local_seed=100)

    # Same seed should produce same pipeline
    assert pipeline1['pipeline_signature'] == pipeline2['pipeline_signature']
    assert pipeline1['architecture_signature'] == pipeline2['architecture_signature']


def test_signature_different_params(temp_super_chain):
    """Test that pipeline signature differs with different params."""
    pipeline1 = [
        {"step": "imputer", "config": {"strategy": "mean"}}
    ]
    pipeline2 = [
        {"step": "imputer", "config": {"strategy": "median"}}
    ]

    sig1 = compute_signature(pipeline1, include_params=True)
    sig2 = compute_signature(pipeline2, include_params=True)

    assert sig1 != sig2, "Different params should produce different signatures"


def test_architecture_signature_same_structure(temp_super_chain):
    """Test that architecture signature is same for same structure."""
    pipeline1 = [
        {"step": "imputer", "variant": "simple", "config": {"strategy": "mean"}}
    ]
    pipeline2 = [
        {"step": "imputer", "variant": "simple", "config": {"strategy": "median"}}
    ]

    arch_sig1 = compute_signature(pipeline1, include_params=False)
    arch_sig2 = compute_signature(pipeline2, include_params=False)

    assert arch_sig1 == arch_sig2, "Same structure should produce same arch signature"


def test_topological_sort_respects_dependencies(temp_super_chain):
    """Test that topological sort places dependencies before dependents."""
    super_chain_path, search_spaces_dir = temp_super_chain
    gen = PipelineGenerator(super_chain_path, search_spaces_dir, seed=42)

    for _ in range(20):
        pipeline = gen.generate_random_pipeline(
            min_len=2,
            max_len=3,
            allow_heavy_steps=False
        )

        # Find positions in full_chain
        positions = {}
        for idx, step in enumerate(pipeline['full_chain']):
            positions[step['group']] = idx

        # If scaler exists, imputer must come before it
        if 'scaler' in positions and 'imputer' in positions:
            assert positions['imputer'] < positions['scaler'], \
                "Imputer must come before scaler (dependency)"


def test_random_toposort_varies(temp_super_chain):
    """Test that random topological sort produces different orders."""
    super_chain_path, search_spaces_dir = temp_super_chain
    gen = PipelineGenerator(super_chain_path, search_spaces_dir, seed=42)

    # Generate multiple pipelines with same core but different seeds
    orders = []
    for seed in range(10):
        pipeline = gen.generate_random_pipeline(
            min_len=3,
            max_len=3,
            allow_heavy_steps=False,
            local_seed=seed
        )

        # Extract order of non-fixed steps
        order = tuple(
            s['group']
            for s in pipeline['full_chain']
            if not s.get('meta', {}).get('fixed')
        )
        orders.append(order)

    # Should have some variety (not all same)
    unique_orders = len(set(orders))
    assert unique_orders > 1, "Random toposort should produce varied orders"


def test_empty_neighborhood_handling(temp_super_chain):
    """Test graceful handling when no eligible groups."""
    super_chain_path, search_spaces_dir = temp_super_chain

    # Create generator with all steps disabled
    gen = PipelineGenerator(super_chain_path, search_spaces_dir, seed=42)

    # Override enabled flags
    for step in gen.super_chain['preprocessors']:
        if not step.get('meta', {}).get('fixed'):
            step['enabled'] = False

    pipeline = gen.generate_random_pipeline(
        min_len=2,
        max_len=3,
        allow_heavy_steps=True
    )

    # Should return empty core
    assert len(pipeline['core']) == 0

    # But should still have fixed harness
    assert len(pipeline['full_chain']) == 2  # fixed_start + fixed_end
