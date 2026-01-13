import pytest
from unittest.mock import MagicMock
from mlarena.modules.mcts.node import PipelineState
from mlarena.modules.mcts.space import SuperChainActionSpace

def test_pipeline_signature_stability():
    """Verify signature is stable and order-dependent."""
    state1 = PipelineState(steps=[
        {"name": "imputer", "variant": "simple", "config": {"strategy": "mean"}}
    ])
    state2 = PipelineState(steps=[
        {"name": "imputer", "variant": "simple", "config": {"strategy": "mean"}}
    ])
    state3 = PipelineState(steps=[
        {"name": "imputer", "variant": "simple", "config": {"strategy": "median"}}
    ])
    
    assert state1.signature == state2.signature
    assert state1.signature != state3.signature

def test_next_actions_ordering():
    """Verify actions respect super chain order."""
    # Mock space with specific steps
    space = MagicMock(spec=SuperChainActionSpace)
    space.steps = [
        {"name": "s1", "group": "g1"},
        {"name": "s2", "group": "g2"},
        {"name": "s3", "group": "g3"},
    ]
    space.search_spaces = {
        "s1": {"variants": [{"name": "v1"}]},
        "s2": {"variants": [{"name": "v2"}]},
        "s3": {"variants": [{"name": "v3"}]},
    }
    
    real_space = SuperChainActionSpace.__new__(SuperChainActionSpace)
    real_space.steps = space.steps
    real_space.search_spaces = space.search_spaces
    real_space.super_chain_config = {}
    
    # Empty state -> can pick s1, s2, s3
    state0 = PipelineState(steps=[])
    actions0 = real_space.next_actions(state0)
    assert len(actions0) > 0
    names0 = [a.step_name for a in actions0]
    assert "s1" in names0
    assert "s2" in names0
    
    # State with s2 (index 1) -> can only pick s3
    state1 = PipelineState(
        steps=[{"name": "s2", "variant": "v2", "config": {}}],
        last_step_index=1 
    )
    
    actions1 = real_space.next_actions(state1)
    names1 = [a.step_name for a in actions1]
    assert "s1" not in names1
    assert "s2" not in names1 # Group g2 already used
    assert "s3" in names1

def test_next_actions_group_constraint():
    """Verify group uniqueness."""
    real_space = SuperChainActionSpace.__new__(SuperChainActionSpace)
    real_space.steps = [
        {"name": "s1a", "group": "g1"},
        {"name": "s1b", "group": "g1"}, # Same group, later
        {"name": "s2", "group": "g2"},
    ]
    real_space.search_spaces = {
        "s1a": {"variants": [{"name": "v"}]},
        "s1b": {"variants": [{"name": "v"}]},
        "s2": {"variants": [{"name": "v"}]},
    }
    real_space.super_chain_config = {}

    state = PipelineState(
        steps=[{"name": "s1a", "group": "g1", "variant": "v", "config": {}}],
        last_step_index=0
    )
    
    actions = real_space.next_actions(state)
    names = [a.step_name for a in actions]
    
    assert "s1b" not in names # Same group g1 used
    assert "s2" in names