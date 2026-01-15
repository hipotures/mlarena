import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from mlarena.modules.mcts.space import SuperChainActionSpace

def test_load_super_chain(tmp_path):
    """Verify loading and parsing of super chain."""
    
    chain_yaml = """
    preprocessors:
      - name: imputer
        template: imputer
        group: imputation
      - name: scaler
        template: scaler
        group: scaling
    
    mcts:
      study_name: test
    """
    chain_path = tmp_path / "super_chain.yaml"
    chain_path.write_text(chain_yaml)
    
    # Mock _load_search_spaces to return empty or specific dict
    with patch("mlarena.modules.mcts.space._load_search_spaces", return_value={}):
        space = SuperChainActionSpace(chain_path)
        
        assert len(space.steps) == 2
        assert space.steps[0]["name"] == "imputer"
        assert space.steps[1]["name"] == "scaler"

def test_load_search_spaces_integration(tmp_path):
    """Verify integration with search space loading."""
    # We'll rely on the existing _load_search_spaces logic, but verify the space uses it.
    
    chain_yaml = """
    preprocessors:
      - name: imputer
        template: imputer
    """
    chain_path = tmp_path / "super_chain.yaml"
    chain_path.write_text(chain_yaml)
    
    dummy_spaces = {"imputer": {"variants": [{"name": "simple"}]}}
    
    with patch("mlarena.modules.mcts.space._load_search_spaces", return_value=dummy_spaces):
        space = SuperChainActionSpace(chain_path)
        assert space.search_spaces == dummy_spaces

def test_next_actions_requires_preproc(tmp_path):
    """Verify that variants with unmet requirements are filtered out."""
    chain_yaml = """
    preprocessors:
      - name: imputer
        template: imputer
        group: imputer
      - name: selector
        template: selector
        group: selector
    """
    chain_path = tmp_path / "super_chain.yaml"
    chain_path.write_text(chain_yaml)
    
    # Selector 'mi' requires 'imputer' group
    dummy_spaces = {
        "imputer": {"variants": [{"name": "simple"}]},
        "selector": {"variants": [
            {"name": "variance"}, # No requirements
            {"name": "mi", "requires_preproc": [{"group": "imputer"}]}
        ]}
    }
    
    with patch("mlarena.modules.mcts.space._load_search_spaces", return_value=dummy_spaces):
        space = SuperChainActionSpace(chain_path)
        
        # 1. State BEFORE any steps: should only see 'imputer' and 'selector:variance'
        from mlarena.modules.mcts.node import PipelineState
        state_empty = PipelineState()
        actions = space.next_actions(state_empty)
        
        vnames = [a.variant_name for a in actions if a.step_name == "selector"]
        assert "variance" in vnames
        assert "mi" not in vnames # Missing imputer!
        
        # 2. State WITH imputer: should see 'mi'
        state_with_imputer = PipelineState(used_groups={"imputer": "imputer"})
        # We need to manually set last_step_index to something that doesn't skip selector
        # index 0 is imputer, index 1 is selector. 
        # PipelineState defaults last_step_index to -1
        actions_after = space.next_actions(state_with_imputer)
        
        vnames_after = [a.variant_name for a in actions_after if a.step_name == "selector"]
        assert "mi" in vnames_after
        assert "variance" in vnames_after
