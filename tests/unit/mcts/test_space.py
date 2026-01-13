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
