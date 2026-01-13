import pytest
from pathlib import Path
from mlarena.modules.mcts.config import MCTSConfig, load_mcts_config

def test_config_defaults():
    """Verify default values are applied."""
    config = MCTSConfig(study_name="test_study")
    assert config.direction == "maximize" # Assuming default
    assert config.budget == 80
    assert config.selection_policy == "puct"
    assert config.exploration_weight == 1.414

def test_config_validation_valid():
    """Verify valid config parsing."""
    data = {
        "study_name": "my_study",
        "direction": "minimize",
        "budget": 100,
        "max_depth": 5,
        "selection_policy": "uct",
        "expansion_width": 3,
        "expansion_alpha": 0.6
    }
    config = MCTSConfig(**data)
    assert config.study_name == "my_study"
    assert config.direction == "minimize"
    assert config.expansion_width == 3

def test_config_validation_invalid():
    """Verify validation errors."""
    with pytest.raises(ValueError):
        MCTSConfig(study_name="test", direction="invalid_direction")
    
    with pytest.raises(ValueError):
        MCTSConfig(study_name="test", selection_policy="invalid_policy")

def test_load_config_from_yaml(tmp_path):
    """Verify loading from mla_super_chain.yaml structure."""
    yaml_content = """
    mcts:
      study_name: "yaml_study"
      budget: 50
    """
    path = tmp_path / "mla_super_chain.yaml"
    path.write_text(yaml_content)
    
    config = load_mcts_config(path)
    assert config.study_name == "yaml_study"
    assert config.budget == 50
