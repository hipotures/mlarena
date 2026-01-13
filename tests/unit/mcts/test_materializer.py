import pytest
from pathlib import Path
import yaml
from mlarena.modules.mcts.materializer import TemplateMaterializer
from mlarena.modules.mcts.node import PipelineState

def test_materialize_state(tmp_path):
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    
    materializer = TemplateMaterializer(
        run_id="run1",
        project_root=tmp_path
    )
    
    state = PipelineState(steps=[
        {"name": "step1", "template": "t1", "variant": "v1", "config": {"a": 1}}
    ])
    
    info = materializer.materialize(state, node_id=123)
    
    # Check return info
    assert info["base_name"].startswith("run1_n000123")
    assert info["chain_path"].exists()
    
    # Check chain YAML content
    chain_data = yaml.safe_load(info["chain_path"].read_text())
    assert len(chain_data["chain"]) == 1
    step_file = chain_data["chain"][0]
    
    # Check step YAML content
    # The chain list usually contains filenames without .yaml extension or relative paths
    # Assuming materializer returns full paths or names resolvable by MLA
    
    # Verify step file exists
    step_path = templates_dir / "preprocess" / f"{step_file}.yaml"
    assert step_path.exists()
    
    step_data = yaml.safe_load(step_path.read_text())
    assert step_data["module"] == "t1"
    assert step_data["config"]["a"] == 1
