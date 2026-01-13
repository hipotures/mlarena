import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from mlarena.core.module import ModuleContext
from mlarena.modules.preprocess_tune import PreprocessTuneModule

@pytest.fixture
def mock_context():
    ctx = MagicMock(spec=ModuleContext)
    ctx.project_root = Path("/tmp/test_project")
    ctx.project_name = "test_project"
    ctx.artifact_dir = Path("/tmp/test_project/artifacts")
    ctx.config_module = MagicMock()
    ctx.config = MagicMock()
    # Ensure model_dump returns a dict to avoid attribute errors if accessed
    ctx.config.model_dump.return_value = {}
    return ctx

def test_mcts_routing_enabled(mock_context):
    """Verify that MCTS runner is invoked when mcts=True is passed."""
    # Setup
    module = PreprocessTuneModule(mock_context)
    module.set_invocation_params({"mcts": True})
    
    # We patch the MCTSRunner class that is imported in preprocess_tune
    # We mock the return value of run()
    with patch("mlarena.modules.preprocess_tune.MCTSRunner") as MockRunner:
        mock_runner_instance = MockRunner.return_value
        mock_runner_instance.run.return_value = MagicMock(success=True, payload={"status": "mcts_stub_completed"})
        
        result = module.execute()
        
        # Assertions
        MockRunner.assert_called_once()
        mock_runner_instance.run.assert_called_once()
        assert result.success is True
        assert result.payload["status"] == "mcts_stub_completed"
