import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from mlarena.modules.mcts.runner import MCTSRunner
from mlarena.modules.mcts.config import MCTSConfig
from mlarena.core.module import ModuleContext
from mlarena.modules.mcts.executor import ExperimentResult

@pytest.fixture
def mock_context():
    ctx = MagicMock(spec=ModuleContext)
    ctx.project_root = Path("/tmp/proj")
    ctx.project_name = "test_proj"
    return ctx

@pytest.fixture
def config():
    return MCTSConfig(study_name="test_run", budget=5, seed=42)

def test_runner_loop(mock_context, config):
    """Verify the main MCTS loop executes budget iterations."""
    
    # Mock components
    with patch("mlarena.modules.mcts.runner.load_mcts_config", return_value=config), \
         patch("mlarena.modules.mcts.runner.SuperChainActionSpace") as MockSpace, \
         patch("mlarena.modules.mcts.runner.MCTSStorage") as MockStorage, \
         patch("mlarena.modules.mcts.runner.MlaCliExecutor") as MockExecutor, \
         patch("mlarena.modules.mcts.runner.TemplateMaterializer") as MockMaterializer:
         
        # Setup mocks
        mock_space_inst = MockSpace.return_value
        mock_space_inst.next_actions.return_value = [] # Prevent infinite expansion in test
        
        mock_storage_inst = MockStorage.return_value
        mock_storage_inst.create_study.return_value = 1
        mock_storage_inst.create_trial.side_effect = range(1, 100) # trial IDs
        
        mock_exec_inst = MockExecutor.return_value
        mock_exec_inst.run.return_value = ExperimentResult(
            experiment_id="exp1", value=0.8, metric="auc", duration=1.0, success=True, details={}
        )
        
        mock_mat_inst = MockMaterializer.return_value
        mock_mat_inst.materialize.return_value = {
            "base_name": "base", 
            "chain_path": Path("chain.yaml"), # Return Path object
            "step_paths": []
        }
        
        runner = MCTSRunner(mock_context, {"study_name": "test_run"})
        result = runner.run()
        
        assert result.success is True
        # Verify interactions
        assert mock_storage_inst.create_study.called
        assert mock_exec_inst.run.called