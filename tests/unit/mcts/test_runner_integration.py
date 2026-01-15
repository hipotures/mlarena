import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from mlarena.modules.mcts.runner import MCTSRunner
from mlarena.modules.mcts.config import MCTSConfig
from mlarena.core.module import ModuleContext
from mlarena.modules.mcts.executor import ExperimentResult

@pytest.fixture
def mock_context(tmp_path):
    ctx = MagicMock(spec=ModuleContext)
    ctx.project_root = tmp_path
    ctx.project_name = "test_proj"
    ctx.config = MagicMock()
    ctx.config.mcts_live = False
    ctx.config.telegram_test = False
    return ctx

@pytest.fixture
def config():
    return MCTSConfig(study_name="test_run", budget=1, seed=42)

def test_runner_loop(mock_context, config):
    """Verify the main MCTS loop executes budget iterations."""
    
    # Create dummy chain file
    chain_path = mock_context.project_root / "chain.yaml"
    chain_path.write_text("dummy: content")
    
    # Mock components
    with patch("mlarena.modules.mcts.runner.load_mcts_config", return_value=config), \
         patch("mlarena.modules.mcts.runner.SuperChainActionSpace") as MockSpace, \
         patch("mlarena.modules.mcts.runner.MCTSStorage") as MockStorage, \
         patch("mlarena.modules.mcts.runner.MlaCliExecutor") as MockExecutor, \
         patch("mlarena.modules.mcts.runner.TemplateMaterializer") as MockMaterializer:
         
        # Setup mocks
        mock_space_inst = MockSpace.return_value
        mock_space_inst.fixed_steps = []
        
        mock_storage_inst = MockStorage.return_value
        mock_storage_inst.create_study.return_value = (1, True)
        mock_storage_inst.create_trial.return_value = (2, 2)
        mock_storage_inst.get_best_trial.return_value = None
        
        # Ensure baseline check in _evaluate_baseline returns None
        mock_storage_inst._connect.return_value.__enter__.return_value.cursor.return_value.fetchone.return_value = None
        
        mock_exec_inst = MockExecutor.return_value
        mock_exec_inst.run.return_value = ExperimentResult(
            experiment_id="exp1", value=0.8, metric="auc", duration=1.0, success=True, details={}
        )
        
        mock_mat_inst = MockMaterializer.return_value
        mock_mat_inst.materialize.return_value = {
            "base_name": "base", 
            "chain_path": chain_path,
            "step_paths": []
        }
        
        runner = MCTSRunner(mock_context, {"study_name": "test_run"})
        
        # KLUCZ: Zamknięcie łańcucha parentów
        runner.tree.root.parent = None
        
        with patch.object(runner.tree, "select") as mock_select, \
             patch.object(runner.tree, "expand") as mock_expand:
             
             mock_child = MagicMock()
             mock_child.state.signature = "child_sig"
             mock_child.state.depth = 1
             mock_child.state.steps = []
             mock_child.parent = runner.tree.root
             
             mock_select.return_value = runner.tree.root
             mock_expand.return_value = mock_child
             
             result = runner.run()
             
             assert result.success is True
             assert mock_storage_inst.create_study.called
             assert mock_exec_inst.run.called
