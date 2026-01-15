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

def test_best_score_logging(mock_context):
    config = MCTSConfig(study_name="test", budget=1)
    
    with patch("mlarena.modules.mcts.runner.load_mcts_config", return_value=config), \
         patch("mlarena.modules.mcts.runner.SuperChainActionSpace"), \
         patch("mlarena.modules.mcts.runner.MCTSStorage") as MockStorage, \
         patch("mlarena.modules.mcts.runner.MCTSTree"), \
         patch("mlarena.modules.mcts.runner.MlaCliExecutor"), \
         patch("mlarena.modules.mcts.runner.TemplateMaterializer"):
         
        mock_storage_inst = MockStorage.return_value
        mock_storage_inst.create_study.return_value = (1, True)
        
        # Ensure baseline check in _evaluate_baseline returns None
        mock_storage_inst._connect.return_value.__enter__.return_value.cursor.return_value.fetchone.return_value = None
        
        runner = MCTSRunner(mock_context, {"study_name": "test"})
        runner.config = config
        
        # Mock storage get_best_trial
        mock_storage_inst.get_best_trial.return_value = {"value": 0.99, "trial_id": 1}
        
        with patch.object(runner, "_evaluate_baseline"), \
             patch.object(runner.tree, "select") as mock_select, \
             patch.object(runner.tree, "expand") as mock_expand, \
             patch.object(runner.storage, "create_trial") as mock_create_trial, \
             patch.object(runner, "_execute_trial_with_templates") as mock_exec:
             
             mock_create_trial.return_value = (2, 2)  # (id, number)
             
             mock_child = MagicMock()
             mock_child.state.signature = "sig1"
             mock_child.state.depth = 1
             mock_child.state.steps = []  # Empty list is JSON serializable
             
             # KLUCZ: Zamknięcie łańcucha parentów
             mock_child.parent = runner.tree.root
             runner.tree.root.parent = None
             
             mock_select.return_value = runner.tree.root
             mock_expand.return_value = mock_child
             
             mock_exec.return_value = ExperimentResult(
                 experiment_id="exp1", value=0.85, metric="auc", duration=10.0, success=True, details={}
             )
             
             res = runner.run()
             assert res.payload["best_trial"]["value"] == 0.99
