import pytest
from unittest.mock import MagicMock, patch
from mlarena.modules.mcts.runner import MCTSRunner
from mlarena.modules.mcts.config import MCTSConfig

def test_best_score_logging():
    config = MCTSConfig(study_name="test", budget=1)
    runner = MCTSRunner(MagicMock(), {"study_name": "test"})
    runner.config = config
    
    # Mock storage
    with patch.object(runner.storage, "get_best_trial") as mock_best:
        mock_best.return_value = {"value": 0.99, "trial_id": 1}
        
        # We assume run() calls get_best_trial at end
        # We can't easily test "logging" without capturing stdout, but we can verify logic flow
        
        # Let's verify _check_new_best logic if we expose it
        # Or just verify result payload
        with patch.object(runner, "_evaluate_baseline"), \
             patch.object(runner.tree, "select"), \
             patch.object(runner.tree, "expand"), \
             patch.object(runner.storage, "create_trial"), \
             patch.object(runner, "_execute_trial"):
             
             # Mock expand to return same node to break loop quickly or just rely on budget=1
             runner.tree.select.return_value = runner.tree.root
             runner.tree.expand.return_value = runner.tree.root # Trigger terminal
             
             res = runner.run()
             assert res.payload["best_trial"]["value"] == 0.99
