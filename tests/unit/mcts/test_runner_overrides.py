import pytest
from unittest.mock import MagicMock
from mlarena.modules.mcts.runner import MCTSRunner
from mlarena.modules.mcts.config import MCTSConfig
from mlarena.core.module import ModuleContext

def test_runner_config_overrides():
    # Mock context
    ctx = MagicMock(spec=ModuleContext)
    ctx.project_root = MagicMock()
    ctx.config = MagicMock()
    ctx.config.mcts_live = False
    
    # Mock load_mcts_config to return a default config
    default_config = MCTSConfig()
    
    # We need to patch load_mcts_config, MCTSStorage, etc. to instantiate Runner
    from unittest.mock import patch
    
    with patch("mlarena.modules.mcts.runner.load_mcts_config", return_value=default_config), \
         patch("mlarena.modules.mcts.runner.MCTSStorage"), \
         patch("mlarena.modules.mcts.runner.SuperChainActionSpace"), \
         patch("mlarena.modules.mcts.runner.ParameterSampler"), \
         patch("mlarena.modules.mcts.runner.MCTSTree"):
         
        # Case 1: Short args
        params = {"budget": "50", "study_name": "test_study"}
        runner = MCTSRunner(ctx, params)
        assert runner.config.budget == 50
        assert runner.config.study_name == "test_study"
        
        # Case 2: Dotted mcts.* args (should override short args if both present? 
        # In implementation: mcts.* are applied first, then shortcuts. 
        # Wait, implementation says:
        # 1. Explicit short args (budget, seed) -> set config.
        # 2. Dotted args -> set config.
        # So dotted args come AFTER in the loop (iterating params), wait.
        # Loop 1: explicit checks. Loop 2: iterate params.
        # Let's check my implementation order.
        
        # My implementation:
        # 1. if "budget" in params -> set.
        # 2. for key, value in params -> if startswith mcts. -> set.
        # So mcts.budget will override budget.
        
        params = {"budget": "50", "mcts.budget": "100"}
        runner = MCTSRunner(ctx, params)
        assert runner.config.budget == 100
        
        # Case 3: Nested dotted args
        params = {"mcts.multi_fidelity.enable": "false"}
        runner = MCTSRunner(ctx, params)
        assert runner.config.multi_fidelity.enable is False
        
        # Case 4: Bool conversion
        params = {"mcts.pruning.enable": "true"}
        runner = MCTSRunner(ctx, params)
        assert runner.config.pruning.enable is True
