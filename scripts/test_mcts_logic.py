import json
import sys
from pathlib import Path

# Add src to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from mlarena.modules.mcts.config import MCTSConfig
from mlarena.modules.mcts.space import SuperChainActionSpace
from mlarena.modules.mcts.node import PipelineState
from mlarena.modules.mcts.tree import MCTSTree
from mlarena.modules.mcts.sampler import ParameterSampler

def test_group_exclusion_logic():
    super_chain_path = REPO_ROOT / "conf/preprocess/mla_super_chain.yaml"
    space = SuperChainActionSpace(super_chain_path)
    config = MCTSConfig()
    sampler = ParameterSampler()
    tree = MCTSTree(config, space, sampler)
    
    violations = 0
    total_simulations = 500
    
    print(f"Starting simulation of {total_simulations} MCTS expansions...")
    
    for i in range(total_simulations):
        node = tree.select(tree.root)
        child = tree.expand(node)
        
        if child == node: continue
        
        # Check for group duplicates in the current state
        groups = [step['group'] for step in child.state.steps]
        if len(groups) != len(set(groups)):
            print(f"VIOLATION FOUND in trial-sim {i}: {groups}")
            violations += 1
            
        # Give some dummy score to allow tree growth
        tree.backpropagate(child, 0.5)
        
    if violations == 0:
        print(f"SUCCESS: Simulated {total_simulations} trials. ZERO group violations found.")
    else:
        print(f"FAILED: Found {violations} group violations.")

if __name__ == "__main__":
    test_group_exclusion_logic()
