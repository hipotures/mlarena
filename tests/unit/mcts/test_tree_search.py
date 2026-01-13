import pytest
from unittest.mock import MagicMock
from mlarena.modules.mcts.tree import MCTSTree, MCTSNode
from mlarena.modules.mcts.node import PipelineState
from mlarena.modules.mcts.config import MCTSConfig

@pytest.fixture
def mock_space():
    space = MagicMock()
    # next_actions returns mock actions
    space.next_actions.return_value = []
    space.total_steps_count = 5
    return space

@pytest.fixture
def mock_sampler():
    sampler = MagicMock()
    sampler.sample.return_value = 1
    return sampler

@pytest.fixture
def config():
    return MCTSConfig(study_name="test", budget=100)

def test_node_stats_update():
    """Verify backprop updates values correctly."""
    node = MCTSNode(state=PipelineState())
    node.update(value=0.5)
    assert node.n_visits == 1
    assert node.value_sum == 0.5
    assert node.value_mean == 0.5
    
    node.update(value=0.8)
    assert node.n_visits == 2
    assert node.value_sum == 1.3
    assert node.value_mean == 0.65

def test_tree_expansion_widening(config, mock_space, mock_sampler):
    """Verify progressive widening limits expansion in select()."""
    config.expansion_width = 2
    config.expansion_alpha = 0.5
    # formula: m(n) = k * N^alpha
    # N=1 -> m = 2 * 1^0.5 = 2 children
    
    tree = MCTSTree(config, mock_space, mock_sampler)
    root = tree.root
    root.n_visits = 1
    
    # Mock next_actions to return many possibilities
    actions = [MagicMock(step_name=f"s{i}") for i in range(10)]
    mock_space.next_actions.return_value = actions
    
    # 1. Select should return root (to expand), because children (0) < limit (2)
    selected1 = tree.select(root)
    assert selected1 == root
    # Simulate expansion
    child1 = tree.expand(selected1)
    assert len(root.children) == 1
    
    # 2. Select should return root again, because children (1) < limit (2)
    selected2 = tree.select(root)
    assert selected2 == root
    child2 = tree.expand(selected2)
    assert len(root.children) == 2
    
    # 3. Select should NOT return root, because children (2) >= limit (2)
    # It should descend to a child.
    selected_final = tree.select(root)
    assert selected_final != root
    assert selected_final in root.children

def test_selection_policy(config, mock_space, mock_sampler):
    """Verify selection picks best child."""
    tree = MCTSTree(config, mock_space, mock_sampler)
    root = tree.root
    
    child1 = MCTSNode(state=PipelineState(), parent=root)
    child1.n_visits = 10
    child1.value_sum = 10.0 # mean 1.0 (Best)
    
    child2 = MCTSNode(state=PipelineState(), parent=root)
    child2.n_visits = 10
    child2.value_sum = 0.0 # mean 0.0 (Worst)
    
    root.children = [child1, child2]
    root.n_visits = 20
    
    # UCT/PUCT should pick child1 (high exploitation)
    selected = tree._best_child(root)
    assert selected == child1