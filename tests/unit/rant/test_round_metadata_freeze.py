"""Unit tests for RANT round metadata freezing."""

import tempfile
from pathlib import Path

import pytest

from mlarena.modules.rant.storage import RANTStorage, StudyDirection


@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_rant.db"
        storage_url = f"sqlite:///{db_path}"
        yield storage_url


@pytest.fixture
def storage(temp_db):
    """Create storage instance."""
    return RANTStorage(temp_db)


@pytest.fixture
def study_id(storage):
    """Create test study."""
    study_id, _ = storage.create_study("test_study", StudyDirection.MAXIMIZE)
    return study_id


def test_round_metadata_freeze_on_create(storage, study_id):
    """Test that round metadata is frozen when round is created."""
    # Create round with initial config
    round_num, metadata = storage.resolve_active_round(
        study_id=study_id,
        n_random=50,
        top_k=10,
        children_per_parent=5,
        min_core_len=3,
        max_core_len=9,
        refinement_config={'grid_size': 20, 'log_grid_size': 20},
        seed=42
    )

    assert round_num == 1
    assert metadata['n_random'] == 50
    assert metadata['top_k'] == 10
    assert metadata['children_per_parent'] == 5
    assert metadata['min_core_len'] == 3
    assert metadata['max_core_len'] == 9
    assert metadata['seed'] == 42


def test_round_metadata_immutable_on_resume(storage, study_id):
    """Test that round metadata is immune to CLI changes on resume."""
    # Create round with initial config
    round_num1, metadata1 = storage.resolve_active_round(
        study_id=study_id,
        n_random=50,
        top_k=10,
        children_per_parent=5,
        min_core_len=3,
        max_core_len=9,
        refinement_config={'grid_size': 20, 'log_grid_size': 20},
        seed=42
    )

    assert round_num1 == 1
    assert metadata1['n_random'] == 50

    # Resume with DIFFERENT CLI config - should return frozen metadata
    round_num2, metadata2 = storage.resolve_active_round(
        study_id=study_id,
        n_random=100,  # Different!
        top_k=20,       # Different!
        children_per_parent=10,  # Different!
        min_core_len=5,  # Different!
        max_core_len=15,  # Different!
        refinement_config={'grid_size': 50, 'log_grid_size': 50},  # Different!
        seed=999  # Different!
    )

    # Should return same round
    assert round_num2 == round_num1

    # Should return FROZEN metadata (original values, not CLI values)
    assert metadata2['n_random'] == 50, "Should return frozen N, not CLI value"
    assert metadata2['top_k'] == 10, "Should return frozen K, not CLI value"
    assert metadata2['children_per_parent'] == 5, "Should return frozen P, not CLI value"
    assert metadata2['min_core_len'] == 3
    assert metadata2['max_core_len'] == 9
    assert metadata2['refinement']['grid_size'] == 20
    assert metadata2['seed'] == 42


def test_new_round_uses_current_config(storage, study_id):
    """Test that new round (after complete) uses current CLI config."""
    # Create and complete first round
    round_num1, metadata1 = storage.resolve_active_round(
        study_id=study_id,
        n_random=50,
        top_k=10,
        children_per_parent=5,
        min_core_len=3,
        max_core_len=9,
        refinement_config={'grid_size': 20, 'log_grid_size': 20},
        seed=42
    )

    # Mark round complete
    storage.mark_round_complete(study_id, round_num1)

    # Create new round with different config
    round_num2, metadata2 = storage.resolve_active_round(
        study_id=study_id,
        n_random=100,  # Different
        top_k=20,       # Different
        children_per_parent=10,  # Different
        min_core_len=5,
        max_core_len=15,
        refinement_config={'grid_size': 50, 'log_grid_size': 50},
        seed=999
    )

    # Should be new round
    assert round_num2 == 2

    # Should use NEW config (not frozen from round 1)
    assert metadata2['n_random'] == 100
    assert metadata2['top_k'] == 20
    assert metadata2['children_per_parent'] == 10
    assert metadata2['min_core_len'] == 5
    assert metadata2['max_core_len'] == 15
    assert metadata2['refinement']['grid_size'] == 50
    assert metadata2['seed'] == 999


def test_round_budget_calculation(storage, study_id):
    """Test that round budget is correctly calculated from frozen metadata."""
    # Create round
    round_num, metadata = storage.resolve_active_round(
        study_id=study_id,
        n_random=10,
        top_k=3,
        children_per_parent=2,
        min_core_len=3,
        max_core_len=5,
        refinement_config={'grid_size': 20},
        seed=42
    )

    # Expected budget: N + K*P = 10 + 3*2 = 16 trials
    expected_budget = metadata['n_random'] + metadata['top_k'] * metadata['children_per_parent']
    assert expected_budget == 16


def test_multiple_studies_independent_rounds(storage):
    """Test that different studies have independent round numbering."""
    # Create two studies
    study_id1, _ = storage.create_study("study1", StudyDirection.MAXIMIZE)
    study_id2, _ = storage.create_study("study2", StudyDirection.MAXIMIZE)

    # Create round 1 for study 1
    round1_s1, _ = storage.resolve_active_round(
        study_id=study_id1,
        n_random=50,
        top_k=10,
        children_per_parent=5,
        min_core_len=3,
        max_core_len=9,
        refinement_config={'grid_size': 20},
        seed=42
    )

    # Create round 1 for study 2
    round1_s2, _ = storage.resolve_active_round(
        study_id=study_id2,
        n_random=100,
        top_k=20,
        children_per_parent=10,
        min_core_len=5,
        max_core_len=15,
        refinement_config={'grid_size': 30},
        seed=999
    )

    # Both should be round 1
    assert round1_s1 == 1
    assert round1_s2 == 1

    # Complete both and create round 2
    storage.mark_round_complete(study_id1, round1_s1)
    storage.mark_round_complete(study_id2, round1_s2)

    round2_s1, _ = storage.resolve_active_round(
        study_id=study_id1,
        n_random=50,
        top_k=10,
        children_per_parent=5,
        min_core_len=3,
        max_core_len=9,
        refinement_config={'grid_size': 20},
        seed=42
    )

    round2_s2, _ = storage.resolve_active_round(
        study_id=study_id2,
        n_random=100,
        top_k=20,
        children_per_parent=10,
        min_core_len=5,
        max_core_len=15,
        refinement_config={'grid_size': 30},
        seed=999
    )

    # Both should be round 2
    assert round2_s1 == 2
    assert round2_s2 == 2


def test_refinement_config_serialization(storage, study_id):
    """Test that refinement config is correctly serialized/deserialized."""
    complex_refinement = {
        'grid_size': 20,
        'log_grid_size': 30,
        'allow_two_param_mutations': True,
        'nested': {
            'key1': 'value1',
            'key2': 42
        }
    }

    round_num, metadata = storage.resolve_active_round(
        study_id=study_id,
        n_random=50,
        top_k=10,
        children_per_parent=5,
        min_core_len=3,
        max_core_len=9,
        refinement_config=complex_refinement,
        seed=42
    )

    # Verify complex config is preserved
    assert metadata['refinement'] == complex_refinement
    assert metadata['refinement']['nested']['key1'] == 'value1'
    assert metadata['refinement']['nested']['key2'] == 42
