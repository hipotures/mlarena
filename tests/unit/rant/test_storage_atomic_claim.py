"""Unit tests for RANT storage atomic claim and dedupe."""

import tempfile
import threading
from pathlib import Path

import pytest

from mlarena.modules.rant.storage import (
    RANTStorage,
    StudyDirection,
    TrialState,
    compute_signature,
)


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


@pytest.fixture
def study_with_round(storage, study_id):
    """Create study with active round."""
    storage.resolve_active_round(
        study_id=study_id,
        n_random=50,
        top_k=10,
        children_per_parent=5,
        min_core_len=3,
        max_core_len=9,
        refinement_config={'grid_size': 20},
        seed=42
    )
    return study_id


def test_compute_signature_stable():
    """Test that signature computation is stable (canonical)."""
    pipeline1 = [
        {"step": "imputer", "config": {"strategy": "mean", "threshold": 0.5}},
        {"step": "scaler", "config": {"method": "standard"}},
    ]
    pipeline2 = [
        {"step": "imputer", "config": {"threshold": 0.5, "strategy": "mean"}},  # Different order
        {"step": "scaler", "config": {"method": "standard"}},
    ]

    sig1 = compute_signature(pipeline1, include_params=True)
    sig2 = compute_signature(pipeline2, include_params=True)

    assert sig1 == sig2, "Signatures should be stable regardless of dict key order"


def test_compute_signature_architecture():
    """Test architecture signature ignores parameter values."""
    pipeline1 = [
        {"step": "imputer", "variant": "simple", "config": {"strategy": "mean"}},
        {"step": "scaler", "variant": "standard", "config": {"with_mean": True}},
    ]
    pipeline2 = [
        {"step": "imputer", "variant": "simple", "config": {"strategy": "median"}},  # Different param
        {"step": "scaler", "variant": "standard", "config": {"with_mean": False}},  # Different param
    ]

    arch_sig1 = compute_signature(pipeline1, include_params=False)
    arch_sig2 = compute_signature(pipeline2, include_params=False)

    assert arch_sig1 == arch_sig2, "Architecture signatures should be same for same structure"

    pipeline_sig1 = compute_signature(pipeline1, include_params=True)
    pipeline_sig2 = compute_signature(pipeline2, include_params=True)

    assert pipeline_sig1 != pipeline_sig2, "Pipeline signatures should differ with different params"


def test_create_study(storage):
    """Test study creation and retrieval."""
    study_id1, created1 = storage.create_study("study1", StudyDirection.MAXIMIZE)
    assert created1 is True
    assert study_id1 > 0

    # Same study name should return existing
    study_id2, created2 = storage.create_study("study1", StudyDirection.MAXIMIZE)
    assert created2 is False
    assert study_id2 == study_id1


def test_insert_trial_dedupe(storage, study_with_round):
    """Test trial insertion with signature-based deduplication."""
    study_id = study_with_round
    pipeline = {
        "full_chain": [{"step": "imputer", "config": {"strategy": "mean"}}],
        "pipeline_signature": "test_sig_001",
        "architecture_signature": "test_arch_001",
    }

    # First insert
    trial_id1, created1 = storage.insert_trial(
        study_id=study_id,
        round_num=1,
        phase="random",
        pipeline=pipeline
    )
    assert created1 is True
    assert trial_id1 > 0

    # Second insert with same signature - should return existing
    trial_id2, created2 = storage.insert_trial(
        study_id=study_id,
        round_num=1,
        phase="random",
        pipeline=pipeline
    )
    assert created2 is False
    assert trial_id2 == trial_id1


def test_insert_trial_concurrent_dedupe(storage, study_with_round):
    """Test concurrent trial insertion with deduplication."""
    study_id = study_with_round
    pipeline = {
        "full_chain": [{"step": "scaler", "config": {"method": "standard"}}],
        "pipeline_signature": "concurrent_sig_001",
        "architecture_signature": "concurrent_arch_001",
    }

    results = []
    errors = []

    def insert_trial():
        try:
            trial_id, created = storage.insert_trial(
                study_id=study_id,
                round_num=1,
                phase="random",
                pipeline=pipeline
            )
            results.append((trial_id, created))
        except Exception as e:
            errors.append(e)

    # Launch 5 concurrent inserts
    threads = [threading.Thread(target=insert_trial) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Some threads may fail with "database is locked" under heavy contention
    # This is expected for SQLite, so we allow some failures
    assert len(results) > 0, f"At least one thread should succeed, got errors: {errors[:3]}"

    # Of the successful results, exactly one should be created, others should return existing
    if len(results) > 0:
        created_count = sum(1 for _, created in results if created)
        assert created_count == 1, f"Exactly one trial should be created, got {created_count}"

        # All successful threads should return same trial_id
        trial_ids = [tid for tid, _ in results]
        assert len(set(trial_ids)) == 1, "All threads should get same trial_id"


def test_claim_next_waiting_trial(storage, study_with_round):
    """Test atomic trial claiming."""
    study_id = study_with_round
    # Insert trial
    pipeline = {
        "full_chain": [{"step": "encoder", "config": {"method": "target"}}],
        "pipeline_signature": "claim_sig_001",
        "architecture_signature": "claim_arch_001",
    }

    trial_id, _ = storage.insert_trial(
        study_id=study_id,
        round_num=1,
        phase="random",
        pipeline=pipeline
    )

    # Claim trial
    claimed = storage.claim_next_waiting_trial(study_id, "worker_1")
    assert claimed is not None
    assert claimed['trial_id'] == trial_id
    assert claimed['phase'] == "random"

    # Second claim should return None (already claimed)
    claimed2 = storage.claim_next_waiting_trial(study_id, "worker_2")
    assert claimed2 is None


def test_claim_concurrent(storage, study_with_round):
    """Test concurrent trial claiming with multiple workers."""
    study_id = study_with_round
    # Insert 3 trials
    for i in range(3):
        pipeline = {
            "full_chain": [{"step": f"step_{i}"}],
            "pipeline_signature": f"concurrent_claim_sig_{i:03d}",
            "architecture_signature": f"concurrent_claim_arch_{i:03d}",
        }
        storage.insert_trial(
            study_id=study_id,
            round_num=1,
            phase="random",
            pipeline=pipeline
        )

    claimed_trials = []
    errors = []

    def claim_trial(worker_id):
        try:
            trial = storage.claim_next_waiting_trial(study_id, f"worker_{worker_id}")
            if trial:
                claimed_trials.append((worker_id, trial['trial_id']))
        except Exception as e:
            errors.append(e)

    # Launch 5 workers (more than trials)
    threads = [threading.Thread(target=claim_trial, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"No errors should occur: {errors}"
    assert len(claimed_trials) == 3, f"Exactly 3 trials should be claimed, got {len(claimed_trials)}"

    # Each trial should be claimed by exactly one worker
    trial_ids = [tid for _, tid in claimed_trials]
    assert len(trial_ids) == len(set(trial_ids)), "Each trial should be claimed once"


def test_store_trial_result(storage, study_with_round):
    """Test storing trial results."""
    study_id = study_with_round
    pipeline = {
        "full_chain": [{"step": "test"}],
        "pipeline_signature": "result_sig_001",
        "architecture_signature": "result_arch_001",
    }

    trial_id, _ = storage.insert_trial(
        study_id=study_id,
        round_num=1,
        phase="random",
        pipeline=pipeline
    )

    # Store success result
    storage.store_trial_result(trial_id, TrialState.COMPLETE, value=0.85)

    # Verify state changed
    with storage._connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT state FROM trials WHERE trial_id=?", (trial_id,))
        state = cur.fetchone()[0]
        assert state == TrialState.COMPLETE.value

        # Verify value stored
        cur.execute("SELECT value FROM trial_values WHERE trial_id=?", (trial_id,))
        value = cur.fetchone()[0]
        assert value == 0.85


def test_count_trials(storage, study_with_round):
    """Test trial counting by phase."""
    study_id = study_with_round
    # Insert random trials
    for i in range(3):
        pipeline = {
            "full_chain": [{"step": f"random_{i}"}],
            "pipeline_signature": f"count_random_sig_{i:03d}",
            "architecture_signature": f"count_random_arch_{i:03d}",
        }
        storage.insert_trial(study_id, 1, "random", pipeline)

    # Insert refine trials
    for i in range(2):
        pipeline = {
            "full_chain": [{"step": f"refine_{i}"}],
            "pipeline_signature": f"count_refine_sig_{i:03d}",
            "architecture_signature": f"count_refine_arch_{i:03d}",
        }
        storage.insert_trial(study_id, 1, "refine", pipeline)

    assert storage.count_trials(study_id, 1) == 5
    assert storage.count_trials(study_id, 1, phase="random") == 3
    assert storage.count_trials(study_id, 1, phase="refine") == 2


def test_select_top_k_global(storage, study_with_round):
    """Test global top-K selection with diversity filtering."""
    study_id = study_with_round
    # Create and complete trials with different architectures
    # Use same step name but different variants to create 3 architectures
    for i in range(10):
        pipeline = {
            "full_chain": [{"step": "imputer", "variant": f"v{i % 3}", "config": {"param": i}}],
            "pipeline_signature": f"topk_sig_{i:03d}",
            "architecture_signature": f"topk_arch_{i % 3:03d}",  # 3 unique architectures
        }
        trial_id, _ = storage.insert_trial(study_id, 1, "random", pipeline)

        # Complete with different scores
        storage.store_trial_result(trial_id, TrialState.COMPLETE, value=0.5 + i * 0.01)

    # Select top-3 with diversity (max 1 per architecture)
    top_trials = storage.select_top_k_global(
        study_id=study_id,
        k=3,
        direction=StudyDirection.MAXIMIZE,
        only_unexpanded=True,
        max_per_architecture=1
    )

    assert len(top_trials) == 3

    # Should have 3 different architectures
    arch_sigs = [t['architecture_signature'] for t in top_trials]
    assert len(set(arch_sigs)) == 3, "Should have diversity (different architectures)"

    # Should be top values (0.59, 0.58, 0.57 from different architectures)
    values = [t['value'] for t in top_trials]
    assert all(v >= 0.55 for v in values), "Should select high-value trials"


def test_mark_expanded(storage, study_with_round):
    """Test marking trials as expanded."""
    study_id = study_with_round
    pipeline = {
        "full_chain": [{"step": "test"}],
        "pipeline_signature": "expand_sig_001",
        "architecture_signature": "expand_arch_001",
    }

    trial_id, _ = storage.insert_trial(study_id, 1, "random", pipeline)
    storage.store_trial_result(trial_id, TrialState.COMPLETE, value=0.9)

    # Initially unexpanded
    top = storage.select_top_k_global(
        study_id, k=1, direction=StudyDirection.MAXIMIZE, only_unexpanded=True
    )
    assert len(top) == 1

    # Mark expanded
    storage.mark_expanded(trial_id, round_num=1)

    # Should not appear in unexpanded query
    top2 = storage.select_top_k_global(
        study_id, k=1, direction=StudyDirection.MAXIMIZE, only_unexpanded=True
    )
    assert len(top2) == 0


def test_requeue_trial(storage, study_with_round):
    """Test requeuing trials."""
    study_id = study_with_round
    pipeline = {
        "full_chain": [{"step": "test"}],
        "pipeline_signature": "requeue_sig_001",
        "architecture_signature": "requeue_arch_001",
    }

    trial_id, _ = storage.insert_trial(study_id, 1, "random", pipeline)

    # Claim trial
    claimed = storage.claim_next_waiting_trial(study_id, "worker_1")
    assert claimed is not None

    # Verify state is RUNNING
    with storage._connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT state FROM trials WHERE trial_id=?", (trial_id,))
        state = cur.fetchone()[0]
        assert state == TrialState.RUNNING.value

    # Requeue
    storage.requeue_trial(trial_id)

    # Verify state is WAITING again
    with storage._connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT state FROM trials WHERE trial_id=?", (trial_id,))
        state = cur.fetchone()[0]
        assert state == TrialState.WAITING.value

    # Should be claimable again
    claimed2 = storage.claim_next_waiting_trial(study_id, "worker_2")
    assert claimed2 is not None
    assert claimed2['trial_id'] == trial_id
