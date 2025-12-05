from mlarena.core.experiment import ExperimentState


def test_experiment_state_roundtrip(project_root):
    state = ExperimentState.load_or_create(project_root, "demo")
    state.start_module("eda", {"n": 1})
    state.complete_module("eda", {"ok": True})
    state.save()

    loaded = ExperimentState.load_or_create(project_root, "demo", experiment_id=state.experiment_id)
    assert "eda" in loaded.modules
    entry = loaded.modules["eda"]
    assert entry.status == "completed"
    assert entry.payload == {"ok": True}
    assert entry.invocation == {"n": 1}


def test_experiment_state_merges_existing(project_root):
    state = ExperimentState.load_or_create(project_root, "demo")
    state.start_module("eda", {"n": 1})
    state.save()

    # Second handle completes module and saves
    second = ExperimentState.load_or_create(project_root, "demo", experiment_id=state.experiment_id)
    second.complete_module("eda", {"done": True})
    second.save()

    # Third handle adds another module without losing previous data
    third = ExperimentState.load_or_create(project_root, "demo", experiment_id=state.experiment_id)
    third.start_module("model", {"alpha": 0.1})
    third.save()

    merged = ExperimentState.load_or_create(project_root, "demo", experiment_id=state.experiment_id)
    assert merged.modules["eda"].status == "completed"
    assert merged.modules["model"].invocation == {"alpha": 0.1}
