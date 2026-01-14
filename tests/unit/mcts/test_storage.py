import pytest
import sqlite3
from mlarena.modules.mcts.storage import MCTSStorage, StudyDirection

def test_create_study(tmp_path):
    db_path = tmp_path / "test.db"
    storage = MCTSStorage(f"sqlite:///{db_path}")
    
    study_id, is_new = storage.create_study("my_study", StudyDirection.MAXIMIZE)
    assert study_id == 1
    assert is_new is True
    
    # Verify DB content
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT study_name FROM studies WHERE study_id=?", (study_id,))
    assert cur.fetchone()[0] == "my_study"
    
    cur.execute("SELECT direction FROM study_directions WHERE study_id=?", (study_id,))
    assert cur.fetchone()[0] == StudyDirection.MAXIMIZE.value

def test_create_trial(tmp_path):
    db_path = tmp_path / "test.db"
    storage = MCTSStorage(f"sqlite:///{db_path}")
    study_id, _ = storage.create_study("my_study", StudyDirection.MINIMIZE)
    
    # Sig: (study_id, pipeline_signature, depth, number=None, params=None)
    trial_id = storage.create_trial(
        study_id=study_id,
        pipeline_signature="sig1",
        depth=0,
        number=0,
        params={"p1": "v1"}
    )
    assert trial_id == 1
    
    # Verify params
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT param_value FROM trial_params WHERE trial_id=? AND param_name='p1'", (trial_id,))
    assert cur.fetchone()[0] == '"v1"' # JSON encoded

def test_get_best_trial(tmp_path):
    db_path = tmp_path / "test.db"
    storage = MCTSStorage(f"sqlite:///{db_path}")
    
    # Maximize
    s1, _ = storage.create_study("max_study", StudyDirection.MAXIMIZE)
    t1 = storage.create_trial(study_id=s1, pipeline_signature="s1", depth=0, number=0)
    storage.set_trial_value(t1, 0.5)
    storage.set_trial_state(t1, "COMPLETE")
    
    t2 = storage.create_trial(study_id=s1, pipeline_signature="s2", depth=0, number=1)
    storage.set_trial_value(t2, 0.9)
    storage.set_trial_state(t2, "COMPLETE")
    
    best = storage.get_best_trial(s1)
    assert best["trial_id"] == t2
    assert best["value"] == 0.9
    
    # Minimize
    s2, _ = storage.create_study("min_study", StudyDirection.MINIMIZE)
    t3 = storage.create_trial(study_id=s2, pipeline_signature="s3", depth=0, number=0)
    storage.set_trial_value(t3, 0.5)
    storage.set_trial_state(t3, "COMPLETE")
    
    t4 = storage.create_trial(study_id=s2, pipeline_signature="s4", depth=0, number=1)
    storage.set_trial_value(t4, 0.9)
    storage.set_trial_state(t4, "COMPLETE")
    
    best_min = storage.get_best_trial(s2)
    assert best_min["trial_id"] == t3
    assert best_min["value"] == 0.5
