from __future__ import annotations
import json
import sqlite3
from enum import IntEnum
from typing import Optional, Dict, Any, List
from pathlib import Path

class StudyDirection(IntEnum):
    NOT_SET = 0
    MINIMIZE = 1
    MAXIMIZE = 2

class TrialState(IntEnum):
    RUNNING = 0
    COMPLETE = 1
    PRUNED = 2
    FAIL = 3
    WAITING = 4

SCHEMA = """
-- Studies
CREATE TABLE IF NOT EXISTS studies (
  study_id   INTEGER PRIMARY KEY AUTOINCREMENT,
  study_name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS study_directions (
  study_id   INTEGER NOT NULL,
  objective  INTEGER NOT NULL DEFAULT 0,
  direction  INTEGER NOT NULL,
  PRIMARY KEY (study_id, objective),
  FOREIGN KEY (study_id) REFERENCES studies(study_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS study_user_attributes (
  study_id   INTEGER NOT NULL,
  key        TEXT NOT NULL,
  value_json TEXT NOT NULL,
  PRIMARY KEY (study_id, key),
  FOREIGN KEY (study_id) REFERENCES studies(study_id) ON DELETE CASCADE
);

-- Trials
CREATE TABLE IF NOT EXISTS trials (
  trial_id          INTEGER PRIMARY KEY AUTOINCREMENT,
  study_id          INTEGER NOT NULL,
  number            INTEGER NOT NULL,
  state             INTEGER NOT NULL,
  datetime_start    TEXT,
  datetime_complete TEXT,
  UNIQUE (study_id, number),
  FOREIGN KEY (study_id) REFERENCES studies(study_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_trials_study_state  ON trials(study_id, state);

CREATE TABLE IF NOT EXISTS trial_values (
  trial_id   INTEGER NOT NULL,
  objective  INTEGER NOT NULL DEFAULT 0,
  value      REAL NOT NULL,
  PRIMARY KEY (trial_id, objective),
  FOREIGN KEY (trial_id) REFERENCES trials(trial_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_trial_values_value ON trial_values(value);

CREATE TABLE IF NOT EXISTS trial_params (
  trial_id    INTEGER NOT NULL,
  param_name  TEXT NOT NULL,
  param_value TEXT NOT NULL,
  PRIMARY KEY (trial_id, param_name),
  FOREIGN KEY (trial_id) REFERENCES trials(trial_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS trial_user_attributes (
  trial_id   INTEGER NOT NULL,
  key        TEXT NOT NULL,
  value_json TEXT NOT NULL,
  PRIMARY KEY (trial_id, key),
  FOREIGN KEY (trial_id) REFERENCES trials(trial_id) ON DELETE CASCADE
);

-- MCTS Nodes
CREATE TABLE IF NOT EXISTS mcts_nodes (
  trial_id          INTEGER PRIMARY KEY,
  depth             INTEGER NOT NULL,
  pipeline_signature TEXT NOT NULL UNIQUE,
  n_visits          INTEGER NOT NULL DEFAULT 0,
  value_sum         REAL NOT NULL DEFAULT 0.0,
  value_best        REAL,
  FOREIGN KEY (trial_id) REFERENCES trials(trial_id) ON DELETE CASCADE
);
"""

class MCTSStorage:
    def __init__(self, storage_url: str):
        self.url = storage_url
        if storage_url.startswith("sqlite:///"):
            self.path = Path(storage_url.replace("sqlite:///", ""))
        else:
            self.path = Path(storage_url)
            
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.path)

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def create_study(self, study_name: str, direction: StudyDirection) -> int:
        with self._connect() as conn:
            cur = conn.cursor()
            # Check if exists
            cur.execute("SELECT study_id FROM studies WHERE study_name=?", (study_name,))
            row = cur.fetchone()
            if row:
                return row[0]
            
            cur.execute("INSERT INTO studies (study_name) VALUES (?)", (study_name,))
            study_id = cur.lastrowid
            
            cur.execute(
                "INSERT INTO study_directions (study_id, objective, direction) VALUES (?, ?, ?)",
                (study_id, 0, direction.value)
            )
            return study_id

    def create_trial(
        self, 
        study_id: int, 
        number: int, 
        pipeline_signature: str,
        depth: int,
        params: Optional[Dict[str, Any]] = None,
        state: TrialState = TrialState.WAITING
    ) -> int:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO trials (study_id, number, state, datetime_start) VALUES (?, ?, ?, datetime('now'))",
                (study_id, number, state.value)
            )
            trial_id = cur.lastrowid
            
            # Insert params
            if params:
                for k, v in params.items():
                    cur.execute(
                        "INSERT INTO trial_params (trial_id, param_name, param_value) VALUES (?, ?, ?)",
                        (trial_id, k, json.dumps(v))
                    )
            
            # Insert MCTS node
            cur.execute(
                "INSERT INTO mcts_nodes (trial_id, depth, pipeline_signature) VALUES (?, ?, ?)",
                (trial_id, depth, pipeline_signature)
            )
            return trial_id

    def set_trial_value(self, trial_id: int, value: float):
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO trial_values (trial_id, objective, value) VALUES (?, ?, ?)",
                (trial_id, 0, value)
            )

    def set_trial_state(self, trial_id: int, state: str):
        # Allow passing string 'COMPLETE' etc
        if isinstance(state, str):
            state_enum = getattr(TrialState, state.upper())
        else:
            state_enum = state
            
        with self._connect() as conn:
            conn.execute(
                "UPDATE trials SET state=? WHERE trial_id=?",
                (state_enum.value, trial_id)
            )
            if state_enum in (TrialState.COMPLETE, TrialState.FAIL, TrialState.PRUNED):
                conn.execute(
                    "UPDATE trials SET datetime_complete=datetime('now') WHERE trial_id=?",
                    (trial_id,)
                )

    def get_best_trial(self, study_id: int) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT direction FROM study_directions WHERE study_id=? AND objective=0", (study_id,))
            row = cur.fetchone()
            if not row:
                return None
            direction = row[0]
            
            order = "DESC" if direction == StudyDirection.MAXIMIZE.value else "ASC"
            
            query = f"""
                SELECT t.trial_id, tv.value
                FROM trials t
                JOIN trial_values tv ON tv.trial_id = t.trial_id AND tv.objective = 0
                WHERE t.study_id = ? AND t.state = ?
                ORDER BY tv.value {order}
                LIMIT 1
            """
            cur.execute(query, (study_id, TrialState.COMPLETE.value))
            res = cur.fetchone()
            if res:
                return {"trial_id": res[0], "value": res[1]}
            return None
