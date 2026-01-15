from __future__ import annotations
import json
import sqlite3
import contextlib
import time
import random
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
  study_id          INTEGER NOT NULL,
  parent_trial_id   INTEGER, -- NULL for root/baseline
  depth             INTEGER NOT NULL,
  pipeline_signature TEXT NOT NULL,
  n_visits          INTEGER NOT NULL DEFAULT 0,
  value_sum         REAL NOT NULL DEFAULT 0.0,
  value_best        REAL,
  UNIQUE (study_id, parent_trial_id, pipeline_signature),
  FOREIGN KEY (trial_id) REFERENCES trials(trial_id) ON DELETE CASCADE,
  FOREIGN KEY (parent_trial_id) REFERENCES trials(trial_id) ON DELETE CASCADE,
  FOREIGN KEY (study_id) REFERENCES studies(study_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_mcts_nodes_study ON mcts_nodes(study_id);
CREATE INDEX IF NOT EXISTS idx_mcts_nodes_sig ON mcts_nodes(study_id, pipeline_signature);
CREATE INDEX IF NOT EXISTS idx_mcts_nodes_parent ON mcts_nodes(parent_trial_id);

-- IMPORTANT (SQLite): UNIQUE constraints treat NULL != NULL, so the table-level UNIQUE
-- does NOT prevent duplicates when parent_trial_id IS NULL (root/baseline).
-- This partial UNIQUE index enforces root-level uniqueness per (study_id, pipeline_signature).
CREATE UNIQUE INDEX IF NOT EXISTS uq_mcts_nodes_root_sig
  ON mcts_nodes(study_id, pipeline_signature)
  WHERE parent_trial_id IS NULL;

-- MCTS Edges (Parent -> Child relations)
CREATE TABLE IF NOT EXISTS mcts_edges (
  parent_trial_id   INTEGER NOT NULL,
  child_trial_id    INTEGER NOT NULL UNIQUE, -- Enforce Tree: one parent per child
  action_json       TEXT NOT NULL,
  PRIMARY KEY (parent_trial_id, child_trial_id),
  FOREIGN KEY (parent_trial_id) REFERENCES trials(trial_id) ON DELETE CASCADE,
  FOREIGN KEY (child_trial_id) REFERENCES trials(trial_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_mcts_edges_parent ON mcts_edges(parent_trial_id);

-- MCTS Evaluations (Multi-fidelity)
CREATE TABLE IF NOT EXISTS mcts_evaluations (
  trial_id        INTEGER NOT NULL,
  fidelity        TEXT NOT NULL,
  status          TEXT NOT NULL,
  value           REAL,
  metric_name     TEXT,
  duration_sec    REAL,
  details_json    TEXT,
  PRIMARY KEY (trial_id, fidelity),
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
        # Higher timeout for NFS/Contention (30 seconds)
        conn = sqlite3.connect(self.path, timeout=30.0)
        
        # Apply performance and integrity pragmas
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        
        return conn

    @contextlib.contextmanager
    def atomic(self):
        """Context manager for a single transaction across multiple storage calls."""
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def create_study(self, study_name: str, direction: StudyDirection) -> tuple[int, bool]:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT study_id FROM studies WHERE study_name=?", (study_name,))
            row = cur.fetchone()
            if row:
                return row[0], False
            
            cur.execute("INSERT INTO studies (study_name) VALUES (?)", (study_name,))
            study_id = cur.lastrowid
            
            cur.execute(
                "INSERT INTO study_directions (study_id, objective, direction) VALUES (?, ?, ?)",
                (study_id, 0, direction.value)
            )
            return study_id, True

    def create_trial(
        self, 
        study_id: int, 
        pipeline_signature: str,
        depth: int,
        parent_trial_id: Optional[int] = None,
        number: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
        state: TrialState = TrialState.WAITING,
        conn: Optional[sqlite3.Connection] = None
    ) -> int:
        
        # Max retries for number collision
        max_retries = 10
        
        def _exec(c):
            cur = c.cursor()
            
            # Assignment and Insert with Retry
            for attempt in range(max_retries):
                # 1. Re-check if signature exists UNDER THIS PARENT (handles concurrent inserts)
                if parent_trial_id is not None:
                    query = "SELECT trial_id FROM mcts_nodes WHERE study_id = ? AND parent_trial_id = ? AND pipeline_signature = ?"
                    cur.execute(query, (study_id, parent_trial_id, pipeline_signature))
                else:
                    query = "SELECT trial_id FROM mcts_nodes WHERE study_id = ? AND parent_trial_id IS NULL AND pipeline_signature = ?"
                    cur.execute(query, (study_id, pipeline_signature))
                
                existing = cur.fetchone()
                if existing:
                    return existing[0]

                try:
                    # Inner savepoint for retry
                    c.execute(f"SAVEPOINT trial_creation_{attempt}")
                    
                    if number is None:
                        cur.execute("SELECT MAX(number) FROM trials WHERE study_id=?", (study_id,))
                        res = cur.fetchone()
                        trial_number = (res[0] or 0) + 1
                    else:
                        trial_number = number

                    cur.execute(
                        "INSERT INTO trials (study_id, number, state, datetime_start) VALUES (?, ?, ?, datetime('now'))",
                        (study_id, trial_number, state.value)
                    )
                    trial_id = cur.lastrowid
                    
                    if params:
                        for k, v in params.items():
                            cur.execute(
                                "INSERT INTO trial_params (trial_id, param_name, param_value) VALUES (?, ?, ?)",
                                (trial_id, k, json.dumps(v))
                            )
                    
                    cur.execute(
                        "INSERT INTO mcts_nodes (trial_id, study_id, parent_trial_id, depth, pipeline_signature) VALUES (?, ?, ?, ?, ?)",
                        (trial_id, study_id, parent_trial_id, depth, pipeline_signature)
                    )
                    
                    c.execute(f"RELEASE SAVEPOINT trial_creation_{attempt}")
                    return trial_id
                    
                except sqlite3.IntegrityError:
                    # Explicitly rollback and release if needed (SQLite release happens automatically on commit/rollback of parent, but we use sub-savepoints)
                    c.execute(f"ROLLBACK TO SAVEPOINT trial_creation_{attempt}")
                    c.execute(f"RELEASE SAVEPOINT trial_creation_{attempt}")
                    # In next iteration of the loop, we will re-check signature 
                    # and potentially return existing trial_id.
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(random.uniform(0.01, 0.1)) # Backoff
            
            return -1 # Should not happen

        if conn:
            return _exec(conn)
        else:
            with self._connect() as c:
                res = _exec(c)
                c.commit()
                return res

    def get_trial_id_by_signature(self, study_id: int, pipeline_signature: str, parent_trial_id: Optional[int] = None, conn: Optional[sqlite3.Connection] = None) -> Optional[int]:
        def _exec(c):
            cur = c.cursor()
            if parent_trial_id is not None:
                query = "SELECT trial_id FROM mcts_nodes WHERE study_id = ? AND parent_trial_id = ? AND pipeline_signature = ?"
                cur.execute(query, (study_id, parent_trial_id, pipeline_signature))
            else:
                query = "SELECT trial_id FROM mcts_nodes WHERE study_id = ? AND parent_trial_id IS NULL AND pipeline_signature = ?"
                cur.execute(query, (study_id, pipeline_signature))
            row = cur.fetchone()
            return row[0] if row else None

        if conn:
            return _exec(conn)
        else:
            with self._connect() as conn_local:
                return _exec(conn_local)

    def add_edge(self, parent_trial_id: int, child_trial_id: int, action: Dict[str, Any], conn: Optional[sqlite3.Connection] = None):
        # Edge should be immutable: (parent, child) must always map to the same action.
        # We use DO NOTHING on conflict and verify semantic equality of JSON objects.
        query = """
            INSERT INTO mcts_edges (parent_trial_id, child_trial_id, action_json)
            VALUES (?, ?, ?)
            ON CONFLICT(parent_trial_id, child_trial_id) DO NOTHING
        """
        def _exec(c):
            try:
                # Canonicalize new writes (stable order, no whitespace). Old rows may still use default formatting.
                action_json = json.dumps(action, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                cur = c.execute(query, (parent_trial_id, child_trial_id, action_json))

                # If conflict happened (row already existed), validate immutability semantically.
                if cur.rowcount == 0:
                    row = c.execute(
                        "SELECT action_json FROM mcts_edges WHERE parent_trial_id=? AND child_trial_id=?",
                        (parent_trial_id, child_trial_id),
                    ).fetchone()
                    if row:
                        existing_raw = row[0]
                        try:
                            existing_obj = json.loads(existing_raw)
                        except Exception:
                            existing_obj = None

                        # Prefer semantic compare; fallback to canonical-string compare if parsing fails.
                        if existing_obj is not None:
                            # Normalize both sides to canonical Action record to avoid legacy-key mismatches
                            from mlarena.modules.mcts.node import Action as MCTSAction
                            existing_norm = MCTSAction.from_record(existing_obj).to_record()
                            action_norm = MCTSAction.from_record(action).to_record()
                            if existing_norm != action_norm:
                                # Detailed diff logging
                                diff_keys = set(existing_norm.keys()) | set(action_norm.keys())
                                diff_report = []
                                for k in sorted(diff_keys):
                                    v_old = existing_norm.get(k)
                                    v_new = action_norm.get(k)
                                    if v_old != v_new:
                                        diff_report.append(f"  [{k}]: DB={v_old!r} vs NEW={v_new!r}")
                                
                                diff_str = "\n".join(diff_report)
                                raise sqlite3.IntegrityError(
                                    f"Edge already exists with different action_json. "
                                    f"(parent={parent_trial_id}, child={child_trial_id})\nDifferences:\n{diff_str}"
                                )
                        else:
                            # Worst case: compare canonicalized strings if the stored JSON is malformed.
                            existing_canon = existing_raw.strip()
                            if existing_canon != action_json:
                                raise sqlite3.IntegrityError(
                                    f"Edge already exists with different action_json. "
                                    f"(parent={parent_trial_id}, child={child_trial_id})"
                                )
            except sqlite3.IntegrityError as e:
                # Check if it's a UNIQUE constraint violation on child_trial_id
                if "UNIQUE constraint failed: mcts_edges.child_trial_id" in str(e):
                    # Tree violation: child already has a different parent
                    cur = c.cursor()
                    cur.execute("SELECT parent_trial_id FROM mcts_edges WHERE child_trial_id=?", (child_trial_id,))
                    row = cur.fetchone()
                    existing_parent = row[0] if row else "unknown"
                    raise sqlite3.IntegrityError(
                        f"Tree Violation: Child trial {child_trial_id} is already attached to parent {existing_parent}. "
                        f"Cannot attach it to parent {parent_trial_id}. MCTS requires a strict tree structure."
                    ) from e
                raise

        if conn:
            _exec(conn)
        else:
            with self._connect() as c:
                _exec(c)
                c.commit()

    def get_all_edges(self, study_id: int) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            query = """
                SELECT e.* FROM mcts_edges e
                JOIN trials t ON t.trial_id = e.child_trial_id
                WHERE t.study_id = ?
            """
            cur.execute(query, (study_id,))
            return [dict(row) for row in cur.fetchall()]

    def get_all_nodes(self, study_id: int) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            query = """
                SELECT n.* FROM mcts_nodes n
                JOIN trials t ON t.trial_id = n.trial_id
                WHERE t.study_id = ?
            """
            cur.execute(query, (study_id,))
            return [dict(row) for row in cur.fetchall()]

    def update_node_stats(self, trial_id: int, n_visits: int, value_sum: float, value_best: float, conn: Optional[sqlite3.Connection] = None):
        if conn:
            conn.execute(
                "UPDATE mcts_nodes SET n_visits=?, value_sum=?, value_best=? WHERE trial_id=?",
                (n_visits, value_sum, value_best, trial_id)
            )
        else:
            with self._connect() as c:
                c.execute(
                    "UPDATE mcts_nodes SET n_visits=?, value_sum=?, value_best=? WHERE trial_id=?",
                    (n_visits, value_sum, value_best, trial_id)
                )
                c.commit()

    def update_node_stats_many(self, updates: List[tuple[int, int, float, float]], conn: Optional[sqlite3.Connection] = None):
        """Batch update node stats. updates is list of (trial_id, n_visits, value_sum, value_best)."""
        query = "UPDATE mcts_nodes SET n_visits=?, value_sum=?, value_best=? WHERE trial_id=?"
        # Reorder params to match query: (n_visits, value_sum, value_best, trial_id)
        params = [(u[1], u[2], u[3], u[0]) for u in updates]
        
        if conn:
            conn.executemany(query, params)
        else:
            with self._connect() as c:
                c.executemany(query, params)
                c.commit()

    def set_trial_value(self, trial_id: int, value: float, conn: Optional[sqlite3.Connection] = None):
        if conn:
            conn.execute(
                "INSERT OR REPLACE INTO trial_values (trial_id, objective, value) VALUES (?, ?, ?)",
                (trial_id, 0, value)
            )
        else:
            with self._connect() as c:
                c.execute(
                    "INSERT OR REPLACE INTO trial_values (trial_id, objective, value) VALUES (?, ?, ?)",
                    (trial_id, 0, value)
                )
                c.commit()

    def set_trial_state(self, trial_id: int, state: str | TrialState, conn: Optional[sqlite3.Connection] = None):
        if isinstance(state, str):
            state_enum = getattr(TrialState, state.upper())
        else:
            state_enum = state
            
        def _exec(c):
            c.execute(
                "UPDATE trials SET state=? WHERE trial_id=?",
                (state_enum.value, trial_id)
            )
            if state_enum in (TrialState.COMPLETE, TrialState.FAIL, TrialState.PRUNED):
                c.execute(
                    "UPDATE trials SET datetime_complete=datetime('now') WHERE trial_id=?",
                    (trial_id,)
                )

        if conn:
            _exec(conn)
        else:
            with self._connect() as c:
                _exec(c)
                c.commit()

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

    def add_evaluation(self, trial_id: int, fidelity: str, status: str, value: Optional[float], metric: str, duration: float, details: Optional[Dict[str, Any]] = None, conn: Optional[sqlite3.Connection] = None):
        details_json = json.dumps(details) if details else None
        if conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO mcts_evaluations 
                (trial_id, fidelity, status, value, metric_name, duration_sec, details_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (trial_id, fidelity, status, value, metric, duration, details_json)
            )
        else:
            with self._connect() as c:
                c.execute(
                    """
                    INSERT OR REPLACE INTO mcts_evaluations 
                    (trial_id, fidelity, status, value, metric_name, duration_sec, details_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (trial_id, fidelity, status, value, metric, duration, details_json)
                )
                c.commit()

    def get_evaluations(self, trial_id: int) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM mcts_evaluations WHERE trial_id=?", (trial_id,))
            return [dict(row) for row in cur.fetchall()]

    def get_fidelity_history(self, study_id: int, fidelity: str) -> List[float]:
        with self._connect() as conn:
            cur = conn.cursor()
            query = """
                SELECT e.value 
                FROM mcts_evaluations e
                JOIN trials t ON t.trial_id = e.trial_id
                WHERE t.study_id = ? AND e.fidelity = ? AND e.value IS NOT NULL
            """
            cur.execute(query, (study_id, fidelity))
            return [row[0] for row in cur.fetchall()]