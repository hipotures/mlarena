from __future__ import annotations
import json
import sqlite3
import contextlib
import time
import random
import hashlib
from enum import IntEnum
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime


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


class RoundStatus(IntEnum):
    ACTIVE = 0
    COMPLETE = 1
    DEGRADED = 2


# Reuse base tables from MCTS + add RANT-specific tables
SCHEMA = """
-- Studies (reused from MCTS)
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

-- Trials (reused from MCTS)
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
CREATE INDEX IF NOT EXISTS idx_trials_study_state ON trials(study_id, state);

CREATE TABLE IF NOT EXISTS trial_values (
  trial_id   INTEGER NOT NULL,
  objective  INTEGER NOT NULL DEFAULT 0,
  value      REAL NOT NULL,
  PRIMARY KEY (trial_id, objective),
  FOREIGN KEY (trial_id) REFERENCES trials(trial_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_trial_values_value ON trial_values(value);

-- RANT-specific tables
CREATE TABLE IF NOT EXISTS rant_rounds (
  study_id            INTEGER NOT NULL,
  round               INTEGER NOT NULL,
  status              INTEGER NOT NULL DEFAULT 0,  -- RoundStatus enum
  n_random            INTEGER NOT NULL,
  top_k               INTEGER NOT NULL,
  children_per_parent INTEGER NOT NULL,
  min_core_len        INTEGER NOT NULL,
  max_core_len        INTEGER NOT NULL,
  refinement_json     TEXT NOT NULL,
  seed                INTEGER NOT NULL,
  created_at          TEXT NOT NULL,
  completed_at        TEXT,
  PRIMARY KEY (study_id, round),
  FOREIGN KEY (study_id) REFERENCES studies(study_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS rant_trials (
  trial_id              INTEGER PRIMARY KEY,
  study_id              INTEGER NOT NULL,
  round                 INTEGER NOT NULL,
  phase                 TEXT NOT NULL,  -- 'random' or 'refine'
  parent_trial_id       INTEGER,  -- NULL for random phase
  expanded_round        INTEGER,  -- NULL = eligible for refinement
  pipeline_signature    TEXT NOT NULL,  -- SHA256(canonical JSON with params)
  architecture_signature TEXT NOT NULL,  -- SHA256(structure only, for diversity)
  pipeline_json         TEXT NOT NULL,
  worker_id             TEXT,
  created_at            TEXT NOT NULL,
  UNIQUE (study_id, pipeline_signature),
  FOREIGN KEY (trial_id) REFERENCES trials(trial_id) ON DELETE CASCADE,
  FOREIGN KEY (study_id) REFERENCES studies(study_id) ON DELETE CASCADE,
  FOREIGN KEY (parent_trial_id) REFERENCES trials(trial_id) ON DELETE CASCADE,
  FOREIGN KEY (study_id, round) REFERENCES rant_rounds(study_id, round) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_rant_trials_study_round ON rant_trials(study_id, round);
CREATE INDEX IF NOT EXISTS idx_rant_trials_phase ON rant_trials(study_id, round, phase);
CREATE INDEX IF NOT EXISTS idx_rant_trials_arch_sig ON rant_trials(study_id, architecture_signature);
CREATE INDEX IF NOT EXISTS idx_rant_trials_expanded ON rant_trials(study_id, round, expanded_round);
"""


def compute_signature(obj: Any, include_params: bool = True) -> str:
    """Compute SHA256 signature of a pipeline.

    Args:
        obj: Pipeline object (dict or list)
        include_params: If True, include parameter values. If False, structure only.

    Returns:
        str: Hex digest of SHA256 hash
    """
    if not include_params:
        # Architecture signature: remove parameter values, keep structure
        if isinstance(obj, list):
            obj_filtered = []
            for item in obj:
                if isinstance(item, dict):
                    filtered_item = {}
                    for k in ('step', 'variant', 'group', 'template'):
                        if k in item:
                            filtered_item[k] = item[k]
                    obj_filtered.append(filtered_item)
            obj = obj_filtered
        elif isinstance(obj, dict):
            obj_filtered = {}
            for k in ('step', 'variant', 'group', 'template'):
                if k in obj:
                    obj_filtered[k] = obj[k]
            obj = obj_filtered

    # Canonical JSON: sorted keys, no whitespace
    canonical = json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


class RANTStorage:
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
        """Create or get existing study.

        Returns:
            (study_id, created): study_id and whether it was newly created
        """
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT study_id FROM studies WHERE study_name=?", (study_name,))
            row = cur.fetchone()
            if row:
                return row[0], False

            cur.execute("INSERT INTO studies (study_name) VALUES (?)", (study_name,))
            study_id = cur.lastrowid
            assert study_id is not None

            cur.execute(
                "INSERT INTO study_directions (study_id, objective, direction) VALUES (?, ?, ?)",
                (study_id, 0, direction.value),
            )
            conn.commit()
            return study_id, True

    def resolve_active_round(
        self,
        study_id: int,
        n_random: int,
        top_k: int,
        children_per_parent: int,
        min_core_len: int,
        max_core_len: int,
        refinement_config: Dict[str, Any],
        seed: int,
    ) -> tuple[int, Dict[str, Any]]:
        """Get or create active round with frozen metadata.

        Returns:
            (round_number, metadata): Round number and frozen metadata from DB
        """
        with self._connect() as conn:
            cur = conn.cursor()

            # Check for existing active round
            cur.execute(
                "SELECT round, n_random, top_k, children_per_parent, min_core_len, max_core_len, refinement_json, seed "
                "FROM rant_rounds WHERE study_id=? AND status=? ORDER BY round DESC LIMIT 1",
                (study_id, RoundStatus.ACTIVE.value)
            )
            row = cur.fetchone()

            if row:
                # Return existing round with its frozen metadata
                round_num, n_r, k, p, min_len, max_len, ref_json, s = row
                metadata = {
                    'n_random': n_r,
                    'top_k': k,
                    'children_per_parent': p,
                    'min_core_len': min_len,
                    'max_core_len': max_len,
                    'refinement': json.loads(ref_json),
                    'seed': s
                }
                return round_num, metadata

            # Create new round with current config
            cur.execute(
                "SELECT MAX(round) FROM rant_rounds WHERE study_id=?",
                (study_id,)
            )
            max_round = cur.fetchone()[0]
            new_round = (max_round or 0) + 1

            refinement_json = json.dumps(refinement_config, sort_keys=True)

            cur.execute(
                "INSERT INTO rant_rounds "
                "(study_id, round, status, n_random, top_k, children_per_parent, min_core_len, max_core_len, refinement_json, seed, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (study_id, new_round, RoundStatus.ACTIVE.value, n_random, top_k, children_per_parent,
                 min_core_len, max_core_len, refinement_json, seed, datetime.now().isoformat())
            )
            conn.commit()

            metadata = {
                'n_random': n_random,
                'top_k': top_k,
                'children_per_parent': children_per_parent,
                'min_core_len': min_core_len,
                'max_core_len': max_core_len,
                'refinement': refinement_config,
                'seed': seed
            }
            return new_round, metadata

    def insert_trial(
        self,
        study_id: int,
        round_num: int,
        phase: str,
        pipeline: Dict[str, Any],
        parent_trial_id: Optional[int] = None,
        max_attempts: int = 10,
    ) -> tuple[int, bool]:
        """Insert trial with dedupe by pipeline_signature.

        Returns:
            (trial_id, created): trial_id and whether it was newly created
        """
        # Compute signatures from full_chain (list of steps)
        full_chain = pipeline.get('full_chain', pipeline)
        pipeline_sig = compute_signature(full_chain, include_params=True)
        arch_sig = compute_signature(full_chain, include_params=False)
        pipeline_json = json.dumps(pipeline, sort_keys=True)

        for attempt in range(max_attempts):
            try:
                with self._connect() as conn:
                    cur = conn.cursor()

                    # Use SAVEPOINT for this attempt
                    cur.execute(f"SAVEPOINT insert_attempt_{attempt}")

                    try:
                        # Check for existing trial with same signature
                        cur.execute(
                            "SELECT trial_id FROM rant_trials WHERE study_id=? AND pipeline_signature=?",
                            (study_id, pipeline_sig)
                        )
                        existing = cur.fetchone()
                        if existing:
                            cur.execute(f"RELEASE SAVEPOINT insert_attempt_{attempt}")
                            return existing[0], False

                        # Create new trial with auto-incrementing number
                        cur.execute(
                            "SELECT COALESCE(MAX(number), 0) + 1 FROM trials WHERE study_id=?",
                            (study_id,)
                        )
                        trial_number = cur.fetchone()[0]

                        cur.execute(
                            "INSERT INTO trials (study_id, number, state, datetime_start) VALUES (?, ?, ?, datetime('now'))",
                            (study_id, trial_number, TrialState.WAITING.value)
                        )
                    except sqlite3.IntegrityError as ie:
                        # Rollback savepoint and retry
                        cur.execute(f"ROLLBACK TO SAVEPOINT insert_attempt_{attempt}")
                        cur.execute(f"RELEASE SAVEPOINT insert_attempt_{attempt}")
                        if "UNIQUE constraint failed: trials.study_id, trials.number" in str(ie):
                            time.sleep(random.uniform(0.01, 0.1))
                            continue
                        raise
                    trial_id = cur.lastrowid
                    assert trial_id is not None

                    # Insert RANT-specific metadata
                    cur.execute(
                        "INSERT INTO rant_trials "
                        "(trial_id, study_id, round, phase, parent_trial_id, pipeline_signature, architecture_signature, pipeline_json, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
                        (trial_id, study_id, round_num, phase, parent_trial_id, pipeline_sig, arch_sig, pipeline_json)
                    )

                    cur.execute(f"RELEASE SAVEPOINT insert_attempt_{attempt}")
                    conn.commit()
                    return trial_id, True

            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed: rant_trials.study_id, rant_trials.pipeline_signature" in str(e):
                    # Another process inserted same signature - retry
                    time.sleep(random.uniform(0.01, 0.1))
                    if attempt == max_attempts - 1:
                        # Final attempt: just return existing
                        with self._connect() as conn:
                            cur = conn.cursor()
                            cur.execute(
                                "SELECT trial_id FROM rant_trials WHERE study_id=? AND pipeline_signature=?",
                                (study_id, pipeline_sig)
                            )
                            existing = cur.fetchone()
                            if existing:
                                return existing[0], False
                            raise  # Should not happen
                else:
                    raise

        raise RuntimeError(f"Failed to insert trial after {max_attempts} attempts")

    def claim_next_waiting_trial(
        self,
        study_id: int,
        worker_id: str,
        max_attempts: int = 5,
        round_num: Optional[int] = None,
        phases: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Atomically claim next WAITING trial and mark as RUNNING.

        Returns:
            Trial dict or None if no trials available
        """
        for attempt in range(max_attempts):
            try:
                with self._connect() as conn:
                    cur = conn.cursor()

                    # Find next waiting trial
                    query = (
                        "SELECT t.trial_id, t.number, rt.pipeline_json, rt.round, rt.phase, rt.parent_trial_id "
                        "FROM trials t "
                        "JOIN rant_trials rt ON t.trial_id = rt.trial_id "
                        "WHERE t.study_id=? AND t.state=?"
                    )
                    params: List[Any] = [study_id, TrialState.WAITING.value]
                    if round_num is not None:
                        query += " AND rt.round=?"
                        params.append(round_num)
                    if phases:
                        placeholders = ",".join(["?"] * len(phases))
                        query += f" AND rt.phase IN ({placeholders})"
                        params.extend(phases)
                    query += " ORDER BY t.trial_id ASC LIMIT 1"

                    cur.execute(query, params)
                    row = cur.fetchone()
                    if not row:
                        return None

                    trial_id, trial_number, pipeline_json, round_num, phase, parent_trial_id = row

                    # Atomically update to RUNNING
                    cur.execute(
                        "UPDATE trials SET state=?, datetime_start=datetime('now') "
                        "WHERE trial_id=? AND state=?",
                        (TrialState.RUNNING.value, trial_id, TrialState.WAITING.value)
                    )

                    if cur.rowcount == 0:
                        # Someone else claimed it - retry
                        conn.rollback()
                        time.sleep(random.uniform(0.01, 0.1 * (2 ** attempt)))
                        continue

                    # Update worker_id
                    cur.execute(
                        "UPDATE rant_trials SET worker_id=? WHERE trial_id=?",
                        (worker_id, trial_id)
                    )

                    conn.commit()

                    return {
                        'trial_id': trial_id,
                        'trial_number': trial_number,
                        'pipeline': json.loads(pipeline_json),
                        'round': round_num,
                        'phase': phase,
                        'parent_trial_id': parent_trial_id
                    }

            except sqlite3.OperationalError as e:
                if "database is locked" in str(e):
                    time.sleep(random.uniform(0.1, 0.5 * (2 ** attempt)))
                else:
                    raise

        return None

    def store_trial_result(
        self, trial_id: int, state: TrialState, value: Optional[float] = None
    ):
        """Store trial result (COMPLETE or FAIL)."""
        with self._connect() as conn:
            cur = conn.cursor()

            cur.execute(
                "UPDATE trials SET state=?, datetime_complete=datetime('now') WHERE trial_id=?",
                (state.value, trial_id)
            )

            if value is not None:
                cur.execute(
                    "INSERT OR REPLACE INTO trial_values (trial_id, objective, value) VALUES (?, 0, ?)",
                    (trial_id, value)
                )

            conn.commit()

    def select_top_k_global(
        self,
        study_id: int,
        k: int,
        direction: StudyDirection,
        only_unexpanded: bool = True,
        max_per_architecture: int = 1,
    ) -> List[Dict[str, Any]]:
        """Select top-K trials from entire study history with diversity filtering.

        Args:
            study_id: Study ID
            k: Number of parents to select
            direction: MINIMIZE or MAXIMIZE
            only_unexpanded: If True, only select trials not yet expanded
            max_per_architecture: Max parents with same architecture_signature

        Returns:
            List of trial dicts with trial_id, pipeline_json, value
        """
        with self._connect() as conn:
            cur = conn.cursor()

            # Build query
            order = "DESC" if direction == StudyDirection.MAXIMIZE else "ASC"
            unexpanded_clause = "AND rt.expanded_round IS NULL" if only_unexpanded else ""

            query = f"""
                SELECT rt.trial_id, rt.pipeline_json, rt.architecture_signature, tv.value
                FROM rant_trials rt
                JOIN trials t ON rt.trial_id = t.trial_id
                JOIN trial_values tv ON rt.trial_id = tv.trial_id
                WHERE rt.study_id=? AND t.state=? {unexpanded_clause}
                ORDER BY tv.value {order}
            """

            cur.execute(query, (study_id, TrialState.COMPLETE.value))
            rows = cur.fetchall()

            # Apply diversity filtering (max per architecture)
            selected = []
            arch_counts: Dict[str, int] = {}

            for trial_id, pipeline_json, arch_sig, value in rows:
                if len(selected) >= k:
                    break

                count = arch_counts.get(arch_sig, 0)
                if count < max_per_architecture:
                    selected.append({
                        'trial_id': trial_id,
                        'pipeline': json.loads(pipeline_json),
                        'value': value,
                        'architecture_signature': arch_sig
                    })
                    arch_counts[arch_sig] = count + 1

            return selected

    def select_best_value(self, study_id: int, direction: StudyDirection) -> Optional[float]:
        """Select best value from completed trials."""
        with self._connect() as conn:
            cur = conn.cursor()
            order = "DESC" if direction == StudyDirection.MAXIMIZE else "ASC"
            cur.execute(
                f"""
                SELECT tv.value
                FROM trial_values tv
                JOIN trials t ON tv.trial_id = t.trial_id
                WHERE t.study_id=? AND t.state=?
                ORDER BY tv.value {order}
                LIMIT 1
                """,
                (study_id, TrialState.COMPLETE.value)
            )
            row = cur.fetchone()
            return row[0] if row else None

    def mark_expanded(self, trial_id: int, round_num: int):
        """Mark trial as expanded in given round."""
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE rant_trials SET expanded_round=? WHERE trial_id=?",
                (round_num, trial_id)
            )
            conn.commit()

    def count_trials(self, study_id: int, round_num: int, phase: Optional[str] = None) -> int:
        """Count trials in a round, optionally filtered by phase."""
        with self._connect() as conn:
            cur = conn.cursor()
            if phase:
                cur.execute(
                    "SELECT COUNT(*) FROM rant_trials WHERE study_id=? AND round=? AND phase=?",
                    (study_id, round_num, phase)
                )
            else:
                cur.execute(
                    "SELECT COUNT(*) FROM rant_trials WHERE study_id=? AND round=?",
                    (study_id, round_num)
                )
            return cur.fetchone()[0]

    def count_total_trials(self, study_id: int) -> int:
        """Count all trials for a study."""
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM trials WHERE study_id=?",
                (study_id,)
            )
            return cur.fetchone()[0]

    def mark_round_complete(self, study_id: int, round_num: int):
        """Mark round as complete."""
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE rant_rounds SET status=?, completed_at=datetime('now') WHERE study_id=? AND round=?",
                (RoundStatus.COMPLETE.value, study_id, round_num)
            )
            conn.commit()

    def get_stale_trials(self, study_id: int, stale_sec: int) -> List[int]:
        """Get trial IDs that have been RUNNING for too long."""
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT trial_id FROM trials "
                "WHERE study_id=? AND state=? "
                "AND datetime_start < datetime('now', ?)",
                (study_id, TrialState.RUNNING.value, f"-{stale_sec} seconds")
            )
            return [row[0] for row in cur.fetchall()]

    def requeue_trial(self, trial_id: int):
        """Reset trial to WAITING state."""
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE trials SET state=?, datetime_start=NULL, datetime_complete=NULL WHERE trial_id=?",
                (TrialState.WAITING.value, trial_id)
            )
            conn.commit()
