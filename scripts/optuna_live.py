#!/usr/bin/env python3
"""Live Optuna SQLite monitor (read-only)."""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

STATE_MAP = {
    0: "RUNNING",
    1: "COMPLETE",
    2: "PRUNED",
    3: "FAIL",
    4: "WAITING",
}

STATE_NAME_TO_CODE = {name: code for code, name in STATE_MAP.items()}


def _normalize_state(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode()
        except Exception:
            return value
    if isinstance(value, str):
        raw = value.strip()
        if raw.isdigit():
            return int(raw)
        key = raw.upper()
        return STATE_NAME_TO_CODE.get(key, raw)
    return value


def _connect_read_only(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path}?mode=ro&cache=shared"
    conn = sqlite3.connect(uri, uri=True, timeout=1, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA query_only = 1")
    except sqlite3.Error:
        pass
    return conn


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _duration_str(start: Optional[str], end: Optional[str]) -> str:
    dt_start = _parse_dt(start)
    dt_end = _parse_dt(end)
    if not dt_start:
        return "-"
    if not dt_end:
        return "running"
    delta = dt_end - dt_start
    seconds = max(0, int(delta.total_seconds()))
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours}h{minutes:02d}m"


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    cols = set()
    for row in conn.execute(f"PRAGMA table_info({table})"):
        cols.add(row["name"])
    return cols


def _fetch_studies(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "studies" not in tables or "trials" not in tables:
        return []

    studies_rows = conn.execute("SELECT study_id, study_name FROM studies").fetchall()
    directions_map: Dict[int, List[int]] = {}
    if "directions" in tables:
        for row in conn.execute("SELECT study_id, direction FROM directions ORDER BY study_id, objective"):
            directions_map.setdefault(row["study_id"], []).append(int(row["direction"]))
    else:
        study_cols = _table_columns(conn, "studies")
        if "direction" in study_cols:
            for row in studies_rows:
                directions_map[row["study_id"]] = [int(row["direction"])]

    trials_cols = _table_columns(conn, "trials")
    has_value_col = "value" in trials_cols
    has_trial_values = "trial_values" in tables

    multi_objective = False
    if has_trial_values:
        row = conn.execute(
            "SELECT MAX(cnt) AS max_cnt FROM (SELECT COUNT(*) AS cnt FROM trial_values GROUP BY trial_id)"
        ).fetchone()
        if row and row["max_cnt"] and row["max_cnt"] > 1:
            multi_objective = True

    studies: List[Dict[str, Any]] = []
    for row in studies_rows:
        study_id = row["study_id"]
        counts = {state: 0 for state in STATE_MAP}
        for c in conn.execute(
            "SELECT state, COUNT(*) AS cnt FROM trials WHERE study_id=? GROUP BY state",
            (study_id,),
        ):
            state_key = _normalize_state(c["state"])
            if isinstance(state_key, int):
                counts[state_key] = int(c["cnt"])

        last_row = conn.execute(
            "SELECT MAX(datetime_start) AS last_start FROM trials WHERE study_id=?",
            (study_id,),
        ).fetchone()
        last_start = last_row["last_start"] if last_row else None

        max_trial_row = conn.execute(
            "SELECT MAX(number) AS max_num FROM trials WHERE study_id=?",
            (study_id,),
        ).fetchone()
        max_num = max_trial_row["max_num"] if max_trial_row else None

        directions = directions_map.get(study_id, [])
        direction_label = ",".join("max" if d == 1 else "min" for d in directions) if directions else "-"

        best_val = None
        best_trial = None
        if not multi_objective:
            if has_trial_values:
                order = "DESC" if directions and directions[0] == 1 else "ASC"
                sql = (
                    "SELECT t.number AS number, tv.value AS value "
                    "FROM trials t "
                    "JOIN trial_values tv ON tv.trial_id = t.trial_id "
                    "WHERE t.study_id=? AND (t.state=1 OR UPPER(CAST(t.state AS TEXT))='COMPLETE') "
                    f"ORDER BY tv.value {order} "
                    "LIMIT 1"
                )
                best_row = conn.execute(sql, (study_id,)).fetchone()
                if best_row:
                    best_trial = best_row["number"]
                    best_val = best_row["value"]
            elif has_value_col:
                order = "DESC" if directions and directions[0] == 1 else "ASC"
                sql = (
                    "SELECT number, value FROM trials "
                    "WHERE study_id=? AND (state=1 OR UPPER(CAST(state AS TEXT))='COMPLETE') "
                    f"ORDER BY value {order} "
                    "LIMIT 1"
                )
                best_row = conn.execute(sql, (study_id,)).fetchone()
                if best_row:
                    best_trial = best_row["number"]
                    best_val = best_row["value"]

        studies.append(
            {
                "study_id": study_id,
                "name": row["study_name"],
                "direction": direction_label,
                "counts": counts,
                "last_start": last_start,
                "max_trial": max_num,
                "best_value": best_val,
                "best_trial": best_trial,
                "multi_objective": multi_objective,
            }
        )
    return studies


def _fetch_recent_trials(
    conn: sqlite3.Connection, study_id: int, limit: int
) -> List[Dict[str, Any]]:
    tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    trials_cols = _table_columns(conn, "trials")
    has_value_col = "value" in trials_cols
    has_trial_values = "trial_values" in tables

    rows = []
    if has_trial_values:
        sql = (
            "SELECT t.number, t.state, t.datetime_start, t.datetime_complete, "
            "GROUP_CONCAT(tv.value) AS values_concat "
            "FROM trials t "
            "LEFT JOIN trial_values tv ON tv.trial_id = t.trial_id "
            "WHERE t.study_id=? "
            "GROUP BY t.trial_id "
            "ORDER BY t.number DESC "
            "LIMIT ?"
        )
        rows = conn.execute(sql, (study_id, limit)).fetchall()
        result = []
        for row in rows:
            result.append(
                {
                    "number": row["number"],
                    "state": row["state"],
                    "datetime_start": row["datetime_start"],
                    "datetime_complete": row["datetime_complete"],
                    "value": row["values_concat"],
                }
            )
        return result

    if has_value_col:
        sql = (
            "SELECT number, state, datetime_start, datetime_complete, value "
            "FROM trials "
            "WHERE study_id=? "
            "ORDER BY number DESC "
            "LIMIT ?"
        )
        rows = conn.execute(sql, (study_id, limit)).fetchall()
        return [dict(row) for row in rows]

    sql = (
        "SELECT number, state, datetime_start, datetime_complete "
        "FROM trials "
        "WHERE study_id=? "
        "ORDER BY number DESC "
        "LIMIT ?"
    )
    rows = conn.execute(sql, (study_id, limit)).fetchall()
    return [dict(row) for row in rows]


def _choose_study(studies: List[Dict[str, Any]], name: Optional[str]) -> Optional[Dict[str, Any]]:
    if not studies:
        return None
    if name:
        for study in studies:
            if study["name"] == name:
                return study
        return None
    if len(studies) == 1:
        return studies[0]
    # pick the most recent (by last_start)
    def _key(item: Dict[str, Any]) -> str:
        return item.get("last_start") or ""

    return sorted(studies, key=_key, reverse=True)[0]


def _render_dashboard(
    db_path: Path,
    studies: List[Dict[str, Any]],
    recent_trials: List[Dict[str, Any]],
    *,
    interval: int,
    study_filter: Optional[str],
    error: Optional[str],
) -> Panel:
    header = Text()
    header.append("Optuna Live Monitor", style="bold cyan")
    header.append("  ")
    header.append(str(db_path), style="white")
    header.append("  ")
    header.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), style="dim")
    header.append(f"  interval={interval}s", style="dim")
    if study_filter:
        header.append(f"  study={study_filter}", style="dim")

    if error:
        body = Panel(Text(error, style="red"), title="Error", border_style="red")
        return Panel(Group(header, body), border_style="bright_blue")

    studies_table = Table(title="Studies", box=box.SIMPLE, expand=True)
    studies_table.add_column("Name", style="cyan", no_wrap=True)
    studies_table.add_column("Dir", style="magenta", width=6)
    studies_table.add_column("Trials", justify="right")
    studies_table.add_column("Complete", justify="right")
    studies_table.add_column("Pruned", justify="right")
    studies_table.add_column("Fail", justify="right")
    studies_table.add_column("Running", justify="right")
    studies_table.add_column("Best", justify="right")
    studies_table.add_column("Best#", justify="right")

    for s in studies:
        counts = s["counts"]
        total = sum(counts.values())
        best_val = "-" if s["best_value"] is None else f"{s['best_value']:.6g}"
        best_trial = "-" if s["best_trial"] is None else str(s["best_trial"])
        studies_table.add_row(
            s["name"],
            s["direction"],
            str(total),
            str(counts.get(1, 0)),
            str(counts.get(2, 0)),
            str(counts.get(3, 0)),
            str(counts.get(0, 0)),
            best_val,
            best_trial,
        )

    trials_table = Table(title="Recent Trials", box=box.SIMPLE, expand=True)
    trials_table.add_column("#", justify="right")
    trials_table.add_column("State", style="yellow")
    trials_table.add_column("Value", justify="right")
    trials_table.add_column("Duration", justify="right")
    trials_table.add_column("Start", style="dim")
    trials_table.add_column("End", style="dim")

    for t in recent_trials:
        state_key = _normalize_state(t.get("state"))
        state_name = STATE_MAP.get(state_key, str(t.get("state")))
        val = t.get("value")
        if isinstance(val, float):
            val_str = f"{val:.6g}"
        elif val is None:
            val_str = "-"
        else:
            val_str = str(val)
        trials_table.add_row(
            str(t.get("number", "-")),
            state_name,
            val_str,
            _duration_str(t.get("datetime_start"), t.get("datetime_complete")),
            str(t.get("datetime_start") or "-"),
            str(t.get("datetime_complete") or "-"),
        )

    group = Group(header, studies_table, trials_table)
    return Panel(group, border_style="bright_blue")


def main() -> int:
    parser = argparse.ArgumentParser(description="Live Optuna SQLite monitor (read-only)")
    parser.add_argument("--db", required=True, help="Path to Optuna sqlite file")
    parser.add_argument("--interval", type=int, default=5, help="Refresh interval in seconds")
    parser.add_argument("--study", default=None, help="Study name to focus on")
    parser.add_argument("--limit", type=int, default=10, help="Number of recent trials to show")
    args = parser.parse_args()

    db_path = Path(args.db).expanduser()
    if not db_path.exists():
        print(f"DB not found: {db_path}", file=sys.stderr)
        return 1

    console = Console()

    def _poll() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[str]]:
        try:
            with _connect_read_only(db_path) as conn:
                studies = _fetch_studies(conn)
                study = _choose_study(studies, args.study)
                recent = []
                if study:
                    recent = _fetch_recent_trials(conn, study["study_id"], args.limit)
                return studies, recent, None
        except sqlite3.OperationalError as exc:
            return [], [], f"SQLite error: {exc}"
        except Exception as exc:
            return [], [], f"Error: {exc}"

    with Live(console=console, refresh_per_second=4) as live:
        while True:
            studies, recent, err = _poll()
            live.update(
                _render_dashboard(
                    db_path,
                    studies,
                    recent,
                    interval=args.interval,
                    study_filter=args.study,
                    error=err,
                )
            )
            time.sleep(max(1, int(args.interval)))


if __name__ == "__main__":
    raise SystemExit(main())
