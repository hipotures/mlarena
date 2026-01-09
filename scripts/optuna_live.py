#!/usr/bin/env python3
"""Live Optuna SQLite monitor (read-only)."""

from __future__ import annotations

import argparse
import select
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

try:
    import termios
    import tty
except ImportError:
    termios = None
    tty = None

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
    
    is_running = False
    if not dt_end:
        dt_end = datetime.now()
        is_running = True

    delta = dt_end - dt_start
    seconds = max(0, int(delta.total_seconds()))
    
    if is_running:
        return f"{seconds}s"

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
    directions_map: Dict[int, List[str]] = {}
    
    # Try the standard Optuna 3.x+ table name first
    target_table = None
    if "study_directions" in tables:
        target_table = "study_directions"
    elif "directions" in tables:
        target_table = "directions"
        
    if target_table:
        for row in conn.execute(f"SELECT study_id, direction FROM {target_table} ORDER BY study_id, objective"):
            d = str(row["direction"]).upper()
            label = "max" if "MAX" in d else "min"
            directions_map.setdefault(row["study_id"], []).append(label)
    else:
        study_cols = _table_columns(conn, "studies")
        if "direction" in study_cols:
            for row in studies_rows:
                d = str(row["direction"]).upper()
                label = "max" if "MAX" in d else "min"
                directions_map[row["study_id"]] = [label]

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
        direction_label = ",".join(directions) if directions else "-"

        best_val = None
        best_trial = None
        if not multi_objective:
            if has_trial_values:
                order = "DESC" if directions and directions[0] == 1 else "ASC"
                sql = (
                    "SELECT t.trial_id AS number, ABS(tv.value) AS value "
                    "FROM trials t "
                    "JOIN trial_values tv ON tv.trial_id = t.trial_id "
                    "WHERE t.study_id=? AND (t.state=1 OR UPPER(CAST(t.state AS TEXT))='COMPLETE') "
                    f"ORDER BY ABS(tv.value) {order} "
                    "LIMIT 1"
                )
                best_row = conn.execute(sql, (study_id,)).fetchone()
                if best_row:
                    best_trial = best_row["number"]
                    best_val = best_row["value"]
            elif has_value_col:
                order = "DESC" if directions and directions[0] == 1 else "ASC"
                sql = (
                    "SELECT trial_id AS number, ABS(value) AS value FROM trials "
                    "WHERE study_id=? AND (state=1 OR UPPER(CAST(state AS TEXT))='COMPLETE') "
                    f"ORDER BY ABS(value) {order} "
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
            "SELECT t.trial_id, t.state, t.datetime_start, t.datetime_complete, "
            "GROUP_CONCAT(ABS(tv.value)) AS values_concat "
            "FROM trials t "
            "LEFT JOIN trial_values tv ON tv.trial_id = t.trial_id "
            "WHERE t.study_id=? "
            "GROUP BY t.trial_id "
            "ORDER BY t.trial_id DESC "
            "LIMIT ?"
        )
        rows = conn.execute(sql, (study_id, limit)).fetchall()
        result = []
        for row in rows:
            result.append(
                {
                    "number": row["trial_id"],
                    "state": row["state"],
                    "datetime_start": row["datetime_start"],
                    "datetime_complete": row["datetime_complete"],
                    "value": row["values_concat"],
                }
            )
        return result

    if has_value_col:
        sql = (
            "SELECT trial_id AS number, state, datetime_start, datetime_complete, ABS(value) AS value "
            "FROM trials "
            "WHERE study_id=? "
            "ORDER BY trial_id DESC "
            "LIMIT ?"
        )
        rows = conn.execute(sql, (study_id, limit)).fetchall()
        return [dict(row) for row in rows]

    sql = (
        "SELECT trial_id AS number, state, datetime_start, datetime_complete "
        "FROM trials "
        "WHERE study_id=? "
        "ORDER BY trial_id DESC "
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
    all_trials: List[Dict[str, Any]],
    *,
    interval: int,
    study_filter: Optional[str],
    active_study: Optional[Dict[str, Any]] = None,
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
    header.append("  ")
    header.append("[q/Esc/Ctrl-C to quit]", style="bold magenta")

    if error:
        body = Panel(Text(error, style="red"), title="Error", border_style="red")
        return Panel(Group(header, body), border_style="bright_blue", expand=False)

    studies_table = Table(title="Studies", box=box.SIMPLE, expand=False)
    studies_table.add_column("Name", style="cyan", no_wrap=True)
  #  studies_table.add_column("Dir", style="magenta", width=6)
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
        best_val = "-" if s["best_value"] is None else f"{s['best_value']:.5f}"
        best_trial = "-" if s["best_trial"] is None else str(s["best_trial"])
        studies_table.add_row(
            s["name"],
  #          s["direction"],
            str(total),
            str(counts.get(1, 0)),
            str(counts.get(2, 0)),
            str(counts.get(3, 0)),
            str(counts.get(0, 0)),
            best_val,
            best_trial,
        )

    # Filter and Sort Trials
    running_trials = []
    completed_trials = []
    
    for t in all_trials:
        state_key = _normalize_state(t.get("state"))
        if state_key == 0: # RUNNING
            running_trials.append(t)
        elif state_key == 1: # COMPLETE
            completed_trials.append(t)

    # Sort Running by number DESC, limit 8
    running_trials = sorted(running_trials, key=lambda x: x.get("number", 0), reverse=True)[:8]

    # Sort Top by value
    if active_study:
        direction = active_study.get("direction", "min")
        is_max = "max" in direction.lower()

        def _val_sort_key(t: Dict[str, Any]) -> float:
            v = t.get("value")
            try:
                if isinstance(v, str) and "," in v:
                    return float(v.split(",")[0])
                return float(v)
            except (ValueError, TypeError):
                return 0.0

        completed_trials = sorted(completed_trials, key=_val_sort_key, reverse=False)[:10]
        
        # Always try to find trial #0 and add it if not already in top 10
        trial_zero = next((t for t in all_trials if t.get("number") == 0 and _normalize_state(t.get("state")) == 1), None)
        if trial_zero and not any(t.get("number") == 0 for t in completed_trials):
            completed_trials.append(trial_zero)

    def _make_trial_table(title: str, trials: List[Dict[str, Any]], color: str = "yellow") -> Table:
        table = Table(title=title, box=box.SIMPLE, expand=False)
        table.add_column("#", justify="right", width=6, no_wrap=True)
        table.add_column("State", style=color, width=10, no_wrap=True)
        table.add_column("Value", justify="right", width=8, no_wrap=True)
        table.add_column("Duration", justify="right", width=12, no_wrap=True)
        table.add_column("Start", style="dim", width=20, no_wrap=True)
        return table

    def _fill_rows(table: Table, trials: List[Dict[str, Any]]):
        for t in trials:
            is_baseline = t.get("number") == 0
            cell_style = "reverse" if is_baseline else None
            
            state_key = _normalize_state(t.get("state"))
            state_name = STATE_MAP.get(state_key, str(t.get("state")))
            
            val = t.get("value")
            if val is None:
                val_str = "-"
            else:
                try:
                    if isinstance(val, str) and "," in val:
                        val_str = ",".join(f"{float(x):.5f}" for x in val.split(","))
                    else:
                        val_str = f"{float(val):.5f}"
                except:
                    val_str = str(val)

            num_renderable = str(t.get("number", "-"))
            state_renderable = state_name
            val_renderable = Text(val_str, style="reverse" if is_baseline else None)
            
            # Truncate microseconds for cleaner alignment
            start_time = str(t.get("datetime_start") or "-")
            if len(start_time) > 19:
                start_time = start_time[:19]

            table.add_row(
                num_renderable,
                state_renderable,
                val_renderable,
                _duration_str(t.get("datetime_start"), t.get("datetime_complete")),
                start_time
            )

    running_table = _make_trial_table("Running Trials (max 8)", running_trials, "cyan")
    _fill_rows(running_table, running_trials)

    top_table = _make_trial_table("Top Trials (max 10)", completed_trials, "green")
    _fill_rows(top_table, completed_trials)

    group = Group(header, studies_table, running_table, top_table)
    return Panel(group, border_style="bright_blue", expand=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Live Optuna SQLite monitor (read-only)")
    parser.add_argument("--db", required=True, help="Path to Optuna sqlite file")
    parser.add_argument("--interval", type=int, default=5, help="Refresh interval in seconds")
    parser.add_argument("--study", default=None, help="Study name to focus on")
    parser.add_argument("--limit", type=int, default=1000, help="Trial buffer size for sorting")
    args = parser.parse_args()

    db_path = Path(args.db).expanduser()
    if not db_path.exists():
        print(f"DB not found: {db_path}", file=sys.stderr)
        return 1

    console = Console()

    def _poll() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str]]:
        try:
            with _connect_read_only(db_path) as conn:
                studies = _fetch_studies(conn)
                study = _choose_study(studies, args.study)
                all_trials = []
                if study:
                    all_trials = _fetch_recent_trials(conn, study["study_id"], args.limit)
                return studies, all_trials, study, None
        except sqlite3.OperationalError as exc:
            return [], [], None, f"SQLite error: {exc}"
        except Exception as exc:
            return [], [], None, f"Error: {exc}"

    def _wait_for_exit(timeout: int) -> bool:
        """Returns True if exit key (q/Esc) pressed."""
        if not sys.stdin.isatty():
            time.sleep(timeout)
            return False
        
        start = time.time()
        while time.time() - start < timeout:
            dr, _, _ = select.select([sys.stdin], [], [], 0.1)
            if dr:
                key = sys.stdin.read(1)
                if key.lower() == "q" or key == "\x1b":
                    return True
            # Short sleep to prevent busy waiting if select returns instantly for some reason
            time.sleep(0.01)
        return False

    old_settings = None
    if sys.stdin.isatty() and termios and tty:
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    try:
        with Live(console=console, refresh_per_second=4) as live:
            while True:
                studies, all_trials, active_study, err = _poll()
                live.update(
                    _render_dashboard(
                        db_path,
                        studies,
                        all_trials,
                        interval=args.interval,
                        study_filter=args.study,
                        active_study=active_study,
                        error=err,
                    )
                )
                if _wait_for_exit(max(1, int(args.interval))):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
