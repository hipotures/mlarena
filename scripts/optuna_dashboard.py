#!/usr/bin/env python3
"""
Optuna Live Dashboard using Textual.
Monitors Optuna studies and allows drill-down into trial artifacts.
"""

import argparse
import hashlib
import json
import sqlite3
import sys
import time
import os
import logging
import gzip
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import requests
from dotenv import load_dotenv
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Label,
    Static,
    TabbedContent,
    TabPane,
    Tree,
    Log,
    LoadingIndicator,
)
from textual.widgets.tree import TreeNode

# Configure logging
logging.basicConfig(
    filename="dashboard.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w"
)

# Bump when UI changes to verify the running file
DASHBOARD_VERSION = "1.1"

# Load environment variables
load_dotenv()

# Telegram notification setup
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
API_BASE = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}" if TELEGRAM_TOKEN else None

def send_telegram_notification(message: str) -> None:
    """Send message to Telegram."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_notification": False,
    }
    try:
        requests.post(f"{API_BASE}/sendMessage", json=payload, timeout=5)
    except Exception:
        pass


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
    if isinstance(value, str):
        raw = value.strip()
        if raw.isdigit():
            return int(raw)
        key = raw.upper()
        if key in STATE_NAME_TO_CODE:
            return STATE_NAME_TO_CODE[key]
        if "." in key:
            tail = key.rsplit(".", 1)[-1]
            return STATE_NAME_TO_CODE.get(tail, raw)
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


class DashboardScreen(Screen):
    BINDINGS = [
        ("q", "app.quit", "Quit"),
        ("c", "copy_dashboard", "Copy All"),
        ("ctrl+insert", "copy_dashboard", "Copy All"),
    ]

    def __init__(self, db_path: Path, project_root: Path, study_name: Optional[str] = None):
        super().__init__()
        self.db_path = db_path
        self.project_root = project_root
        self.target_study_name = study_name
        self.active_study_name = study_name
        self.last_best_val = None
        self.first_run = True
        self.last_trial_row_key = None

    def action_copy_dashboard(self) -> None:
        """Serializes the entire dashboard state to text and copies to clipboard."""
        try:
            lines = []
            lines.append(f"Optuna Dashboard Export - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"Database: {self.db_path}")
            lines.append("-" * 40)
            
            # Study Stats
            header = str(self.query_one("#study_header").content)
            body = str(self.query_one("#study_body").content)
            
            # Simple markup removal
            def clean_markup(t):
                return t.replace("[bold]", "").replace("[/bold]", "").replace("[bold red]", "").replace("[/bold red]", "").replace("[bold yellow]", "").replace("[/bold yellow]", "")

            lines.append(clean_markup(header))
            lines.append(clean_markup(body))
            lines.append("-" * 40)
            
            # Trials Table
            table = self.query_one("#trials_table")
            lines.append("TRIALS TABLE")
            
            # Header
            col_labels = [col.label for col in table.columns.values()]
            lines.append(" | ".join(map(str, col_labels)))
            lines.append("-" * 80)
            
            # Rows
            for row_idx in range(table.row_count):
                row_data = [table.get_cell_at((row_idx, col_idx)) for col_idx in range(len(table.columns))]
                lines.append(" | ".join(map(str, row_data)))
            
            final_text = "\n".join(lines)
            self.app.copy_to_clipboard(final_text)
            self.app.notify("Dashboard summary copied to clipboard!")
        except Exception as e:
            self.app.notify(f"Failed to copy dashboard: {e}", severity="error")

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Container(
            Label(f"Database: {self.db_path}", id="db_label"),
            Horizontal(
                Vertical(
                    Static(id="study_header"),
                    Static(id="study_body"),
                    id="study_stats",
                    classes="box"
                ),
                DataTable(
                    id="trials_table",
                    cursor_type="row",
                    show_row_labels=False,
                    zebra_stripes=False,
                    classes="box"
                ),
                id="row1"
            ),
            id="main_container"
        )
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#trials_table")
        table.add_column("#", width=6)
        table.add_column("State", width=10)
        table.add_column("Local CV", width=10)
        table.add_column("Duration", width=12)
        table.add_column("CfgHash", width=10)
        table.add_column("Start", width=19)
        self.set_interval(5, self.update_data)
        # Ensure data is loaded immediately after the first refresh pass
        self.call_after_refresh(self.update_data)

    def update_data(self) -> None:
        try:
            if not self.db_path.exists():
                msg = f"Database not found at:\n{self.db_path}"
                self.query_one("#study_header").update("[bold red]⚠ DB Missing[/bold red]")
                self.query_one("#study_body").update(msg)
                logging.warning(msg)
                return

            with _connect_read_only(self.db_path) as conn:
                studies = self._fetch_studies(conn)
                logging.debug(f"Fetched {len(studies)} studies")
                
                study = self._choose_study(studies, self.target_study_name)
                
                if not study:
                    msg = "No study found in database."
                    if self.target_study_name:
                        msg += f"\nTarget: {self.target_study_name}"
                    self.query_one("#study_header").update("[bold yellow]❓ No Study[/bold yellow]")
                    self.query_one("#study_body").update(msg)
                    logging.warning(msg)
                    return

                # Update active study name
                self.active_study_name = study["name"]

                logging.debug(f"Selected study: {study['name']} (ID: {study['study_id']})")
                all_trials = self._fetch_recent_trials(conn, study["study_id"], 1000)
                logging.debug(f"Fetched {len(all_trials)} trials")
                
                self._update_ui(study, all_trials)
                self._check_best_score(study)
        except Exception as e:
            err_msg = f"Error in update_data: {e}"
            logging.error(err_msg, exc_info=True)
            try:
                self.query_one("#study_header").update("[bold red]❌ Error[/bold red]")
                self.query_one("#study_body").update(str(e))
            except:
                pass

    def _update_ui(self, study: Dict[str, Any], trials: List[Dict[str, Any]]) -> None:
        # Update Study Stats
        best_val = study.get("best_value")
        if isinstance(best_val, (int, float)):
            best_val_str = f"{best_val:.5f}"
        elif best_val is None:
            best_val_str = "-"
        else:
            best_val_str = str(best_val)
        header_text = f"[bold]Study:[/bold] {study['name']}"
        body_text = (
            f"[bold]Direction:[/bold] {study.get('direction', 'MINIMIZE')}\n"
            f"[bold]Total:[/bold] {sum(study['counts'].values())}\n"
            f"[bold]Running:[/bold] {study['counts'].get(0, 0)}\n"
            f"[bold]Complete:[/bold] {study['counts'].get(1, 0)}\n"
            f"[bold]Waiting:[/bold] {study['counts'].get(4, 0)}\n"
            f"[bold]Fail/Pruned:[/bold] {study['counts'].get(3, 0)}/{study['counts'].get(2, 0)}\n"
            f"[bold]Best Value:[/bold] {best_val_str}\n"
            f"[bold]Best Trial:[/bold] {study['best_trial']}"
        )
        self.query_one("#study_header").update(header_text)
        self.query_one("#study_body").update(body_text)

        # Update Trials Table (Running + Top Completed)
        trials_table = self.query_one("#trials_table")
        prev_row_key = self.last_trial_row_key
        trials_table.clear()
        
        # Calculate dynamic limit based on actual table height
        try:
            table_widget = self.query_one("#trials_table")
            h = table_widget.content_size.height
            # Use a slightly more aggressive fallback or actual height
            available_height = h - 1 if h > 0 else (self.size.height - 7)
        except:
            available_height = 25
            
        if available_height < 5: available_height = 5

        running_trials = [
            t for t in trials
            if _normalize_state(t.get("state")) in (0, 4)
        ]
        running_trials.sort(key=lambda x: x["number"], reverse=True)
        
        for t in running_trials:
            state_code = _normalize_state(t.get("state"))
            state_label = STATE_MAP.get(state_code, str(t.get("state")))
            start_str = str(t["datetime_start"]) if t["datetime_start"] else "-"
            if len(start_str) > 19:
                start_str = start_str[:19]
            trials_table.add_row(
                str(t["number"]),
                state_label,
                "-",
                _duration_str(t["datetime_start"], None),
                str(t.get("params_hash", "-")),
                start_str,
                key=str(t["number"])
            )

        # How many slots left for completed trials?
        slots_for_completed = max(5, available_height - len(running_trials))

        completed_trials = [t for t in trials if _normalize_state(t.get("state")) == 1]

        direction = (study.get("direction") or "MINIMIZE").upper()
        reverse = direction == "MAXIMIZE"

        def _val_key(t):
            v = t.get("value")
            try:
                return float(v)
            except Exception:
                return float("-inf") if reverse else float("inf")

        completed_trials.sort(key=_val_key, reverse=reverse)

        for t in completed_trials[:slots_for_completed]:
            val_str = f"{float(t['value']):.5f}" if t['value'] is not None else "-"
            start_str = str(t["datetime_start"]) if t["datetime_start"] else "-"
            if len(start_str) > 19:
                start_str = start_str[:19]
            trials_table.add_row(
                str(t["number"]),
                "COMPLETE",
                val_str,
                _duration_str(t["datetime_start"], t["datetime_complete"]),
                str(t.get("params_hash", "-")),
                start_str,
                key=str(t["number"]) # Store number in key for selection
            )

        if prev_row_key is not None:
            try:
                row_index = trials_table._row_locations.get(prev_row_key)
                if row_index is not None:
                    trials_table.move_cursor(row=row_index, column=0)
            except Exception:
                pass

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id in ("trials_table",):
            trial_id = event.row_key.value
            if trial_id:
                self.app.push_screen(TrialInspector(self.project_root, self.active_study_name, trial_id))

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id == "trials_table":
            self.last_trial_row_key = event.row_key

    def _check_best_score(self, study: Dict[str, Any]) -> None:
        current_best = study.get("best_value")
        if not self.first_run and current_best is not None and self.last_best_val is not None:
            if current_best != self.last_best_val:
                self.app.bell()
                
                project_name = self.project_root.name if self.project_root else "Unknown"
                score_fmt = f"{current_best:.5f}" if isinstance(current_best, (int, float)) else str(current_best)
                prev_fmt = f"{self.last_best_val:.5f}" if isinstance(self.last_best_val, (int, float)) else str(self.last_best_val)
                
                msg = (
                    f"🚀 <b>New Best Score!</b>\n\n"
                    f"<b>Project:</b> {project_name}\n"
                    f"<b>Study:</b> {study['name']}\n"
                    f"<b>Score:</b> {score_fmt}\n"
                    f"<b>Previous:</b> {prev_fmt}"
                )
                send_telegram_notification(msg)
        
        if current_best is not None:
            self.last_best_val = current_best
        self.first_run = False

    # --- Data Fetching Methods (Adapted from optuna_live.py) ---
    def _table_columns(self, conn: sqlite3.Connection, table: str) -> set[str]:
        cols = set()
        for row in conn.execute(f"PRAGMA table_info({table})"):
            cols.add(row["name"])
        return cols

    def _fetch_studies(self, conn: sqlite3.Connection) -> List[Dict[str, Any]]:
        tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "studies" not in tables or "trials" not in tables:
            return []

        studies_rows = conn.execute("SELECT study_id, study_name FROM studies").fetchall()
        
        trials_cols = self._table_columns(conn, "trials")
        has_value_col = "value" in trials_cols
        has_trial_values = "trial_values" in tables

        direction_by_study = {}
        if "study_directions" in tables:
            for row in conn.execute("SELECT study_id, direction FROM study_directions"):
                direction_by_study[row["study_id"]] = row["direction"]

        studies: List[Dict[str, Any]] = []
        for row in studies_rows:
            study_id = row["study_id"]
            raw_dir = direction_by_study.get(study_id, "MINIMIZE")
            # Convert Optuna enum code to text if needed
            if str(raw_dir) == "1": direction = "MAXIMIZE"
            elif str(raw_dir) == "0": direction = "MINIMIZE"
            else: direction = str(raw_dir).upper()

            order = "DESC" if direction == "MAXIMIZE" else "ASC"
            counts = {state: 0 for state in STATE_MAP}
            for c in conn.execute(
                "SELECT state, COUNT(*) AS cnt FROM trials WHERE study_id=? GROUP BY state",
                (study_id,),
            ):
                state_key = _normalize_state(c["state"])
                if isinstance(state_key, int):
                    counts[state_key] = int(c["cnt"])

            best_val = None
            best_trial = None
            
            if has_trial_values:
                sql = (
                    "SELECT t.number, tv.value "
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
                    "counts": counts,
                    "best_value": best_val,
                    "best_trial": best_trial,
                    "direction": direction_by_study.get(study_id, "MINIMIZE"),
                }
            )
        return studies

    def _choose_study(self, studies: List[Dict[str, Any]], name: Optional[str]) -> Optional[Dict[str, Any]]:
        if not studies:
            return None
        if name:
            for s in studies:
                if s["name"] == name:
                    return s
            # Fallback to fuzzy match or first
            for s in studies:
                if name in s["name"]:
                    return s
        # Return last created/updated? Or just first
        return studies[-1] if studies else None

    def _fetch_recent_trials(self, conn: sqlite3.Connection, study_id: int, limit: int) -> List[Dict[str, Any]]:
        # limit is ignored in favor of specific 50/ALL split
        tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        has_trial_values = "trial_values" in tables

        # Get direction for sorting Top Completed
        raw_dir = "MINIMIZE"
        if "study_directions" in tables:
            d_row = conn.execute("SELECT direction FROM study_directions WHERE study_id=?", (study_id,)).fetchone()
            if d_row:
                raw_dir = d_row["direction"]
        
        # Robust direction mapping
        if str(raw_dir) == "1": direction = "MAXIMIZE"
        elif str(raw_dir) == "0": direction = "MINIMIZE"
        else: direction = str(raw_dir).upper()
        
        order = "DESC" if direction == "MAXIMIZE" else "ASC"

        result = []
        
        # 1. Fetch ALL RUNNING/WAITING trials
        sql_running = (
            "SELECT t.trial_id, t.number, t.state, t.datetime_start, t.datetime_complete, NULL as value "
            "FROM trials t "
            "WHERE t.study_id=? AND (t.state=0 OR UPPER(CAST(t.state AS TEXT))='RUNNING' OR t.state=4 OR UPPER(CAST(t.state AS TEXT))='WAITING') "
            "ORDER BY t.number DESC"
        )
        rows_running = conn.execute(sql_running, (study_id,)).fetchall()
        result.extend([dict(r) for r in rows_running])

        # 2. Fetch TOP COMPLETE trials (limit 50)
        if has_trial_values:
            sql_complete = (
                "SELECT t.trial_id, t.number, t.state, t.datetime_start, t.datetime_complete, tv.value "
                "FROM trials t "
                "JOIN trial_values tv ON tv.trial_id = t.trial_id "
                "WHERE t.study_id=? AND (t.state=1 OR UPPER(CAST(t.state AS TEXT))='COMPLETE') "
                f"ORDER BY tv.value {order} LIMIT 50"
            )
        else:
            sql_complete = (
                "SELECT t.trial_id, t.number, t.state, t.datetime_start, t.datetime_complete, t.value "
                "FROM trials t "
                "WHERE t.study_id=? AND (t.state=1 OR UPPER(CAST(t.state AS TEXT))='COMPLETE') "
                f"ORDER BY t.value {order} LIMIT 50"
            )
        
        rows_complete = conn.execute(sql_complete, (study_id,)).fetchall()
        # Deduplicate by number
        seen_nums = {r["number"] for r in result}
        for r in rows_complete:
            if r["number"] not in seen_nums:
                result.append(dict(r))
                seen_nums.add(r["number"])

        # Enrich with params hash
        if "trial_params" in tables and result:
            trial_ids = [r["trial_id"] for r in result if r.get("trial_id") is not None]
            if not trial_ids:
                return result
                
            ids_str = ",".join(map(str, trial_ids))
            try:
                params_rows = conn.execute(
                    f"SELECT trial_id, param_name, param_value FROM trial_params WHERE trial_id IN ({ids_str})"
                ).fetchall()
                
                params_by_trial = {}
                for pr in params_rows:
                    tid = pr["trial_id"]
                    if tid not in params_by_trial:
                        params_by_trial[tid] = {}
                    params_by_trial[tid][pr["param_name"]] = pr["param_value"]
                
                for r in result:
                    tid = r.get("trial_id")
                    p = params_by_trial.get(tid, {})
                    if p:
                        s = json.dumps(p, sort_keys=True)
                        h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
                        r["params_hash"] = h
                    else:
                        r["params_hash"] = "-"
            except Exception:
                for r in result:
                    r["params_hash"] = "-"
        else:
            for r in result:
                r["params_hash"] = "-"

        return result


class TrialInspector(Screen):
    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
        ("c", "copy_tree", "Copy Branch"),
        ("ctrl+insert", "copy_tree", "Copy Branch"),
    ]
    # Full perimeter spinner including bottom dots (7, 8) for maximum height
    SPINNER_FRAMES = ["⠁", "⠈", "⠐", "⠠", "⢀", "⡀", "⠄", "⠂"]

    def __init__(self, project_root: Path, study_name: str, trial_id: str):
        super().__init__()
        self.project_root = project_root
        self.study_name = study_name
        self.trial_id = int(trial_id)
        self.trial_dir = self._find_trial_dir()
        self.refresh_timer = None
        self.state_cache = {}  # Cache for state.json contents
        self.spinner_idx = 0
        self._last_data_refresh = 0.0
        self._last_full_rebuild = 0
        self._running_nodes = []  # List of (node, base_label) for animation

    def _animate_running_nodes(self) -> None:
        """Updates labels of running nodes with the current spinner frame."""
        frame = self.SPINNER_FRAMES[self.spinner_idx % len(self.SPINNER_FRAMES)]
        for node, base_label in self._running_nodes:
            try:
                node.set_label(f"{base_label} {frame}")
            except:
                pass

    def action_copy_tree(self) -> None:
        """Serializes the selected node structure to text and copies to clipboard with proper branch characters."""
        try:
            tree = self.query_one("#flow_tree")
            start_node = tree.cursor_node if tree.cursor_node else tree.root
            lines = []

            def walk(node: TreeNode, prefix: str = "", is_last: bool = True, is_root: bool = True):
                label = str(node.label)
                # Clean up icons, spinners and bold tags
                clean_label = label.replace("📁 ", "").replace("📐 ", "").replace("⚙ ", "").replace("📊 ", "").replace("📄 ", "")
                clean_label = clean_label.replace("[bold]", "").replace("[/bold]", "")
                for frame in self.SPINNER_FRAMES:
                    clean_label = clean_label.replace(f" {frame}", "")

                if is_root:
                    lines.append(clean_label)
                    new_prefix = ""
                else:
                    connector = "└── " if is_last else "├── "
                    lines.append(f"{prefix}{connector}{clean_label}")
                    new_prefix = prefix + ("    " if is_last else "│   ")

                children = list(node.children)
                for i, child in enumerate(children):
                    walk(child, new_prefix, is_last=(i == len(children) - 1), is_root=False)

            walk(start_node)
            tree_text = "\n".join(lines)
            self.app.copy_to_clipboard(tree_text)
            self.app.notify("Tree structure copied to clipboard!")
        except Exception as e:
            self.app.notify(f"Failed to copy tree: {e}", severity="error")

    def _find_trial_dir(self) -> Optional[Path]:
        # Search pattern: experiments/optuna_<study_name>/trial_<id>
        # trial_id here is actually the 0-based 'number'
        
        # Try exact match first
        base = self.project_root / "experiments" / f"optuna_{self.study_name}" / f"trial_{self.trial_id:04d}"
        
        try:
            resolved = base.resolve()
            logging.debug(f"Checking path: {base} -> {resolved} (Exists: {base.exists()})")
        except Exception as e:
            logging.debug(f"Resolution failed for {base}: {e}")

        if base.exists():
            return base
            
        # Try listing experiments dir and finding one that matches
        exp_root = self.project_root / "experiments"
        logging.debug(f"Searching in exp_root: {exp_root} (Exists: {exp_root.exists()})")
        
        if exp_root.exists():
            for d in exp_root.iterdir():
                if d.name.startswith("optuna_") and self.study_name in d.name:
                    candidate = d / f"trial_{self.trial_id:04d}"
                    try:
                        res_cand = candidate.resolve()
                        logging.debug(f"Checking candidate: {candidate} -> {res_cand} (Exists: {candidate.exists()})")
                    except:
                        pass
                        
                    if candidate.exists():
                        return candidate
        return None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Horizontal(
                Label(f"Trial {self.trial_id} Inspector", classes="title"),
                LoadingIndicator(id="loading_spinner"),
                id="inspector_header"
            ),
            Label(f"Path: {self.trial_dir}", classes="subtitle"),
            Tree("Pipeline Flow", id="flow_tree"),
            id="inspector_container"
        )
        yield Footer()

    def on_mount(self) -> None:
        # Spinner refresh is frequent; heavy I/O throttled inside _build_tree
        self.refresh_timer = self.set_interval(0.1, self._build_tree)
        self._build_tree()

    def _build_tree(self) -> None:
        self.spinner_idx += 1

        tree = self.query_one("#flow_tree")
        current_time = time.time()
        if current_time - self._last_data_refresh <= 1.0:
            self._animate_running_nodes()
            return
        self._last_data_refresh = current_time

        tree.clear()
        root = tree.root
        root.expand()
        self._running_nodes = []
        
        if not self.trial_dir or not self.trial_dir.exists():
            # Try finding it again
            self.trial_dir = self._find_trial_dir()
        
        if not self.trial_dir or not self.trial_dir.exists():
            root.add("[dim]Trial directory initializing...[/dim]")
            return

        # 1. Pipeline Steps (numbered dirs)
        steps = sorted([d for d in self.trial_dir.iterdir() if d.is_dir() and d.name[0].isdigit()], key=lambda x: int(x.name.split("-")[0]))
        
        pre_node = root.add("[bold]Preprocessing[/bold]", expand=True)
        
        for step_dir in steps:
            step_node = pre_node.add(f"{step_dir.name}", expand=False)
            self._add_step_details(step_node, step_dir)

        # 2. Metrics
        metrics_file = self.trial_dir / "metrics.json"
        if metrics_file.exists():
            # Stop refreshing if metrics found
            if self.refresh_timer:
                self.refresh_timer.stop()
                self.refresh_timer = None
            
            # Hide spinner
            try:
                self.query_one("#loading_spinner").display = False
            except:
                pass

            try:
                data = json.loads(metrics_file.read_text())
                m_node = root.add("[bold]Metrics[/bold]", expand=True)
                for k, v in data.items():
                    if isinstance(v, float):
                        if k in ["total_sec", "preprocess_sec"]:
                            val_str = f"{v:.1f}"
                        else:
                            val_str = f"{v:.5f}"
                    else:
                        val_str = str(v)
                    m_node.add(f"{k}: {val_str}", allow_expand=False)
            except:
                pass

        # 3. Model
        model_dir = self.trial_dir / "model" # Or optuna_model
        if not model_dir.exists():
            model_dir = self.trial_dir / "optuna_model"
        
        if model_dir.exists():
            mod_node = root.add("[bold]Model[/bold]", expand=True)
            self._add_step_details(mod_node, model_dir)

    def _add_step_details(self, node: TreeNode, step_dir: Path) -> None:
        state_path = step_dir / "state.json"
        if not state_path.exists():
            node.add("No state.json found", allow_expand=False)
            return

        try:
            # 0. Caching logic
            mtime = state_path.stat().st_mtime
            cache_key = str(state_path)
            if cache_key in self.state_cache and self.state_cache[cache_key]["mtime"] == mtime:
                state = self.state_cache[cache_key]["data"]
            else:
                state = json.loads(state_path.read_text())
                self.state_cache[cache_key] = {"mtime": mtime, "data": state}

            # Find the most relevant module (preprocess or model)
            modules = state.get("modules", {})
            mod_name = "preprocess" if "preprocess" in modules else "model" if "model" in modules else list(modules.keys())[-1] if modules else "unknown"
            
            mod_info = modules.get(mod_name, {})
            status = mod_info.get("status", "unknown")
            
            if status == "failed":
                node.set_label(f"{node.label} [FAILED]")
            elif status == "running":
                base_label = str(node.label)
                self._running_nodes.append((node, base_label))
                frame = self.SPINNER_FRAMES[self.spinner_idx % len(self.SPINNER_FRAMES)]
                node.set_label(f"{base_label} {frame}")

            # 1. Duration
            start_str = mod_info.get("started_at")
            end_str = mod_info.get("finished_at")
            if start_str and end_str:
                try:
                    d1 = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                    d2 = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                    diff = (d2 - d1).total_seconds()
                    node.add(f"Duration: {diff:.1f}s", allow_expand=False)
                except Exception:
                    pass

            # 2. Model-specific Details
            if mod_name == "model":
                payload = mod_info.get("payload", {})
                cms = payload.get("custom_module_state", {})
                
                init_mod = modules.get("init", {})
                init_payload = init_mod.get("payload", {})
                problem_type = init_payload.get("problem_type")
                metric = init_payload.get("metric")

                p_node = node.add("Info")
                if problem_type: p_node.add(f"Problem: {problem_type}", allow_expand=False)
                if metric: p_node.add(f"Metric: {metric}", allow_expand=False)
                p_node.add(f"Preset: {payload.get('preset')}", allow_expand=False)
                p_node.add(f"Time Limit: {payload.get('time_limit')}", allow_expand=False)
                p_node.add(f"Use GPU: {payload.get('use_gpu')}", allow_expand=False)
                p_node.add(f"Model Template: {payload.get('template')}", allow_expand=False)
                p_node.add(f"Preprocess Template: {payload.get('preprocess_template')}", allow_expand=False)
                t_rows = payload.get("tuning_rows") or cms.get("tuning_rows")
                p_node.add(f"Tuning Rows: {t_rows}", allow_expand=False)

            # 3. Shapes
            payload = mod_info.get("payload", {})
            shapes = payload.get("shapes", {})
            if shapes:
                s_node = node.add("Shapes")
                prefixes = sorted({k[:-7] if k.endswith("_before") else k[:-6] for k in shapes if k.endswith(('_before', '_after'))})
                for p in prefixes:
                    b, a = shapes.get(f"{p}_before"), shapes.get(f"{p}_after")
                    if b or a: s_node.add(f"{p.capitalize()}: {b} -> {a}", allow_expand=False)

            # 4. Config / Invocation (Pretty Print)
            invocation = mod_info.get("invocation", {})
            if invocation:
                i_node = node.add("Invocation")
                for line in json.dumps(invocation, indent=2).splitlines():
                    i_node.add(line, allow_expand=False)

            config_data = invocation.get("preprocess_template_config", {}).get("config") or invocation.get("model_template_config", {}).get("config") or invocation.get("config")
            if config_data:
                c_node = node.add("Config")
                for line in json.dumps(config_data, indent=2).splitlines():
                    c_node.add(line, allow_expand=False)

            # 5. Leaderboard (Artifact)
            lb_path = step_dir / "artifacts" / mod_name / "leaderboard.csv.gz"
            if lb_path.exists():
                try:
                    with gzip.open(lb_path, "rt") as f:
                        reader = csv.DictReader(f)
                        lb_node = node.add("Leaderboard")
                        count = 0
                        for row in reader:
                            m_name = row.get("model", "unknown")
                            s_val = row.get("score_val", "-")
                            try: s_val = f"{float(s_val):.5f}"
                            except: pass
                            lb_node.add(f"{m_name}: {s_val}", allow_expand=False)
                            count += 1
                            if count >= 10: break
                except Exception as e:
                    node.add(f"Error reading leaderboard: {e}", allow_expand=False)

        except Exception as e:
            node.add(f"Error: {e}", allow_expand=False)

    def _animate_running_nodes(self) -> None:
        if not self._running_nodes:
            return
        frame = self.SPINNER_FRAMES[self.spinner_idx % len(self.SPINNER_FRAMES)]
        for node, base_label in self._running_nodes:
            try:
                node.set_label(f"{base_label} {frame}")
            except Exception:
                pass


class OptunaDashboard(App):
    CSS = """
    #db_label {
        background: $primary;
        color: $text;
        padding: 1 1;
        width: 100%;
        height: 3;
    }
    #main_container {
        padding-top: 0;
        height: 100%;
    }
    .box {
        border: none;
        padding: 0 1;
        margin: 0;
    }
    #row1 {
        height: 1fr;
        min-height: 10;
    }
    #study_stats {
        width: 30%;
        height: 1fr;
        min-height: 6;
        overflow: auto;
        border: solid $accent;
    }
    #study_header {
        background: #2a2a2a;
        padding: 0 1;
    }
    #study_body {
        padding: 0 1;
    }
    #running_container {
        width: 70%;
        height: 1fr;
        min-height: 6;
        border: solid $accent;
        padding: 0;
        margin: 0;
    }
    .section-title {
        text-align: center;
        text-style: bold;
        background: $secondary;
        color: $text;
        margin: 0;
        padding: 0;
        height: 1;
    }
    #trials_table {
        width: auto;
        height: 1fr;
        min-height: 6;
        border: solid $accent;
        background: transparent;
        scrollbar-size: 0 0;
        scrollbar-gutter: auto;
    }
    #trials_table > .datatable--header {
        background: #2a2a2a;
        border-bottom: solid #3a3a3a;
    }
    #inspector_header {
        height: 2;
        width: 100%;
    }
    #loading_spinner {
        width: 1fr;
        height: 1;
        content-align: center middle;
    }
    .title {
        text-align: center;
        text-style: bold;
    }
    .subtitle {
        text-align: center;
        color: $text-muted;
        margin-bottom: 1;
    }
    """

    def __init__(self, db_path: str, project_root: str, study_name: str = None):
        super().__init__()
        self.db_path = Path(db_path)
        self.project_root = Path(project_root)
        self.study_name = study_name

    def on_mount(self) -> None:
        self.push_screen(DashboardScreen(self.db_path, self.project_root, self.study_name))


def main():
    parser = argparse.ArgumentParser(description="Optuna Live Dashboard")
    parser.add_argument("--db", required=True, help="Path to Optuna sqlite DB")
    parser.add_argument("--project-root", required=True, help="Project root directory")
    parser.add_argument("--study-name", help="Specific study name to monitor")
    parser.add_argument("--telegram-test-message", action="store_true", help="Send a test Telegram message on startup")
    args = parser.parse_args()

    if args.telegram_test_message:
        print("Sending test Telegram message...", file=sys.stderr)
        project_name = Path(args.project_root).name
        send_telegram_notification(f"🔔 <b>Test Message</b>\n\nOptuna Dashboard is connected.\n<b>Project:</b> {project_name}")

    app = OptunaDashboard(args.db, args.project_root, args.study_name)
    app.run()


if __name__ == "__main__":
    main()
