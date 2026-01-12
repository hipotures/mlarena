#!/usr/bin/env python3
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
from rich.json import JSON
from textual.app import App, ComposeResult
from textual import events
from textual import events
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen, Screen
from textual.reactive import reactive
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

logging.basicConfig(
    filename="dashboard.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w"
)

load_dotenv()

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


from textual.containers import Container, Horizontal, Vertical, VerticalScroll

class JSONModal(ModalScreen):
    BINDINGS = [
        ("escape", "app.pop_screen", "Close"),
        ("c", "copy_json", "Copy"),
    ]

    def __init__(self, title: str, data: Any):
        super().__init__()
        self.modal_title = title
        self.data = data

    def action_copy_json(self) -> None:
        """Copies the JSON data to clipboard."""
        try:
            json_str = json.dumps(self.data, indent=2)
            self.app.copy_to_clipboard(json_str)
            self.app.notify("JSON copied to clipboard!")
        except Exception as e:
            self.app.notify(f"Failed to copy: {e}", severity="error")

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            event.stop()
            event.prevent_default()
            self.app.pop_screen()

    def compose(self) -> ComposeResult:
        yield Container(
            Label(self.modal_title, classes="modal-title"),
            VerticalScroll(
                Static(JSON.from_data(self.data), classes="modal-text"),
                classes="modal-scroll-area"
            ),
            classes="modal-window"
        )


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
            
            header = str(self.query_one("#study_header").content)
            body = str(self.query_one("#study_body").content)
            
            def clean_markup(t):
                return t.replace("[bold]", "").replace("[/bold]", "").replace("[bold red]", "").replace("[/bold red]", "").replace("[bold yellow]", "").replace("[/bold yellow]", "")

            lines.append(clean_markup(header))
            lines.append(clean_markup(body))
            lines.append("-" * 40)
            
            table = self.query_one("#trials_table")
            lines.append("TRIALS TABLE")
            
            col_labels = [col.label for col in table.columns.values()]
            lines.append(" | ".join(map(str, col_labels)))
            lines.append("-" * 80)
            
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

        trials_table = self.query_one("#trials_table")
        prev_row_key = self.last_trial_row_key
        trials_table.clear()
        
        try:
            table_widget = self.query_one("#trials_table")
            h = table_widget.content_size.height
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

        for i, t in enumerate(completed_trials[:slots_for_completed]):
            val_str = f"{float(t['value']):.5f}" if t['value'] is not None else "-"
            start_str = str(t["datetime_start"]) if t["datetime_start"] else "-"
            if len(start_str) > 19:
                start_str = start_str[:19]
            
            # Highlight best trial (index 0) in green, others dim
            if i == 0:
                style_tag = "green"
            else:
                style_tag = "dim"
            
            trials_table.add_row(
                f"[{style_tag}]{t['number']}[/{style_tag}]",
                f"[{style_tag}]COMPLETE[/{style_tag}]",
                f"[{style_tag}]{val_str}[/{style_tag}]",
                f"[{style_tag}]{_duration_str(t['datetime_start'], t['datetime_complete'])}[/{style_tag}]",
                f"[{style_tag}]{t.get('params_hash', '-')}[/{style_tag}]",
                f"[{style_tag}]{start_str}[/{style_tag}]",
                key=str(t["number"])
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
                self.app.push_screen(TrialInspector(self.project_root, self.active_study_name, trial_id, self.db_path))

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
            
            for s in studies:
                if name in s["name"]:
                    return s
        
        return studies[-1] if studies else None

    def _fetch_recent_trials(self, conn: sqlite3.Connection, study_id: int, limit: int) -> List[Dict[str, Any]]:
        tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        has_trial_values = "trial_values" in tables

        raw_dir = "MINIMIZE"
        if "study_directions" in tables:
            d_row = conn.execute("SELECT direction FROM study_directions WHERE study_id=?", (study_id,)).fetchone()
            if d_row:
                raw_dir = d_row["direction"]
        
        if str(raw_dir) == "1": direction = "MAXIMIZE"
        elif str(raw_dir) == "0": direction = "MINIMIZE"
        else: direction = str(raw_dir).upper()
        
        order = "DESC" if direction == "MAXIMIZE" else "ASC"

        result = []
        
        sql_running = (
            "SELECT t.trial_id, t.number, t.state, t.datetime_start, t.datetime_complete, NULL as value "
            "FROM trials t "
            "WHERE t.study_id=? AND (t.state=0 OR UPPER(CAST(t.state AS TEXT))='RUNNING' OR t.state=4 OR UPPER(CAST(t.state AS TEXT))='WAITING') "
            "ORDER BY t.number DESC"
        )
        rows_running = conn.execute(sql_running, (study_id,)).fetchall()
        result.extend([dict(r) for r in rows_running])

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
        
        seen_nums = {r["number"] for r in result}
        for r in rows_complete:
            if r["number"] not in seen_nums:
                result.append(dict(r))
                seen_nums.add(r["number"])

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
        ("E", "expand_all", "Expand All"),
        ("e", "reset_view_tree", "Reset View"),
    ]
    SPINNER_FRAMES = ["⠁", "⠈", "⠐", "⠠", "⢀", "⡀", "⠄", "⠂"]

    def __init__(self, project_root: Path, study_name: str, trial_id: str, db_path: Path):
        super().__init__()
        self.project_root = project_root
        self.study_name = study_name
        self.trial_id = int(trial_id)
        self.db_path = db_path
        self.trial_dir = self._find_trial_dir()
        self.refresh_timer = None
        self.state_cache = {}  # Cache for state.json contents
        self.spinner_idx = 0
        self._last_data_refresh = 0.0
        self._last_full_rebuild = 0
        self._running_nodes = []  # List of (node, base_label) for animation
        self._tree_fully_expanded = False # Tracking state
        self._pre_node = None
        self._model_node = None
        self._opt_node = None
        self._metrics_node = None
        self._preprocess_running = False
        self._model_running = False
        self._last_preprocess_refresh = 0.0
        self._last_model_refresh = 0.0
        self._last_optuna_refresh = 0.0
        self._tree_initialized = False
        self._completed_view = False
        self._pending_expand_all = False
        self._pending_reset_view = False
        self._suppress_modal_on_expand = False
        self._last_modal_key = None
        self._last_modal_time = 0.0
        self._trial_state_cache = {}
        self._last_trial_state_refresh = 0.0
        self._full_trial_path = None

    def _get_optuna_params(self) -> Dict[str, Any]:
        """Fetches Optuna params from SQLite for the current trial number."""
        try:
            with _connect_read_only(self.db_path) as conn:
                # 1. Get study_id
                study_row = conn.execute("SELECT study_id FROM studies WHERE study_name=?", (self.study_name,)).fetchone()
                if not study_row:
                    return {"error": "Study not found"}
                study_id = study_row["study_id"]

                # 2. Get trial_id (PK) from trial_number (self.trial_id is the number)
                trial_row = conn.execute(
                    "SELECT trial_id FROM trials WHERE study_id=? AND number=?", 
                    (study_id, self.trial_id)
                ).fetchone()
                
                if not trial_row:
                    return {"error": f"Trial number {self.trial_id} not found in DB"}
                
                real_trial_id = trial_row["trial_id"]

                # 3. Get params
                params = {}
                rows = conn.execute(
                    "SELECT param_name, param_value, distribution_json FROM trial_params WHERE trial_id=?", 
                    (real_trial_id,)
                ).fetchall()
                
                for r in rows:
                    try:
                        dist = json.loads(r["distribution_json"])
                        val = r["param_value"]
                        if dist.get("name") == "CategoricalDistribution":
                            choices = dist.get("attributes", {}).get("choices", [])
                            if isinstance(val, float) and val.is_integer() and 0 <= int(val) < len(choices):
                                val = choices[int(val)]
                        params[r["param_name"]] = val
                    except:
                        params[r["param_name"]] = r["param_value"]
                
                return params
        except Exception as e:
            return {"error": str(e)}

    def check_action(self, action: str, parameters: tuple[Any, ...]) -> bool | None:
        """Controls visibility of 'e' actions based on state."""
        if action == "expand_all":
            return not self._tree_fully_expanded
        if action == "collapse_all":
            return self._tree_fully_expanded
        return True

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Handle node selection to open modals."""
        if self._suppress_modal_on_expand:
            return
        self._handle_json_node(event.node)

    def on_tree_node_expanded(self, event: Tree.NodeExpanded) -> None:
        """Handle node expansion to open modals."""
        if self._suppress_modal_on_expand:
            return
        self._handle_json_node(event.node)

    def _handle_json_node(self, node: TreeNode) -> None:
        if self._suppress_modal_on_expand:
            return
        if node.data and isinstance(node.data, dict) and node.data.get("type") == "json_modal":
            modal_key = node.data.get("title", str(node.label))
            now = time.time()
            if self._last_modal_key == modal_key and (now - self._last_modal_time) < 0.25:
                return
            self._last_modal_key = modal_key
            self._last_modal_time = now
            title = node.data.get("title", "Data View")
            payload = node.data.get("payload", {})
            
            # Collapse immediately to keep the arrow state "ready to expand"
            # We use call_after_refresh to ensure the UI update doesn't fight the event loop
            self.call_after_refresh(node.collapse)
            
            self.app.push_screen(JSONModal(title, payload))

    def action_expand_all(self) -> None:
        """Expands all nodes in the tree."""
        try:
            tree = self.query_one("#flow_tree")
            self._suppress_modal_on_expand = True
            self._expand_all_non_modal(tree.root)
            self._tree_fully_expanded = True
            self._pending_expand_all = True
            self._pending_reset_view = False
            self.app.notify("Tree fully expanded.")
            try:
                self.set_timer(0.5, self._clear_expand_suppression)
            except Exception:
                self.call_after_refresh(self._clear_expand_suppression)
        except Exception: pass

    def _clear_expand_suppression(self) -> None:
        self._suppress_modal_on_expand = False

    def _expand_all_non_modal(self, node: TreeNode) -> None:
        if isinstance(node.data, dict) and node.data.get("type") == "json_modal":
            return
        try:
            node.expand()
        except Exception:
            pass
        for child in node.children:
            self._expand_all_non_modal(child)

    def action_reset_view_tree(self) -> None:
        """Resets tree to default view (headings only)."""
        try:
            tree = self.query_one("#flow_tree")
            tree.root.collapse_all()
            tree.root.expand()
            for child in tree.root.children:
                child.expand()
            self._tree_fully_expanded = False
            self._pending_reset_view = True
            self._pending_expand_all = False
            self.app.notify("Tree view reset to default.")
        except Exception: pass

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
        base = self.project_root / "experiments" / f"optuna_{self.study_name}" / f"trial_{self.trial_id:04d}"
        
        try:
            resolved = base.resolve()
            logging.debug(f"Checking path: {base} -> {resolved} (Exists: {base.exists()})")
        except Exception as e:
            logging.debug(f"Resolution failed for {base}: {e}")

        if base.exists():
            return base
            
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
                Label("Path: -", id="path_label"),
                LoadingIndicator(id="loading_spinner"),
                id="path_bar",
            ),
            Container(
                Tree("Pipeline Flow", id="flow_tree"),
                classes="box",
                id="flow_panel",
            ),
            id="inspector_container"
        )
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_timer = self.set_interval(0.1, self._build_tree)
        self.app.title = f"OptunaInspector/Trial {self.trial_id}"
        self._update_path_label()
        self._build_tree()

    def on_unmount(self) -> None:
        self.app.title = f"OptunaDashboard/{self.project_root.name}"

    def on_click(self, event: events.Click) -> None:
        if event.widget.id != "path_label":
            return
        if self._full_trial_path:
            self.app.copy_to_clipboard(str(self._full_trial_path))
            self.app.notify("Full path copied to clipboard.")

    def _update_path_label(self) -> None:
        label = self.query_one("#path_label", Label)
        if not self.trial_dir:
            label.update("Path: -")
            self._full_trial_path = None
            return
        full_path = Path(self.trial_dir)
        self._full_trial_path = full_path
        display_path = self._shorten_path(full_path)
        label.update(f"Path: {display_path}")

    def _shorten_path(self, path: Path) -> str:
        parts = list(path.parts)
        if "experiments" in parts:
            idx = parts.index("experiments")
            return "/".join(parts[idx:])
        return str(path)

    def _build_tree(self) -> None:
        self.spinner_idx += 1

        tree = self.query_one("#flow_tree")
        current_time = time.time()
        if current_time - self._last_data_refresh <= 0.1:
            self._animate_running_nodes()
            return
        self._last_data_refresh = current_time

        root = tree.root
        root.expand()
        
        if not self.trial_dir or not self.trial_dir.exists():
            self.trial_dir = self._find_trial_dir()
            self._update_path_label()
        
        if not self.trial_dir or not self.trial_dir.exists():
            if not self._tree_initialized:
                root.remove_children()
                root.add("[dim]Trial directory initializing...[/dim]")
                self._tree_initialized = True
            return

        if not self._tree_initialized:
            root.remove_children()
            self._pre_node = root.add("[bold]Preprocessing[/bold]", expand=True)
            self._opt_node = root.add("[bold]Optuna[/bold]", expand=False)
            self._tree_initialized = True

        force_refresh = self._last_preprocess_refresh == 0.0 and self._last_model_refresh == 0.0
        trial_state = self._load_trial_state()
        pipeline_status = (trial_state.get("pipeline_progress", {}) or {}).get("status")
        pipeline_running = str(pipeline_status).lower() == "running"

        refresh_pre = force_refresh or (self._preprocess_running and current_time - self._last_preprocess_refresh >= 0.5)
        pending_model = (not self._model_dir_exists()) and not (self.trial_dir / "metrics.json").exists()
        refresh_model = (
            force_refresh
            or (self._model_running and current_time - self._last_model_refresh >= 30.0)
            or (self._model_node is None and self._model_dir_exists())
            or (pending_model and current_time - self._last_model_refresh >= 2.0)
        )
        refresh_optuna = (self._preprocess_running or self._model_running or pipeline_running) and (
            force_refresh or (current_time - self._last_optuna_refresh >= 10.0)
        )

        if refresh_pre:
            self._render_preprocess(self._pre_node)
            self._last_preprocess_refresh = current_time
        if refresh_model:
            self._render_model()
            self._last_model_refresh = current_time
        if refresh_optuna:
            self._render_optuna()
            self._last_optuna_refresh = current_time

        self._running_nodes = [
            *getattr(self, "_running_nodes_pre", []),
            *getattr(self, "_running_nodes_model", []),
        ]
        self._animate_running_nodes()
        if self._pending_expand_all:
            try:
                self._expand_all_non_modal(tree.root)
            except Exception:
                pass
            self._pending_expand_all = False
        elif self._pending_reset_view:
            try:
                tree.root.collapse_all()
                tree.root.expand()
                for child in tree.root.children:
                    child.expand()
            except Exception:
                pass
            self._pending_reset_view = False

        any_running = self._preprocess_running or self._model_running or pipeline_running
        metrics_file = self.trial_dir / "metrics.json"
        if metrics_file.exists() and not any_running:
            if not self._completed_view:
                try:
                    self.query_one("#loading_spinner").display = False
                except Exception:
                    pass
                self._metrics_node = root.add("[bold]Metrics[/bold]", expand=True)
                self._render_metrics(self._metrics_node, metrics_file)
                self._completed_view = True
            if self.refresh_timer:
                self.refresh_timer.stop()
                self.refresh_timer = None
            return

        if not any_running and not force_refresh and not pending_model:
            if self.refresh_timer:
                self.refresh_timer.stop()
                self.refresh_timer = None
            return

    def _render_preprocess(self, pre_node: TreeNode) -> None:
        if pre_node is None:
            return
        pre_node.remove_children()
        trial_state = self._load_trial_state()
        steps_meta = {
            s.get("name"): s
            for s in (trial_state.get("pipeline_progress", {}) or {}).get("steps", [])
            if isinstance(s, dict)
        }
        steps = sorted(
            [d for d in self.trial_dir.iterdir() if d.is_dir() and d.name[0].isdigit()],
            key=lambda x: int(x.name.split("-")[0]),
        )
        running = False
        self._running_nodes_pre = []
        for step_dir in steps:
            status_override = None
            meta = steps_meta.get(step_dir.name)
            if isinstance(meta, dict):
                status_override = meta.get("status")
            step_node = pre_node.add(f"{step_dir.name}", expand=False)
            status = self._add_step_details(step_node, step_dir, self._running_nodes_pre, status_override=status_override)
            if status == "running" or status_override == "running":
                running = True
        self._preprocess_running = running

    def _render_metrics(self, metrics_node: TreeNode, metrics_file: Path) -> None:
        if metrics_node is None or not metrics_file.exists():
            return
        metrics_node.remove_children()
        try:
            data = json.loads(metrics_file.read_text())
            for k, v in data.items():
                if isinstance(v, float):
                    if k in ["total_sec", "preprocess_sec"]:
                        val_str = f"{v:.1f}"
                    else:
                        val_str = f"{v:.5f}"
                else:
                    val_str = str(v)
                metrics_node.add(f"{k}: {val_str}", allow_expand=False)
        except Exception:
            pass

    def _render_model(self) -> None:
        model_dir = self.trial_dir / "model" # Or optuna_model
        if not model_dir.exists():
            model_dir = self.trial_dir / "optuna_model"

        if not model_dir.exists():
            if self._model_node is not None:
                try:
                    self._model_node.remove()
                except Exception:
                    pass
                self._model_node = None
            self._model_running = False
            self._running_nodes_model = []
            return

        if self._model_node is None or self._model_node.parent is None:
            root = self.query_one("#flow_tree").root
            if self._opt_node is not None and self._opt_node.parent is root:
                self._model_node = root.add("[bold]Model[/bold]", expand=True, before=self._opt_node)
            else:
                self._model_node = root.add("[bold]Model[/bold]", expand=True)

        self._model_node.remove_children()
        self._running_nodes_model = []
        status = self._add_step_details(self._model_node, model_dir, self._running_nodes_model)
        self._model_running = status == "running"

    def _render_optuna(self) -> None:
        if self._opt_node is None or self._opt_node.parent is None:
            root = self.query_one("#flow_tree").root
            if self._model_node is not None and self._model_node.parent is root:
                self._opt_node = root.add("[bold]Optuna[/bold]", expand=False, after=self._model_node)
            else:
                self._opt_node = root.add("[bold]Optuna[/bold]", expand=False)
        self._opt_node.remove_children()
        params = self._get_optuna_params()
        if params:
            tp_node = self._opt_node.add("trial_params", expand=False)
            tp_node.data = {"type": "json_modal", "title": "Optuna Trial Params", "payload": params}
            tp_node.add_leaf("") # Dummy leaf for arrow

    def _model_dir_exists(self) -> bool:
        if not self.trial_dir:
            return False
        model_dir = self.trial_dir / "model"
        if model_dir.exists():
            return True
        model_dir = self.trial_dir / "optuna_model"
        return model_dir.exists()

    def _add_step_details(
        self,
        node: TreeNode,
        step_dir: Path,
        running_nodes: List[Tuple[TreeNode, str]],
        status_override: Optional[str] = None,
    ) -> str:
        state_path = step_dir / "state.json"
        if not state_path.exists():
            if status_override:
                status_norm = str(status_override).lower()
                if status_norm == "failed":
                    node.set_label(f"{node.label} [FAILED]")
                elif status_norm == "running":
                    base_label = str(node.label)
                    running_nodes.append((node, base_label))
                    frame = self.SPINNER_FRAMES[self.spinner_idx % len(self.SPINNER_FRAMES)]
                    node.set_label(f"{base_label} {frame}")
            node.add("No state.json found", allow_expand=False)
            return str(status_override or "unknown")

        try:
            mtime = state_path.stat().st_mtime
            cache_key = str(state_path)
            if cache_key in self.state_cache and self.state_cache[cache_key]["mtime"] == mtime:
                state = self.state_cache[cache_key]["data"]
            else:
                state = json.loads(state_path.read_text())
                self.state_cache[cache_key] = {"mtime": mtime, "data": state}

            modules = state.get("modules", {})
            mod_name = "preprocess" if "preprocess" in modules else "model" if "model" in modules else list(modules.keys())[-1] if modules else "unknown"
            
            mod_info = modules.get(mod_name, {})
            status = mod_info.get("status", "unknown")
            any_running = any(
                isinstance(m, dict) and m.get("status") == "running"
                for m in modules.values()
            )
            any_failed = any(
                isinstance(m, dict) and m.get("status") == "failed"
                for m in modules.values()
            )
            
            status_effective = status_override or status
            status_norm = str(status_effective).lower()
            if any_failed or status_norm == "failed":
                node.set_label(f"{node.label} [FAILED]")
            elif any_running or status_norm == "running":
                base_label = str(node.label)
                running_nodes.append((node, base_label))
                frame = self.SPINNER_FRAMES[self.spinner_idx % len(self.SPINNER_FRAMES)]
                node.set_label(f"{base_label} {frame}")

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

            payload = mod_info.get("payload", {})
            shapes = payload.get("shapes", {})
            if shapes:
                s_node = node.add("Shapes")
                prefixes = sorted({k[:-7] if k.endswith("_before") else k[:-6] for k in shapes if k.endswith(('_before', '_after'))})
                for p in prefixes:
                    b, a = shapes.get(f"{p}_before"), shapes.get(f"{p}_after")
                    if b or a: s_node.add(f"{p.capitalize()}: {b} -> {a}", allow_expand=False)

            # 4. Config / Invocation (Modal View)
            invocation = mod_info.get("invocation", {})
            if invocation:
                i_node = node.add("Invocation", expand=False)
                i_node.data = {"type": "json_modal", "title": "Invocation Data", "payload": invocation}
                i_node.add_leaf("") # Dummy leaf to show arrow

            config_data = invocation.get("preprocess_template_config", {}).get("config") or invocation.get("model_template_config", {}).get("config") or invocation.get("config")
            if config_data:
                c_node = node.add("Config", expand=False)
                c_node.data = {"type": "json_modal", "title": "Configuration Data", "payload": config_data}
                c_node.add_leaf("") # Dummy leaf to show arrow

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
        if any_running or str(status_override or status).lower() == "running":
            return "running"
        return str(status_override or status)

    def _load_trial_state(self) -> Dict[str, Any]:
        if not self.trial_dir:
            return {}
        state_path = self.trial_dir / "state.json"
        if not state_path.exists():
            return {}
        now = time.time()
        if now - self._last_trial_state_refresh < 0.5:
            cached = self._trial_state_cache.get(str(state_path))
            if cached:
                return cached.get("data", {})
        cache_key = str(state_path)
        try:
            mtime = state_path.stat().st_mtime
        except Exception:
            mtime = None
        cached = self._trial_state_cache.get(cache_key)
        if cached and cached.get("mtime") == mtime:
            return cached.get("data", {})
        try:
            data = json.loads(state_path.read_text())
        except Exception:
            data = {}
        self._trial_state_cache[cache_key] = {"mtime": mtime, "data": data}
        self._last_trial_state_refresh = now
        return data

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
    #path_label {
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
    #path_bar {
        background: $primary;
        color: $text;
        width: 100%;
        height: 3;
    }
    #path_label {
        width: 1fr;
        padding: 1 1;
    }
    #path_label > .label--text {
        text-style: bold;
    }
    #flow_panel {
        border: solid $accent;
        padding: 0 1;
        margin: 0;
    }
    #flow_tree {
        background: transparent;
    }
    #loading_spinner {
        display: none;
        width: 14;
        height: 1;
    }
    JSONModal {
        align: center middle;
    }
    .modal-window {
        width: 90%;
        height: 90%;
        background: $surface;
        border: solid $accent;
        padding: 1 2;
    }
    .modal-title {
        text-style: bold;
        background: $primary;
        color: $text;
        width: 100%;
        text-align: center;
        margin-bottom: 1;
    }
    .modal-scroll-area {
        width: 100%;
        height: 1fr;
        scrollbar-size: 1 1;
    }
    .modal-text {
        width: 100%;
    }
    """

    def __init__(self, db_path: str, project_root: str, study_name: str = None):
        super().__init__()
        self.db_path = Path(db_path)
        self.project_root = Path(project_root)
        self.study_name = study_name
        self.title = f"OptunaDashboard/{self.project_root.name}"

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
