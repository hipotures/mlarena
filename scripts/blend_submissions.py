#!/usr/bin/env python3
"""
Blend top-N Kaggle submissions using public scores as weights.

Default flow:
1) Fetch submissions from Kaggle API (CSV format)
2) Exclude ensemble submissions (blend/stack/voting) by default
3) Select top-N by public score (tie-breaker: newer date)
4) Map Kaggle filenames to local submissions in /mnt/mlarena
5) Blend predictions with weighted average
6) Save blended submission + manifest

Example (auto-fetch from Kaggle API):
python scripts/blend_submissions.py \
  --project playground-series-s5e12 \
  --top-n 5 \
  --weighting public \
  --output-name submission-blend-top5-public.csv

With pre-downloaded CSV:
python scripts/blend_submissions.py \
  --project playground-series-s5e12 \
  --kaggle-csv /tmp/submissions.csv \
  --top-n 5

To include ensemble submissions in blending:
python scripts/blend_submissions.py \
  --project playground-series-s5e12 \
  --top-n 5 \
  --include-ensembles
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Set, Optional

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich import box
from rich.progress import Progress


DATE_FMT = "%Y-%m-%d %H:%M:%S.%f"


def load_weighted_blender(repo_root: Path):
    blender_path = repo_root / "src" / "kaggle_tools" / "stacking" / "blender.py"
    if not blender_path.exists():
        raise FileNotFoundError(f"Missing blender module at {blender_path}")
    spec = importlib.util.spec_from_file_location("kaggle_tools.stacking.blender", blender_path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Failed to load blender module from {blender_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WeightedBlender


def fetch_kaggle_submissions(project: str) -> pd.DataFrame:
    """Fetch submissions from Kaggle CLI in CSV format."""
    try:
        result = subprocess.run(
            ["kaggle", "competitions", "submissions", "-c", project, "--csv"],
            capture_output=True,
            text=True,
            check=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Kaggle CLI failed: {result.stderr}")

        # Parse CSV from stdout
        from io import StringIO
        df = pd.read_csv(StringIO(result.stdout))
        return df

    except subprocess.CalledProcessError as e:
        print(f"Error: Kaggle CLI failed: {e.stderr}", file=sys.stderr)
        raise RuntimeError("Kaggle authentication failed or competition not found. Run: kaggle competitions list")
    except FileNotFoundError:
        raise RuntimeError("Kaggle CLI not found. Install with: pip install kaggle")


def parse_kaggle_csv(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Parse Kaggle submissions CSV into rows."""
    rows: List[Dict[str, Any]] = []

    # Convert to list of dicts for easier access
    records = df.to_dict('records')

    for record in records:
        filename = record.get("fileName", "")
        public_score = record.get("publicScore", None)
        description = record.get("description", "")
        date_str = record.get("date", "")

        # Skip if no public score
        if pd.isna(public_score):
            continue

        try:
            public_score_val = float(public_score)
        except (ValueError, TypeError):
            continue

        # Parse date
        try:
            date = pd.to_datetime(date_str)
        except (ValueError, TypeError):
            date = None

        # Handle NaN description
        if pd.isna(description):
            description = ""

        rows.append({
            "filename": filename,
            "public_score": public_score_val,
            "date": date,
            "description": str(description),
        })

    if not rows:
        raise RuntimeError("No submission rows with public scores found")
    return rows


def is_ensemble_submission(filename: str, description: str) -> bool:
    """Check if submission is already blended/stacked/ensembled."""
    ensemble_keywords = ["blend", "stack", "ensemble", "voting", "stacking"]
    filename_str = str(filename).lower() if filename else ""
    description_str = str(description).lower() if description and not pd.isna(description) else ""
    text_to_check = f"{filename_str} {description_str}"
    return any(keyword in text_to_check for keyword in ensemble_keywords)


def select_top_n(rows: List[Dict[str, Any]], top_n: int, exclude_ensembles: bool = True) -> List[Dict[str, Any]]:
    """Select top-N submissions by score, optionally excluding ensemble submissions."""
    filtered = rows
    if exclude_ensembles:
        filtered = [r for r in rows if not is_ensemble_submission(r["filename"], r["description"])]
        excluded_count = len(rows) - len(filtered)
        if excluded_count > 0:
            print(f"Excluded {excluded_count} ensemble submission(s) from blending")

    rows_sorted = sorted(
        filtered,
        key=lambda r: (r["public_score"], r["date"] or datetime.min),
        reverse=True,
    )
    return rows_sorted[:top_n]


def resolve_local_file(submissions_dir: Path, kaggle_filename: str) -> Path:
    direct = submissions_dir / kaggle_filename
    if direct.exists():
        return direct
    stem = Path(kaggle_filename).stem
    # Support both .csv and .csv.gz files
    matches = sorted(submissions_dir.glob(f"{stem}-*.csv*"))
    if not matches:
        raise FileNotFoundError(f"No local submission found for {kaggle_filename}")
    if len(matches) > 1:
        print(f"Warning: multiple matches for {kaggle_filename}, using {matches[0].name}")
    return matches[0]


def compute_weights(
    scores: List[float],
    mode: str,
    power: float,
    temp: float,
    eps: float,
    manual: List[float] | None,
) -> List[float]:
    n = len(scores)
    if n == 0:
        raise ValueError("No scores provided for weight computation")

    if mode == "public":
        raw = scores[:]
    elif mode == "rank":
        raw = [float(n - i) for i in range(n)]
    elif mode == "power":
        raw = [float(s) ** power for s in scores]
    elif mode == "softmax":
        if temp <= 0:
            raise ValueError("temp must be > 0 for softmax weighting")
        raw = [math.exp(float(s) / temp) for s in scores]
    elif mode == "offset":
        min_score = min(scores)
        raw = [max(float(s) - min_score + eps, 0.0) for s in scores]
    elif mode == "uniform":
        raw = [1.0] * n
    elif mode == "manual":
        if not manual or len(manual) != n:
            raise ValueError("manual weights must be provided and match top-n")
        raw = manual[:]
    else:
        raise ValueError(f"Unknown weighting mode: {mode}")

    total = sum(raw)
    if total <= 0:
        raise ValueError("Weights sum to zero; adjust weighting settings")
    return [float(w) / total for w in raw]


def blend_predictions(
    model_files: List[Path],
    weights: List[float],
    id_column: str | None,
    target_column: str | None,
) -> pd.DataFrame:
    first_df = pd.read_csv(model_files[0], compression='infer')
    id_col = id_column or first_df.columns[0]
    target_col = target_column or first_df.columns[-1]
    base_ids = first_df[id_col]

    preds = []
    for path in model_files:
        df = pd.read_csv(path, compression='infer')
        if id_col not in df.columns or target_col not in df.columns:
            raise ValueError(f"Missing columns in {path.name}: expected {id_col}, {target_col}")
        if df[id_col].equals(base_ids):
            series = df[target_col]
        else:
            aligned = df.set_index(id_col).reindex(base_ids)
            if aligned[target_col].isna().any():
                raise ValueError(f"ID mismatch when aligning {path.name}")
            series = aligned[target_col].reset_index(drop=True)
        preds.append(series)

    WeightedBlender = load_weighted_blender(Path(__file__).resolve().parents[1])
    blender = WeightedBlender()
    ensemble = blender.blend(preds, weights)
    return pd.DataFrame({id_col: base_ids, target_col: ensemble})


# ============================================================================
# Interactive Mode
# ============================================================================

class InteractiveBlender:
    """Interactive CLI for blending Kaggle submissions using rich library."""

    def __init__(self, project: str, submissions_dir: Optional[Path], console: Console):
        self.project = project
        repo_root = Path(__file__).resolve().parents[1]  # scripts/blend_submissions.py -> repo root
        self.submissions_dir = (
            submissions_dir
            if submissions_dir
            else repo_root / "projects" / "kaggle" / project / "submissions"
        )
        self.console = console

        # Data
        self.all_submissions: List[Dict] = []
        self.filtered_submissions: List[Dict] = []
        self.current_view: List[Dict] = []
        self.local_scores: Dict[str, float] = {}
        self.submission_metadata: Dict[str, Dict[str, str]] = {}  # Template info

        # Selection state
        self.selected_count: int = 0
        self.selected_indices: Set[int] | None = None  # None = top-N mode, Set = specific mode
        self.show_all: bool = False
        self.excluded_indices: Set[int] = set()
        self.display_count: int = 10  # Configurable display count

        # Strategy state
        self.strategy: str = "public"
        self.strategy_params: Dict[str, Any] = {"power": 2.0, "temp": 0.001, "eps": 1e-6}

        # History
        self.history: List[Dict] = []

    def fetch_and_parse(self) -> None:
        """Fetch submissions from Kaggle and load local scores."""
        # Fetch from Kaggle
        df = fetch_kaggle_submissions(self.project)
        self.all_submissions = parse_kaggle_csv(df)

        # Sort by public score (descending), then by date (newer first)
        self.all_submissions = sorted(
            self.all_submissions,
            key=lambda r: (r["public_score"], r["date"] or datetime.min),
            reverse=True,
        )

        # Filter ensembles
        self.filtered_submissions = [
            s for s in self.all_submissions
            if not is_ensemble_submission(s["filename"], s["description"])
        ]
        self.current_view = self.filtered_submissions

        # Extract template info from Kaggle descriptions
        self._extract_templates_from_descriptions()

        # Load local scores and history if available
        self._load_local_scores()
        self._load_history()

    def _extract_templates_from_descriptions(self) -> None:
        """Extract local CV score and templates from Kaggle submission descriptions.

        Expected format: "CV_score | feat: N | exp-ID | model_template | preprocess_template | filename"
        Example: "0.71509 | feat: 84 | exp-20251225-202731 | catboost-binned-hpo-gap | catboost-full-clean-rfe-gap-p99-wm08 | submission.csv.gz"
        """
        for submission in self.all_submissions:
            filename = submission.get("filename", "")
            description = submission.get("description", "")

            if not filename or not description:
                continue

            # Parse description format: "score | feat: N | exp-ID | model_tpl | preprocess_tpl | filename"
            parts = [p.strip() for p in description.split("|")]

            model_tpl = "-"
            preprocess_tpl = "-"

            # Expected positions: [0]=score, [1]=feat, [2]=exp-ID, [3]=model_tpl, [4]=preprocess_tpl, [5]=filename
            if len(parts) >= 1:
                # Extract local CV score from position [0]
                try:
                    local_cv = float(parts[0])
                    self.local_scores[filename] = local_cv
                except (ValueError, IndexError):
                    pass

            if len(parts) >= 5:
                model_tpl = parts[3] if parts[3] else "-"
                preprocess_tpl = parts[4] if parts[4] else "-"

            self.submission_metadata[filename] = {
                "model_template": model_tpl,
                "preprocess_template": preprocess_tpl
            }

    def _load_local_scores(self) -> None:
        """Load local CV scores from submissions.json (and template info as fallback)."""
        submissions_json = self.submissions_dir / "submissions.json"
        if not submissions_json.exists():
            return

        try:
            data = json.loads(submissions_json.read_text())
            # Handle both list and dict formats
            entries = data if isinstance(data, list) else data.get("submissions", [])

            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                filename = entry.get("submission_file", "")
                local_score = entry.get("local_cv_score")

                # Store CV score
                if filename and local_score is not None:
                    self.local_scores[filename] = float(local_score)

                # Template info as fallback (only if not already set from Kaggle description)
                if filename and filename not in self.submission_metadata:
                    model_tpl = entry.get("config", {}).get("system", {}).get("template")
                    notes = entry.get("notes", "")

                    # Parse preprocessing template from notes
                    prep_tpl = self._parse_preprocess_template(notes)

                    # Fallback: extract model template from notes if config missing
                    if not model_tpl and notes:
                        model_tpl = self._parse_model_template(notes)

                    self.submission_metadata[filename] = {
                        "model_template": model_tpl or "-",
                        "preprocess_template": prep_tpl or "-"
                    }
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            pass  # Silently ignore malformed submissions.json

    def _load_history(self) -> None:
        """Load ensemble history from JSON if available."""
        history_json = self.submissions_dir / "ensemble_history.json"
        if not history_json.exists():
            return

        try:
            data = json.loads(history_json.read_text())
            for entry in data:
                # Convert timestamp string back to datetime
                entry["timestamp"] = datetime.fromisoformat(entry["timestamp"])
                self.history.append(entry)
        except (json.JSONDecodeError, ValueError):
            pass  # Silently ignore malformed history

    def _save_history(self) -> None:
        """Save ensemble history to JSON."""
        history_json = self.submissions_dir / "ensemble_history.json"

        # Convert to serializable format
        serializable = []
        for entry in self.history:
            serializable.append({
                "timestamp": entry["timestamp"].isoformat(),
                "ensemble_method": entry.get("ensemble_method", "blend"),  # blend/voting/stacking
                "strategy": entry["strategy"],  # For blend: public/power/rank/etc
                "selected_count": entry["selected_count"],
                "output_file": entry["output_file"],
                "avg_score": entry["avg_score"],
                "submitted": entry.get("submitted", False),  # Was it submitted to Kaggle?
                "public_score": entry.get("public_score"),  # From Kaggle leaderboard (None if not fetched)
            })

        history_json.write_text(json.dumps(serializable, indent=2))

    def _parse_preprocess_template(self, notes: str) -> str | None:
        """Extract preprocessing template from notes field.

        Expected format: "preprocess=template_name" or "template=X; preprocess=Y"
        Returns template name or None if not found.
        """
        if not notes or not isinstance(notes, str):
            return None

        # Look for "preprocess=XXX" pattern
        import re
        match = re.search(r'preprocess=([^\s;]+)', notes)
        if match:
            return match.group(1)
        return None

    def _parse_model_template(self, notes: str) -> str | None:
        """Extract model template from notes field as fallback.

        Expected format: "template=template_name"
        Returns template name or None if not found.
        """
        if not notes or not isinstance(notes, str):
            return None

        import re
        match = re.search(r'template=([^\s;]+)', notes)
        if match:
            return match.group(1)
        return None

    def _truncate(self, text: str, max_len: int) -> str:
        """Truncate text to max length with ellipsis."""
        if len(text) <= max_len:
            return text
        return text[:max_len-1] + "…"

    def _is_selected(self, idx: int) -> bool:
        """Check if index is selected (handles both top-N and specific modes)."""
        if self.selected_indices is not None:
            # Specific ID mode
            return idx in self.selected_indices
        else:
            # Top-N mode
            return idx <= self.selected_count

    def render_main_screen(self) -> None:
        """Render the main interactive screen with submissions table and panels."""
        self.console.clear()

        # Build display rows first to size columns to content
        display_count = max(self.display_count, self.selected_count)
        rows = []
        for idx, sub in enumerate(self.current_view[:display_count], start=1):
            if idx in self.excluded_indices:
                continue  # Skip excluded

            filename = sub["filename"]
            public_score = f"{sub['public_score']:.5f}"
            local_score = f"{self.local_scores.get(filename, 0.0):.5f}" if filename in self.local_scores else "-"
            date_str = sub["date"].strftime("%Y-%m-%d") if sub["date"] else "-"

            metadata = self.submission_metadata.get(filename, {})
            model_tpl = metadata.get("model_template", "-")
            prep_tpl = metadata.get("preprocess_template", "-")
            selected = "✓" if self._is_selected(idx) else ""

            rows.append({
                "idx": str(idx),
                "filename": filename,
                "model_tpl": model_tpl,
                "prep_tpl": prep_tpl,
                "public": public_score,
                "local": local_score,
                "date": date_str,
                "selected": selected,
                "idx_end": str(idx),
            })

        headers = {
            "idx": "#",
            "filename": "Filename",
            "model_tpl": "Model Tpl",
            "prep_tpl": "Prep Tpl",
            "public": "Public",
            "local": "Local",
            "date": "Date",
            "selected": "✓",
            "idx_end": "#",
        }

        def _max_len(key: str) -> int:
            values = [row[key] for row in rows]
            return max([len(headers[key])] + [len(str(v)) for v in values]) if values else len(headers[key])

        widths = {key: _max_len(key) for key in headers}

        # If the table is wider than the console, shrink flexible columns first.
        console_width = self.console.width
        column_keys = ["idx", "filename", "model_tpl", "prep_tpl", "public", "local", "date", "selected", "idx_end"]
        content_width = sum(widths[k] for k in column_keys)
        column_count = len(column_keys)
        overhead = (column_count * 2) + (column_count + 1)
        max_content_width = max(console_width - overhead, 0)

        if max_content_width and content_width > max_content_width:
            flex_cols = ["filename", "model_tpl", "prep_tpl"]
            min_widths = {k: len(headers[k]) for k in flex_cols}
            excess = content_width - max_content_width
            while excess > 0:
                shrunk = False
                for key in flex_cols:
                    if widths[key] > min_widths[key] and excess > 0:
                        widths[key] -= 1
                        excess -= 1
                        shrunk = True
                if not shrunk:
                    break

        # Build submissions table
        table = Table(title="Top Submissions", box=box.ROUNDED, show_header=True, expand=False)
        table.add_column(headers["idx"], justify="right", style="cyan", width=widths["idx"])
        table.add_column(headers["filename"], style="white", width=widths["filename"])
        table.add_column(headers["model_tpl"], style="cyan", width=widths["model_tpl"])
        table.add_column(headers["prep_tpl"], style="green", width=widths["prep_tpl"])
        table.add_column(headers["public"], justify="right", style="yellow", width=widths["public"])
        table.add_column(headers["local"], justify="right", style="magenta", width=widths["local"])
        table.add_column(headers["date"], style="blue", width=widths["date"])
        table.add_column(headers["selected"], justify="center", style="green", width=widths["selected"])
        table.add_column(headers["idx_end"], justify="right", style="cyan", width=widths["idx_end"])

        for row in rows:
            row_style = "on grey15" if int(row["idx"]) % 2 == 0 else None
            table.add_row(
                row["idx"],
                self._truncate(row["filename"], widths["filename"]),
                self._truncate(row["model_tpl"], widths["model_tpl"]),
                self._truncate(row["prep_tpl"], widths["prep_tpl"]),
                row["public"],
                row["local"],
                row["date"],
                row["selected"],
                row["idx_end"],
                style=row_style,
            )

        self.console.print(table)

        # Commands panel
        commands_text = """[1-9] Select top N  [0] Top 10 (confirm)  [1,3,5] Select IDs
[A] Toggle show all                 [W] Choose weighting strategy
[D] Set display count               [E] Exclude specific
[C] Create CSV                      [S] Create and submit
[M] Multi-strategy batch            [X] Clear selection
[F] Fetch scores from Kaggle        [Q] Quit"""
        commands_panel = Panel(commands_text, title="Commands", box=box.ROUNDED)
        self.console.print(commands_panel)

        # Current config panel
        strategy_display = self.strategy
        if self.strategy == "power":
            strategy_display += f" (power={self.strategy_params['power']:.1f})"
        elif self.strategy == "softmax":
            strategy_display += f" (temp={self.strategy_params['temp']:.4f})"
        elif self.strategy == "offset":
            strategy_display += f" (eps={self.strategy_params['eps']:.1e})"

        # Current config panel
        selection_mode = "specific IDs" if self.selected_indices else f"top {self.selected_count}"
        if self.selected_count > 0 or self.selected_indices:
            selected_subs = self._get_selected_submissions()
            if selected_subs:
                scores = [s["public_score"] for s in selected_subs]
                avg_score = sum(scores) / len(scores) if scores else 0.0
                min_score = min(scores) if scores else 0.0
                max_score = max(scores) if scores else 0.0
                score_range_text = f"Mode: {selection_mode} | Range: {min_score:.5f} - {max_score:.5f} (avg: {avg_score:.5f})"
            else:
                score_range_text = f"Mode: {selection_mode} | No selections"
        else:
            score_range_text = "No selections"

        ensemble_info = "Hidden" if not self.show_all else "Shown"
        filtered_count = len(self.all_submissions) - len(self.filtered_submissions)
        if filtered_count > 0:
            ensemble_info += f" ({filtered_count} filtered)"

        config_text = f"""Strategy: {strategy_display}
{score_range_text}
Display: {self.display_count} rows
Ensembles: {ensemble_info}"""
        config_panel = Panel(config_text, title="Current Config", box=box.ROUNDED)
        self.console.print(config_panel)

        # History panel
        if self.history:
            history_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
            history_table.add_column("Time", style="dim")
            history_table.add_column("Method", style="cyan")
            history_table.add_column("Strategy")
            history_table.add_column("N")
            history_table.add_column("Public", justify="right", style="yellow")
            history_table.add_column("File", style="green")
            history_table.add_column("", style="dim", width=3)  # Submit status

            for h in self.history[-5:]:  # Show last 5
                time_str = h["timestamp"].strftime("%H:%M")
                method_str = h.get("ensemble_method", "blend")  # blend/voting/stacking
                strat_str = h["strategy"]
                n_str = f"top{h['selected_count']}"
                public_score = h.get("public_score")
                public_str = f"{public_score:.5f}" if public_score else "-"
                file_str = h["output_file"]
                submitted_str = "→K" if h.get("submitted", False) else ""  # →K = submitted to Kaggle
                history_table.add_row(time_str, method_str, strat_str, n_str, public_str, file_str, submitted_str)

            history_panel = Panel(history_table, title="Recent Ensembles", box=box.ROUNDED)
            self.console.print(history_panel)

    def handle_number_key(self, n: int) -> None:
        """Handle number key press for top-N selection."""
        if n == self.selected_count and self.selected_indices is None:
            # Pressing same number in top-N mode deselects all
            self.selected_count = 0
        else:
            # Switch to top-N mode and select N submissions
            max_available = len([s for idx, s in enumerate(self.current_view, 1)
                               if idx not in self.excluded_indices])
            self.selected_count = min(n, max_available)
            self.selected_indices = None  # Clear specific mode
            self.excluded_indices.clear()  # Clear exclusions

    def handle_specific_selection(self, cmd: str) -> None:
        """Handle comma-separated ID selection (e.g., '1,3,5')."""
        try:
            # Parse comma-separated indices
            indices = set()
            for part in cmd.split(','):
                idx = int(part.strip())
                if not (1 <= idx <= len(self.current_view)):
                    self.console.print(f"[red]Invalid ID: {idx} (valid range: 1-{len(self.current_view)})[/red]")
                    Prompt.ask("Press Enter to continue", console=self.console)
                    return
                indices.add(idx)

            # Switch to specific selection mode
            self.selected_indices = indices
            self.excluded_indices.clear()  # Clear exclusions when switching modes

            self.console.print(f"[green]Selected IDs: {sorted(indices)}[/green]")
            Prompt.ask("Press Enter to continue", console=self.console)

        except ValueError:
            self.console.print("[red]Invalid format. Use comma-separated numbers (e.g., 1,3,5)[/red]")
            Prompt.ask("Press Enter to continue", console=self.console)

    def toggle_show_all(self) -> None:
        """Toggle between showing all submissions and filtered (no ensembles)."""
        self.show_all = not self.show_all
        self.current_view = self.all_submissions if self.show_all else self.filtered_submissions
        # Reset selection if it exceeds new view
        max_available = len([s for idx, s in enumerate(self.current_view, 1) if idx not in self.excluded_indices])
        self.selected_count = min(self.selected_count, max_available)

    def choose_strategy_menu(self) -> None:
        """Show strategy selection menu and update strategy."""
        self.console.print("\n")
        strategy_panel_text = """1. public    - Direct score-based (default)
2. power     - Emphasize best (requires power)
3. rank      - Position-based (robust)
4. softmax   - Temperature-controlled
5. offset    - From minimum score
6. uniform   - Equal weights for all
7. manual    - Custom weights (advanced)
0. Cancel    - Keep current strategy"""
        strategy_panel = Panel(strategy_panel_text, title="Choose Weighting Strategy", box=box.ROUNDED)
        self.console.print(strategy_panel)

        choice = Prompt.ask("Choice (0-7)", choices=["0", "1", "2", "3", "4", "5", "6", "7"], default="0", console=self.console)

        # Cancel - keep current strategy
        if choice == "0":
            self.console.print(f"[dim]Keeping current strategy: {self.strategy}[/dim]")
            return

        strategy_map = {
            "1": "public",
            "2": "power",
            "3": "rank",
            "4": "softmax",
            "5": "offset",
            "6": "uniform",
            "7": "manual",
        }
        self.strategy = strategy_map[choice]

        # Prompt for parameters if needed
        if self.strategy == "power":
            power = Prompt.ask("Enter power value", default="10.0", console=self.console)
            self.strategy_params["power"] = float(power)
        elif self.strategy == "softmax":
            temp = Prompt.ask("Enter temperature", default="0.001", console=self.console)
            self.strategy_params["temp"] = float(temp)
        elif self.strategy == "offset":
            eps = Prompt.ask("Enter epsilon", default="1e-6", console=self.console)
            self.strategy_params["eps"] = float(eps)
        elif self.strategy == "manual":
            selected_count = len(self._get_selected_submissions())
            if selected_count == 0:
                self.console.print("[red]Error: No submissions selected[/red]")
                self.strategy = "public"
                return
            self.console.print(f"[yellow]Manual weights require {selected_count} values[/yellow]")
            weights_str = Prompt.ask(f"Enter {selected_count} weights (space-separated)", console=self.console)
            try:
                weights = [float(w) for w in weights_str.split()]
                if len(weights) != selected_count:
                    self.console.print(f"[red]Error: Expected {selected_count} weights, got {len(weights)}[/red]")
                    self.strategy = "public"  # Fall back
                else:
                    self.strategy_params["manual_weights"] = weights
            except ValueError:
                self.console.print("[red]Error: Invalid weights format[/red]")
                self.strategy = "public"  # Fall back

    def exclude_submissions(self) -> None:
        """Allow user to manually exclude specific submissions."""
        self.console.print("\n")
        exclude_str = Prompt.ask("Enter submission numbers to exclude (comma-separated)", default="", console=self.console)

        if not exclude_str.strip():
            return

        try:
            indices = [int(x.strip()) for x in exclude_str.split(",")]
            self.excluded_indices.update(indices)
            # Adjust selected_count if we excluded some selected ones
            valid_selected = len([i for i in range(1, self.selected_count + 1) if i not in self.excluded_indices])
            self.selected_count = valid_selected
        except ValueError:
            self.console.print("[red]Error: Invalid format. Use comma-separated numbers like: 1,3,5[/red]")

    def set_display_count(self) -> None:
        """Prompt user to set display count (1-100)."""
        self.console.print("\n")
        current_str = str(self.display_count)

        count_str = Prompt.ask(
            f"Enter display count (1-100, current: {self.display_count})",
            default=current_str,
            console=self.console
        )

        try:
            count = int(count_str)
            if 1 <= count <= 100:
                self.display_count = count
                self.console.print(f"[green]Display count set to {count}[/green]")
            else:
                self.console.print("[red]Display count must be between 1 and 100[/red]")
        except ValueError:
            self.console.print("[red]Invalid number[/red]")

        Prompt.ask("Press Enter to continue", console=self.console)

    def clear_selection(self) -> None:
        """Clear all selection state."""
        self.selected_count = 0
        self.selected_indices = None
        self.excluded_indices.clear()

    def _get_selected_submissions(self) -> List[Dict]:
        """Get the currently selected submissions."""
        selected = []

        if self.selected_indices is not None:
            # Specific ID mode
            for idx in sorted(self.selected_indices):
                if idx <= len(self.current_view) and idx not in self.excluded_indices:
                    selected.append(self.current_view[idx - 1])
        else:
            # Top-N mode
            for idx, sub in enumerate(self.current_view[:self.selected_count], start=1):
                if idx not in self.excluded_indices:
                    selected.append(sub)

        return selected

    def _compute_weights_for_selected(self) -> List[float]:
        """Compute weights for selected submissions."""
        selected = self._get_selected_submissions()
        scores = [s["public_score"] for s in selected]

        if self.strategy == "manual":
            return self.strategy_params.get("manual_weights", [1.0] * len(scores))
        else:
            return compute_weights(
                scores,
                self.strategy,
                self.strategy_params.get("power", 2.0),
                self.strategy_params.get("temp", 0.001),
                self.strategy_params.get("eps", 1e-6),
                None,
            )

    def _preview_blend(self) -> Optional[tuple]:
        """Show preview of blend with weights. Returns (selected, weights, output_name) if confirmed."""
        selected = self._get_selected_submissions()
        if not selected:
            self.console.print("[red]Error: No submissions selected[/red]")
            return None

        weights = self._compute_weights_for_selected()

        self.console.print("\n")

        # Preview table
        preview_table = Table(title=f"Blend Preview - Strategy: {self.strategy}", box=box.ROUNDED)
        preview_table.add_column("#", justify="right", style="cyan", width=4)
        preview_table.add_column("Filename", style="white", width=30)
        preview_table.add_column("Public", justify="right", style="yellow", width=8)
        preview_table.add_column("Weight", justify="right", style="green", width=10)
        preview_table.add_column("%", justify="right", style="magenta", width=6)

        for idx, (sub, weight) in enumerate(zip(selected, weights), start=1):
            filename = sub["filename"]
            public_score = f"{sub['public_score']:.5f}"
            weight_str = f"{weight:.5f}"
            percent_str = f"{weight * 100:.1f}%"
            preview_table.add_row(str(idx), filename, public_score, weight_str, percent_str)

        self.console.print(preview_table)

        # Weight visualization (ASCII bar chart)
        self.console.print("\n[bold]Weight Distribution:[/bold]")
        max_weight = max(weights) if weights else 1.0
        for sub, weight in zip(selected, weights):
            bar_length = int((weight / max_weight) * 40)
            bar = "█" * bar_length
            color = "green" if weight == max_weight else "yellow" if weight > sum(weights) / len(weights) else "red"
            self.console.print(f"[{color}]{bar}[/{color}] {weight:.5f} ({weight * 100:.1f}%) {sub['filename']}")

        # Output name with timestamp: submission-YYYYMMDDHHMMSS-topN-ensemble_method-strategy.csv.gz
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_name = f"submission-{timestamp}-top{len(selected)}-blend-{self.strategy}.csv.gz"
        self.console.print(f"\n[bold]Output:[/bold] {output_name}\n")

        return (selected, weights, output_name)

    def create_csv(self) -> None:
        """Create blended CSV file."""
        preview_result = self._preview_blend()
        if not preview_result:
            return

        selected, weights, output_name = preview_result

        # No confirmation needed - creating CSV is not destructive
        try:
            # Resolve local files
            model_files = [resolve_local_file(self.submissions_dir, r["filename"]) for r in selected]

            # Blend
            with self.console.status("[bold blue]Blending predictions..."):
                submission = blend_predictions(model_files, weights, None, None)

            # Save
            output_path = self.submissions_dir / output_name
            submission.to_csv(output_path, index=False, compression='infer')

            # Save manifest
            manifest = {
                "project": self.project,
                "submissions_dir": str(self.submissions_dir),
                "top_n": len(model_files),
                "weighting": self.strategy,
                **self.strategy_params,
                "weights": weights,
                "excluded_ensembles": not self.show_all,
                "models": [
                    {
                        "kaggle_filename": r["filename"],
                        "public_score": r["public_score"],
                        "date": r["date"].strftime("%Y-%m-%d %H:%M:%S") if r["date"] else None,
                        "weight": w,
                    }
                    for r, w in zip(selected, weights)
                ],
                "output": output_name,
            }
            manifest_path = Path(str(output_path) + ".manifest.json")
            manifest_path.write_text(json.dumps(manifest, indent=2))

            # Add to history and save
            self.history.append({
                "timestamp": datetime.now(),
                "ensemble_method": "blend",  # Future: voting/stacking
                "strategy": self.strategy,
                "selected_count": len(selected),
                "output_file": output_name,
                "avg_score": sum([s["public_score"] for s in selected]) / len(selected),
                "submitted": False,  # Only generated locally
            })
            self._save_history()

            success_panel = Panel(
                f"[green]✓ Created:[/green] {output_path}\n[green]✓ Manifest:[/green] {manifest_path}",
                title="Success",
                box=box.ROUNDED
            )
            self.console.print(success_panel)
            Prompt.ask("Press Enter to continue", console=self.console)

        except Exception as e:
            self.console.print(f"[red]Error: {e}[/red]")
            Prompt.ask("Press Enter to continue", console=self.console)

    def create_and_submit(self) -> None:
        """Create blended CSV and submit to Kaggle."""
        preview_result = self._preview_blend()
        if not preview_result:
            return

        selected, weights, output_name = preview_result

        # Generate detailed description
        desc_parts = [f"blend:{self.strategy}"]
        if self.strategy == "power":
            desc_parts[0] += f"({self.strategy_params['power']:.1f})"
        elif self.strategy == "softmax":
            desc_parts[0] += f"(temp={self.strategy_params['temp']:.4f})"

        desc_parts.append(f"top{len(selected)}")

        # Add file:weight pairs
        file_weights = ", ".join([f"{s['filename'].replace('submission-', '')}({w:.3f})" for s, w in zip(selected, weights)])
        desc_parts.append(file_weights)

        # Add average score
        avg_score = sum([s["public_score"] for s in selected]) / len(selected)
        desc_parts.append(f"avg_public={avg_score:.5f}")

        description = " | ".join(desc_parts)

        # Show submit confirmation
        self.console.print("\n")
        submit_panel_text = f"""File: {output_name}

Description:
{description}

⚠️  This will upload to Kaggle competition: {self.project}"""
        submit_panel = Panel(submit_panel_text, title="Submit to Kaggle", box=box.ROUNDED, border_style="yellow")
        self.console.print(submit_panel)

        if not Confirm.ask("Proceed with submission?", console=self.console):
            self.console.print("[yellow]Cancelled[/yellow]")
            return

        # First create the CSV
        try:
            # Resolve local files
            model_files = [resolve_local_file(self.submissions_dir, r["filename"]) for r in selected]

            # Blend
            with self.console.status("[bold blue]Blending predictions..."):
                submission = blend_predictions(model_files, weights, None, None)

            # Save
            output_path = self.submissions_dir / output_name
            submission.to_csv(output_path, index=False, compression='infer')

            # Decompress .csv.gz to .csv for Kaggle submission
            # Kaggle API requires uncompressed .csv files
            submission_file_for_upload = output_path
            if str(output_path).endswith('.csv.gz'):
                import gzip
                import shutil

                csv_path = Path(str(output_path)[:-3])  # Remove .gz extension
                if not csv_path.exists():
                    self.console.print(f"[dim]Decompressing {output_path.name} for Kaggle upload...[/dim]")
                    with gzip.open(output_path, 'rb') as f_in:
                        with open(csv_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                submission_file_for_upload = csv_path

            # Submit to Kaggle
            with self.console.status("[bold blue]Submitting to Kaggle..."):
                result = subprocess.run(
                    ["kaggle", "competitions", "submit", "-c", self.project, "-f", str(submission_file_for_upload), "-m", description],
                    capture_output=True,
                    text=True,
                    check=True,
                )

            # Add to history and save
            self.history.append({
                "timestamp": datetime.now(),
                "ensemble_method": "blend",  # Future: voting/stacking
                "strategy": self.strategy,
                "selected_count": len(selected),
                "output_file": output_name,
                "avg_score": avg_score,
                "submitted": True,  # Successfully submitted to Kaggle
            })
            self._save_history()

            success_panel = Panel(
                f"[green]✓ Submitted:[/green] {output_name}\n[green]✓ Description:[/green] {description}\n\n{result.stdout}",
                title="Success",
                box=box.ROUNDED
            )
            self.console.print(success_panel)
            Prompt.ask("Press Enter to continue", console=self.console)

        except subprocess.CalledProcessError as e:
            self.console.print(f"[red]Kaggle submission failed:[/red] {e.stderr}")
            Prompt.ask("Press Enter to continue", console=self.console)
        except Exception as e:
            self.console.print(f"[red]Error: {e}[/red]")
            Prompt.ask("Press Enter to continue", console=self.console)

    def multi_strategy_batch(self) -> None:
        """Generate N CSVs with different strategies, submit max 5 to Kaggle."""
        if not self._get_selected_submissions():
            self.console.print("[red]Error: No submissions selected[/red]")
            Prompt.ask("Press Enter to continue", console=self.console)
            return

        self.console.print("\n")
        # Show strategy selection menu
        strategy_menu = """Select strategies (comma-separated numbers):
1. public    - Direct score-based
2. power     - Emphasize best (power=10)
3. rank      - Position-based
4. softmax   - Temperature-controlled (temp=0.001)
5. offset    - From minimum score
6. uniform   - Equal weights"""
        strategy_panel = Panel(strategy_menu, title="Multi-Strategy Batch", box=box.ROUNDED)
        self.console.print(strategy_panel)

        choice_str = Prompt.ask("Select strategies (e.g., 1,2,3)", console=self.console)

        try:
            choices = [int(x.strip()) for x in choice_str.split(",")]
            if not all(1 <= c <= 6 for c in choices):
                raise ValueError("Invalid choice")
        except ValueError:
            self.console.print("[red]Error: Invalid input. Use comma-separated numbers 1-6[/red]")
            Prompt.ask("Press Enter to continue", console=self.console)
            return

        # Map choices to strategies
        strategy_map = {
            1: ("public", {}),
            2: ("power", {"power": 10.0}),
            3: ("rank", {}),
            4: ("softmax", {"temp": 0.001}),
            5: ("offset", {"eps": 1e-6}),
            6: ("uniform", {}),
        }

        strategies = [strategy_map[c] for c in choices]

        self.console.print(f"\n[bold]Generating {len(strategies)} CSV files...[/bold]")
        for i, (strat, _) in enumerate(strategies, 1):
            self.console.print(f"  {i}. {strat}")

        if len(strategies) > 5:
            self.console.print(f"[yellow]Note: Kaggle daily limit is 5. Will submit first 5 only.[/yellow]")

        self.console.print()  # Empty line

        # Generate all CSVs
        generated = []
        selected = self._get_selected_submissions()

        for strat_name, strat_params in strategies:
            try:
                # Compute weights
                scores = [s["public_score"] for s in selected]
                weights = compute_weights(
                    scores,
                    strat_name,
                    strat_params.get("power", 2.0),
                    strat_params.get("temp", 0.001),
                    strat_params.get("eps", 1e-6),
                    None,
                )

                # Generate output name
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                output_name = f"submission-{timestamp}-top{len(selected)}-blend-{strat_name}.csv.gz"

                # Resolve local files and blend
                model_files = [resolve_local_file(self.submissions_dir, r["filename"]) for r in selected]
                submission = blend_predictions(model_files, weights, None, None)

                # Save
                output_path = self.submissions_dir / output_name
                submission.to_csv(output_path, index=False, compression='infer')

                generated.append({
                    "strategy": strat_name,
                    "output_name": output_name,
                    "output_path": output_path,
                    "weights": weights,
                })

                self.console.print(f"[green]✓[/green] Generated: {output_name}")

                # Small delay to ensure unique timestamps
                time.sleep(0.1)

            except Exception as e:
                self.console.print(f"[red]✗ Failed {strat_name}: {e}[/red]")

        if not generated:
            self.console.print("[red]No files generated successfully[/red]")
            Prompt.ask("Press Enter to continue", console=self.console)
            return

        # Add all generated files to history (submitted=False initially)
        avg_score = sum([s["public_score"] for s in selected]) / len(selected)
        for gen in generated:
            self.history.append({
                "timestamp": datetime.now(),
                "ensemble_method": "blend",
                "strategy": gen["strategy"],
                "selected_count": len(selected),
                "output_file": gen["output_name"],
                "avg_score": avg_score,
                "submitted": False,  # Not submitted yet
            })
        self._save_history()

        # Ask about submission
        self.console.print(f"\n[bold]Generated {len(generated)} files successfully[/bold]")

        if not Confirm.ask("\nSubmit to Kaggle?", console=self.console):
            self.console.print("[yellow]Files saved but not submitted[/yellow]")
            Prompt.ask("Press Enter to continue", console=self.console)
            return

        # Submit (max 5)
        to_submit = generated[:5]
        self.console.print(f"\n[bold]Submitting {len(to_submit)} files to Kaggle...[/bold]")

        for i, gen in enumerate(to_submit, 1):
            try:
                # Generate description
                avg_score = sum([s["public_score"] for s in selected]) / len(selected)
                desc_parts = [f"blend:{gen['strategy']}", f"top{len(selected)}", f"avg_public={avg_score:.5f}"]
                description = " | ".join(desc_parts)

                self.console.print(f"\n[{i}/{len(to_submit)}] Submitting {gen['output_name']}...")

                # Decompress .csv.gz to .csv for Kaggle submission
                submission_file_for_upload = gen["output_path"]
                if str(gen["output_path"]).endswith('.csv.gz'):
                    import gzip
                    import shutil

                    csv_path = Path(str(gen["output_path"])[:-3])  # Remove .gz extension
                    if not csv_path.exists():
                        self.console.print(f"[dim]Decompressing {gen['output_name']} for Kaggle upload...[/dim]")
                        with gzip.open(gen["output_path"], 'rb') as f_in:
                            with open(csv_path, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                    submission_file_for_upload = csv_path

                result = subprocess.run(
                    ["kaggle", "competitions", "submit", "-c", self.project, "-f", str(submission_file_for_upload), "-m", description],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                self.console.print(f"[green]✓[/green] Submitted successfully")

                # Update history entry to mark as submitted
                for h in self.history:
                    if h.get("output_file") == gen["output_name"]:
                        h["submitted"] = True
                        break

                # Sleep between submissions (except after last one)
                if i < len(to_submit):
                    self.console.print("[dim]Waiting 10 seconds...[/dim]")
                    time.sleep(10)

            except subprocess.CalledProcessError as e:
                self.console.print(f"[red]✗ Submission failed: {e.stderr}[/red]")

        # Save history
        self._save_history()

        self.console.print("\n[bold green]Batch submission complete![/bold green]")
        Prompt.ask("Press Enter to continue", console=self.console)

    def fetch_and_update_scores(self) -> None:
        """Fetch public scores from Kaggle and update history."""
        self.console.print("\n")

        try:
            with self.console.status("[bold blue]Fetching submissions from Kaggle..."):
                df = fetch_kaggle_submissions(self.project)

            records = df.to_dict('records')

            # Build filename -> public_score mapping
            kaggle_scores = {}
            for record in records:
                filename = record.get("fileName", "")
                public_score = record.get("publicScore")
                if filename and public_score is not None and not pd.isna(public_score):
                    try:
                        kaggle_scores[filename] = float(public_score)
                    except (ValueError, TypeError):
                        pass

            # Update history entries that are submitted but missing public_score
            updated_count = 0
            for h in self.history:
                if h.get("submitted") and not h.get("public_score"):
                    filename = h.get("output_file")
                    # Try exact match first, then without .gz extension
                    if filename in kaggle_scores:
                        h["public_score"] = kaggle_scores[filename]
                        updated_count += 1
                    elif filename.endswith('.csv.gz'):
                        filename_csv = filename[:-3]  # Remove .gz
                        if filename_csv in kaggle_scores:
                            h["public_score"] = kaggle_scores[filename_csv]
                            updated_count += 1

            # Save updated history
            if updated_count > 0:
                self._save_history()
                self.console.print(f"[green]✓ Updated {updated_count} score(s) from Kaggle[/green]")
            else:
                self.console.print("[yellow]No new scores to update[/yellow]")

        except Exception as e:
            self.console.print(f"[red]Error fetching scores: {e}[/red]")

        Prompt.ask("Press Enter to continue", console=self.console)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend top-N Kaggle submissions.")
    parser.add_argument("--project", required=True, help="Kaggle project slug (e.g., playground-series-s5e12)")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode with rich CLI (default if no blend args provided)",
    )
    parser.add_argument(
        "--kaggle-csv",
        default=None,
        help="Path to pre-downloaded Kaggle CSV (optional, will fetch from API if not provided)",
    )
    parser.add_argument(
        "--submissions-dir",
        default=None,
        help="Override submissions directory (defaults to {repo_root}/projects/kaggle/<project>/submissions)",
    )
    parser.add_argument("--top-n", type=int, default=5, help="Number of top submissions to blend")
    parser.add_argument(
        "--include-ensembles",
        action="store_true",
        help="Include already blended/stacked submissions (default: exclude)",
    )
    parser.add_argument(
        "--weighting",
        choices=["public", "rank", "power", "softmax", "offset", "uniform", "manual"],
        default="public",
        help="Weighting strategy for blending",
    )
    parser.add_argument("--power", type=float, default=2.0, help="Power for power weighting")
    parser.add_argument("--temp", type=float, default=0.001, help="Temperature for softmax weighting")
    parser.add_argument("--eps", type=float, default=1e-6, help="Epsilon for offset weighting")
    parser.add_argument("--weights", nargs="+", type=float, help="Manual weights (only for --weighting manual)")
    parser.add_argument("--output-name", default=None, help="Output CSV filename")
    parser.add_argument("--id-column", default=None, help="ID column name override")
    parser.add_argument("--target-column", default=None, help="Target column name override")
    return parser.parse_args()


def main_interactive(args: argparse.Namespace) -> None:
    """Run interactive mode with rich CLI."""
    console = Console()

    if args.submissions_dir:
        submissions_dir = Path(args.submissions_dir)
    else:
        repo_root = Path(__file__).resolve().parents[1]
        submissions_dir = repo_root / "projects" / "kaggle" / args.project / "submissions"

    blender = InteractiveBlender(args.project, submissions_dir, console)

    # Fetch data
    try:
        with console.status("[bold blue]Fetching submissions from Kaggle..."):
            blender.fetch_and_parse()
    except Exception as e:
        console.print(f"[red]Error fetching submissions: {e}[/red]")
        return

    # Main loop
    while True:
        blender.render_main_screen()
        cmd = Prompt.ask("\nCommand", console=console).strip()

        # Check for comma-separated selection (e.g., "1,3,5") before upper()
        if ',' in cmd:
            blender.handle_specific_selection(cmd)
        elif cmd.upper() == 'Q':
            console.print("[yellow]Goodbye![/yellow]")
            break
        elif cmd == '0':
            if Confirm.ask("0 selects top 10. Continue?", console=console):
                blender.handle_number_key(10)
        elif cmd.upper() in '123456789':
            blender.handle_number_key(int(cmd))
        elif cmd.upper() == 'A':
            blender.toggle_show_all()
        elif cmd.upper() == 'W':
            blender.choose_strategy_menu()
        elif cmd.upper() == 'D':
            blender.set_display_count()
        elif cmd.upper() == 'E':
            blender.exclude_submissions()
        elif cmd.upper() == 'X':
            blender.clear_selection()
        elif cmd.upper() == 'C':
            blender.create_csv()
        elif cmd.upper() == 'S':
            blender.create_and_submit()
        elif cmd.upper() == 'M':
            blender.multi_strategy_batch()
        elif cmd.upper() == 'F':
            blender.fetch_and_update_scores()
        else:
            # Try parsing as plain number (e.g., "15" for top 15)
            try:
                n = int(cmd)
                if n <= 0:
                    console.print("[red]Invalid selection: use a positive number (e.g., 10) or X to clear.[/red]")
                    Prompt.ask("Press Enter to continue", console=console)
                else:
                    blender.handle_number_key(n)
            except ValueError:
                console.print(f"[red]Unknown command: {cmd}[/red]")
                Prompt.ask("Press Enter to continue", console=console)


def main_non_interactive(args: argparse.Namespace) -> None:
    """Run non-interactive mode (original CLI behavior)."""
    if args.submissions_dir:
        submissions_dir = Path(args.submissions_dir)
    else:
        repo_root = Path(__file__).resolve().parents[1]
        submissions_dir = repo_root / "projects" / "kaggle" / args.project / "submissions"

    if not submissions_dir.exists():
        raise FileNotFoundError(f"Submissions dir not found: {submissions_dir}")

    # Fetch or load submissions
    if args.kaggle_csv:
        print(f"Loading submissions from {args.kaggle_csv}")
        df = pd.read_csv(args.kaggle_csv, compression='infer')
    else:
        print(f"Fetching submissions from Kaggle API for {args.project}...")
        df = fetch_kaggle_submissions(args.project)

    rows = parse_kaggle_csv(df)
    print(f"Found {len(rows)} submissions with public scores")

    selected = select_top_n(rows, args.top_n, exclude_ensembles=not args.include_ensembles)
    if not selected:
        raise RuntimeError("No submissions selected for blending")

    model_files = [resolve_local_file(submissions_dir, r["filename"]) for r in selected]
    scores = [r["public_score"] for r in selected]
    weights = compute_weights(scores, args.weighting, args.power, args.temp, args.eps, args.weights)

    output_name = args.output_name or f"submission-blend-top{len(model_files)}-{args.weighting}.csv.gz"
    output_path = submissions_dir / output_name

    submission = blend_predictions(model_files, weights, args.id_column, args.target_column)
    submission.to_csv(output_path, index=False, compression='infer')

    manifest = {
        "project": args.project,
        "submissions_dir": str(submissions_dir),
        "top_n": len(model_files),
        "weighting": args.weighting,
        "power": args.power,
        "temp": args.temp,
        "eps": args.eps,
        "weights": weights,
        "excluded_ensembles": not args.include_ensembles,
        "models": [
            {
                "kaggle_filename": r["filename"],
                "local_filename": f.name,
                "public_score": r["public_score"],
                "date": r["date"].strftime("%Y-%m-%d %H:%M:%S") if r["date"] else None,
            }
            for r, f in zip(selected, model_files)
        ],
        "output": output_path.name,
    }
    manifest_path = Path(str(output_path) + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("Top submissions:")
    for r, f, w in zip(selected, model_files, weights):
        date = r["date"].strftime("%Y-%m-%d %H:%M:%S") if r["date"] else "-"
        print(f"- {r['filename']} | public={r['public_score']:.5f} | date={date} | local={f.name} | weight={w:.6f}")
    print(f"\nSaved blended submission: {output_path}")
    print(f"Saved manifest: {manifest_path}")


def main() -> None:
    """Main entry point - detects interactive vs non-interactive mode."""
    args = parse_args()

    # Detect mode: interactive if --interactive flag OR if no blend-specific args provided
    is_interactive = args.interactive or (
        args.top_n == 5
        and args.weighting == "public"
        and not args.kaggle_csv
        and not args.output_name
    )

    if is_interactive:
        main_interactive(args)
    else:
        main_non_interactive(args)


if __name__ == "__main__":
    main()
