"""Experiments listing module (compat with legacy experiment_manager list)."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.align import Align
from rich.panel import Panel
from rich.table import Table

from mlarena.core.module import BaseModule, ModuleResult
from mlarena.core.registry import ModuleRegistry

console = Console()


def _extract_module_argv(argv: List[str], module_name: str) -> List[str]:
    if module_name in argv:
        idx = argv.index(module_name)
        return argv[idx + 1 :]
    return list(argv)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", default="list")
    parser.add_argument("--show-table", action="store_true")
    parser.add_argument("--show-table-compact", action="store_true")
    parser.add_argument("--status", default="completed", help="Filter by status (default: completed). Use 'all' for everything.")
    parser.add_argument("--show-submission-name", action="store_true", help="Show full submission filename instead of checkmark.")
    parser.add_argument("--sort-by", choices=["id", "local", "public", "started", "template", "module"], default="started", help="Sort by column.")
    parser.add_argument("--reverse", action="store_true", help="Reverse sort order.")
    return parser


def _parse_args(argv: List[str]) -> Tuple[argparse.Namespace, List[str]]:
    parser = _build_parser()
    return parser.parse_known_args(argv)


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        clean = value.replace("Z", "+00:00")
        return datetime.fromisoformat(clean)
    except Exception:
        return None


def _format_duration(start: Optional[datetime], end: Optional[datetime]) -> str:
    if not start:
        return "-"
    end_time = end or datetime.now(timezone.utc)
    delta = end_time - start
    seconds = int(delta.total_seconds())
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}h {minutes:02d}m"
    return f"{minutes:02d}m {sec:02d}s"


def _format_ts(ts_str: Optional[str]) -> str:
    dt = _parse_timestamp(ts_str)
    if not dt:
        return "-"
    return dt.strftime("%Y%m%d %H%M%S")


def _format_time_limit(time_limit_val: Optional[float]) -> str:
    if not isinstance(time_limit_val, (int, float)) or not time_limit_val:
        return "-"
    if time_limit_val >= 3600:
        hours = time_limit_val / 3600
        return f"{hours:.1f}h"
    if time_limit_val >= 60:
        minutes = time_limit_val / 60
        return f"{minutes:.0f}m"
    return f"{int(time_limit_val)}s"


def _format_status_icon(status: str) -> str:
    if status == "completed":
        return "[green]✓[/green]"
    if status == "failed":
        return "[red]✗[/red]"
    if status == "running":
        return "[blue]⟳[/blue]"
    if status == "pending":
        return "[yellow]•[/yellow]"
    return status or "-"


def _module_icon(module: str) -> str:
    if not module:
        return "-"
    key = module.lower().replace("_", "-")
    icon_map = {
        "preprocess": "🔧",
        "model": "🧠",
        "eda": "🔍",
        "predict": "🎯",
        "submit": "📨",
        "fetch-score": "📈",
        "init": "⏻",
        "blend": "🧬",
        "ensemble": "🧬",
    }
    return icon_map.get(key, "-")


@ModuleRegistry.register
class ExperimentsModule(BaseModule):
    name = "experiments"
    description = "List experiments and results"

    def execute(self) -> ModuleResult:
        raw_argv = []
        if self.context and self.context.state:
            raw_argv = self.context.state.run.get("argv", [])

        module_argv = _extract_module_argv(raw_argv, "experiments")
        args, _ = _parse_args(module_argv)
        if args.command != "list":
            return ModuleResult(success=False, error=f"Unknown command: {args.command}")

        project_label = self.context.project_name
        project_root = self.context.project_root
        base_dir = project_root / "experiments"
        if not base_dir.exists():
            console.print("[yellow]No experiments found.[/yellow]")
            return ModuleResult(success=True, payload={"count": 0})

        view_table = bool(args.show_table)
        view_table_compact = bool(args.show_table_compact)
        if not view_table and not view_table_compact:
            view_table_compact = True
        use_vertical = False

        status_filter = args.status
        show_sub_name = args.show_submission_name

        table = Table(title=f"Experiments for {project_label}", show_lines=False)
        if view_table or view_table_compact:
            table.add_column("Experiment", style="cyan", no_wrap=True)
            table.add_column("State", style="white")
            table.add_column("Module", style="cyan")
            table.add_column("Template", style="magenta")
            if view_table:
                table.add_column("Preset", style="magenta")
                table.add_column("GPU")
                table.add_column("TimeLimit")
            table.add_column("Local CV")
            table.add_column("Public")
            table.add_column("Started", style="dim")
            table.add_column("Elapsed", style="dim")
            if view_table:
                table.add_column("Submission", overflow="fold", justify="center")
                table.add_column("Git", style="dim")

        experiments_list = []

        for dir_path in base_dir.glob("exp-*"):
            state_path = dir_path / "state.json"
            if not state_path.exists():
                continue
            try:
                data = json.loads(state_path.read_text())
            except Exception:
                continue

            modules = data.get("modules", {})
            model_mod = modules.get("model", {})
            predict_mod = modules.get("predict", {})
            submit_mod = modules.get("submit", {})
            fetch_mod = modules.get("fetch-score") or modules.get("fetch_score", {})

            last_module = "-"
            last_status = "-"
            last_ts = None
            last_entry: Dict[str, Any] = {}
            for name, mod in modules.items():
                ts = mod.get("updated_at") or mod.get("finished_at") or mod.get("started_at")
                if ts and (last_ts is None or ts > last_ts):
                    last_ts = ts
                    last_module = name
                    last_status = mod.get("status", "-")
                    last_entry = mod

            # Status filtering
            if status_filter != "all" and last_status != status_filter:
                continue

            model_payload = model_mod.get("payload", {}) or {}
            predict_payload = predict_mod.get("payload", {}) or {}
            submit_payload = submit_mod.get("payload", {}) or {}
            fetch_payload = fetch_mod.get("payload", {}) or {}

            template = (
                model_payload.get("template")
                or model_mod.get("template")
                or (model_mod.get("invocation") or {}).get("model_template")
                or predict_payload.get("template")
                or (predict_mod.get("invocation") or {}).get("model_template")
                or "-"
            )

            local_cv = (
                model_payload.get("local_cv")
                or model_payload.get("local_cv_score")
                or model_mod.get("local_cv")
            )
            if isinstance(local_cv, (int, float)) and not math.isnan(local_cv):
                local_cv_str = f"{local_cv:.5f}"
            else:
                local_cv_str = "-"

            public_score = None
            if fetch_mod.get("status") == "completed":
                public_score = fetch_payload.get("score")
            elif submit_mod.get("public_score") is not None:
                public_score = submit_mod.get("public_score")
            public_str = f"{public_score:.5f}" if isinstance(public_score, (int, float)) else "-"

            preset_val = model_payload.get("preset") or (model_payload.get("training_summary") or {}).get("preset")
            if not preset_val:
                config_data = model_mod.get("config") or {}
                hyper = config_data.get("hyperparameters") or {}
                preset_val = hyper.get("presets") or hyper.get("preset")
            preset_str = preset_val or "-"

            use_gpu_val = model_payload.get("use_gpu")
            if use_gpu_val is None:
                use_gpu_val = (model_payload.get("training_summary") or {}).get("use_gpu")
            if use_gpu_val is None:
                config_data = model_mod.get("config") or {}
                hyper = config_data.get("hyperparameters") or {}
                use_gpu_val = hyper.get("use_gpu")
            use_gpu_str = "-" if use_gpu_val is None else str(int(bool(use_gpu_val)))

            time_limit_val = model_payload.get("time_limit")
            if time_limit_val is None:
                time_limit_val = (model_payload.get("training_summary") or {}).get("time_limit")
            if time_limit_val is None:
                config_data = model_mod.get("config") or {}
                hyper = config_data.get("hyperparameters") or {}
                time_limit_val = hyper.get("time_limit")
            time_limit_str = _format_time_limit(time_limit_val)

            started_ts = (
                last_entry.get("started_at")
                or model_mod.get("started_at")
                or predict_mod.get("started_at")
                or data.get("created_at")
            )
            finished_ts = last_entry.get("finished_at")
            started_dt = _parse_timestamp(started_ts)
            finished_dt = _parse_timestamp(finished_ts)
            elapsed = _format_duration(started_dt, finished_dt)

            submission_file = (
                predict_payload.get("submission_file")
                or predict_payload.get("predictions")
                or model_mod.get("submission_file")
                or submit_payload.get("submission_file")
            )
            submission_display = "-"
            if submission_file:
                if show_sub_name:
                    submission_name = Path(str(submission_file)).name
                    submission_display = f".../{submission_name}"
                else:
                    submission_display = "✓"

            experiments_list.append({
                "id": data.get("experiment_id", "-"),
                "status": _format_status_icon(last_status),
                "module": _module_icon(last_module),
                "module_name": last_module,
                "template": template,
                "preset": preset_str,
                "gpu": use_gpu_str,
                "time": time_limit_str,
                "local": local_cv_str,
                "public": public_str,
                "started": _format_ts(started_ts),
                "elapsed": elapsed,
                "submission": submission_display,
                "git": (data.get("git") or {}).get("hash", (data.get("git") or {}).get("commit", "-"))[:7],
                "started_dt": started_dt, # For sorting
                "local_val": local_cv if isinstance(local_cv, (int, float)) else -float('inf'),
                "public_val": public_score if isinstance(public_score, (int, float)) else -float('inf'),
            })

        # Dynamic sorting
        sort_key_map = {
            "id": lambda x: x["id"],
            "local": lambda x: x["local_val"],
            "public": lambda x: x["public_val"],
            "started": lambda x: x["started_dt"] or datetime.min.replace(tzinfo=timezone.utc),
            "template": lambda x: x["template"],
            "module": lambda x: x["module_name"],
        }
        
        sort_fn = sort_key_map.get(args.sort_by, sort_key_map["started"])
        
        # If user didn't specify --reverse, we use smart defaults:
        # - started: newest first (reverse=True)
        # - local/public: highest first (reverse=True) - assuming higher is better for now, or just consistent with scores
        # - others: ascending
        
        is_reverse = args.reverse
        if not is_reverse and args.sort_by in ["started", "local", "public"]:
            is_reverse = True
            
        experiments_list.sort(key=sort_fn, reverse=is_reverse)

        count = 0
        for item in experiments_list:
            if use_vertical:
                lines = [
                    f"[cyan]{item['id']}[/cyan]",
                    f"  [dim]state[/dim]: [white]{item['status']}[/white] ([white]{item['module_name']}[/white])",
                    f"  [dim]template[/dim]: [white]{item['template']}[/white]",
                    f"  [dim]preset[/dim]: [white]{item['preset']}[/white]",
                    f"  [dim]gpu[/dim]: [white]{item['gpu']}[/white]",
                    f"  [dim]time_limit[/dim]: [white]{item['time']}[/white]",
                    f"  [dim]local_cv[/dim]: [white]{item['local']}[/white]",
                    f"  [dim]public[/dim]: [white]{item['public']}[/white]",
                    f"  [dim]started[/dim]: [white]{item['started']}[/white]",
                    f"  [dim]elapsed[/dim]: [white]{item['elapsed']}[/white]",
                    f"  [dim]submission[/dim]: [white]{item['submission'] if item['submission'] != '-' else '<none>'}[/white]",
                    f"  [dim]git[/dim]: [white]{item['git']}[/white]",
                ]
                console.print(
                    Panel.fit(
                        "\n".join(lines),
                        title=None,
                        border_style="blue",
                    )
                )
            else:
                row = [
                    item['id'],
                    Align.center(item['status']),
                    Align.center(item['module']),
                    item['template'],
                ]
                if view_table:
                    row.extend([
                        item['preset'],
                        Align.right(item['gpu']),
                        Align.right(item['time']),
                    ])
                row.extend(
                    [
                        Align.right(item['local']),
                        Align.right(item['public']),
                        item['started'],
                        item['elapsed'],
                    ]
                )
                if view_table:
                    row.extend([item['submission'], item['git']])
                table.add_row(*row)

            count += 1

        if not use_vertical:
            console.print(table)
            console.print(
                "[dim]Legend: 🔧 preprocess | 🧠 model | 🔍 eda | 🎯 predict | 📨 submit | "
                "📈 fetch-score | ⏻ init | 🧬 blend/ensemble[/dim]"
            )

        return ModuleResult(success=True, payload={"count": count})
