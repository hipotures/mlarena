"""Exploratory Data Analysis module."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from rich.console import Console
from rich.table import Table

from mlarena.core.module import BaseModule, ModuleResult
from mlarena.core.registry import ModuleRegistry
from mlarena.utils.project import data_paths, load_project_config


def _safe_profile(df: "pd.DataFrame", title: str, output_html: Path, output_json: Path) -> Dict[str, Any]:
    """
    Run ydata-profiling when available; otherwise save describe() summary.
    Returns a lightweight summary dict.
    """
    try:
        from ydata_profiling import ProfileReport
    except Exception:
        summary = df.describe(include="all").to_dict()
        output_json.write_text(json.dumps(summary, indent=2))
        output_html.write_text("<html><body><p>Profiling unavailable.</p></body></html>")
        return {"summary": summary, "html": str(output_html), "json": str(output_json)}

    profile = ProfileReport(df, title=title, minimal=True, infer_dtypes=True, progress_bar=False)
    profile.to_file(str(output_html))
    raw = profile.to_json()
    output_json.write_text(raw)
    try:
        payload = json.loads(raw)
        trimmed = {
            "table": payload.get("table", {}),
            "variables": payload.get("variables", {}),
            "alerts": payload.get("alerts", []),
        }
    except Exception:
        trimmed = {}
    return {
        "summary": trimmed,
        "html": str(output_html),
        "json": str(output_json),
    }


@ModuleRegistry.register
class EDAModule(BaseModule):
    name = "eda"
    description = "Exploratory data analysis"

    @classmethod
    def register_cli_args(cls, parser) -> None:
        parser.add_argument("--eda-notes", type=str, default="", help="Optional notes saved with the EDA artifact.")

    def execute(self) -> ModuleResult:
        import pandas as pd

        artifact_dir: Path = self.context.artifact_dir
        artifact_dir.mkdir(parents=True, exist_ok=True)

        config = self.context.config_module or load_project_config(self.context.project_root)
        train_path, test_path = data_paths(config)

        notes: str = self.invocation_params.get("eda_notes", "")

        if not train_path.exists() or not test_path.exists():
            placeholder = artifact_dir / "eda_placeholder.json"
            payload = {
                "status": "skipped",
                "reason": "train/test missing",
                "notes": notes,
            }
            placeholder.write_text(json.dumps(payload, indent=2))
            return ModuleResult(success=True, payload=payload, artifacts=[placeholder])

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        train_html = artifact_dir / "train_profile.html"
        train_json = artifact_dir / "train_profile.json"
        test_html = artifact_dir / "test_profile.html"
        test_json = artifact_dir / "test_profile.json"

        train_summary = _safe_profile(train_df, f"{self.context.project_name} - Train", train_html, train_json)
        test_summary = _safe_profile(test_df, f"{self.context.project_name} - Test", test_html, test_json)

        target_col = getattr(config, "TARGET_COLUMN", None)
        target_analysis = None
        if target_col and target_col in train_df.columns:
            series = train_df[target_col]
            target_analysis = {
                "unique": series.nunique(),
                "nulls": int(series.isna().sum()),
                "sample": series.head(5).tolist(),
            }

        summary_path = artifact_dir / "eda_summary.json"
        summary_payload = {
            "train": train_summary,
            "test": test_summary,
            "notes": notes,
            "target_column": target_col,
            "target_analysis": target_analysis,
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2))

        # Display Rich output
        console = Console(force_terminal=True)

        # Dataset info table
        data_table = Table(title="Dataset Information", show_header=True)
        data_table.add_column("Dataset", style="cyan")
        data_table.add_column("Rows", style="green", justify="right")
        data_table.add_column("Columns", style="green", justify="right")
        data_table.add_row("Train", str(len(train_df)), str(len(train_df.columns)))
        data_table.add_row("Test", str(len(test_df)), str(len(test_df.columns)))
        console.print(data_table)

        # Target column info
        if target_analysis:
            console.print(f"\n[bold]Target Column:[/bold] [cyan]{target_col}[/cyan]")
            console.print(f"  Unique values: {target_analysis['unique']}")
            console.print(f"  Null values:   {target_analysis['nulls']}")
            console.print(f"  Sample:        {target_analysis['sample']}")

        # Generated files
        console.print(f"\n[bold green]✓[/bold green] EDA profiles generated:")
        console.print(f"  Train: [cyan]{train_html.relative_to(self.context.project_root)}[/cyan]")
        console.print(f"  Test:  [cyan]{test_html.relative_to(self.context.project_root)}[/cyan]")
        console.print(f"  Summary: [cyan]{summary_path.relative_to(self.context.project_root)}[/cyan]")

        # Print next steps
        from mlarena.core.module import print_next_steps
        print_next_steps("eda", self.context.project_name, self.context.experiment_id, console)

        return ModuleResult(
            success=True,
            payload={
                "summary_file": str(summary_path),
                "train_profile": train_summary,
                "test_profile": test_summary,
                "target": target_analysis,
            },
            artifacts=[summary_path, train_html, train_json, test_html, test_json],
        )
