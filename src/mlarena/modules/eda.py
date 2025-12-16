"""Exploratory Data Analysis module.

Generates profiling reports for train/test and captures lightweight summaries in state.
"""

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


# Keys to remove from ydata profiling output
PROFILE_REMOVE_KEYS = {
    "value_counts_without_nan",
    "value_counts_index_sorted",
    "histogram",
    "histogram_length",
    "character_counts",
    "block_alias_values",
    "category_alias_values",
    "block_alias_char_counts",
    "script_char_counts",
    "category_alias_char_counts",
    "package",
    "analysis",
    "time_index_analysis",
}
WORD_COUNT_LIMIT = 50


def _sanitize_payload(payload: Any) -> Any:
    """Recursively remove heavy keys from nested dict/list structure."""
    def _clean(node: Any) -> Any:
        if isinstance(node, dict):
            cleaned = {}
            for key, value in node.items():
                if key in PROFILE_REMOVE_KEYS:
                    continue
                if key == "word_counts" and isinstance(value, dict):
                    cleaned[key] = value if len(value) <= WORD_COUNT_LIMIT else {}
                    continue
                cleaned[key] = _clean(value)
            return cleaned
        if isinstance(node, list):
            return [_clean(item) for item in node]
        return node

    return _clean(payload)


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
        # Remove heavy keys from profiling data
        trimmed = _sanitize_payload(trimmed)
    except Exception:
        trimmed = {}
    return {
        "summary": trimmed,
        "html": str(output_html),
        "json": str(output_json),
    }


@ModuleRegistry.register
class EDAModule(BaseModule):
    """Run exploratory data analysis for a project."""

    name = "eda"
    description = "Exploratory data analysis"

    @classmethod
    def register_cli_args(cls, parser) -> None:
        """
        Register CLI arguments for the EDA module.

        Args:
            parser: Argparse parser for the ``eda`` subcommand.
        """
        parser.add_argument("--eda-notes", type=str, default="", help="Optional notes saved with the EDA artifact.")

    def execute(self) -> ModuleResult:
        """
        Profile train/test datasets and persist summary artifacts.

        Returns:
            ModuleResult containing profile paths and summary payload.

        Raises:
            RuntimeError: When profiling fails unexpectedly.
        """
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

        return ModuleResult(
            success=True,
            payload={
                "summary_file": str(summary_path),
                "train_profile": train_summary,
                "test_profile": test_summary,
                "train_profile_path": str(train_html),
                "test_profile_path": str(test_html),
                "target": target_analysis,
            },
            artifacts=[summary_path, train_html, train_json, test_html, test_json],
        )
