"""Data preprocessing module."""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

from mlarena.core.module import BaseModule, ModuleResult
from mlarena.core.registry import ModuleRegistry
from mlarena.core.config import TemplateLoader
from mlarena.utils.project import data_paths, load_project_config


@ModuleRegistry.register
class PreprocessModule(BaseModule):
    name = "preprocess"
    description = "Data preprocessing"

    @classmethod
    def register_cli_args(cls, parser) -> None:
        parser.add_argument("--preprocess-template", type=str, default="identity", help="Name of preprocessing template.")
        parser.add_argument("--cache", action="store_true", help="Reuse existing processed files if present.")

    def _drop_columns(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        return df.drop(columns=[c for c in cols if c in df.columns], errors="ignore")

    def _apply_template(self, df: pd.DataFrame, template: Dict[str, Any]) -> pd.DataFrame:
        # Basic operations: drop_columns, fillna
        drop_cols = template.get("drop_columns", [])
        if drop_cols:
            df = self._drop_columns(df, drop_cols)
        fills = template.get("fillna", {})
        if fills:
            df = df.fillna(fills)
        return df

    def execute(self) -> ModuleResult:
        artifact_dir: Path = self.context.artifact_dir
        artifact_dir.mkdir(parents=True, exist_ok=True)
        config = self.context.config_module or load_project_config(self.context.project_root)
        train_path, test_path = data_paths(config)
        template_name = self.invocation_params.get("preprocess_template", "identity")
        cache_ok = bool(self.invocation_params.get("cache"))

        processed_train = artifact_dir / "train_processed.csv"
        processed_test = artifact_dir / "test_processed.csv"

        if cache_ok and processed_train.exists() and processed_test.exists():
            return ModuleResult(
                success=True,
                payload={
                    "train_processed": str(processed_train),
                    "test_processed": str(processed_test),
                    "cached": True,
                    "template": template_name,
                },
                artifacts=[processed_train, processed_test],
            )

        if not train_path.exists() or not test_path.exists():
            marker = artifact_dir / "preprocess_skipped.txt"
            marker.write_text("Missing train/test; preprocess skipped.")
            return ModuleResult(success=True, payload={"skipped": True}, artifacts=[marker])

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        ignored = getattr(config, "IGNORED_COLUMNS", []) or []
        train_df = self._drop_columns(train_df, ignored)
        test_df = self._drop_columns(test_df, ignored)

        template_cfg = TemplateLoader(self.context.project_root).load(template_name)
        if template_cfg:
            train_df = self._apply_template(train_df, template_cfg)
            test_df = self._apply_template(test_df, template_cfg)

        train_df.to_csv(processed_train, index=False)
        test_df.to_csv(processed_test, index=False)

        return ModuleResult(
            success=True,
            payload={
                "train_processed": str(processed_train),
                "test_processed": str(processed_test),
                "ignored_columns": ignored,
                "template": template_name,
                "cached": False,
            },
            artifacts=[processed_train, processed_test],
        )
