"""Feature engineering module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# pandas imported in execute()

from mlarena.core.module import BaseModule, ModuleResult
from mlarena.core.registry import ModuleRegistry
from mlarena.core.config import TemplateLoader
from mlarena.utils.project import data_paths, load_project_config


@ModuleRegistry.register
class FeatureModule(BaseModule):
    name = "feat"
    description = "Feature engineering"

    @classmethod
    def register_cli_args(cls, parser) -> None:
        parser.add_argument("--feat-template", type=str, default="identity", help="Feature template name.")

    def _apply_ops(self, df: pd.DataFrame, template: Dict[str, Any]) -> pd.DataFrame:
        df = df.copy()
        for col in template.get("log1p", []):
            if col in df.columns:
                df[col] = df[col].apply(lambda x: x if x is None else pd.Series([x]).log1p().iloc[0])
        for ratio in template.get("ratios", []):
            num = ratio.get("numerator")
            den = ratio.get("denominator")
            out = ratio.get("name") or f"{num}_over_{den}"
            if num in df.columns and den in df.columns:
                df[out] = df[num] / df[den].replace(0, pd.NA)
        drop_cols = template.get("drop_columns", [])
        if drop_cols:
            df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
        return df

    def execute(self) -> ModuleResult:
        import pandas as pd
        artifact_dir: Path = self.context.artifact_dir
        artifact_dir.mkdir(parents=True, exist_ok=True)
        config = self.context.config_module or load_project_config(self.context.project_root)

        template_name = self.invocation_params.get("feat_template", "identity")
        template_cfg = TemplateLoader(self.context.project_root).load(template_name)

        train_path, test_path = data_paths(config)
        if not train_path.exists() or not test_path.exists():
            marker = artifact_dir / "feat_skipped.txt"
            marker.write_text("Missing train/test; feat skipped.")
            return ModuleResult(success=True, payload={"skipped": True}, artifacts=[marker])

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        train_df = self._apply_ops(train_df, template_cfg)
        test_df = self._apply_ops(test_df, template_cfg)

        train_out = artifact_dir / "train_features.csv"
        test_out = artifact_dir / "test_features.csv"
        train_df.to_csv(train_out, index=False)
        test_df.to_csv(test_out, index=False)

        meta = {
            "template": template_name,
            "columns": train_df.columns.tolist(),
        }
        meta_path = artifact_dir / "features_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))

        return ModuleResult(
            success=True,
            payload={
                "train_features": str(train_out),
                "test_features": str(test_out),
                "template": template_name,
            },
            artifacts=[train_out, test_out, meta_path],
        )
