"""Stacking / ensembling module.

Averages multiple prediction CSVs (simple arithmetic mean) to create an ensemble submission.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

# pandas imported in execute()


from mlarena.core.module import BaseModule, ModuleResult
from mlarena.core.registry import ModuleRegistry


@ModuleRegistry.register
class StackModule(BaseModule):
    """Ensemble predictions by averaging provided submission files."""

    name = "stack"
    description = "Stacking/ensembling"
    dependencies = {"predict"}

    def execute(self) -> ModuleResult:
        """
        Average multiple prediction CSVs into a single submission.

        Returns:
            ModuleResult containing the stacked submission path and input file list.
        """
        import pandas as pd
        artifact_dir: Path = self.context.artifact_dir
        artifact_dir.mkdir(parents=True, exist_ok=True)

        files = self.invocation_params.get("prediction_files")
        if not files:
            predict_entry = self.context.state.modules.get("predict")
            if predict_entry and getattr(predict_entry, "payload", None):
                files = [predict_entry.payload["submission_file"]]
        if not files:
            marker = artifact_dir / "stack_failed.txt"
            marker.write_text("No prediction files provided.")
            return ModuleResult(success=False, error="no predictions", artifacts=[marker])

        dfs = []
        for f in files:
            path = Path(f)
            if not path.exists():
                continue
            dfs.append(pd.read_csv(path, compression='infer'))
        if not dfs:
            marker = artifact_dir / "stack_failed.txt"
            marker.write_text("Provided prediction files missing.")
            return ModuleResult(success=False, error="missing files", artifacts=[marker])

        id_col = self.invocation_params.get("id_column")
        target_col = self.invocation_params.get("target_column")
        if not id_col:
            id_col = dfs[0].columns[0]
        if not target_col:
            target_col = dfs[0].columns[-1]

        base = dfs[0][[id_col]].copy()
        preds = [df[target_col] for df in dfs]
        base[target_col] = sum(preds) / len(preds)

        out_path = artifact_dir / "stacked_submission.csv.gz"
        base.to_csv(out_path, index=False, compression='infer')

        return ModuleResult(
            success=True,
            payload={"stacked_submission": str(out_path), "inputs": files},
            artifacts=[out_path],
        )
