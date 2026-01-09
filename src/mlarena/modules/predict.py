"""Prediction module.

Loads a trained model, generates predictions, and prepares submission files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

# pandas imported in execute()
from rich.console import Console

from mlarena.core.module import BaseModule, ModuleResult
from mlarena.core.registry import ModuleRegistry
from mlarena.utils.project import data_paths, load_project_config, load_submission_module
from mlarena.modules.model import _load_processed_or_raw  # reuse loader logic


def _load_predictor(model_artifact: Path):
    from autogluon.tabular import TabularPredictor

    return TabularPredictor.load(str(model_artifact))


@ModuleRegistry.register
class PredictModule(BaseModule):
    """Generate predictions for test data and write a submission CSV."""

    name = "predict"
    description = "Generate predictions"
    dependencies = {"model"}

    def execute(self) -> ModuleResult:
        """
        Load the model artifact and run inference on (optionally) preprocessed test data.

        Returns:
            ModuleResult containing submission file path and feature count metadata.
        """
        import pandas as pd
        artifact_dir: Path = self.context.artifact_dir
        artifact_dir.mkdir(parents=True, exist_ok=True)
        config = self.context.config_module or load_project_config(self.context.project_root)
        submission_helper = load_submission_module(self.context.project_root)

        model_entry = self.context.state.modules.get("model")
        if not model_entry or not getattr(model_entry, "payload", None):
            marker = artifact_dir / "predict_failed.txt"
            marker.write_text("Model step missing payload.")
            return ModuleResult(success=False, error="model not run", artifacts=[marker])

        model_artifact = Path(model_entry.payload["model_artifact"])  # type: ignore
        predictor = _load_predictor(model_artifact)

        # Try to use preprocessed test set from the same chain as training
        # Load ID column BEFORE preprocessing (preprocessing may drop it)
        _, test_path_raw = data_paths(config)
        test_df_raw = pd.read_csv(test_path_raw, compression='infer')
        id_col = getattr(config, "ID_COLUMN", None)
        id_series = test_df_raw[id_col].copy() if id_col and id_col in test_df_raw.columns else None

        # Load preprocessed or raw test data
        processed_test: Optional[Path] = None
        pp_template = self.invocation_params.get("preprocess_template")
        pp_exp_dir = self.invocation_params.get("preprocess_exp_dir")

        # If not provided explicitly (common when running `predict` with `--exp-id`),
        # fall back to whatever preprocessing the model was trained with.
        if not pp_template:
            pp_template = (model_entry.payload or {}).get("preprocess_template") or (model_entry.invocation or {}).get("preprocess_template")  # type: ignore[union-attr]
        if not pp_exp_dir:
            pp_exp_dir = (model_entry.payload or {}).get("preprocess_exp_dir") or (model_entry.invocation or {}).get("preprocess_exp_dir")  # type: ignore[union-attr]

        if pp_template:
            _, test_df_pp, _, _, _ = _load_processed_or_raw(
                self.context,
                config,
                pp_template,
                preprocess_exp_dir=pp_exp_dir,
            )
            processed_test = Path(pp_exp_dir) / "artifacts" / "preprocess" / "test_processed.csv.gz" if pp_exp_dir else None
            test_df = test_df_pp if test_df_pp is not None else test_df_raw
        else:
            test_df = test_df_raw

        # Drop ID column from features if still present
        if id_col and id_col in test_df.columns:
            test_df = test_df.drop(columns=[id_col])

        # Count features after preprocessing and ID removal
        feature_count = len(test_df.columns)

        submit_probas = bool(getattr(config, "SUBMISSION_PROBAS", False))
        if submit_probas and predictor.problem_type != "regression":
            preds = predictor.predict_proba(test_df, as_multiclass=False)
        else:
            preds = predictor.predict(test_df)

        submission_df = pd.DataFrame()
        if id_series is not None:
            submission_df[id_col] = id_series
        else:
            submission_df["id"] = range(len(preds))
        submission_df[getattr(config, "TARGET_COLUMN", "target")] = preds

        # Generate filename with timestamp (like old code)
        from datetime import datetime
        suffix = self.invocation_params.get("predict_suffix")
        if suffix:
            fname = f"submission-{suffix}.csv.gz"
        else:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            fname = f"submission-{timestamp}.csv.gz"
        submission_path = artifact_dir / fname
        submission_df.to_csv(submission_path, index=False, compression='infer')

        # Create submission entry via helper (tracks experiment if available)
        try:
            # Resolve competition name and submission paths from config
            competition = getattr(config, "COMPETITION_NAME", self.context.project_name)
            submissions_dir = Path(getattr(config, "SUBMISSIONS_DIR", self.context.project_root / "submissions"))
            sample_sub_path = Path(getattr(config, "SAMPLE_SUBMISSION_PATH", self.context.project_root / "data" / "sample_submission.csv"))

            submission_helper.create_submission(
                predictions=preds,
                project_root=self.context.project_root,
                competition_name=competition,
                submissions_dir=submissions_dir,
                sample_submission_path=sample_sub_path,
                test_ids=id_series if id_series is not None else submission_df.iloc[:, 0],
                model_name="mlarena",
                local_cv_score=None,
                notes="Generated by MLArena",
                filename_prefix=submission_path.stem,
                track=False,
            )
        except Exception:
            # Best-effort; creation is optional here.
            pass

        return ModuleResult(
            success=True,
            payload={
                "predictions": str(submission_path),
                "submission_file": str(submission_path),
                "feature_count": feature_count,
            },
            artifacts=[submission_path],
        )
