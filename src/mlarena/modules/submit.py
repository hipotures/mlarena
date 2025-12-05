"""Submission upload module."""

from __future__ import annotations

import subprocess
from pathlib import Path

from mlarena.core.module import BaseModule, ModuleResult
from mlarena.core.registry import ModuleRegistry
from mlarena.utils.project import load_project_config


@ModuleRegistry.register
class SubmitModule(BaseModule):
    name = "submit"
    description = "Submit predictions to Kaggle"
    dependencies = {"predict"}

    @classmethod
    def register_cli_args(cls, parser) -> None:
        parser.add_argument("--skip-submit", action="store_true", help="Skip Kaggle submission (placeholder).")
        parser.add_argument("--message", type=str, default="MLArena submission", help="Submission message.")

    def execute(self) -> ModuleResult:
        artifact_dir: Path = self.context.artifact_dir
        artifact_dir.mkdir(parents=True, exist_ok=True)
        skip = bool(self.invocation_params.get("skip_submit", False))
        message = self.invocation_params.get("message", "MLArena submission")

        predict_payload = self.context.state.modules.get("predict")
        if not predict_payload or not getattr(predict_payload, "payload", None):
            marker = artifact_dir / "submit_failed.txt"
            marker.write_text("Predict step missing payload.")
            return ModuleResult(success=False, error="predict not run", artifacts=[marker])

        submission_file = Path(predict_payload.payload["submission_file"])  # type: ignore
        if skip:
            marker = artifact_dir / "submit_skipped.txt"
            marker.write_text("Skipped Kaggle submission.")
            return ModuleResult(success=True, payload={"submitted": False}, artifacts=[marker])

        config = self.context.config_module or load_project_config(self.context.project_root)
        competition = getattr(config, "COMPETITION_NAME", self.context.project_name)

        try:
            subprocess.check_call(
                [
                    "kaggle",
                    "competitions",
                    "submit",
                    "-c",
                    competition,
                    "-f",
                    str(submission_file),
                    "-m",
                    message,
                ]
            )
            marker = artifact_dir / "submit_success.txt"
            marker.write_text(f"Submitted {submission_file} to {competition}")
            return ModuleResult(
                success=True,
                payload={"submitted": True, "competition": competition, "submission_file": str(submission_file)},
                artifacts=[marker],
            )
        except Exception as exc:
            marker = artifact_dir / "submit_failed.txt"
            marker.write_text(f"Submission failed: {exc}")
            return ModuleResult(success=False, error=str(exc), artifacts=[marker])
