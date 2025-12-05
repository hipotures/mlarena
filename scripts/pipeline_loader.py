"""
Helpers for loading declarative pipeline definitions (list of modules + optional handlers).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
GLOBAL_PIPELINES_DIR = REPO_ROOT / "config" / "pipelines"


class PipelineValidationError(RuntimeError):
    """Raised when pipeline files fail validation."""


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise PipelineValidationError(f"Invalid pipeline file {path}: expected mapping at root.")
    return data


def _normalize_pipeline(data: Dict[str, Any], pipeline_name: str, source: Path) -> Dict[str, Any]:
    pipeline_data = data.get("pipeline", data)
    if not isinstance(pipeline_data, dict):
        raise PipelineValidationError(f"Invalid pipeline structure in {source}: expected mapping under 'pipeline' or root.")

    modules_raw = pipeline_data.get("modules")
    if not modules_raw:
        raise PipelineValidationError(f"Pipeline '{pipeline_name}' in {source} missing required 'modules' list.")
    if not isinstance(modules_raw, list):
        raise PipelineValidationError(f"Pipeline '{pipeline_name}' in {source} must define 'modules' as a list.")

    modules: List[Dict[str, str]] = []
    for idx, entry in enumerate(modules_raw):
        if isinstance(entry, str):
            modules.append({"name": entry})
            continue
        if not isinstance(entry, dict):
            raise PipelineValidationError(
                f"Pipeline '{pipeline_name}' in {source}: module #{idx + 1} must be a string or mapping."
            )
        name = entry.get("name")
        if not name:
            raise PipelineValidationError(f"Pipeline '{pipeline_name}' in {source}: module #{idx + 1} missing 'name'.")
        module_entry: Dict[str, str] = {"name": name}
        handler = entry.get("handler")
        if handler:
            module_entry["handler"] = handler
        description = entry.get("description")
        if description:
            module_entry["description"] = description
        modules.append(module_entry)

    return {
        "name": pipeline_data.get("name") or pipeline_name,
        "description": pipeline_data.get("description"),
        "modules": modules,
        "source": str(source),
    }


def load_pipeline(
    pipeline_name: str,
    project_root: Path | None = None,
    *,
    suppress_warnings: bool | None = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Load a pipeline definition from global/project locations.

    Global:   config/pipelines/<name>.yaml
    Project:  <project_root>/pipelines/<name>.yaml (overrides global)
    """
    warnings: List[str] = []
    no_warn_env = os.getenv("KAGGLE_PIPELINE_NO_WARN")
    no_warn = bool(suppress_warnings) or (no_warn_env and no_warn_env.strip().lower() not in {"", "0", "false"})

    global_path = GLOBAL_PIPELINES_DIR / f"{pipeline_name}.yaml"
    local_root = project_root or REPO_ROOT
    local_path = local_root / "pipelines" / f"{pipeline_name}.yaml"

    pipeline: Dict[str, Any] | None = None

    if global_path.exists():
        pipeline = _normalize_pipeline(_read_yaml(global_path), pipeline_name, global_path)

    if local_path.exists():
        local_pipeline = _normalize_pipeline(_read_yaml(local_path), pipeline_name, local_path)
        if pipeline and not no_warn:
            warnings.append(
                f"Pipeline '{pipeline_name}' overridden by project pipeline ({local_path} > {global_path})"
            )
        pipeline = local_pipeline

    if not pipeline:
        raise PipelineValidationError(
            f"Pipeline '{pipeline_name}' not found. Add {global_path} or {local_path}."
        )

    return pipeline, warnings
