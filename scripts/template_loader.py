"""
Utilities for loading and merging global + project templates with validation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
GLOBAL_TEMPLATE_DIR = REPO_ROOT / "config" / "templates"


class TemplateValidationError(RuntimeError):
    """Raised when template files fail validation."""


def _read_templates(path: Path, kind: str) -> Dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    templates = data.get("templates") or {}
    if templates and not isinstance(templates, dict):
        raise TemplateValidationError(f"Invalid {kind} templates in {path}: 'templates' must be a mapping.")
    return templates


def _validate_templates(templates: Dict[str, Any], kind: str, *, source: str) -> Dict[str, Dict[str, Any]]:
    required_key = "model" if kind == "model" else "module"
    validated: Dict[str, Dict[str, Any]] = {}
    for name, payload in templates.items():
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            raise TemplateValidationError(f"Template '{name}' in {source} must be a mapping.")
        if required_key not in payload:
            raise TemplateValidationError(f"Template '{name}' missing required key '{required_key}' in {source}.")
        entry = dict(payload)
        config = entry.get("config", {})
        if config is None:
            config = {}
        if not isinstance(config, dict):
            raise TemplateValidationError(f"Template '{name}' has non-dict config in {source}.")
        entry["config"] = config
        validated[name] = entry
    return validated


def load_templates(kind: str, project_root: Path, *, suppress_warnings: bool | None = None) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """
    Load templates from global + project locations with project overrides.
    Returns merged templates and a list of override warnings.
    """
    if kind not in {"model", "preprocess"}:
        raise ValueError(f"Unsupported template kind '{kind}'")

    env_no_warn = os.getenv("KAGGLE_TEMPLATE_NO_WARN")
    env_no_warn_flag = env_no_warn is not None and env_no_warn.strip().lower() not in {"", "0", "false"}
    no_warn = bool(suppress_warnings) or env_no_warn_flag
    global_path = GLOBAL_TEMPLATE_DIR / f"{kind}.yaml"
    local_path = project_root / "templates" / f"{kind}.yaml"

    global_templates = _read_templates(global_path, kind)
    local_templates = _read_templates(local_path, kind)

    merged: Dict[str, Any] = dict(global_templates)
    warnings: List[str] = []

    for name, payload in local_templates.items():
        if name in merged and not no_warn:
            warnings.append(
                f"Template '{name}' overridden by project/{project_root.name} template ({local_path} > {global_path})"
            )
        merged[name] = payload

    if not merged:
        raise TemplateValidationError(
            f"No {kind} templates found. Add entries to {global_path} or {local_path} before running."
        )

    validated = _validate_templates(merged, kind, source=f"{local_path} or {global_path}")
    return validated, warnings
