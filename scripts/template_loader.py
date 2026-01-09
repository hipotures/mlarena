"""
Utilities for loading and merging global + project templates with validation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
GLOBAL_TEMPLATE_DIR = REPO_ROOT / "src" / "mlarena" / "templates"


class TemplateValidationError(RuntimeError):
    """Raised when template files fail validation."""


def _read_templates_from_directory(directory: Path, kind: str) -> Dict[str, Any]:
    """
    Read all *.yaml files from directory, return {name: content}.
    Template name is derived from filename (without .yaml extension).
    """
    if not directory.exists():
        return {}

    templates = {}
    for file in directory.glob("*.yaml"):
        name = file.stem  # filename without .yaml
        try:
            content = yaml.safe_load(file.read_text()) or {}
            if not isinstance(content, dict):
                raise TemplateValidationError(f"Invalid {kind} template in {file}: content must be a mapping.")
            templates[name] = content
        except yaml.YAMLError as e:
            raise TemplateValidationError(f"Failed to parse {file}: {e}")

    return templates


def _validate_case_insensitive(templates: Dict[str, Any], directory: Path) -> None:
    """Validate no case-insensitive duplicates in template names."""
    seen = {}
    for name in templates.keys():
        name_lower = name.lower()
        if name_lower in seen:
            raise TemplateValidationError(
                f"Case-insensitive conflict in {directory}:\n"
                f"  - {seen[name_lower]}\n"
                f"  - {name}"
            )
        seen[name_lower] = name


def _validate_global_uniqueness() -> None:
    """
    Validate global name uniqueness across model and preprocess types.
    Note: This is now a no-op - we allow same names in different types.
    """
    # Allow same template names in model/ and preprocess/
    # They are distinguished by context (type)
    pass


def _validate_templates(templates: Dict[str, Any], kind: str, *, source: str) -> Dict[str, Dict[str, Any]]:
    """Validate templates: regular templates need 'module'/'model', meta-templates need 'chain'."""
    required_key = "model" if kind == "model" else "module"
    validated: Dict[str, Dict[str, Any]] = {}
    for name, payload in templates.items():
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            raise TemplateValidationError(f"Template '{name}' in {source} must be a mapping.")

        # Check if meta-template (has "chain" key)
        if "chain" in payload:
            # Meta-template: validate chain is list of strings
            if not isinstance(payload["chain"], list):
                raise TemplateValidationError(f"Meta-template '{name}' chain must be a list in {source}.")
            for item in payload["chain"]:
                if not isinstance(item, str):
                    raise TemplateValidationError(f"Meta-template '{name}' chain items must be strings in {source}.")
            # Meta-templates don't need config
            entry = dict(payload)
            validated[name] = entry
        else:
            # Regular template: validate required key
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


def _merge_template_payload(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge a project template into a global template.
    - Meta-templates (chain) are not merged; the override replaces the base.
    - Regular templates: shallow merge, with dict-merge for "config".
    """
    if "chain" in base or "chain" in override:
        return dict(override)

    merged = dict(base)
    base_cfg = base.get("config", {})
    override_cfg = override.get("config", {})
    if isinstance(base_cfg, dict) and isinstance(override_cfg, dict):
        merged["config"] = {**base_cfg, **override_cfg}
    elif "config" in override:
        merged["config"] = override_cfg

    for key, value in override.items():
        if key == "config":
            continue
        merged[key] = value
    return merged


def load_templates(kind: str, project_root: Path, *, suppress_warnings: bool | None = None) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """
    Load templates from global + project locations with project overrides.
    Returns merged templates and a list of override warnings.
    """
    if kind not in {"model", "preprocess"}:
        raise ValueError(f"Unsupported template kind '{kind}'")

    # Validate global uniqueness first
    _validate_global_uniqueness()

    env_no_warn = os.getenv("KAGGLE_TEMPLATE_NO_WARN")
    env_no_warn_flag = env_no_warn is not None and env_no_warn.strip().lower() not in {"", "0", "false"}
    no_warn = bool(suppress_warnings) or env_no_warn_flag

    global_dir = GLOBAL_TEMPLATE_DIR / kind
    local_dir = project_root / "templates" / kind

    global_templates = _read_templates_from_directory(global_dir, kind)
    local_templates = _read_templates_from_directory(local_dir, kind)

    # Validate case-insensitive in each directory
    _validate_case_insensitive(global_templates, global_dir)
    _validate_case_insensitive(local_templates, local_dir)

    merged: Dict[str, Any] = dict(global_templates)
    warnings: List[str] = []

    for name, payload in local_templates.items():
        if name in merged:
            if not no_warn:
                warnings.append(
                    f"Template '{name}' overridden by project/{project_root.name} template ({local_dir} > {global_dir})"
                )
            base_payload = merged.get(name, {})
            if isinstance(base_payload, dict) and isinstance(payload, dict):
                merged[name] = _merge_template_payload(base_payload, payload)
            else:
                merged[name] = payload
        else:
            merged[name] = payload

    if not merged:
        raise TemplateValidationError(
            f"No {kind} templates found. Add entries to {global_dir} or {local_dir} before running."
        )

    if kind == "preprocess":
        module_bases = dict(merged)
        for name, payload in list(merged.items()):
            if not isinstance(payload, dict) or "chain" in payload:
                continue
            module_name = payload.get("module")
            if not module_name or module_name == name:
                continue
            base_payload = module_bases.get(module_name)
            if isinstance(base_payload, dict) and "chain" not in base_payload:
                merged[name] = _merge_template_payload(base_payload, payload)

    validated = _validate_templates(merged, kind, source=f"{local_dir} or {global_dir}")
    return validated, warnings
