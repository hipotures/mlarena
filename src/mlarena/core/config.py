"""
Configuration helpers for MLArena.

Loads pipeline definitions and (model/preprocess) templates from project config
directories. YAML support is optional; JSON is primary.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


def load_pipeline_def(name: str, project_root: Optional[Path] = None) -> Tuple[Dict[str, Any], List[str]]:
    """
    Load pipeline definition; returns default when not present.
    """
    project_root = Path(project_root) if project_root else Path(".")
    path = project_root / "config" / f"pipeline_{name}.json"
    if path.exists():
        data = json.loads(path.read_text())
        warnings = _validate_pipeline(data, name)
        return data, warnings
    return {"name": name, "modules": []}, []


class TemplateLoader:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)

    def load(self, template_name: str) -> Dict[str, Any]:
        """
        Load a JSON/YAML template if present under config/. Returns empty dict
        when not found to allow graceful defaults.
        """
        config_dir = self.project_root / "config"
        candidates = [
            config_dir / f"{template_name}.json",
            config_dir / f"{template_name}.yaml",
            config_dir / f"{template_name}.yml",
        ]
        for path in candidates:
            if path.exists():
                if path.suffix == ".json":
                    data = json.loads(path.read_text())
                    _validate_template(data, template_name)
                    return data
                if path.suffix in {".yaml", ".yml"} and yaml:
                    data = yaml.safe_load(path.read_text())
                    _validate_template(data, template_name)
                    return data
        return {}


def _validate_pipeline(data: Dict[str, Any], name: str) -> List[str]:
    if not isinstance(data, dict):
        raise ValueError(f"Pipeline '{name}' must be a dict.")
    modules = data.get("modules", [])
    if not isinstance(modules, list):
        raise ValueError(f"Pipeline '{name}' modules must be a list.")
    warnings: List[str] = []
    allowed = {
        "eda",
        "preprocess",
        "feat",
        "model",
        "predict",
        "tune",
        "stack",
        "submit",
        "fetch-score",
    }
    for m in modules:
        if not isinstance(m, str):
            raise ValueError(f"Pipeline '{name}' module entries must be strings. Got {m!r}")
    # Optional: warn if referenced modules not registered (checked lazily to avoid circular import).
    try:
        from mlarena.core.registry import ModuleRegistry

        ModuleRegistry.discover()
        registered = set(ModuleRegistry.available())
        for m in modules:
            if m not in registered:
                warnings.append(f"Pipeline '{name}': module '{m}' not registered.")
    except Exception:
        warnings.append(f"Pipeline '{name}': could not verify module registrations.")

    for m in modules:
        if m not in allowed:
            warnings.append(f"Pipeline '{name}': module '{m}' not in default allowed set.")
    return warnings


def _validate_template(data: Dict[str, Any], name: str) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"Template '{name}' must be a dict.")
