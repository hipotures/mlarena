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
    def __init__(self, project_root: Path, template_type: str = "model"):
        self.project_root = Path(project_root)
        self.template_type = template_type
        self._validated = False

    def _scan_directory(self, directory: Path) -> Dict[str, Path]:
        """
        Scan directory for *.yaml files, return {name: path}.
        Template name is derived from filename (without .yaml extension).
        """
        if not directory.exists():
            return {}

        templates = {}
        for file in directory.glob("*.yaml"):
            name = file.stem  # filename without .yaml
            templates[name] = file
        return templates

    def _validate_case_insensitive(self, templates: Dict[str, Path], source: str) -> None:
        """Validate no case-insensitive duplicates in template names."""
        seen = {}
        for name, path in templates.items():
            name_lower = name.lower()
            if name_lower in seen:
                raise ValueError(
                    f"Case-insensitive conflict in {source}:\n"
                    f"  - {seen[name_lower]}\n"
                    f"  - {path}"
                )
            seen[name_lower] = path

    def _validate_global_uniqueness(self) -> None:
        """
        Validate global name uniqueness across model and preprocess types.
        Note: This is now a no-op - we allow same names in different types.
        """
        # Allow same template names in model/ and preprocess/
        # They are distinguished by context (type)
        self._validated = True

    def list_available(self) -> List[str]:
        """List available templates from global and project-local directories."""
        if yaml is None:
            return []

        # Validate global uniqueness on first call
        try:
            self._validate_global_uniqueness()
        except ValueError:
            # If validation fails, return empty list (graceful degradation)
            return []

        templates = {}

        # Load global templates
        repo_root = Path(__file__).resolve().parents[3]
        global_dir = repo_root / "config" / "templates" / self.template_type
        global_templates = self._scan_directory(global_dir)

        # Validate case-insensitive in global
        try:
            self._validate_case_insensitive(global_templates, f"global {self.template_type}")
        except ValueError:
            return []

        templates.update(global_templates)

        # Load project-local templates (override globals)
        local_dir = self.project_root / "templates" / self.template_type
        local_templates = self._scan_directory(local_dir)

        # Validate case-insensitive in local
        try:
            self._validate_case_insensitive(local_templates, f"local {self.template_type}")
        except ValueError:
            return []

        # Project-local templates override global ones (by name)
        templates.update(local_templates)

        return sorted(templates.keys())

    def list_available_with_source(self) -> List[Dict[str, str]]:
        """List templates with source information (global/local/conflict).

        Returns:
            List of dicts with keys: name, source ('global', 'local', 'conflict')
        """
        if yaml is None:
            return []

        # Validate global uniqueness
        try:
            self._validate_global_uniqueness()
        except ValueError:
            return []

        repo_root = Path(__file__).resolve().parents[3]
        global_dir = repo_root / "config" / "templates" / self.template_type
        local_dir = self.project_root / "templates" / self.template_type

        global_templates = set(self._scan_directory(global_dir).keys())
        local_templates = set(self._scan_directory(local_dir).keys())

        # Validate case-insensitive
        try:
            self._validate_case_insensitive(
                self._scan_directory(global_dir),
                f"global {self.template_type}"
            )
            self._validate_case_insensitive(
                self._scan_directory(local_dir),
                f"local {self.template_type}"
            )
        except ValueError:
            return []

        conflicts = global_templates & local_templates

        result = []

        # Add global-only templates first
        for name in sorted(global_templates - conflicts):
            result.append({"name": name, "source": "global"})

        # Add conflicts
        for name in sorted(conflicts):
            result.append({"name": name, "source": "conflict"})

        # Add local-only templates last
        for name in sorted(local_templates - conflicts):
            result.append({"name": name, "source": "local"})

        return result

    def load(self, template_name: str) -> Dict[str, Any]:
        """
        Load template from global or project-local directory.
        Project-local templates override global ones.
        Returns empty dict when not found to allow graceful defaults.
        """
        if yaml is None:
            return {}

        # Validate global uniqueness
        try:
            self._validate_global_uniqueness()
        except ValueError:
            return {}

        template_data = {}

        # Load from global first
        repo_root = Path(__file__).resolve().parents[3]
        global_file = repo_root / "config" / "templates" / self.template_type / f"{template_name}.yaml"
        if global_file.exists():
            try:
                template_data = yaml.safe_load(global_file.read_text()) or {}
            except Exception:
                pass

        # Override with project-local if exists
        local_file = self.project_root / "templates" / self.template_type / f"{template_name}.yaml"
        if local_file.exists():
            try:
                template_data = yaml.safe_load(local_file.read_text()) or {}
            except Exception:
                pass

        # Extract config from nested structure, preserving top-level fields
        if "config" in template_data:
            config = template_data["config"]
            # Ensure config is a dict before copying
            if isinstance(config, dict):
                result = config.copy()
                # Preserve ALL top-level fields (not just "model")
                for key, value in template_data.items():
                    if key != "config":  # Don't re-add config itself
                        result[key] = value
                return result
            # If config is not a dict, return as-is
            return config
        return template_data


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
