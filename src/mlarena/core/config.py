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
    Load a pipeline definition JSON for a given project.

    Args:
        name: Pipeline name suffix (``pipeline_<name>.json``).
        project_root: Project root directory.

    Returns:
        Tuple of (pipeline_definition, warnings).

    Examples:
        >>> load_pipeline_def("default", Path("."))[0]
        {'name': 'default', 'modules': []}
    """
    project_root = Path(project_root) if project_root else Path(".")
    path = project_root / "config" / f"pipeline_{name}.json"
    if path.exists():
        data = json.loads(path.read_text())
        warnings = _validate_pipeline(data, name)
        return data, warnings
    return {"name": name, "modules": []}, []


class TemplateLoader:
    """Utility class that resolves model and preprocess templates."""

    def __init__(self, project_root: Path, template_type: str = "model"):
        """
        Initialize loader.

        Args:
            project_root: Project root directory.
            template_type: ``"model"`` or ``"preprocess"``.
        """
        self.project_root = Path(project_root)
        self.template_type = template_type
        self._validated = False

    def _scan_directory(self, directory: Path) -> Dict[str, Path]:
        """
        Scan a directory for ``*.yaml`` templates.

        Args:
            directory: Directory to scan.

        Returns:
            Mapping of template name to file path.
        """
        if not directory.exists():
            return {}

        templates = {}
        for file in directory.glob("*.yaml"):
            name = file.stem  # filename without .yaml
            templates[name] = file
        return templates

    def _validate_case_insensitive(self, templates: Dict[str, Path], source: str) -> None:
        """
        Ensure template names are unique in a case-insensitive comparison.

        Args:
            templates: Mapping of template names to file paths.
            source: Human-readable source label for error reporting.

        Raises:
            ValueError: When duplicates are detected.
        """
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
        """Validate name uniqueness across template types (currently a no-op)."""
        # Allow same template names in model/ and preprocess/
        # They are distinguished by context (type)
        self._validated = True

    def list_available(self) -> List[str]:
        """
        List available templates from global and project-local directories.

        Returns:
            Sorted list of template names (project overrides included).
        """
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
        global_dir = repo_root / "src" / "mlarena" / "templates" / self.template_type
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
        """
        List templates along with their source.

        Returns:
            List of dictionaries with keys: ``name`` and ``source`` (``global``, ``local``, or ``conflict``).
        """
        if yaml is None:
            return []

        # Validate global uniqueness
        try:
            self._validate_global_uniqueness()
        except ValueError:
            return []

        repo_root = Path(__file__).resolve().parents[3]
        global_dir = repo_root / "src" / "mlarena" / "templates" / self.template_type
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
        Load a template file, preferring project-local overrides.

        Args:
            template_name: Name of the template (filename without ``.yaml``).

        Returns:
            Parsed template dictionary (config and metadata), or empty dict if not found.
        """
        if yaml is None:
            return {}

        # Validate global uniqueness
        try:
            self._validate_global_uniqueness()
        except ValueError:
            return {}

        template_data = {}

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

        def _load_by_name(name: str) -> Dict[str, Any]:
            repo_root = Path(__file__).resolve().parents[3]
            data: Dict[str, Any] = {}
            global_file = repo_root / "src" / "mlarena" / "templates" / self.template_type / f"{name}.yaml"
            if global_file.exists():
                try:
                    data = yaml.safe_load(global_file.read_text()) or {}
                except Exception:
                    data = {}

            local_file = self.project_root / "templates" / self.template_type / f"{name}.yaml"
            if local_file.exists():
                try:
                    local_data = yaml.safe_load(local_file.read_text()) or {}
                    if isinstance(data, dict) and isinstance(local_data, dict):
                        data = _merge_template_payload(data, local_data)
                    else:
                        data = local_data
                except Exception:
                    pass
            return data

        template_data = _load_by_name(template_name)

        # For preprocess templates, inherit defaults from the global template named after the module.
        if (
            self.template_type == "preprocess"
            and isinstance(template_data, dict)
            and "chain" not in template_data
        ):
            module_name = template_data.get("module")
            if module_name and module_name != template_name:
                base_data = _load_by_name(module_name)
                if isinstance(base_data, dict) and "chain" not in base_data:
                    template_data = _merge_template_payload(base_data, template_data)

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
    """
    Validate the shape and contents of a pipeline definition.

    Args:
        data: Pipeline dictionary loaded from disk.
        name: Pipeline name (for error context).

    Returns:
        List of warning strings (non-fatal issues).

    Raises:
        ValueError: When the pipeline structure is invalid.
    """
    if not isinstance(data, dict):
        raise ValueError(f"Pipeline '{name}' must be a dict.")
    modules = data.get("modules", [])
    if not isinstance(modules, list):
        raise ValueError(f"Pipeline '{name}' modules must be a list.")
    warnings: List[str] = []
    allowed = {
        "eda",
        "preprocess",
        "preprocess-tune",
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
    """
    Validate that a template structure is a dictionary.

    Args:
        data: Template payload.
        name: Template name for error context.

    Raises:
        ValueError: When the template is not a dictionary.
    """
    if not isinstance(data, dict):
        raise ValueError(f"Template '{name}' must be a dict.")
