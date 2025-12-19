"""
Experiment state management for MLArena.

Persists module execution metadata to experiments/<id>/state.json.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

from filelock import FileLock, Timeout

from mlarena.utils.git import get_git_info
from mlarena.utils.time import utc_now_iso

# Repository root (resolve to absolute path)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _relativize_paths(data: Any, project_root: Path) -> Any:
    """
    Recursively convert absolute paths within a project to project-relative paths.
    """
    if isinstance(data, dict):
        return {k: _relativize_paths(v, project_root) for k, v in data.items()}
    elif isinstance(data, list):
        return [_relativize_paths(v, project_root) for v in data]
    elif isinstance(data, (str, Path)) and data:
        try:
            p = Path(data)
            if p.is_absolute() and project_root in p.parents or p == project_root:
                return str(p.relative_to(project_root))
            # Also try relative to REPO_ROOT for paths outside project but inside repo
            if p.is_absolute() and REPO_ROOT in p.parents:
                return str(p.relative_to(REPO_ROOT))
        except (ValueError, TypeError):
            pass
    return data


def _reconstruct_paths(data: Any, project_root: Path) -> Any:
    """
    Recursively convert project-relative paths back to absolute paths.
    Only converts strings that look like relative paths and exist relative to project_root or REPO_ROOT.
    """
    if isinstance(data, dict):
        return {k: _reconstruct_paths(v, project_root) for k, v in data.items()}
    elif isinstance(data, list):
        return [_reconstruct_paths(v, project_root) for v in data]
    elif isinstance(data, str) and data and not os.path.isabs(data):
        # Safety check: if it contains newlines, is too long, or has null bytes, it is not a path
        if "\n" in data or "\0" in data or len(data) > 1024:
            return data
            
        try:
            # Try relative to project_root
            p_proj = project_root / data
            if p_proj.exists():
                return p_proj
            # Try relative to REPO_ROOT
            p_repo = REPO_ROOT / data
            if p_repo.exists():
                return p_repo
        except (OSError, ValueError):
            pass
    return data


def _generate_experiment_id() -> str:
    ts = utc_now_iso().replace(":", "").replace("-", "")
    return f"exp-{ts[:8]}-{ts[9:15]}"


@dataclass
class ModuleEntry:
    """
    Serialized record of a single module execution.

    Attributes:
        name: Module name.
        status: Execution status (pending|running|completed|failed).
        started_at: ISO timestamp when execution began.
        finished_at: ISO timestamp when execution ended.
        pid: Process ID that executed the module.
        invocation: CLI invocation parameters.
        payload: Arbitrary module output payload.
        error: Error message if the module failed.
    """
    name: str
    status: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    pid: Optional[int] = None
    invocation: Dict[str, Any] = field(default_factory=dict)
    payload: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the entry to a JSON-serializable dictionary."""
        return {
            "name": self.name,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "pid": self.pid,
            "invocation": self.invocation,
            "payload": self.payload,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModuleEntry":
        """
        Build a ``ModuleEntry`` from persisted state.

        Args:
            data: Raw dictionary loaded from ``state.json``.

        Returns:
            ModuleEntry populated with available fields.
        """
        return cls(
            name=data.get("name", "unknown"),
            status=data.get("status", "pending"),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            pid=data.get("pid"),
            invocation=data.get("invocation", {}),
            payload=data.get("payload", {}),
            error=data.get("error"),
        )


@dataclass
class ExperimentState:
    """
    Aggregate state for a single experiment, persisted in ``state.json``.

    Attributes:
        experiment_id: Experiment identifier.
        project: Project slug.
        project_root: Root directory of the project.
        experiment_dir: Directory where state/artifacts are written.
        created_at: ISO timestamp of state creation.
        pipeline: Pipeline definition snapshot.
        modules: Mapping of module names to ``ModuleEntry`` records.
        run: CLI invocation metadata.
        git: Git metadata captured at start.
    """
    experiment_id: str
    project: str
    project_root: Path
    experiment_dir: Path
    created_at: str
    pipeline: Dict[str, Any] = field(default_factory=dict)
    modules: Dict[str, ModuleEntry] = field(default_factory=dict)
    run: Dict[str, Any] = field(default_factory=dict)
    git: Dict[str, Any] = field(default_factory=dict)

    @property
    def state_path(self) -> Path:
        """Filesystem path to ``state.json``."""
        return self.experiment_dir / "state.json"

    @staticmethod
    def _load_setup_modules(project_root: Path) -> Dict[str, ModuleEntry]:
        """
        Load frozen init and eda module entries from fixed experiment directories.

        Args:
            project_root: Root directory of the project.

        Returns:
            Dictionary of module entries keyed by module name.
        """
        setup_modules = {}

        # Load init module
        init_state_path = project_root / "experiments" / "init" / "state.json"
        if init_state_path.exists():
            try:
                init_data = json.loads(init_state_path.read_text())
                init_module = init_data.get("modules", {}).get("init")
                if init_module:
                    setup_modules["init"] = ModuleEntry.from_dict(init_module)
            except (json.JSONDecodeError, KeyError):
                pass

        # Load eda module (optional)
        eda_state_path = project_root / "experiments" / "eda" / "state.json"
        if eda_state_path.exists():
            try:
                eda_data = json.loads(eda_state_path.read_text())
                eda_module = eda_data.get("modules", {}).get("eda")
                if eda_module:
                    setup_modules["eda"] = ModuleEntry.from_dict(eda_module)
            except (json.JSONDecodeError, KeyError):
                pass

        return setup_modules

    @classmethod
    def load_or_create(
        cls,
        project_root: Path,
        project_name: str,
        experiment_id: Optional[str] = None,
        pipeline: Optional[Dict[str, Any]] = None,
        git_info: Optional[Dict[str, Any]] = None,
        run_invocation: Optional[Dict[str, Any]] = None,
        create_dirs: bool = True,
        setup_module_name: Optional[str] = None,
    ) -> "ExperimentState":
        """
        Create or reload an ``ExperimentState`` instance.

        Args:
            project_root: Project root (used for relative path resolution).
            project_name: Competition slug.
            experiment_id: Optional explicit identifier; autogenerated when omitted.
            pipeline: Pipeline definition snapshot.
            git_info: Git metadata snapshot.
            run_invocation: CLI args snapshot.
            create_dirs: When False, avoid creating directories (used by ``mla init``).
            setup_module_name: When set to ``"init"`` or ``"eda"``, use fixed experiment directories.

        Returns:
            ExperimentState ready for mutation and persistence.
        """
        experiments_dir = project_root / "experiments"
        if create_dirs:
            experiments_dir.mkdir(parents=True, exist_ok=True)

        # Setup modules (init/eda) use fixed directories
        if setup_module_name in ("init", "eda"):
            exp_id = setup_module_name
            experiment_dir = experiments_dir / setup_module_name
        else:
            exp_id = experiment_id or _generate_experiment_id()
            experiment_dir = experiments_dir / exp_id

        if create_dirs:
            experiment_dir.mkdir(parents=True, exist_ok=True)
        state_path = experiment_dir / "state.json"
        lock_path = experiment_dir / "state.json.lock"

        # For setup modules (init/eda), allow reloading if already exists
        # For regular experiments, always load existing state
        if state_path.exists() and setup_module_name not in ("init", "eda"):
            with FileLock(str(lock_path), timeout=10):
                return cls._from_file(state_path)

        # For regular experiments (not init/eda), merge setup modules
        initial_modules = {}
        if setup_module_name not in ("init", "eda"):
            initial_modules = cls._load_setup_modules(project_root)

        state = cls(
            experiment_id=exp_id,
            project=project_name,
            project_root=project_root,
            experiment_dir=experiment_dir,
            created_at=utc_now_iso(),
            pipeline=pipeline or {},
            modules=initial_modules,
            run=run_invocation or {},
            git=git_info or get_git_info(project_root),
        )
        if create_dirs:
            state.save()
        return state

    def start_module(self, name: str, invocation: Dict[str, Any]) -> None:
        """Mark a module as running and capture invocation metadata."""
        entry = self.modules.get(name, ModuleEntry(name=name, status="pending"))
        entry.status = "running"
        entry.started_at = utc_now_iso()
        entry.finished_at = None
        entry.pid = os.getpid()
        entry.invocation = invocation or {}
        entry.error = None
        self.modules[name] = entry

    def complete_module(self, name: str, payload: Dict[str, Any]) -> None:
        """Mark a module as completed and attach its payload."""
        entry = self.modules.get(name, ModuleEntry(name=name, status="pending"))
        entry.status = "completed"
        entry.finished_at = utc_now_iso()
        entry.payload = payload or {}
        entry.error = None
        self.modules[name] = entry

    def fail_module(self, name: str, error: str) -> None:
        """Mark a module as failed and record the error message."""
        entry = self.modules.get(name, ModuleEntry(name=name, status="pending"))
        entry.status = "failed"
        entry.finished_at = utc_now_iso()
        entry.error = error
        self.modules[name] = entry

    def to_dict(self, relative: bool = False) -> Dict[str, Any]:
        """
        Serialize experiment state to a JSON-compatible dictionary.
        
        Args:
            relative: If True, convert all paths to be relative to project_root or REPO_ROOT.
        """
        if relative:
            # Format project_root: use "/slug" if under projects/kaggle, otherwise relative to REPO_ROOT
            try:
                rel_to_kaggle = self.project_root.relative_to(REPO_ROOT / "projects" / "kaggle")
                project_root_str = "/" + rel_to_kaggle.parts[0]
            except (ValueError, IndexError):
                try:
                    project_root_str = str(self.project_root.relative_to(REPO_ROOT))
                except ValueError:
                    project_root_str = str(self.project_root)

            # Format experiment_dir: relative to project_root
            try:
                experiment_dir_str = str(self.experiment_dir.relative_to(self.project_root))
            except ValueError:
                experiment_dir_str = str(self.experiment_dir)
        else:
            project_root_str = str(self.project_root)
            experiment_dir_str = str(self.experiment_dir)

        data = {
            "experiment_id": self.experiment_id,
            "project": self.project,
            "project_root": project_root_str,
            "experiment_dir": experiment_dir_str,
            "created_at": self.created_at,
            "pipeline": self.pipeline,
            "modules": {k: v.to_dict() for k, v in self.modules.items()},
            "run": self.run,
            "git": self.git,
        }
        
        if relative:
            return _relativize_paths(data, self.project_root)
        return data

    @classmethod
    def _from_file(cls, path: Path) -> "ExperimentState":
        """
        Load state from ``state.json``, reconstructing absolute paths.

        Args:
            path: Path to the state file.

        Returns:
            ExperimentState loaded from disk.
        """
        data = json.loads(path.read_text())
        
        project_name = data["project"]
        
        # Reconstruct project_root
        project_root_raw = data.get("project_root")
        if project_root_raw:
            if project_root_raw.startswith("/") and project_root_raw.count("/") == 1:
                # "/slug" -> REPO_ROOT/projects/kaggle/slug
                project_root = REPO_ROOT / "projects" / "kaggle" / project_root_raw[1:]
            else:
                p = Path(project_root_raw)
                if p.is_absolute():
                    project_root = p
                else:
                    project_root = REPO_ROOT / p
        else:
            # Fallback based on file location
            # projects/kaggle/P/experiments/E/state.json -> project_root is 3 levels up
            # projects/kaggle/P/experiments/pre-T/S/state.json -> project_root is 4 levels up
            # We use the project name to find it reliably if it's under our projects dir
            project_root = REPO_ROOT / "projects" / "kaggle" / project_name
            if not project_root.exists():
                # Generic fallback: 3 levels up from state.json
                project_root = path.parent.parent.parent

        # Reconstruct experiment_dir
        experiment_dir_raw = data.get("experiment_dir")
        if experiment_dir_raw:
            p = Path(experiment_dir_raw)
            if p.is_absolute():
                experiment_dir = p
            else:
                experiment_dir = project_root / p
        else:
            experiment_dir = path.parent

        # Reconstruct absolute paths in modules and other data
        reconstructed_data = _reconstruct_paths(data, project_root)
        
        modules = {
            k: ModuleEntry.from_dict(v) 
            for k, v in reconstructed_data.get("modules", {}).items()
        }
        
        return cls(
            experiment_id=data["experiment_id"],
            project=project_name,
            project_root=project_root,
            experiment_dir=experiment_dir,
            created_at=data.get("created_at", utc_now_iso()),
            pipeline=reconstructed_data.get("pipeline", {}),
            modules=modules,
            run=reconstructed_data.get("run", {}),
            git=reconstructed_data.get("git", {}),
        )

    def save(self) -> None:
        """
        Persist the current state to disk with file locking, using relative paths for storage
        but maintaining absolute paths in the in-memory state.

        Raises:
            RuntimeError: When the file lock cannot be acquired.
        """
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.state_path.with_suffix(".lock")

        try:
            with FileLock(str(lock_path), timeout=10):
                # Always work with absolute paths for merging
                current_abs_payload = self.to_dict(relative=False)
                
                if self.state_path.exists():
                    try:
                        # Load existing, and reconstruct to absolute for merging
                        raw_existing = json.loads(self.state_path.read_text())
                        existing_abs = _reconstruct_paths(raw_existing, self.project_root)
                        
                        existing_modules = existing_abs.get("modules", {})
                        # Merge modules (absolute)
                        current_abs_payload["modules"] = {**existing_modules, **current_abs_payload["modules"]}
                        
                        # Update in-memory modules with merged absolute data
                        self.modules = {
                            k: ModuleEntry.from_dict({**v, "name": v.get("name", k)})
                            for k, v in current_abs_payload["modules"].items()
                        }
                    except (json.JSONDecodeError, Exception):
                        pass
                
                # Now relativize the entire payload for saving
                relative_payload = _relativize_paths(current_abs_payload, self.project_root)
                
                # Special handling for project_root and experiment_dir in the final relative_payload
                try:
                    rel_to_kaggle = self.project_root.relative_to(REPO_ROOT / "projects" / "kaggle")
                    relative_payload["project_root"] = "/" + rel_to_kaggle.parts[0]
                except (ValueError, IndexError):
                    try:
                        relative_payload["project_root"] = str(self.project_root.relative_to(REPO_ROOT))
                    except ValueError:
                        relative_payload["project_root"] = str(self.project_root)

                try:
                    relative_payload["experiment_dir"] = str(self.experiment_dir.relative_to(self.project_root))
                except ValueError:
                    relative_payload["experiment_dir"] = str(self.experiment_dir)

                self.state_path.write_text(json.dumps(relative_payload, indent=2))
        except Timeout:
            raise RuntimeError(f"Could not acquire lock for state file at {self.state_path}")
