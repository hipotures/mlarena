"""
Experiment state management for MLArena.

Persists module execution metadata to experiments/<id>/state.json.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from filelock import FileLock, Timeout

from mlarena.utils.git import get_git_info
from mlarena.utils.time import utc_now_iso


def _generate_experiment_id() -> str:
    ts = utc_now_iso().replace(":", "").replace("-", "")
    return f"exp-{ts[:8]}-{ts[9:15]}"


@dataclass
class ModuleEntry:
    name: str
    status: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    pid: Optional[int] = None
    invocation: Dict[str, Any] = field(default_factory=dict)
    payload: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
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
        # Handle legacy state.json that might not have 'name' field
        # Infer name from dict key if missing
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
        return self.experiment_dir / "state.json"

    @staticmethod
    def _load_setup_modules(project_root: Path) -> Dict[str, ModuleEntry]:
        """Load frozen init and eda modules from experiments/init/ and experiments/eda/."""
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
        Create or load ExperimentState.

        Args:
            project_root: Final project root (for metadata in state.json).
            project_name: Competition slug.
            experiment_id: Optional explicit id.
            pipeline: Pipeline definition snapshot.
            git_info: Git metadata snapshot.
            run_invocation: CLI args snapshot.
            create_dirs: When False, do not create project/experiment
                directories yet (used by `mla init` to avoid pre-creating the
                project before legacy init runs).
            setup_module_name: If "init" or "eda", use fixed directory
                (experiments/init/ or experiments/eda/). Otherwise create
                timestamped experiment and merge setup modules.
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
        entry = self.modules.get(name, ModuleEntry(name=name, status="pending"))
        entry.status = "running"
        entry.started_at = utc_now_iso()
        entry.finished_at = None
        entry.pid = os.getpid()
        entry.invocation = invocation or {}
        entry.error = None
        self.modules[name] = entry

    def complete_module(self, name: str, payload: Dict[str, Any]) -> None:
        entry = self.modules.get(name, ModuleEntry(name=name, status="pending"))
        entry.status = "completed"
        entry.finished_at = utc_now_iso()
        entry.payload = payload or {}
        entry.error = None
        self.modules[name] = entry

    def fail_module(self, name: str, error: str) -> None:
        entry = self.modules.get(name, ModuleEntry(name=name, status="pending"))
        entry.status = "failed"
        entry.finished_at = utc_now_iso()
        entry.error = error
        self.modules[name] = entry

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "project": self.project,
            "project_root": str(self.project_root),
            "experiment_dir": str(self.experiment_dir),
            "created_at": self.created_at,
            "pipeline": self.pipeline,
            "modules": {k: v.to_dict() for k, v in self.modules.items()},
            "run": self.run,
            "git": self.git,
        }

    @classmethod
    def _from_file(cls, path: Path) -> "ExperimentState":
        data = json.loads(path.read_text())
        modules = {k: ModuleEntry.from_dict(v) for k, v in data.get("modules", {}).items()}
        return cls(
            experiment_id=data["experiment_id"],
            project=data["project"],
            project_root=Path(data.get("project_root", path.parent.parent)),
            experiment_dir=Path(data.get("experiment_dir", path.parent)),
            created_at=data.get("created_at", utc_now_iso()),
            pipeline=data.get("pipeline", {}),
            modules=modules,
            run=data.get("run", {}),
            git=data.get("git", {}),
        )

    def save(self) -> None:
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        lock_path = self.state_path.with_suffix(".lock")

        try:
            with FileLock(str(lock_path), timeout=10):
                if self.state_path.exists():
                    try:
                        existing = json.loads(self.state_path.read_text())
                        existing_modules = existing.get("modules", {})
                        payload["modules"] = {**existing_modules, **payload["modules"]}
                        # Use dict key as name if not present in data
                        self.modules = {
                            k: ModuleEntry.from_dict({**v, "name": v.get("name", k)})
                            for k, v in payload["modules"].items()
                        }
                    except json.JSONDecodeError:
                        pass
                self.state_path.write_text(json.dumps(payload, indent=2))
        except Timeout:
            raise RuntimeError(f"Could not acquire lock for state file at {self.state_path}")
