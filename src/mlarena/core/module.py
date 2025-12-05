"""
Core module abstractions for MLArena.

Defines the BaseModule interface alongside lightweight result/context
containers shared across all pipeline modules.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class ModuleResult:
    """Execution result container for modules."""

    success: bool
    payload: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[Path] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class ModuleContext:
    """Context passed to modules during execution."""

    project_name: str
    project_root: Path
    experiment_id: str
    experiment_dir: Path
    artifact_dir: Path
    cli_args: Dict[str, Any]
    state: Any
    config_module: Any


class BaseModule(ABC):
    """
    Base class for all MLArena modules.

    Subclasses should define `name`, `description`, `dependencies`,
    and override `execute`.
    """

    name: str = ""
    description: str = ""
    dependencies: Set[str] = set()

    def __init__(self, context: ModuleContext) -> None:
        self.context = context
        self.invocation_params: Dict[str, Any] = {}

    @classmethod
    def register_cli_args(cls, parser) -> None:
        """Hook for argparse registration; optional per module."""

    def set_invocation_params(self, params: Dict[str, Any]) -> None:
        """Store invocation parameters for state tracking."""
        self.invocation_params = params or {}

    @abstractmethod
    def execute(self) -> ModuleResult:
        """Execute module logic and return a ModuleResult."""

    def can_run(self) -> Tuple[bool, str]:
        """
        Optional pre-flight validation.

        Returns:
            (ok, reason) where ok=False blocks execution and reason is stored.
        """
        return True, ""
