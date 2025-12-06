"""
Core module abstractions for MLArena.

Defines the BaseModule interface alongside lightweight result/context
containers shared across all pipeline modules.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import Console


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


def suggest_next_steps(
    current_module: str,
    project_name: str,
    experiment_id: str,
) -> List[str]:
    """
    Suggest next module(s) to run based on dependency graph.

    Args:
        current_module: Name of the module that just completed (e.g., "model")
        project_name: Project name
        experiment_id: Experiment ID

    Returns:
        List of command strings to run next modules (formatted with line continuations)
    """
    from mlarena.core.registry import ModuleRegistry

    # Discover all modules
    ModuleRegistry.discover()

    # Find modules that depend on current_module
    next_modules = []
    for module_name in ModuleRegistry.available():
        module_cls = ModuleRegistry.get(module_name)
        if current_module in module_cls.dependencies:
            next_modules.append(module_name)

    # Generate commands with line continuations for easy copying
    commands = []
    for next_module in sorted(next_modules):
        cmd = (
            f"python scripts/mla.py {next_module} \\\n"
            f"  --project {project_name} \\\n"
            f"  --experiment-id {experiment_id}"
        )
        commands.append(cmd)

    return commands


def print_next_steps(
    current_module: str,
    project_name: str,
    experiment_id: str,
    console: Optional["Console"] = None,
) -> None:
    """
    Print suggested next steps after module completion.

    Args:
        current_module: Name of the module that just completed
        project_name: Project name
        experiment_id: Experiment ID
        console: Optional Rich console instance
    """
    if console is None:
        from rich.console import Console
        console = Console()

    next_steps = suggest_next_steps(current_module, project_name, experiment_id)

    if next_steps:
        console.print(f"\n[bold]Next steps:[/bold]")
        for step in next_steps:
            # Print without leading spaces to avoid bash history issues
            console.print(f"[dim]{step}[/dim]")
        console.print()
