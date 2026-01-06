"""
Core module abstractions for MLArena.

Defines the BaseModule interface alongside lightweight result/context
containers shared across all pipeline modules.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from rich.console import Console
    from mlarena.core.conf import GlobalConfig


@dataclass
class ModuleResult:
    """
    Execution result container for modules.

    Attributes:
        success: Whether the module finished successfully.
        payload: Optional dictionary with module-specific outputs.
        artifacts: List of produced artifact paths.
        error: Error message when ``success`` is False.
    """

    success: bool
    payload: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[Path] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class ModuleContext:
    """
    Context passed to modules during execution.

    Attributes:
        project_name: Kaggle project slug.
        project_root: Project root directory.
        experiment_id: Current experiment identifier.
        experiment_dir: Directory where state/artifacts are stored.
        artifact_dir: Directory dedicated to the module's artifacts.
        cli_args: Parsed CLI arguments for the module.
        state: ``ExperimentState`` object associated with this run.
        config_module: Loaded project configuration module.
        config: Unified GlobalConfig object.
    """

    project_name: str
    project_root: Path
    experiment_id: str
    experiment_dir: Path
    artifact_dir: Path
    cli_args: Dict[str, Any]
    state: Any
    config_module: Any
    config: "GlobalConfig" = None


class BaseModule(ABC):
    """Base class for MLArena pipeline modules."""

    name: str = ""
    description: str = ""
    dependencies: Set[str] = set()

    def __init__(self, context: ModuleContext) -> None:
        self.context = context
        self.invocation_params: Dict[str, Any] = {}

    def set_invocation_params(self, params: Dict[str, Any]) -> None:
        """
        Store invocation parameters for state tracking and execution.

        Args:
            params: Dictionary of CLI-provided parameters.
        """
        self.invocation_params = params or {}

    @abstractmethod
    def execute(self) -> ModuleResult:
        """
        Execute the module logic.

        Returns:
            ModuleResult capturing success status, payload, and artifacts.
        """

    def can_run(self) -> Tuple[bool, str]:
        """
        Optional pre-flight validation.

        Returns:
            Tuple of (is_allowed, reason). When ``is_allowed`` is False, the pipeline aborts with ``reason``.
        """
        return True, ""


def suggest_next_steps(
    current_module: str,
    project_name: str,
    experiment_id: str,
) -> List[str]:
    """
    Suggest follow-up CLI invocations based on module dependencies.

    Args:
        current_module: Recently completed module name (for example ``"model"``).
        project_name: Kaggle project slug.
        experiment_id: Experiment identifier to reuse.

    Returns:
        List of formatted CLI commands for dependent modules.

    Examples:
        >>> isinstance(suggest_next_steps("model", "titanic", "exp-123"), list)
        True
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
            f"  --exp-id {experiment_id}"
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
    Render follow-up command suggestions to the console.

    Args:
        current_module: Recently completed module name.
        project_name: Kaggle project slug.
        experiment_id: Experiment identifier to reuse.
        console: Optional Rich console instance.
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
