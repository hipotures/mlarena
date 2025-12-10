"""
Centralized display utility for module output formatting.

Provides consistent header/footer panels across all modules using Rich library
with standardized styling and information presentation.
"""

from typing import Optional, Dict, Any, Union
from pathlib import Path
from datetime import datetime, timezone
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
import re


# Color scheme
COLOR_HEADER_BORDER = "#3498DB"  # Blue
COLOR_FOOTER_SUCCESS = "#2C3E50"  # Dark gray
COLOR_FOOTER_ERROR = "#E74C3C"  # Red
COLOR_KEY = "bold cyan"
COLOR_VALUE = "white"
COLOR_OVERRIDE = "yellow"
COLOR_METRIC_KEY = "cyan"


def format_path_relative(path: Union[str, Path], project_root: Path) -> str:
    """
    Convert absolute path to project-relative path.

    Args:
        path: Path to format (absolute or relative)
        project_root: Project root directory

    Returns:
        Relative path string or original if not under project_root

    Examples:
        /mnt/ml/kaggle/projects/kaggle/proj/data/train.csv -> data/train.csv
        /home/user/projects/kaggle/proj/experiments/... -> experiments/...
    """
    if not path:
        return ""

    path = Path(path)
    try:
        rel = path.relative_to(project_root)
        # Simplify if starts with "projects/kaggle/project-name"
        parts = rel.parts
        if len(parts) > 2 and parts[0] == "projects" and parts[1] == "kaggle":
            # Remove "projects/kaggle/" prefix
            return str(Path(*parts[2:]))
        return str(rel)
    except ValueError:
        # Path not under project_root, return as-is
        return str(path)


def extract_template_overrides(
    template_config: Dict[str, Any],
    cli_invocation: Dict[str, Any]
) -> Dict[str, tuple]:
    """
    Compare template config vs CLI invocation to identify overrides.

    Args:
        template_config: Configuration from template YAML
        cli_invocation: CLI arguments that invoked the module

    Returns:
        Dictionary mapping parameter names to (template_value, cli_value) tuples

    Example:
        template_config = {"time_limit": 300, "preset": "medium"}
        cli_invocation = {"time_limit": 28800}
        Returns: {"time_limit": (300, 28800)}
    """
    overrides = {}

    for key, cli_value in cli_invocation.items():
        if key in template_config and template_config[key] != cli_value:
            overrides[key] = (template_config[key], cli_value)

    return overrides


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string like "5m 30s" or "2h 15m 30s"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")

    return " ".join(parts)


def format_time_limit(seconds: int | None) -> str:
    """
    Format time limit with both seconds and human-readable form.

    Args:
        seconds: Time limit in seconds (can be None)

    Returns:
        Formatted string like "28800s (8.0h)" or "300s (5.0m)"
    """
    if seconds is None:
        return "N/A"

    hours = seconds / 3600
    minutes = seconds / 60

    if seconds >= 3600:
        return f"{seconds}s ({hours:.1f}h)"
    elif seconds >= 60:
        return f"{seconds}s ({minutes:.1f}m)"
    else:
        return f"{seconds}s"


def truncate_path(path: str, max_length: int = 50) -> str:
    """
    Truncate long paths with ellipsis in the middle.

    Args:
        path: Path string
        max_length: Maximum length before truncation

    Returns:
        Truncated path if necessary

    Example:
        "experiments/pre-av_weights_mix_best/artifacts/preprocess/file.csv"
        -> "experiments/.../artifacts/preprocess/file.csv"
    """
    if len(path) <= max_length:
        return path

    # Keep start and end, truncate middle
    parts = Path(path).parts
    if len(parts) <= 2:
        return path[:max_length-3] + "..."

    # Keep first and last 2 parts, add ellipsis
    return str(Path(parts[0], "...", parts[-2], parts[-1]))


def format_progress_message(message: str, project_root: Path) -> str:
    """
    Transform custom module output into rich-styled format.

    Removes module prefixes like [module_name] and formats paths/key-value pairs.

    Args:
        message: Raw message from custom module
        project_root: Project root for path formatting

    Returns:
        Rich-formatted string

    Examples:
        "[av_weights_mix] Loading dataset: /path"
        -> "[bold cyan]Loading dataset:[/bold cyan] data/file.csv"

        "[av_weights_mix] AV stats: mean=1.02, min=0.01"
        -> "[bold cyan]AV stats:[/bold cyan] mean=[green]1.02[/green] min=[yellow]0.01[/yellow]"
    """
    # Remove module prefix [module_name]
    message = re.sub(r'^\[[\w_-]+\]\s*', '', message)

    # Format paths in message
    def replace_path(match):
        path = match.group(0)
        return format_path_relative(path, project_root)

    # Replace absolute paths (starting with / or containing /mnt/, /home/)
    message = re.sub(r'(?:/[\w/./_-]+)', replace_path, message)

    # Format key:value pairs
    if ':' in message and '=' not in message:
        key, value = message.split(':', 1)
        return f"[{COLOR_KEY}]{key.strip()}:[/{COLOR_KEY}] [{COLOR_VALUE}]{value.strip()}[/{COLOR_VALUE}]"

    # Format key=value pairs (e.g., "mean=1.02, min=0.01")
    if '=' in message and ':' in message:
        # Extract prefix before stats
        prefix, stats = message.split(':', 1)
        formatted_prefix = f"[{COLOR_KEY}]{prefix.strip()}:[/{COLOR_KEY}]"

        # Format each stat
        formatted_stats = []
        for stat in stats.split(','):
            stat = stat.strip()
            if '=' in stat:
                k, v = stat.split('=', 1)
                formatted_stats.append(f"{k.strip()}=[{COLOR_METRIC_KEY}]{v.strip()}[/{COLOR_METRIC_KEY}]")
            else:
                formatted_stats.append(stat)

        return f"{formatted_prefix} {', '.join(formatted_stats)}"

    return message


def print_module_header(
    module_name: str,
    started_at: str,
    experiment_id: Optional[str] = None,
    template_name: Optional[str] = None,
    template_config: Optional[Dict[str, Any]] = None,
    cli_overrides: Optional[Dict[str, tuple]] = None,
    cli_invocation: Optional[Dict[str, Any]] = None,
    input_paths: Optional[Dict[str, str]] = None,
    output_paths: Optional[Dict[str, str]] = None,
    project_root: Optional[Path] = None,
    console: Optional[Console] = None
) -> None:
    """
    Print standardized module header panel.

    Args:
        module_name: Name of the module (e.g., "preprocess", "model")
        started_at: ISO-8601 timestamp of module start
        template_name: Template name if applicable
        template_config: Configuration from template YAML
        cli_overrides: Parameters overridden by CLI (from extract_template_overrides)
        input_paths: Input file paths dict
        output_paths: Expected output file paths dict
        project_root: Project root for path formatting
        console: Rich console instance (creates new if None)
    """
    if console is None:
        console = Console(force_terminal=True)

    if project_root is None:
        project_root = Path.cwd()

    template_config = template_config or {}
    cli_overrides = cli_overrides or {}
    input_paths = input_paths or {}
    output_paths = output_paths or {}

    # Parse timestamp
    try:
        dt = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
        formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        formatted_time = started_at

    # Build content lines
    lines = []
    lines.append(f"[{COLOR_KEY}]Started:[/{COLOR_KEY}] {formatted_time}")
    if experiment_id:
        lines.append(f"[{COLOR_KEY}]Experiment:[/{COLOR_KEY}] {experiment_id}")

    # Template configuration
    if template_name:
        lines.append("")
        lines.append(f"[{COLOR_KEY}]Template:[/{COLOR_KEY}] {template_name}")

        # Show key config parameters
        config_params = [
            ("preset", "Preset"),
            ("presets", "Preset"),
            ("time_limit", "Time Limit"),
            ("use_gpu", "GPU"),
            ("mode", "Mode"),
            ("included_model_types", "Model Types"),
        ]

        for param_key, param_label in config_params:
            if param_key in template_config:
                value = template_config[param_key]

                # Format value
                if param_key == "time_limit":
                    value_str = format_time_limit(value)
                elif param_key == "included_model_types" and isinstance(value, list):
                    value_str = str(value)
                elif param_key == "use_gpu":
                    value_str = "Yes" if value else "No"
                else:
                    value_str = str(value)

                # Check if overridden
                if param_key in cli_overrides:
                    _, cli_value = cli_overrides[param_key]
                    if param_key == "time_limit":
                        cli_value_str = format_time_limit(cli_value)
                    else:
                        cli_value_str = str(cli_value)

                    # Check if from convenience flag
                    convenience_flag = cli_invocation.get("_convenience_flag") if cli_invocation else None
                    if convenience_flag:
                        flag_label = f"--{convenience_flag}"
                    else:
                        flag_label = "CLI override"

                    lines.append(
                        f"  [{COLOR_KEY}]{param_label}:[/{COLOR_KEY}] "
                        f"[{COLOR_OVERRIDE}]{cli_value_str}[/{COLOR_OVERRIDE}] "
                        f"[dim][{flag_label}][/dim]"
                    )
                else:
                    lines.append(f"  [{COLOR_KEY}]{param_label}:[/{COLOR_KEY}] {value_str}")

    # Input paths (with optional shapes)
    if input_paths:
        lines.append("")
        lines.append(f"[{COLOR_KEY}]Input:[/{COLOR_KEY}]")
        for key, path in input_paths.items():
            if key == "__shapes__":
                continue
            label = key.replace('_', ' ').title()
            lines.append(f"  [{COLOR_KEY}]{label}:[/{COLOR_KEY}] {path}")
        # Show shapes if provided
        shapes_meta = input_paths.get("__shapes__") if "__shapes__" in input_paths else None
        if shapes_meta:
            train_shape = shapes_meta.get("train")
            test_shape = shapes_meta.get("test")
            if train_shape:
                lines.append(f"  [{COLOR_KEY}]Shapes:[/{COLOR_KEY}] Train: {train_shape[0]:,} × {train_shape[1]}")
            if test_shape:
                lines.append(f"  [{COLOR_KEY}]Shapes:[/{COLOR_KEY}] Test: {test_shape[0]:,} × {test_shape[1]}")

    # Output paths (expected)
    if output_paths:
        lines.append("")
        lines.append(f"[{COLOR_KEY}]Output:[/{COLOR_KEY}]")
        for key, path in output_paths.items():
            label = key.replace('_', ' ').title()
            lines.append(f"  [{COLOR_KEY}]{label}:[/{COLOR_KEY}] {path}")

    content = "\n".join(lines)

    # Create panel
    panel = Panel(
        content,
        title=f"[bold]{module_name.upper()}[/bold]",
        border_style=COLOR_HEADER_BORDER,
        box=box.HEAVY,
        padding=(0, 1)
    )

    console.print(panel)


def print_module_footer(
    module_name: str,
    finished_at: str,
    duration: float,
    output_paths: Optional[Dict[str, str]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    shapes: Optional[Dict[str, list]] = None,
    error: Optional[str] = None,
    project_root: Optional[Path] = None,
    console: Optional[Console] = None
) -> None:
    """
    Print standardized module footer panel.

    Args:
        module_name: Name of the module
        finished_at: ISO-8601 timestamp of completion
        duration: Duration in seconds
        output_paths: Actual output file paths dict
        metrics: Module-specific metrics dict
        shapes: Dataset shape changes (for preprocess)
        error: Error message if failed
        project_root: Project root for path formatting
        console: Rich console instance (creates new if None)
    """
    if console is None:
        console = Console(force_terminal=True)

    if project_root is None:
        project_root = Path.cwd()

    output_paths = output_paths or {}
    metrics = metrics or {}

    # Parse timestamp
    try:
        dt = datetime.fromisoformat(finished_at.replace('Z', '+00:00'))
        formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        formatted_time = finished_at

    # Determine title and border style
    if error:
        title = f"[bold]{module_name.upper()} FAILED[/bold]"
        border_style = COLOR_FOOTER_ERROR
    else:
        title = f"[bold]{module_name.upper()} COMPLETED[/bold]"
        border_style = COLOR_FOOTER_SUCCESS

    # Build content lines
    lines = []
    lines.append(f"[{COLOR_KEY}]Finished:[/{COLOR_KEY}] {formatted_time}")
    lines.append(f"[{COLOR_KEY}]Duration:[/{COLOR_KEY}] {format_duration(duration)}")

    if error:
        # Show error message
        lines.append("")
        lines.append(f"[bold red]Error:[/bold red] {error}")
    else:
        # Output files
        if output_paths:
            lines.append("")
            lines.append(f"[{COLOR_KEY}]Output Files:[/{COLOR_KEY}]")
            for key, path in output_paths.items():
                label = key.replace('_', ' ').title()
                lines.append(f"  [{COLOR_KEY}]{label}:[/{COLOR_KEY}] {path}")

        # Dataset shapes (for preprocess)
        if shapes:
            lines.append("")
            lines.append(f"[{COLOR_KEY}]Dataset Shapes:[/{COLOR_KEY}]")
            if "train_before" in shapes and "train_after" in shapes:
                train_before = shapes["train_before"]
                train_after = shapes["train_after"]
                lines.append(
                    f"  [{COLOR_KEY}]Train:[/{COLOR_KEY}] "
                    f"{train_before[0]:,} × {train_before[1]} → "
                    f"{train_after[0]:,} × {train_after[1]}"
                )
            if "test_before" in shapes and "test_after" in shapes:
                test_before = shapes["test_before"]
                test_after = shapes["test_after"]
                lines.append(
                    f"  [{COLOR_KEY}]Test:[/{COLOR_KEY}] "
                    f"{test_before[0]:,} × {test_before[1]} → "
                    f"{test_after[0]:,} × {test_after[1]}"
                )

        # Module-specific metrics
        if metrics:
            # AV statistics (for preprocess with adversarial validation)
            if "av_stats" in metrics:
                av_stats = metrics["av_stats"]
                lines.append("")
                lines.append(f"[{COLOR_KEY}]AV Statistics:[/{COLOR_KEY}]")
                lines.append(
                    f"  Mean: [{COLOR_METRIC_KEY}]{av_stats.get('mean', 'N/A'):.4f}[/{COLOR_METRIC_KEY}]  "
                    f"Min: [{COLOR_METRIC_KEY}]{av_stats.get('min', 'N/A'):.4f}[/{COLOR_METRIC_KEY}]  "
                    f"Max: [{COLOR_METRIC_KEY}]{av_stats.get('max', 'N/A'):.4f}[/{COLOR_METRIC_KEY}]"
                )
                if "clipped_rows" in metrics:
                    lines.append(f"  Clipped: {metrics['clipped_rows']:,} rows")

            # Model metrics (local CV, best model, etc.)
            if "local_cv" in metrics:
                lines.append("")
                lines.append(f"[{COLOR_KEY}]Model Performance:[/{COLOR_KEY}]")
                lines.append(f"  Local CV: [{COLOR_METRIC_KEY}]{metrics['local_cv']:.6f}[/{COLOR_METRIC_KEY}]")

            if "best_model" in metrics:
                lines.append(f"  Best Model: {metrics['best_model']}")

            # Other metrics
            for key, value in metrics.items():
                if key not in ["av_stats", "clipped_rows", "local_cv", "best_model"]:
                    label = key.replace('_', ' ').title()
                    lines.append(f"  [{COLOR_KEY}]{label}:[/{COLOR_KEY}] {value}")

    # Enforce max 15 lines
    if len(lines) > 15:
        lines = lines[:14] + ["[dim]... (output truncated)[/dim]"]

    content = "\n".join(lines)

    # Create panel
    panel = Panel(
        content,
        title=title,
        border_style=border_style,
        box=box.HEAVY,
        padding=(0, 1)
    )

    console.print(panel)
