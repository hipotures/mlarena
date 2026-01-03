"""
Centralized display utility for module output formatting.

Provides consistent header/footer panels across all modules using Rich library
with standardized styling and information presentation.
"""

from typing import Optional, Dict, Any, Union
from pathlib import Path
from datetime import datetime, timezone
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.text import Text
from rich import box
import json
import re
import os
import shutil
import subprocess
import sys
import termios
import tty
import select


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


def get_directory_size(path: Path) -> int:
    """
    Calculate total size of directory in bytes.

    Args:
        path: Directory path

    Returns:
        Total size in bytes
    """
    if not path.exists() or not path.is_dir():
        return 0

    total = 0
    for entry in path.rglob('*'):
        if entry.is_file():
            try:
                total += entry.stat().st_size
            except (OSError, PermissionError):
                pass
    return total


def format_size(size_bytes: int) -> str:
    """
    Format size in bytes to human-readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string like "352KB", "29MB", "1.2GB"
    """
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.0f}KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.0f}MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f}GB"


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


def copy_to_clipboard(text: str) -> bool:
    """
    Copy text to system clipboard.

    Supports:
        - Wayland: wl-copy
        - X11: xclip, xsel (fallback)

    Args:
        text: Text to copy to clipboard

    Returns:
        True if successful, False otherwise
    """
    # Detect session type
    if os.environ.get("XDG_SESSION_TYPE") == "wayland":
        if shutil.which("wl-copy"):
            cmd = ["wl-copy"]
        else:
            return False
    else:
        # X11 or unknown
        if shutil.which("xclip"):
            cmd = ["xclip", "-selection", "clipboard"]
        elif shutil.which("xsel"):
            cmd = ["xsel", "--clipboard", "--input"]
        else:
            return False

    try:
        subprocess.run(
            cmd,
            input=text.encode(),
            check=True,
            timeout=2,
            stderr=subprocess.DEVNULL
        )
        return True
    except (subprocess.SubprocessError, OSError):
        return False


def build_path_tree_for_chain(
    current_paths: Dict[str, str],
    current_shapes: Optional[Dict] = None,
    chain_exp_id: Optional[str] = None,
    input_source: Optional[str] = None,
    current_step: Optional[str] = None,
    is_header: bool = True,
    project_root: Optional[Path] = None
) -> Optional[Tree]:
    """
    Build Rich.Tree showing incremental history of preprocessing chain.

    For preprocessing chains, displays all steps with color-coded highlighting:
    - INPUT: previous step (input_source)
    - OUTPUT: current step (current_step)
    If the current step has no state yet (header), uses current_paths/current_shapes
    to inject a synthetic OUTPUT node so input/output are shown together.

    Args:
        current_paths: Paths for current step (e.g., {"train": "path/to/train.csv"})
        current_shapes: Shape annotations for current step (optional)
        chain_exp_id: Chain identifier (e.g., "pre-catboost-full/abc123")
        input_source: Previous step in chain (e.g., "1-encoder")
        current_step: Current step in chain (e.g., "2-feature_selector")
        is_header: True for header, False for footer (kept for compatibility)
        project_root: Root path for normalization

    Returns:
        Rich.Tree object or None (fallback to flat list)
    """
    # Fallback: Not a preprocessing chain
    if not chain_exp_id or not project_root:
        return None

    try:
        # Reconstruct chain history
        chain_base_dir = project_root / "experiments" / chain_exp_id

        if not chain_base_dir.exists():
            return None

        def _step_sort_key(name: str) -> tuple[int, str]:
            match = re.match(r'(\d+)-', name)
            if match:
                return (int(match.group(1)), name)
            return (10**9, name)

        def _normalize_processed_paths(paths: Dict[str, str] | None) -> Dict[str, str]:
            if not paths:
                return {}
            normalized: Dict[str, str] = {}
            mapping = {
                "train_processed": "train_processed",
                "train": "train_processed",
                "test_processed": "test_processed",
                "test": "test_processed",
                "orig_processed": "orig_processed",
                "orig": "orig_processed",
                "original": "orig_processed",
                "tuning_processed": "tuning_processed",
                "tuning": "tuning_processed",
            }
            for key, value in paths.items():
                if key == "__shapes__":
                    continue
                if not value:
                    continue
                mapped = mapping.get(key)
                if mapped:
                    normalized[mapped] = value
            return normalized

        # List all submodules (0-imputer, 1-encoder, 2-feature_selector, ...)
        all_steps = sorted(
            [
                d for d in chain_base_dir.iterdir()
                if d.is_dir() and re.match(r'\d+-', d.name)
            ],
            key=lambda d: _step_sort_key(d.name)
        )

        if len(all_steps) == 0:
            # No steps found
            return None

        # Read state.json from each step
        chain_history = []
        for step_dir in all_steps:
            state_file = step_dir / "state.json"
            if state_file.exists():
                try:
                    state = json.loads(state_file.read_text())
                    # Payload is in modules.preprocess.payload (new structure)
                    preprocess_module = state.get("modules", {}).get("preprocess", {})
                    payload = preprocess_module.get("payload", {})
                    processed_paths = _normalize_processed_paths(payload)
                    chain_history.append({
                        "step": step_dir.name,  # "0-imputer"
                        "paths": processed_paths,
                        "shapes": payload.get("shapes", {}),
                        "custom_module_state": payload.get("custom_module_state", {})
                    })
                except (json.JSONDecodeError, KeyError):
                    # Skip malformed state files
                    continue

        current_paths_norm = _normalize_processed_paths(current_paths)

        # Inject/override current step so OUTPUT appears in the tree (header or footer)
        if current_step and current_paths_norm:
            existing = {step["step"]: step for step in chain_history}
            if current_step in existing:
                existing[current_step]["paths"].update(current_paths_norm)
                if current_shapes:
                    existing[current_step]["shapes"] = {
                        **existing[current_step].get("shapes", {}),
                        **current_shapes
                    }
            else:
                chain_history.append({
                    "step": current_step,
                    "paths": current_paths_norm,
                    "shapes": current_shapes or {}
                })

        if not chain_history:
            return None

        # Build tree
        root_label = f"📁 experiments/{chain_exp_id}/"
        tree = Tree(root_label, guide_style="dim")

        input_step = input_source
        output_step = current_step

        def _style_label(label: str, style: Optional[str], dim: bool) -> str:
            if dim:
                return f"[dim]{label}[/dim]"
            if style:
                return f"[{style}]{label}[/{style}]"
            return label

        for step_info in sorted(chain_history, key=lambda item: _step_sort_key(item["step"])):
            step_name = step_info["step"]
            is_input = (step_name == input_step)
            is_output = (step_name == output_step)
            is_active = is_input or is_output

            if not step_info["paths"] and not is_active:
                continue

            # Add step node (color branch for input/output)
            step_style = "green" if is_input else "cyan" if is_output else None
            step_label = _style_label(f"📁 {step_name}", step_style, dim=not is_active)
            step_node = tree.add(step_label)

            # Add artifacts/preprocess hierarchy
            artifacts_label = _style_label("📁 artifacts", step_style, dim=not is_active)
            preprocess_label = _style_label("📁 preprocess", step_style, dim=not is_active)
            artifacts_node = step_node.add(artifacts_label)
            preprocess_node = artifacts_node.add(preprocess_label)

            # Add files
            paths_dict = step_info.get("paths", {})
            shapes_dict = step_info.get("shapes", {})

            # Process train/test/orig/tuning processed files
            for file_key in ["train_processed", "test_processed", "orig_processed", "tuning_processed"]:
                if file_key in paths_dict:
                    file_path = paths_dict[file_key]
                    if file_path:  # Skip None values
                        filename = Path(file_path).name

                        # Shape annotation
                        shape_annotation = ""
                        shape_key = f"{file_key.replace('_processed', '')}_after"
                        if shapes_dict and shape_key in shapes_dict:
                            rows, cols = shapes_dict[shape_key]
                            shape_annotation = f" ({rows:,} × {cols})"

                        # Add file node with color coding
                        file_label = _style_label(f"📄 {filename}{shape_annotation}", step_style, dim=not is_active)
                        preprocess_node.add(file_label)

            # Add eval_processed if present in custom_module_state
            custom_state = step_info.get("custom_module_state", {})
            if "eval_path" in custom_state:
                eval_path_str = custom_state["eval_path"]
                if eval_path_str:
                    filename = Path(eval_path_str).name
                    # Try to get row count from custom_module_state
                    eval_rows = custom_state.get("eval_rows")
                    shape_annotation = f" ({eval_rows:,} rows)" if eval_rows else ""
                    file_label = _style_label(f"📄 {filename}{shape_annotation}", step_style, dim=not is_active)
                    preprocess_node.add(file_label)

        return tree

    except Exception:
        # Fallback to None on any error (caller will use flat list)
        return None


def show_path_copy_menu(
    paths_dict: Dict[str, str],
    project_root: Path,
    console: Console
) -> None:
    """
    Show interactive clipboard copy menu for file paths.

    Uses termios raw mode for single-key input (like submit.py).
    Allows copying individual files or all paths to clipboard.

    Args:
        paths_dict: {"train": "path/to/train.csv", ...}
        project_root: Root path for normalization
        console: Rich Console instance
    """
    # Skip if not interactive
    if not sys.stdin.isatty():
        return

    # Normalize paths to absolute
    abs_paths = {}
    for label, path_str in paths_dict.items():
        if not Path(path_str).is_absolute():
            abs_path = (project_root / path_str).resolve()
        else:
            abs_path = Path(path_str).resolve()
        abs_paths[label] = abs_path

    # Build numbered list
    menu_items = list(abs_paths.items())  # [(label, abs_path), ...]
    if not menu_items:
        return

    # Compute base directory (common parent)
    common_path = Path(os.path.commonpath([str(p) for _, p in menu_items]))
    base_dir = common_path if common_path.is_dir() else common_path.parent

    # Print menu
    console.print()
    for idx, (label, abs_path) in enumerate(menu_items, 1):
        rel_name = abs_path.name
        console.print(f"  {idx}) {rel_name}")

    all_idx = len(menu_items) + 1
    console.print(f"  {all_idx}) [cyan]Copy all paths[/cyan]")
    base_idx = all_idx + 1
    console.print(f"  {base_idx}) [cyan]Copy base directory[/cyan]")

    # Get terminal settings
    fd = sys.stdin.fileno()
    try:
        old_settings = termios.tcgetattr(fd)
    except termios.error:
        # Terminal doesn't support termios (e.g., non-POSIX)
        return

    try:
        # Enter raw mode
        tty.setraw(fd)

        console.print()
        console.print("[dim]Press number (or Enter to skip): [/dim]", end="")

        # Read single character (non-blocking with timeout)
        ready, _, _ = select.select([sys.stdin], [], [], 10.0)  # 10s timeout

        if not ready:
            console.print("[yellow]Timeout[/yellow]")
            return

        char = sys.stdin.read(1)

        # Handle Enter/Ctrl+C/Esc
        if char in ('\n', '\r', '\x03', '\x1b'):
            console.print("[yellow]Skipped[/yellow]")
            return

        # Parse choice
        try:
            choice = int(char)
        except ValueError:
            console.print(f"[red]Invalid: {char!r}[/red]")
            return

        # Execute choice
        if 1 <= choice <= len(menu_items):
            # Individual file
            label, abs_path = menu_items[choice - 1]
            if copy_to_clipboard(str(abs_path)):
                # Show truncated path (just filename for readability)
                console.print(f"[green]✓ Copied:[/green] {abs_path.name}")
            else:
                console.print(f"[red]✗ Clipboard failed (install wl-copy or xclip)[/red]")

        elif choice == all_idx:
            # All paths
            all_paths_str = "\n".join(str(p) for _, p in menu_items)
            if copy_to_clipboard(all_paths_str):
                console.print(f"[green]✓ Copied {len(menu_items)} paths[/green]")
            else:
                console.print(f"[red]✗ Clipboard failed[/red]")
        elif choice == base_idx:
            # Base directory
            if copy_to_clipboard(str(base_dir)):
                console.print(f"[green]✓ Copied base directory[/green]")
            else:
                console.print(f"[red]✗ Clipboard failed[/red]")
        else:
            console.print(f"[red]Invalid choice: {choice}[/red]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

    finally:
        # Always restore terminal
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def format_progress_message(message: str, project_root: Path) -> str:
    """
    Transform custom module output into rich-styled format.
    """
    message = re.sub(r'^\[[\w_-]+\]\s*', '', message)

    def replace_path(match):
        path = match.group(0)
        return format_path_relative(path, project_root)

    message = re.sub(r'(?:/[\w/./_-]+)', replace_path, message)

    if ':' in message and '=' not in message:
        key, value = message.split(':', 1)
        return f"[{COLOR_KEY}]{key.strip()}:[/{COLOR_KEY}] [{COLOR_VALUE}]{value.strip()}[/{COLOR_VALUE}]"

    if '=' in message and ':' in message:
        prefix, stats = message.split(':', 1)
        formatted_prefix = f"[{COLOR_KEY}]{prefix.strip()}:[/{COLOR_KEY}]"

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
    project_name: Optional[str] = None,
    console: Optional[Console] = None
) -> None:
    """Standardized module header panel."""
    if console is None:
        console = Console(force_terminal=True)

    if project_root is None:
        project_root = Path.cwd()

    template_config = template_config or {}
    cli_overrides = cli_overrides or {}
    input_paths = input_paths or {}
    output_paths = output_paths or {}

    try:
        dt_val = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
        formatted_time = dt_val.strftime("%Y-%m-%d %H:%M:%S")
    except:
        formatted_time = started_at

    lines = []
    if project_name:
        lines.append(f"[{COLOR_KEY}]Project:[/{COLOR_KEY}] {project_name}")
    lines.append(f"[{COLOR_KEY}]Started:[/{COLOR_KEY}] {formatted_time}")
    if experiment_id:
        lines.append(f"[{COLOR_KEY}]Experiment:[/{COLOR_KEY}] {experiment_id}")

    if template_name:
        lines.append("")
        lines.append(f"[{COLOR_KEY}]Template:[/{COLOR_KEY}] {template_name}")

        implementation_name = template_config.get("module") or template_config.get("model")
        if implementation_name:
            label = "Module" if "module" in template_config else "Model"
            lines.append(f"[{COLOR_KEY}]{label}:[/{COLOR_KEY}] {implementation_name}")

        hpo_preset = template_config.get("hpo_preset")
        if hpo_preset:
            lines.append(f"[{COLOR_KEY}]HPO Preset:[/{COLOR_KEY}] [{COLOR_OVERRIDE}]{hpo_preset}[/{COLOR_OVERRIDE}]")

        config_dict = template_config.get("config", {})

        def format_config_value(key: str, value: Any, max_length: int = 70) -> str:
            if value is None:
                return "null"
            if isinstance(value, bool):
                return "Yes" if value else "No"
            if isinstance(value, list):
                if len(value) == 0:
                    return "[]"
                items_str = " | ".join(str(item) for item in value)
                count_prefix = f"[{len(value)}] " if len(value) > 1 else ""
                if len(items_str) > max_length:
                    return f"{count_prefix}{items_str[:max_length - 3]}..."
                return f"{count_prefix}{items_str}"
            if isinstance(value, dict):
                if len(value) == 0:
                    return "{}"
                items = [f"{k}:{v}" for k, v in value.items()]
                dict_str = " | ".join(items)
                count_prefix = f"[{len(value)}] " if len(value) > 1 else ""
                if len(dict_str) > max_length:
                    return f"{count_prefix}{dict_str[:max_length - 3]}..."
                return f"{count_prefix}{dict_str}"
            if key == "time_limit":
                return format_time_limit(value)
            if key == "use_gpu":
                return "Yes" if bool(value) else "No"
            value_str = str(value)
            if len(value_str) > max_length:
                return value_str[:max_length - 3] + "..."
            return value_str

        for param_key in sorted(config_dict.keys()):
            value = config_dict[param_key]
            if value is None:
                continue
            param_label = param_key.replace('_', ' ').title()
            value_str = format_config_value(param_key, value)

            if param_key in cli_overrides:
                _, cli_value = cli_overrides[param_key]
                if cli_value is None:
                    lines.append(f"  [{COLOR_KEY}]{param_label}:[/{COLOR_KEY}] {value_str}")
                else:
                    cli_value_str = format_config_value(param_key, cli_value)
                    convenience_flag = cli_invocation.get("_convenience_flag") if cli_invocation else None
                    flag_label = f"--{convenience_flag}" if convenience_flag else "CLI override"
                    lines.append(
                        f"  [{COLOR_KEY}]{param_label}:[/{COLOR_KEY}] "
                        f"[{COLOR_OVERRIDE}]{cli_value_str}[/{COLOR_OVERRIDE}] "
                        f"[dim][{flag_label}][/dim]"
                    )
            else:
                lines.append(f"  [{COLOR_KEY}]{param_label}:[/{COLOR_KEY}] {value_str}")

    if input_paths:
        lines.append("")
        if module_name == "predict":
            preprocessing = input_paths.get("preprocessing")
            model_template = input_paths.get("model_template")
            if preprocessing or model_template:
                lines.append(f"[{COLOR_KEY}]Configuration:[/{COLOR_KEY}]")
                if preprocessing:
                    lines.append(f"  [{COLOR_KEY}]Preprocessing:[/{COLOR_KEY}] {preprocessing}")
                if model_template:
                    lines.append(f"  [{COLOR_KEY}]Model Template:[/{COLOR_KEY}] {model_template}")
                lines.append("")

        lines.append(f"[{COLOR_KEY}]Input:[/{COLOR_KEY}]")
        for key, path in input_paths.items():
            if key == "__shapes__":
                continue
            if module_name == "predict" and key in ["preprocessing", "model_template"]:
                continue
            label = key.replace('_', ' ').title()
            lines.append(f"  [{COLOR_KEY}]{label}:[/{COLOR_KEY}] {path}")
        
        shapes_meta = input_paths.get("__shapes__")
        if shapes_meta:
            for s_key, s_val in shapes_meta.items():
                if s_val:
                    label = s_key.title()
                    if s_val[1] is not None:
                        lines.append(f"  [{COLOR_KEY}]Shapes:[/{COLOR_KEY}] {label}: {s_val[0]:,} × {s_val[1]}")
                    else:
                        lines.append(f"  [{COLOR_KEY}]Shapes:[/{COLOR_KEY}] {label}: {s_val[0]:,} rows")

    if output_paths:
        lines.append("")
        lines.append(f"[{COLOR_KEY}]Output:[/{COLOR_KEY}]")
        for key, path in output_paths.items():
            label = key.replace('_', ' ').title()
            lines.append(f"  [{COLOR_KEY}]{label}:[/{COLOR_KEY}] {path}")

    panel = Panel(
        "\n".join(lines),
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
    project_name: Optional[str] = None,
    experiment_id: Optional[str] = None,
    cli_invocation: Optional[Dict] = None,
    console: Optional[Console] = None
) -> None:
    """Standardized module footer panel."""
    if console is None:
        console = Console(force_terminal=True)

    if project_root is None:
        project_root = Path.cwd()

    output_paths = output_paths or {}
    metrics = metrics or {}
    path_tree = None

    if module_name == "preprocess" and not error:
        chain_exp_id = cli_invocation.get("chain_exp_id") if cli_invocation else None
        input_source = cli_invocation.get("input_source") if cli_invocation else None
        current_step = experiment_id.split("/")[-1] if experiment_id and "/" in experiment_id else None

        path_tree = build_path_tree_for_chain(
            current_paths=output_paths,
            current_shapes=shapes,
            chain_exp_id=chain_exp_id,
            input_source=input_source,
            current_step=current_step,
            is_header=False,
            project_root=project_root
        )

    try:
        dt_val = datetime.fromisoformat(finished_at.replace('Z', '+00:00'))
        formatted_time = dt_val.strftime("%Y-%m-%d %H:%M:%S")
    except:
        formatted_time = finished_at

    if error:
        title = f"[bold]{module_name.upper()} FAILED[/bold]"
        border_style = COLOR_FOOTER_ERROR
    else:
        title = f"[bold]{module_name.upper()} COMPLETED[/bold]"
        border_style = COLOR_FOOTER_SUCCESS

    lines = []
    if project_name:
        lines.append(f"[{COLOR_KEY}]Project:[/{COLOR_KEY}] {project_name}")
    lines.append(f"[{COLOR_KEY}]Finished:[/{COLOR_KEY}] {formatted_time}")
    lines.append(f"[{COLOR_KEY}]Duration:[/{COLOR_KEY}] {format_duration(duration)}")

    if error:
        lines.append("")
        lines.append(f"[bold red]Error:[/bold red] {error}")
    else:
        if output_paths and not path_tree:
            lines.append("")
            lines.append(f"[{COLOR_KEY}]Output Files:[/{COLOR_KEY}]")
            for key, path in output_paths.items():
                label = key.replace('_', ' ').title()
                if module_name == "model" and key == "model":
                    model_path = Path(path)
                    if not model_path.is_absolute():
                        model_path = project_root / path
                    size_str = format_size(get_directory_size(model_path))
                    lines.append(f"  [{COLOR_KEY}]{label}:[/{COLOR_KEY}] {path} [dim]({size_str})[/dim]")
                else:
                    lines.append(f"  [{COLOR_KEY}]{label}:[/{COLOR_KEY}] {path}")

        if shapes and not path_tree:
            lines.append("")
            if shapes.get("pass_through", False):
                lines.append(f"[{COLOR_KEY}]Dataset Shapes:[/{COLOR_KEY}] [dim](pass-through, unchanged)[/dim]")
                if "train_before" in shapes:
                    lines.append(f"  [{COLOR_KEY}]Train:[/{COLOR_KEY}] {shapes['train_before'][0]:,} × {shapes['train_before'][1]}")
                if "test_before" in shapes:
                    lines.append(f"  [{COLOR_KEY}]Test:[/{COLOR_KEY}] {shapes['test_before'][0]:,} × {shapes['test_before'][1]}")
            else:
                lines.append(f"[{COLOR_KEY}]Dataset Shapes:[/{COLOR_KEY}]")
                if "train_before" in shapes and "train_after" in shapes:
                    lines.append(f"  [{COLOR_KEY}]Train:[/{COLOR_KEY}] {shapes['train_before'][0]:,} × {shapes['train_before'][1]} → {shapes['train_after'][0]:,} × {shapes['train_after'][1]}")
                if "test_before" in shapes and "test_after" in shapes:
                    lines.append(f"  [{COLOR_KEY}]Test:[/{COLOR_KEY}] {shapes['test_before'][0]:,} × {shapes['test_before'][1]} → {shapes['test_after'][0]:,} × {shapes['test_after'][1]}")

        if metrics:
            if "av_stats" in metrics:
                av_stats = metrics["av_stats"]
                lines.append(f"\n[{COLOR_KEY}]AV Statistics:[/{COLOR_KEY}]")
                lines.append(f"  Mean: [{COLOR_METRIC_KEY}]{av_stats.get('mean', 'N/A'):.4f}[/{COLOR_METRIC_KEY}]  Min: [{COLOR_METRIC_KEY}]{av_stats.get('min', 'N/A'):.4f}[/{COLOR_METRIC_KEY}]  Max: [{COLOR_METRIC_KEY}]{av_stats.get('max', 'N/A'):.4f}[/{COLOR_METRIC_KEY}]")
                if "clipped_rows" in metrics:
                    lines.append(f"  Clipped: {metrics['clipped_rows']:,} rows")

            if "local_cv_score" in metrics:
                lines.append(f"\n[{COLOR_KEY}]Model Performance:[/{COLOR_KEY}]")
                lines.append(f"  Local CV: [{COLOR_METRIC_KEY}]{metrics['local_cv_score']:.6f}[/{COLOR_METRIC_KEY}]")
            if "best_model" in metrics:
                lines.append(f"  Best Model: {metrics['best_model']}")

            for key, value in metrics.items():
                if key not in ["av_stats", "clipped_rows", "local_cv_score", "best_model", "cache_status", "status"]:
                    lines.append(f"  [{COLOR_KEY}]{key.replace('_', ' ').title()}:[/{COLOR_KEY}] {value}")

    if len(lines) > 15:
        lines = lines[:14] + ["[dim]... (output truncated)[/dim]"]

    content = "\n".join(lines)
    status_text = None
    if metrics:
        cache_status = metrics.get("cache_status")
        status = metrics.get("status")
        if cache_status:
            cache_text = str(cache_status)
            cached_idx = cache_text.find("Cached")
            if cached_idx != -1:
                cache_text = f"{cache_text[:cached_idx]}[red]{cache_text[cached_idx:]}[/red]"
            status_text = f"[{COLOR_KEY}]Status:[/{COLOR_KEY}] {cache_text}"
        elif status:
            status_text = f"[{COLOR_KEY}]Status:[/{COLOR_KEY}] {status}"

    renderables = [Text.from_markup(content), Text("")]
    if path_tree:
        renderables.append(path_tree)
        renderables.append(Text(""))
    if status_text:
        renderables.append(Text.from_markup(status_text))

    panel = Panel(
        Group(*renderables),
        title=title,
        border_style=border_style,
        box=box.HEAVY,
        padding=(0, 1)
    )
    console.print(panel)
