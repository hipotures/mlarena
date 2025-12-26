"""Core initialization logic orchestrating all init steps."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .ai import detect_problem_type_and_metric, validate_and_fix_metric
from .cdp import fetch_kaggle_evaluation
from .config import generate_config_py
from .files import copy_templates, create_directory_structure, download_kaggle_data

# Import template loader from scripts
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from template_loader import TemplateValidationError, load_templates  # noqa: E402


def init_project(
    *,
    project_root: Path,
    competition_slug: str,
    skip_download: bool = False,
    force: bool = False,
    target_column: Optional[str] = None,
    problem_type: Optional[str] = None,
    metric: Optional[str] = None,
    id_column: Optional[str] = None,
    ignore_columns: Optional[list[str]] = None,
    submit_probas: Optional[bool] = None,
    submit_labels: bool = False,
    cdp_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Initialize Kaggle competition project with AI detection and Rich formatting.

    Returns:
        dict with keys:
          - success: bool
          - stats: detected dataset stats (if available)
          - error: optional error message
    """
    import pandas as pd

    console = Console(force_terminal=True)

    # Check if project is already initialized (by checking for config.py)
    config_path = project_root / "code" / "utils" / "config.py"
    if config_path.exists() and not force:
        console.rule(f"[bold yellow]Project '{competition_slug}' is already initialized[/bold yellow]", style="yellow")

        # Load and display existing config
        try:
            sys.path.insert(0, str(project_root / "code"))
            config_module = __import__("utils.config", fromlist=["dummy"])

            table = Table(title="Existing Project Configuration", show_header=True)
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Project Name", competition_slug)
            table.add_row("Location", str(project_root))
            table.add_row("Target Column", getattr(config_module, "TARGET_COLUMN", "N/A"))
            table.add_row("Problem Type", getattr(config_module, "AUTOGLUON_PROBLEM_TYPE", "N/A"))
            table.add_row("Metric", getattr(config_module, "AUTOGLUON_EVAL_METRIC", "N/A"))
            table.add_row("ID Column", getattr(config_module, "ID_COLUMN", "N/A"))
            console.print(table)

            next_steps = (
                f"[bold]To work with this project:[/]\n"
                f"[bold]1.[/] Run EDA: [cyan]uv run python scripts/mla.py eda --project {competition_slug}[/cyan]\n"
                f"[bold]2.[/] Train model: [cyan]uv run python scripts/mla.py model --project {competition_slug} --model-template gpu-dev-5m[/cyan]\n\n"
                f"[bold]To reinitialize:[/] Use [cyan]--force[/cyan] flag"
            )
            console.print(Panel(next_steps, title="Project Already Initialized", border_style="yellow"))

            return {
                "success": True,
                "stats": {
                    "target": getattr(config_module, "TARGET_COLUMN", "N/A"),
                    "problem_type": getattr(config_module, "AUTOGLUON_PROBLEM_TYPE", "N/A"),
                    "metric": getattr(config_module, "AUTOGLUON_EVAL_METRIC", "N/A"),
                    "already_initialized": True
                },
            }
        except Exception:
            # Fallback if config can't be loaded
            console.print(f"[red]Error: Project '{competition_slug}' already exists. Use --force to overwrite.[/red]")
            return {"success": False, "error": "project_exists"}

    # Create directory structure
    project_root.mkdir(parents=True, exist_ok=True)
    create_directory_structure(project_root, console)

    # Copy templates
    copy_templates(project_root, console)

    # Download data
    if not skip_download:
        success = download_kaggle_data(competition_slug, project_root, console)
        if not success:
            return {"success": False, "error": "kaggle_download_failed"}
    else:
        console.print("\n[yellow]Skipping data download (--skip-download)[/yellow]")

    # Detect from sample_submission
    sample_path = None
    submission_files = list((project_root / "data").glob("*submission*.csv*"))
    if submission_files:
        sample_path = submission_files[0]

    sample_columns = []
    if sample_path:
        try:
            sample_preview = pd.read_csv(sample_path, nrows=1, compression='infer')
            sample_columns = sample_preview.columns.tolist()
        except Exception:
            pass

    if sample_columns and not target_column and len(sample_columns) >= 2:
        detected_target = sample_columns[-1]
        console.print(f"\n[cyan]Detected target column:[/cyan] [green]'{detected_target}'[/green] [dim]from {sample_path.name}[/dim]")
        target_column = detected_target

    if not id_column:
        if sample_columns:
            id_column = sample_columns[0]
            console.print(f"[cyan]Detected ID column:[/cyan] [green]'{id_column}'[/green]")
        else:
            id_column = "id"
            console.print(f"[yellow]ID column defaulting to[/yellow] [green]'{id_column}'[/green]")

    # AI-based detection
    eval_text = ""
    submit_probabilities = None
    if submit_probas:
        submit_probabilities = True
    elif submit_labels:
        submit_probabilities = False

    if not problem_type or not metric:
        from rich.progress import Progress, SpinnerColumn, TextColumn

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Fetching competition details from Kaggle...", total=None)
            try:
                eval_text = fetch_kaggle_evaluation(competition_slug, cdp_url)
                progress.remove_task(task)
            except RuntimeError as exc:
                progress.remove_task(task)
                console.print(f"[yellow]Skipping AI detection: {exc}[/yellow]")
                eval_text = ""

        if eval_text:
            # Show evaluation text in a panel (max 1024 chars)
            display_text = eval_text[:1024]
            if len(eval_text) > 1024:
                display_text += "..."

            eval_panel = Panel(
                display_text,
                title="[cyan]Evaluation Section from Kaggle[/cyan]",
                border_style="dim",
                padding=(0, 1),
            )
            console.print(eval_panel)

            ai_problem, ai_metric, ai_submit_proba, ai_log = detect_problem_type_and_metric(
                eval_text, competition_slug, project_root, console
            )

            if ai_problem:
                problem_type = problem_type or ai_problem
                metric = metric or ai_metric
                if submit_probabilities is None and ai_submit_proba is not None:
                    submit_probabilities = ai_submit_proba
        else:
            ai_log = None
    else:
        ai_log = None

    # Track original Kaggle metric name for documentation
    kaggle_metric_name = metric

    # Fallbacks
    if not target_column:
        target_column = "target"
    if not problem_type:
        problem_type = "binary"
    if not metric:
        default_metrics = {"binary": "roc_auc", "regression": "mean_absolute_error", "multiclass": "accuracy"}
        metric = default_metrics.get(problem_type, "roc_auc")
        console.print(f"Using default metric for {problem_type}: {metric}")
    if submit_probabilities is None:
        proba_metrics = {"roc_auc", "log_loss", "brier_score"}
        submit_probabilities = metric in proba_metrics

    # Build ignored columns list
    ignored_columns = list(ignore_columns or [])
    if id_column and id_column != target_column:
        ignored_columns.append(id_column)
    ignored_columns = list(dict.fromkeys(ignored_columns))  # dedup

    # Validate AutoGluon metric and suggest alternative if needed
    console.print()  # blank line for readability
    validated_metric, metric_ai_log = validate_and_fix_metric(
        problem_type, metric, kaggle_metric_name, project_root, console
    )

    # Use validated metric for AutoGluon, keep original for Kaggle reference
    autogluon_metric = validated_metric

    # Generate config.py
    sample_submission_name = sample_path.name if sample_path else "sample_submission.csv"
    generate_config_py(
        project_root,
        competition_slug,
        target_column,
        id_column,
        problem_type,
        autogluon_metric,
        submit_probabilities,
        ignored_columns,
        sample_submission_name,
        console,
        kaggle_metric=kaggle_metric_name,
    )

    # Print summary
    console.rule(f"[bold green]✓ Project '{competition_slug}' initialized successfully![/bold green]", style="green")

    table = Table(title="Project Configuration", show_header=True)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Project Name", competition_slug)
    table.add_row("Target Column", target_column)
    table.add_row("Problem Type", problem_type)
    table.add_row("Metric", metric)
    table.add_row("Location", str(project_root))
    console.print(table)

    # Show model templates
    try:
        model_templates, _ = load_templates("model", project_root, suppress_warnings=True)
        if model_templates:
            console.rule("[cyan]Model Templates[/cyan]")
            tpl_table = Table(show_header=True, header_style="bold cyan")
            tpl_table.add_column("Name", style="cyan")
            tpl_table.add_column("Preset", style="green")
            tpl_table.add_column("Time Limit", style="magenta")
            tpl_table.add_column("GPU", style="yellow")
            for name, payload in model_templates.items():
                hyper = (payload.get("config") or {}).get("hyperparameters") or {}
                preset = hyper.get("presets") or hyper.get("preset") or "-"
                time_limit = hyper.get("time_limit")
                time_str = f"{time_limit}s" if time_limit is not None else "-"
                use_gpu = hyper.get("use_gpu")
                gpu_str = "yes" if use_gpu else ("no" if use_gpu is not None else "-")
                tpl_table.add_row(name, str(preset), time_str, gpu_str)
            console.print(tpl_table)
    except TemplateValidationError as exc:
        console.print(f"[yellow]Could not read model templates: {exc}[/yellow]")

    # Show preprocess templates
    try:
        preprocess_templates, _ = load_templates("preprocess", project_root, suppress_warnings=True)
        if preprocess_templates:
            console.rule("[cyan]Preprocess Templates[/cyan]")
            pre_table = Table(show_header=True, header_style="bold cyan")
            pre_table.add_column("Name", style="cyan")
            pre_table.add_column("Module", style="green")
            pre_table.add_column("Cache", style="yellow")
            for name, payload in preprocess_templates.items():
                module = payload.get("module", "identity")
                cache = payload.get("cache", True)
                pre_table.add_row(name, module, "yes" if cache else "no")
            console.print(pre_table)
    except TemplateValidationError as exc:
        console.print(f"[yellow]Could not read preprocess templates: {exc}[/yellow]")

    # Next steps
    next_steps = (
        f"[bold]1.[/] Review configuration: [cyan]{project_root}/code/utils/config.py[/cyan]\n"
        f"[bold]2.[/] Run EDA: [cyan]uv run python scripts/mla.py eda --project {competition_slug}[/cyan]\n"
        f"[bold]3.[/] Train baseline: [cyan]uv run python scripts/mla.py model --project {competition_slug} --model-template gpu-dev-5m[/cyan]"
    )
    console.print(Panel(next_steps, title="Next Steps", border_style="yellow"))

    stats = {
        "target": target_column,
        "problem_type": problem_type,
        "metric": metric,
    }

    # Add AI interaction logs if available
    if ai_log:
        stats["ai_detection"] = ai_log
    if metric_ai_log:
        stats["ai_metric_validation"] = metric_ai_log

    return {
        "success": True,
        "stats": stats,
    }
