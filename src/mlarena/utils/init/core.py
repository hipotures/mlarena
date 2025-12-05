"""Core initialization logic orchestrating all init steps."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .ai import detect_problem_type_and_metric
from .cdp import fetch_kaggle_evaluation
from .config import generate_config_py
from .files import copy_templates, create_directory_structure, customize_readme, download_kaggle_data

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

    console = Console()

    # Check if exists (ignore experiments/ dir created by ExperimentState)
    if project_root.exists() and not force:
        existing_items = [item for item in project_root.iterdir() if item.name != "experiments"]
        if existing_items:
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
    submission_files = list((project_root / "data").glob("*submission*.csv"))
    if submission_files:
        sample_path = submission_files[0]

    sample_columns = []
    if sample_path:
        try:
            sample_preview = pd.read_csv(sample_path, nrows=1)
            sample_columns = sample_preview.columns.tolist()
        except Exception:
            pass

    if sample_columns and not target_column and len(sample_columns) >= 2:
        detected_target = sample_columns[-1]
        console.print(f"\n[cyan]Detected target column: '{detected_target}' from {sample_path.name}[/cyan]")
        target_column = detected_target

    if not id_column:
        if sample_columns:
            id_column = sample_columns[0]
            console.print(f"[cyan]Detected ID column: '{id_column}'[/cyan]")
        else:
            id_column = "id"
            console.print(f"[yellow]ID column defaulting to '{id_column}'[/yellow]")

    # AI-based detection
    eval_text = ""
    submit_probabilities = None
    if submit_probas:
        submit_probabilities = True
    elif submit_labels:
        submit_probabilities = False

    if not problem_type or not metric:
        try:
            console.print(f"\n[cyan]Fetching competition details from Kaggle...[/cyan]")
            eval_text = fetch_kaggle_evaluation(competition_slug, cdp_url)
        except RuntimeError as exc:
            console.print(f"[yellow]Skipping AI detection: {exc}[/yellow]")

        if eval_text:
            console.print(f"[dim]Evaluation section: {eval_text[:100]}...[/dim]")
            console.print(f"[cyan]Asking AI to detect problem type and metric...[/cyan]")

            ai_problem, ai_metric, ai_submit_proba = detect_problem_type_and_metric(
                eval_text, competition_slug, project_root, console
            )

            if ai_problem:
                problem_type = problem_type or ai_problem
                metric = metric or ai_metric
                if submit_probabilities is None and ai_submit_proba is not None:
                    submit_probabilities = ai_submit_proba

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

    # Generate config.py
    sample_submission_name = sample_path.name if sample_path else "sample_submission.csv"
    generate_config_py(
        project_root,
        competition_slug,
        target_column,
        id_column,
        problem_type,
        metric,
        submit_probabilities,
        ignored_columns,
        sample_submission_name,
        console,
    )

    # Customize README
    readme_replacements = {
        "playground-series-s5e11": competition_slug,
        "{{COMPETITION_NAME}}": competition_slug,
        "{{TARGET_COLUMN}}": target_column,
        "{{ID_COLUMN}}": id_column,
        "{{METRIC_NAME}}": metric,
    }
    customize_readme(project_root, readme_replacements, console)

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
        f"[bold]3.[/] Train baseline: [cyan]uv run python scripts/mla.py model --project {competition_slug} --model-template dev-gpu[/cyan]"
    )
    console.print(Panel(next_steps, title="Next Steps", border_style="yellow"))

    return {
        "success": True,
        "stats": {"target": target_column, "problem_type": problem_type, "metric": metric},
    }
