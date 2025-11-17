"""
Reusable AutoGluon training pipeline driven by competition configs.

Usage:
    uv run python tools/autogluon_runner.py --project playground-series-s5e11 \
        --template dev-gpu --auto-submit
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from autogluon.tabular import TabularPredictor
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from experiment_manager import ExperimentManager, ModuleStateError
from submission_workflow import SubmissionRunner

TOOLS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_ROOT.parent

console = Console()


TEMPLATES: Dict[str, Dict[str, Any]] = {
    "fast-cpu": {
        "time_limit": 60,
        "preset": "medium_quality",
        "use_gpu": False,
        # XGB-only smoke test for quick validation.
        "hyperparameters": {"XGB": [{}]},
    },
    "dev-cpu": {"time_limit": 300, "preset": "medium_quality", "use_gpu": False},
    "dev-gpu": {"time_limit": 300, "preset": "medium_quality", "use_gpu": True},
    "best-cpu": {"time_limit": 3600, "preset": "best_quality", "use_gpu": False},
    "best-gpu": {"time_limit": 3600, "preset": "best_quality", "use_gpu": True},
    "extreme-gpu": {"time_limit": 24 * 3600, "preset": "extreme_quality", "use_gpu": True},
    "time8-cpu": {"time_limit": 8 * 3600, "preset": "best_quality", "use_gpu": False},
}


@dataclass
class ProjectContext:
    name: str
    root: Path
    config: Any
    submission_module: Any


def load_project_context(project_name: str) -> ProjectContext:
    project_root = (REPO_ROOT / project_name).resolve()
    if not project_root.exists():
        raise FileNotFoundError(f"Project directory '{project_name}' not found at {project_root}")

    code_dir = project_root / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    config = importlib.import_module("utils.config")
    submission_module = importlib.import_module("utils.submission")
    return ProjectContext(name=project_name, root=project_root, config=config, submission_module=submission_module)


def parse_args(default_project: Optional[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AutoGluon baseline for any competition project")
    parser.add_argument("--project", default=default_project, help="Competition directory name (e.g., playground-series-s5e11)")
    parser.add_argument(
        "--template",
        choices=list(TEMPLATES.keys()),
        default="dev-gpu",
        help="Predefined compute template",
    )
    parser.add_argument("--time-limit", type=int, help="Override time limit (seconds)")
    parser.add_argument("--preset", help="Override AutoGluon preset")
    parser.add_argument("--use-gpu", dest="use_gpu", type=int, choices=[0, 1], help="Override GPU usage (1 or 0)")
    parser.add_argument(
        "--force-extreme",
        action="store_true",
        help="Skip dataset size confirmation when using extreme template",
    )

    # Submission workflow flags
    parser.add_argument("--skip-submit", action="store_true", help="Do not run Kaggle submission workflow (alias for --auto-submit=0)")
    parser.add_argument("--auto-submit", action="store_true", help="Submit to Kaggle without asking (non-interactive)")
    parser.add_argument("--kaggle-message", help="Custom Kaggle submission message")
    parser.add_argument("--wait-seconds", type=int, default=30, help="Seconds to wait before scraping score")
    parser.add_argument("--cdp-url", default="http://localhost:9222", help="Playwright CDP endpoint")
    parser.add_argument("--skip-score-fetch", action="store_true", help="Skip Playwright scraping of latest score")
    parser.add_argument("--skip-git", action="store_true", help="Do not stage/commit git changes automatically")
    parser.add_argument("--experiment-id", help="Existing experiment identifier (auto-generated if omitted)")
    parser.add_argument("--require-eda", action="store_true", help="Require EDA module completion before training")
    parser.add_argument("--skip-eda-check", action="store_true", help="Deprecated: EDA enforcement is opt-in")

    args = parser.parse_args()
    if not args.project:
        parser.error("--project is required (competition directory name).")
    return args


def resolve_template(args: argparse.Namespace) -> Dict[str, Any]:
    params = TEMPLATES[args.template].copy()
    if args.time_limit is not None:
        params["time_limit"] = args.time_limit
    if args.preset is not None:
        params["preset"] = args.preset
    if args.use_gpu is not None:
        params["use_gpu"] = bool(args.use_gpu)
    return params


def confirm_extreme(train_rows: int, force: bool) -> bool:
    if force:
        return True
    console.print(
        Panel.fit(
            f"[bold red]Extreme template warning[/bold red]\n"
            f"Train rows: {train_rows}\n"
            "Recommended limit: ≤ 30,000 rows.\n"
            "Continue anyway? [y/N]",
            title="⚠️ Long Training Job",
        )
    )
    answer = input("Proceed with extreme template? [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def train_autogluon(context: ProjectContext, params: Dict[str, Any]) -> Dict[str, Any]:
    cfg = context.config

    console.print(Panel.fit(
        "[bold magenta]AutoGluon Baseline[/bold magenta]\n"
        f"Preset: {params['preset']}\n"
        f"Time Limit: {params['time_limit']}s\n"
        f"GPU: {'Yes' if params['use_gpu'] else 'No'}",
        title=f"{context.name}",
    ))

    train_df = pd.read_csv(cfg.TRAIN_PATH)
    test_df = pd.read_csv(cfg.TEST_PATH)

    console.print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    console.print(f"Target distribution:\n{train_df[cfg.TARGET_COLUMN].value_counts(normalize=True)}")

    if params["preset"] == "extreme_quality" and len(train_df) > 30000:
        if not confirm_extreme(len(train_df), params.get("force_extreme", False)):
            console.print("[yellow]Stopping because extreme template was not confirmed.[/yellow]")
            sys.exit(1)

    predictor = TabularPredictor(
        label=cfg.TARGET_COLUMN,
        problem_type=getattr(cfg, "AUTOGLUON_PROBLEM_TYPE", None),
        eval_metric=getattr(cfg, "AUTOGLUON_EVAL_METRIC", None),
        path=str(cfg.PROJECT_ROOT / "AutogluonModels"),
        verbosity=2,
    )

    ignored_columns = list(getattr(cfg, "IGNORED_COLUMNS", []))
    id_column = getattr(cfg, "ID_COLUMN", None)
    if not id_column:
        if ignored_columns:
            id_column = ignored_columns[0]
        elif "id" in train_df.columns:
            id_column = "id"
    if not id_column or id_column not in test_df.columns:
        raise RuntimeError(
            f"ID column '{id_column or 'unknown'}' not found in test.csv; "
            "set --id-column during init-project or update config."
        )
    test_ids = test_df[id_column]

    drop_candidates = [col for col in ignored_columns if col != cfg.TARGET_COLUMN]
    train_no_id = train_df.drop(columns=[c for c in drop_candidates if c in train_df.columns])
    test_no_id = test_df.drop(columns=[c for c in drop_candidates if c in test_df.columns])

    predictor.fit(
        train_no_id,
        presets=params["preset"],
        time_limit=params["time_limit"],
        num_cpus="auto",
        num_gpus=1 if params["use_gpu"] else 0,
        hyperparameters=params.get("hyperparameters"),
    )

    leaderboard = predictor.leaderboard(train_no_id, silent=True)
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Time", justify="right")
    for _, row in leaderboard.head(10).iterrows():
        table.add_row(row["model"], f"{row['score_val']:.5f}", f"{row['fit_time']:.1f}s")
    console.print(table)

    best_score = leaderboard.iloc[0]["score_val"]
    console.print(f"[bold green]Best model:[/bold green] {leaderboard.iloc[0]['model']} @ {best_score:.5f}")

    submit_probas = getattr(cfg, "SUBMISSION_PROBAS", True)
    problem_type = getattr(cfg, "AUTOGLUON_PROBLEM_TYPE", None)
    if problem_type == "regression":
        predictions = predictor.predict(test_no_id)
    elif submit_probas:
        predictions = predictor.predict_proba(test_no_id, as_multiclass=False)
    else:
        predictions = predictor.predict(test_no_id)

    submission_artifact = context.submission_module.create_submission(
        predictions=predictions,
        test_ids=test_ids,
        model_name=f"autogluon-{params['preset']}",
        local_cv_score=best_score,
        notes=f"AutoGluon {params['preset']} ({params['time_limit']}s, gpu={params['use_gpu']})",
        config={
            "preset": params["preset"],
            "time_limit": params["time_limit"],
            "use_gpu": params["use_gpu"],
            "template": params.get("template"),
        },
        track=False,  # Disable legacy ExperimentLogger - use experiment_manager tracking instead
    )

    validate_fn = getattr(context.submission_module, "validate_submission", None)
    if callable(validate_fn):
        console.print("[bold blue]Validating submission format...[/bold blue]")
        validate_fn(submission_artifact.path)

    return {
        "submission": submission_artifact,
        "best_score": best_score,
        "train_rows": len(train_df),
    }


def run(args: argparse.Namespace, default_project: Optional[str] = None):
    context = load_project_context(args.project)
    manager = ExperimentManager.load_or_create(args.project, args.experiment_id)
    require_eda = args.require_eda
    if args.skip_eda_check:
        require_eda = False
    if require_eda:
        try:
            manager.require("eda")
        except ModuleStateError as exc:
            console.print(f"[yellow]{exc}[/yellow]")
            return

    params = resolve_template(args)
    params["template"] = args.template
    params["force_extreme"] = args.force_extreme

    try:
        manager.start_module(
            "model",
            {
                "template": params["template"],
                "compute": params,
            },
            allow_restart=True,
        )
    except ModuleStateError as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        return

    try:
        result = train_autogluon(context, params)
    except Exception as exc:
        manager.fail_module("model", str(exc))
        raise

    console.print(f"[bold green]✓ Training complete[/bold green]")
    console.print(f"Submission file: {result['submission'].path}")
    console.print(f"Local CV: {result['best_score']:.5f}")

    # Save code snapshot for reproducibility
    exp_dir = context.root / "experiments" / manager.experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = exp_dir / "autogluon_runner.py"
    import shutil
    shutil.copy2(Path(__file__), snapshot_path)
    console.print(f"[dim]Code snapshot: {snapshot_path.relative_to(context.root)}[/dim]")

    manager.complete_module(
        "model",
        {
            "template": params["template"],
            "local_cv": result["best_score"],
            "submission_file": str(result["submission"].path.relative_to(context.root)),
            "config": params,
            "code_snapshot": str(snapshot_path.relative_to(context.root)),
        },
    )

    if args.skip_submit or os.environ.get("KAGGLE_SKIP_SUBMIT"):
        console.print("[yellow]Skipping Kaggle submission workflow (--skip-submit or KAGGLE_SKIP_SUBMIT).[/yellow]")
        return

    runner = SubmissionRunner(
        artifact=result["submission"],
        kaggle_message=args.kaggle_message or f"{context.name} | {params['preset']} | local {result['best_score']:.5f}",
        wait_seconds=args.wait_seconds,
        cdp_url=args.cdp_url,
        auto_submit=args.auto_submit,
        prompt=not args.auto_submit,
        skip_browser=args.skip_score_fetch,
        skip_git=args.skip_git,
        experiment_id=manager.experiment_id,
    )
    runner.execute()


def cli_entry(default_project: Optional[str] = None):
    args = parse_args(default_project)
    run(args, default_project=default_project)


if __name__ == "__main__":
    cli_entry()
