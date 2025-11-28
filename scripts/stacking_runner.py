"""
Model stacking and ensembling runner.

Supports two ensemble strategies:
1. Blending - Weighted/Rank/Power averaging of predictions
2. Meta-learning - Stacking with out-of-fold predictions

Usage:
    # Blending (weighted average)
    uv run python scripts/stacking_runner.py \
        --project playground-series-s5e11 \
        --experiment-id exp-20251117-020830 \
        --strategy blend \
        --blend-method weighted \
        --models xgb_baseline.csv lgb_baseline.csv catboost_baseline.csv

    # Meta-learning (stacking)
    uv run python scripts/stacking_runner.py \
        --project playground-series-s5e11 \
        --experiment-id exp-20251117-020830 \
        --strategy meta \
        --meta-model logistic \
        --models xgb_baseline.csv lgb_baseline.csv

    # Auto-optimize blend weights
    uv run python scripts/stacking_runner.py \
        --project playground-series-s5e11 \
        --strategy blend \
        --optimize \
        --models model1.csv model2.csv model3.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from experiment_manager import ExperimentManager, ModuleStateError

TOOLS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_ROOT.parent

# Add src to path
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from kaggle_tools.stacking import WeightedBlender, RankBlender, PowerBlender, MetaLearner

console = Console()


def load_project_context(project_name: str) -> Dict[str, Any]:
    """Load project configuration and paths."""
    project_root = (REPO_ROOT / "projects" / "kaggle" / project_name).resolve()
    if not project_root.exists():
        raise FileNotFoundError(f"Project directory '{project_name}' not found at {project_root}")

    code_dir = project_root / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    import importlib
    config_module = importlib.import_module("utils.config")

    return {
        "name": project_name,
        "root": project_root,
        "config": config_module,
        "submissions_dir": project_root / "submissions",
        "experiments_dir": project_root / "experiments",
    }


def load_model_predictions(
    project_ctx: Dict[str, Any],
    model_files: List[str]
) -> tuple[List[pd.Series], List[str]]:
    """
    Load predictions from model submission files.

    Args:
        project_ctx: Project context
        model_files: List of submission CSV filenames

    Returns:
        (predictions_list, model_names)
    """
    predictions = []
    model_names = []

    submissions_dir = project_ctx["submissions_dir"]
    target_column = project_ctx["config"].TARGET_COLUMN

    for model_file in model_files:
        # Try submissions/ directory first
        file_path = submissions_dir / model_file
        if not file_path.exists():
            # Try as absolute path
            file_path = Path(model_file)

        if not file_path.exists():
            raise FileNotFoundError(f"Model predictions not found: {model_file}")

        # Load predictions
        df = pd.read_csv(file_path)
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in {model_file}")

        predictions.append(df[target_column])
        model_names.append(file_path.stem)

        console.print(f"[green]✓[/green] Loaded {model_file}: {len(df)} predictions")

    return predictions, model_names


def optimize_blend_weights(
    predictions: List[pd.Series],
    train_labels: pd.Series,
    method: str = "nelder-mead"
) -> List[float]:
    """
    Optimize blend weights using validation data.

    Args:
        predictions: List of prediction series
        train_labels: True labels for optimization
        method: Optimization method

    Returns:
        Optimized weights
    """
    from scipy.optimize import minimize

    def objective(weights):
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)

        # Compute weighted average
        ensemble_pred = sum(w * pred for w, pred in zip(weights, predictions))

        # Compute metric (minimize negative ROC-AUC)
        from sklearn.metrics import roc_auc_score
        try:
            score = roc_auc_score(train_labels, ensemble_pred)
            return -score  # Minimize negative score
        except:
            return 1.0  # Penalty for invalid predictions

    # Initial weights (equal)
    n_models = len(predictions)
    initial_weights = [1.0 / n_models] * n_models

    # Optimize
    console.print(f"[cyan]Optimizing blend weights using {method}...[/cyan]")
    result = minimize(
        objective,
        initial_weights,
        method=method,
        bounds=[(0, 1)] * n_models,
        options={"maxiter": 1000}
    )

    # Normalize weights
    optimized_weights = result.x / np.sum(result.x)

    console.print(f"[green]✓[/green] Optimization completed (score: {-result.fun:.5f})")

    return optimized_weights.tolist()


def run_stacking(
    project_name: str,
    model_files: List[str],
    experiment_id: str | None = None,
    strategy: str = "blend",
    blend_method: str = "weighted",
    blend_weights: List[float] | None = None,
    blend_power: float = 2.0,
    meta_model: str = "logistic",
    optimize: bool = False,
    output_name: str | None = None,
    force: bool = False,
) -> None:
    """
    Run model stacking/ensembling.

    Args:
        project_name: Competition project name
        model_files: List of submission CSV files to ensemble
        experiment_id: Existing experiment ID (or None to create new)
        strategy: Ensemble strategy (blend or meta)
        blend_method: Blending method (weighted, rank, power)
        blend_weights: Manual blend weights (or None for equal/optimized)
        blend_power: Power parameter for power blending
        meta_model: Meta-learner model (logistic, ridge, xgboost)
        optimize: Optimize blend weights using validation data
        output_name: Output submission filename (auto-generated if None)
        force: Force re-run if already completed
    """
    # Load project
    project_ctx = load_project_context(project_name)

    # Initialize experiment manager
    exp_manager = ExperimentManager.load_or_create(project_name, experiment_id)
    experiment_id = exp_manager.experiment_id

    if not experiment_id:
        # Create new experiment
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        experiment_id = f"exp-{timestamp}"
        exp_manager.create_experiment(
            experiment_id=experiment_id,
            module="stack",
            notes=f"Ensemble: {strategy} ({blend_method if strategy == 'blend' else meta_model})",
        )

    # Check module state
    try:
        exp_manager.start_module("stack", allow_restart=force)
    except ModuleStateError as e:
        console.print(f"[red]✗[/red] {e}")
        sys.exit(1)

    # Display configuration
    console.print(Panel(
        f"[bold]Model Stacking/Ensembling[/bold]\n"
        f"Project: {project_name}\n"
        f"Experiment: {experiment_id}\n"
        f"Strategy: {strategy}\n"
        f"Models: {len(model_files)}\n"
        f"Method: {blend_method if strategy == 'blend' else meta_model}\n"
        f"Optimize: {optimize}",
        title="Configuration",
        border_style="blue"
    ))

    try:
        # Load predictions
        predictions, model_names = load_model_predictions(project_ctx, model_files)

        # Validate all have same length
        if len(set(len(p) for p in predictions)) > 1:
            raise ValueError("All model predictions must have the same length")

        n_predictions = len(predictions[0])
        console.print(f"[green]✓[/green] Loaded {len(predictions)} models with {n_predictions} predictions each")

        # Run ensemble strategy
        if strategy == "blend":
            # Blending strategy
            if optimize:
                # Optimize weights (requires validation data)
                # For now, use equal weights - TODO: implement validation split
                console.print("[yellow]⚠[/yellow] Weight optimization requires validation data (not implemented yet)")
                console.print("[yellow]⚠[/yellow] Using equal weights")
                blend_weights = [1.0 / len(predictions)] * len(predictions)

            elif blend_weights is None:
                # Equal weights
                blend_weights = [1.0 / len(predictions)] * len(predictions)

            # Normalize weights
            blend_weights = np.array(blend_weights) / np.sum(blend_weights)

            # Apply blending method
            if blend_method == "weighted":
                blender = WeightedBlender()
                ensemble_pred = blender.blend(predictions, blend_weights.tolist())
                console.print(f"[green]✓[/green] Weighted blending with weights: {blend_weights}")

            elif blend_method == "rank":
                blender = RankBlender()
                ensemble_pred = blender.blend(predictions)
                console.print(f"[green]✓[/green] Rank averaging (robust to outliers)")

            elif blend_method == "power":
                blender = PowerBlender()
                ensemble_pred = blender.blend(predictions, power=blend_power)
                console.print(f"[green]✓[/green] Power averaging (power={blend_power})")

            else:
                raise ValueError(f"Unknown blend method: {blend_method}")

        elif strategy == "meta":
            # Meta-learning strategy (requires OOF predictions)
            console.print("[yellow]⚠[/yellow] Meta-learning requires out-of-fold predictions")
            console.print("[yellow]⚠[/yellow] This feature requires model training integration (not fully implemented)")

            # For now, fall back to simple averaging
            console.print("[yellow]⚠[/yellow] Falling back to simple averaging")
            ensemble_pred = sum(predictions) / len(predictions)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Create submission DataFrame
        target_column = project_ctx["config"].TARGET_COLUMN
        id_column = project_ctx["config"].ID_COLUMN

        # Load ID column from first model file
        first_model_path = project_ctx["submissions_dir"] / model_files[0]
        if not first_model_path.exists():
            first_model_path = Path(model_files[0])

        id_df = pd.read_csv(first_model_path)[[id_column]]

        submission = id_df.copy()
        submission[target_column] = ensemble_pred

        # Generate output filename
        if output_name is None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            output_name = f"ensemble-{strategy}-{timestamp}.csv"

        # Save submission
        output_path = project_ctx["submissions_dir"] / output_name
        submission.to_csv(output_path, index=False)

        console.print(f"[green]✓[/green] Saved ensemble submission: {output_path}")

        # Display ensemble statistics
        display_ensemble_stats(predictions, ensemble_pred, model_names)

        # Complete module
        metadata = {
            "strategy": strategy,
            "n_models": len(predictions),
            "model_files": model_files,
            "output_file": output_name,
        }

        if strategy == "blend":
            metadata["blend_method"] = blend_method
            if blend_weights is not None:
                metadata["blend_weights"] = blend_weights.tolist()
            if blend_method == "power":
                metadata["blend_power"] = blend_power

        exp_manager.complete_module("stack", metadata)

        console.print(f"\n[green]✓[/green] Ensemble completed!")
        console.print(f"[dim]Experiment ID: {experiment_id}[/dim]")
        console.print(f"[dim]Output: {output_path}[/dim]")

    except Exception as e:
        exp_manager.fail_module("stack", str(e))
        console.print(f"\n[red]✗ Error:[/red] {e}")
        raise


def display_ensemble_stats(
    predictions: List[pd.Series],
    ensemble_pred: pd.Series,
    model_names: List[str]
) -> None:
    """Display ensemble statistics."""
    table = Table(title="Ensemble Statistics")
    table.add_column("Model", style="cyan")
    table.add_column("Mean", style="yellow")
    table.add_column("Std", style="yellow")
    table.add_column("Correlation", style="green")

    # Individual models
    for i, (pred, name) in enumerate(zip(predictions, model_names)):
        corr = pred.corr(ensemble_pred)
        table.add_row(
            name,
            f"{pred.mean():.5f}",
            f"{pred.std():.5f}",
            f"{corr:.5f}"
        )

    # Ensemble
    table.add_row(
        "[bold]Ensemble[/bold]",
        f"[bold]{ensemble_pred.mean():.5f}[/bold]",
        f"[bold]{ensemble_pred.std():.5f}[/bold]",
        "[bold]1.00000[/bold]"
    )

    console.print(table)

    # Diversity metrics
    console.print(f"\n[cyan]Diversity Metrics:[/cyan]")
    n_models = len(predictions)
    correlations = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            corr = predictions[i].corr(predictions[j])
            correlations.append(corr)

    avg_corr = np.mean(correlations)
    console.print(f"  Average pairwise correlation: {avg_corr:.5f}")
    console.print(f"  Min correlation: {min(correlations):.5f}")
    console.print(f"  Max correlation: {max(correlations):.5f}")

    if avg_corr > 0.95:
        console.print(f"  [yellow]⚠ High correlation - models may be too similar[/yellow]")
    elif avg_corr < 0.7:
        console.print(f"  [green]✓ Good diversity - models are complementary[/green]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Model stacking and ensembling"
    )
    parser.add_argument(
        "--project",
        required=True,
        help="Competition project name (e.g., playground-series-s5e11)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Submission CSV files to ensemble"
    )
    parser.add_argument(
        "--experiment-id",
        help="Existing experiment ID (auto-generated if omitted)"
    )
    parser.add_argument(
        "--strategy",
        choices=["blend", "meta"],
        default="blend",
        help="Ensemble strategy (blend: averaging, meta: stacking)"
    )
    parser.add_argument(
        "--blend-method",
        choices=["weighted", "rank", "power"],
        default="weighted",
        help="Blending method (for blend strategy)"
    )
    parser.add_argument(
        "--blend-weights",
        nargs="+",
        type=float,
        help="Manual blend weights (e.g., 0.5 0.3 0.2)"
    )
    parser.add_argument(
        "--blend-power",
        type=float,
        default=2.0,
        help="Power parameter for power blending"
    )
    parser.add_argument(
        "--meta-model",
        choices=["logistic", "ridge", "xgboost"],
        default="logistic",
        help="Meta-learner model (for meta strategy)"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Optimize blend weights using validation data"
    )
    parser.add_argument(
        "--output-name",
        help="Output submission filename (auto-generated if omitted)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run if module already completed"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate weights if provided
    if args.blend_weights and len(args.blend_weights) != len(args.models):
        console.print(f"[red]✗ Error:[/red] Number of weights ({len(args.blend_weights)}) must match number of models ({len(args.models)})")
        sys.exit(1)

    try:
        run_stacking(
            project_name=args.project,
            model_files=args.models,
            experiment_id=args.experiment_id,
            strategy=args.strategy,
            blend_method=args.blend_method,
            blend_weights=args.blend_weights,
            blend_power=args.blend_power,
            meta_model=args.meta_model,
            optimize=args.optimize,
            output_name=args.output_name,
            force=args.force,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Fatal error:[/red] {e}")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
