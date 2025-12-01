"""
Feature engineering pipeline runner with data leakage protection.

Usage:
    # Run feature engineering for specific experiment
    uv run python scripts/feature_runner.py \
        --project playground-series-s5e11 \
        --experiment-id exp-20251117-020830 \
        --feature-set baseline

    # Preview transformations without saving
    uv run python scripts/feature_runner.py \
        --project playground-series-s5e11 \
        --preview
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from experiment_manager import ExperimentManager, ModuleStateError

TOOLS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_ROOT.parent

# Add src to path for imports
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from kaggle_tools.preprocessing import FeaturePipeline
from kaggle_tools.config_models import PreprocessingConfig

console = Console()


def load_project_context(project_name: str) -> Dict[str, Any]:
    """Load project configuration and paths."""
    project_root = (REPO_ROOT / "projects" / "kaggle" / project_name).resolve()
    if not project_root.exists():
        raise FileNotFoundError(f"Project directory '{project_name}' not found at {project_root}")

    # Load project config
    code_dir = project_root / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    import importlib
    config_module = importlib.import_module("utils.config")

    return {
        "name": project_name,
        "root": project_root,
        "config": config_module,
        "data_dir": project_root / "data",
        "experiments_dir": project_root / "experiments",
    }


def load_preprocessing_config(
    project_ctx: Dict[str, Any],
    feature_set: str = "baseline"
) -> PreprocessingConfig:
    """
    Load preprocessing configuration.

    Priority:
    1. configs/preprocessing/{feature_set}.yaml
    2. configs/project.yaml (preprocessing section)
    3. Default config
    """
    config_dir = project_ctx["root"] / "configs"

    # Try feature-set specific config
    feature_config_path = config_dir / "preprocessing" / f"{feature_set}.yaml"
    if feature_config_path.exists():
        import yaml
        with open(feature_config_path) as f:
            config_dict = yaml.safe_load(f)
        console.print(f"[green]✓[/green] Loaded preprocessing config: {feature_config_path}")
        return PreprocessingConfig(**config_dict.get("preprocessing", {}))

    # Try project.yaml
    project_config_path = config_dir / "project.yaml"
    if project_config_path.exists():
        import yaml
        with open(project_config_path) as f:
            config_dict = yaml.safe_load(f)
        if "preprocessing" in config_dict:
            console.print(f"[green]✓[/green] Loaded preprocessing from: {project_config_path}")
            return PreprocessingConfig(**config_dict["preprocessing"])

    # Default config
    console.print("[yellow]⚠[/yellow] Using default preprocessing config")
    return PreprocessingConfig()


def run_feature_engineering(
    project_name: str,
    experiment_id: str | None = None,
    feature_set: str = "baseline",
    preview: bool = False,
    force: bool = False,
) -> None:
    """
    Run feature engineering pipeline.

    Args:
        project_name: Competition project name
        experiment_id: Existing experiment ID (or None to create new)
        feature_set: Name of feature set to apply (e.g., 'baseline', 'advanced')
        preview: If True, show transformations without saving
        force: If True, overwrite existing outputs
    """
    # Load project
    project_ctx = load_project_context(project_name)
    config_module = project_ctx["config"]

    # Initialize experiment manager
    exp_manager = ExperimentManager.load_or_create(project_name, experiment_id)
    experiment_id = exp_manager.experiment_id

    # Check module state
    try:
        exp_manager.start_module("feat", extra={"feature_set": feature_set}, allow_restart=force)
    except ModuleStateError as e:
        console.print(f"[red]✗[/red] {e}")
        sys.exit(1)

    # Load preprocessing config
    preprocessing_config = load_preprocessing_config(project_ctx, feature_set)

    # Display config
    console.print(Panel(
        f"[bold]Feature Engineering[/bold]\n"
        f"Project: {project_name}\n"
        f"Experiment: {experiment_id}\n"
        f"Feature Set: {feature_set}\n"
        f"Preview Mode: {preview}",
        title="Configuration",
        border_style="blue"
    ))

    try:
        # Load data
        train_path = project_ctx["data_dir"] / "train.csv"
        test_path = project_ctx["data_dir"] / "test.csv"

        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_path}")

        console.print(f"[cyan]Loading data...[/cyan]")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path) if test_path.exists() else None

        console.print(f"[green]✓[/green] Train shape: {train_df.shape}")
        if test_df is not None:
            console.print(f"[green]✓[/green] Test shape: {test_df.shape}")

        # Initialize pipeline
        pipeline = FeaturePipeline(preprocessing_config)

        # Fit feat_stage (global transformers)
        console.print(f"\n[cyan]Fitting feat_stage transformers...[/cyan]")
        pipeline.fit_feat_stage(train_df)

        # Transform train data
        console.print(f"[cyan]Transforming training data...[/cyan]")
        train_transformed = pipeline.transform_feat_stage(train_df)

        # Transform test data if available
        test_transformed = None
        if test_df is not None:
            console.print(f"[cyan]Transforming test data...[/cyan]")
            test_transformed = pipeline.transform_feat_stage(test_df)

        # Display feature statistics
        display_feature_stats(train_df, train_transformed)

        if preview:
            console.print("\n[yellow]Preview mode - not saving outputs[/yellow]")
            exp_manager.complete_module("feat", {
                "preview": True,
                "feature_set": feature_set,
            })
            return

        # Save transformed data
        output_dir = project_ctx["experiments_dir"] / experiment_id / "features"
        output_dir.mkdir(parents=True, exist_ok=True)

        console.print(f"\n[cyan]Saving transformed data...[/cyan]")

        # Save pipeline
        pipeline.save(output_dir / "pipeline")

        # Save transformed data + metadata (parquet + feature_metadata.json)
        pipeline.save_transformed_data(
            output_dir / "pipeline",
            train_transformed,
            test_transformed if test_transformed is not None else train_transformed,
        )

        # Save transformed dataframes
        storage_format = preprocessing_config.storage_format
        if storage_format == "parquet":
            train_transformed.to_parquet(
                output_dir / "train_transformed.parquet",
                compression=preprocessing_config.compression,
                index=False
            )
            if test_transformed is not None:
                test_transformed.to_parquet(
                    output_dir / "test_transformed.parquet",
                    compression=preprocessing_config.compression,
                    index=False
                )
        else:  # csv
            train_transformed.to_csv(output_dir / "train_transformed.csv", index=False)
            if test_transformed is not None:
                test_transformed.to_csv(output_dir / "test_transformed.csv", index=False)

        console.print(f"[green]✓[/green] Saved to: {output_dir}")

        # Complete module
        exp_manager.complete_module("feat", {
            "feature_set": feature_set,
            "train_shape": list(train_transformed.shape),
            "test_shape": list(test_transformed.shape) if test_transformed is not None else None,
            "storage_format": storage_format,
            "n_features_original": len(train_df.columns),
            "n_features_transformed": len(train_transformed.columns),
        })

        console.print(f"\n[green]✓[/green] Feature engineering completed!")
        console.print(f"[dim]Experiment ID: {experiment_id}[/dim]")

    except Exception as e:
        exp_manager.fail_module("feat", str(e))
        console.print(f"\n[red]✗ Error:[/red] {e}")
        raise


def display_feature_stats(original: pd.DataFrame, transformed: pd.DataFrame) -> None:
    """Display feature statistics comparison."""
    table = Table(title="Feature Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Original", style="yellow")
    table.add_column("Transformed", style="green")

    table.add_row("Number of features", str(len(original.columns)), str(len(transformed.columns)))
    table.add_row(
        "Feature types",
        f"{original.select_dtypes(include='number').shape[1]} num, "
        f"{original.select_dtypes(include='object').shape[1]} cat",
        f"{transformed.select_dtypes(include='number').shape[1]} num, "
        f"{transformed.select_dtypes(include='object').shape[1]} cat"
    )
    table.add_row("Memory (MB)", f"{original.memory_usage(deep=True).sum() / 1024**2:.2f}",
                  f"{transformed.memory_usage(deep=True).sum() / 1024**2:.2f}")

    console.print(table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature engineering pipeline with leakage protection"
    )
    parser.add_argument(
        "--project",
        required=True,
        help="Competition project name (e.g., playground-series-s5e11)"
    )
    parser.add_argument(
        "--experiment-id",
        help="Existing experiment ID (auto-generated if omitted)"
    )
    parser.add_argument(
        "--feature-set",
        default="baseline",
        help="Feature set name (looks for configs/preprocessing/{name}.yaml)"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview transformations without saving"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing feature engineering outputs"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        run_feature_engineering(
            project_name=args.project,
            experiment_id=args.experiment_id,
            feature_set=args.feature_set,
            preview=args.preview,
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
