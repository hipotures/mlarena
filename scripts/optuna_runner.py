"""
Optuna hyperparameter tuning runner for XGBoost/LightGBM/CatBoost.

Usage:
    # Run tuning with preset
    uv run python scripts/optuna_runner.py \
        --project playground-series-s5e11 \
        --experiment-id exp-20251117-020830 \
        --model xgboost \
        --preset thorough

    # Run with custom parameters
    uv run python scripts/optuna_runner.py \
        --project playground-series-s5e11 \
        --model lightgbm \
        --n-trials 50 \
        --timeout 3600 \
        --cv-folds 5

    # Resume existing study
    uv run python scripts/optuna_runner.py \
        --project playground-series-s5e11 \
        --experiment-id exp-20251117-020830 \
        --model xgboost \
        --resume
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table
from sklearn.model_selection import StratifiedKFold, KFold

from experiment_manager import ExperimentManager, ModuleStateError

TOOLS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_ROOT.parent

# Add src to path
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from kaggle_tools.optuna import StudyManager, CVObjective, xgboost_param_space, lightgbm_param_space, catboost_param_space
from kaggle_tools.config_models import OptunaConfig

console = Console()


# Model imports (lazy loaded to avoid unnecessary dependencies)
def get_model_class(model_name: str):
    """Lazy import model class."""
    if model_name == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier
    elif model_name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier
    elif model_name == "catboost":
        import catboost as cb
        return cb.CatBoostClassifier
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_param_space_fn(model_name: str):
    """Get parameter space function for model."""
    if model_name == "xgboost":
        return xgboost_param_space
    elif model_name == "lightgbm":
        return lightgbm_param_space
    elif model_name == "catboost":
        return catboost_param_space
    else:
        raise ValueError(f"Unknown model: {model_name}")


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
        "data_dir": project_root / "data",
        "experiments_dir": project_root / "experiments",
    }


def load_optuna_config(
    project_ctx: Dict[str, Any],
    model_name: str,
    preset: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> OptunaConfig:
    """
    Load Optuna configuration with priority:
    1. CLI overrides
    2. Preset YAML (configs/presets/{preset}.yaml)
    3. Project YAML (configs/project.yaml)
    4. Default config
    """
    config_dir = project_ctx["root"] / "configs"
    config_dict = {}

    # Load preset if specified
    if preset:
        preset_path = config_dir / "presets" / f"{preset}.yaml"
        if not preset_path.exists():
            # Try template presets
            template_preset_path = REPO_ROOT / "config" / "templates" / "kaggle_competition" / "configs" / "presets" / f"{preset}.yaml"
            if template_preset_path.exists():
                preset_path = template_preset_path
            else:
                raise FileNotFoundError(f"Preset not found: {preset}.yaml")

        import yaml
        with open(preset_path) as f:
            preset_dict = yaml.safe_load(f)
            config_dict = preset_dict.get("optuna", {})
        console.print(f"[green]✓[/green] Loaded preset: {preset_path}")

    # Try project.yaml if no preset
    elif (config_dir / "project.yaml").exists():
        import yaml
        with open(config_dir / "project.yaml") as f:
            project_dict = yaml.safe_load(f)
            config_dict = project_dict.get("optuna", {})
        console.print(f"[green]✓[/green] Loaded config from: project.yaml")

    # Apply overrides
    if overrides:
        config_dict.update(overrides)

    # Set defaults for storage and study_name if not present
    if "storage" not in config_dict:
        config_dict["storage"] = f"sqlite:///{project_ctx['experiments_dir']}/optuna.db"

    if "study_name" not in config_dict:
        config_dict["study_name"] = f"{model_name}_study"

    return OptunaConfig(**config_dict)


def load_data(
    project_ctx: Dict[str, Any],
    experiment_id: Optional[str] = None,
    use_transformed: bool = False
) -> tuple[pd.DataFrame, str]:
    """
    Load training data.

    Args:
        project_ctx: Project context dictionary
        experiment_id: Experiment ID (for transformed features)
        use_transformed: If True, load transformed features from feat module

    Returns:
        (train_df, target_column)
    """
    if use_transformed and experiment_id:
        # Load transformed features
        feature_dir = project_ctx["experiments_dir"] / experiment_id / "features"
        train_path = feature_dir / "train_transformed.parquet"

        if not train_path.exists():
            train_path = feature_dir / "train_transformed.csv"

        if not train_path.exists():
            raise FileNotFoundError(
                f"Transformed features not found. Run feat module first:\n"
                f"  uv run python scripts/experiment_manager.py feat --project {project_ctx['name']} --experiment-id {experiment_id}"
            )

        console.print(f"[green]✓[/green] Loading transformed features from: {train_path}")
        if train_path.suffix == ".parquet":
            train_df = pd.read_parquet(train_path)
        else:
            train_df = pd.read_csv(train_path)
    else:
        # Load raw data
        train_path = project_ctx["data_dir"] / "train.csv"
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_path}")

        console.print(f"[green]✓[/green] Loading raw training data")
        train_df = pd.read_csv(train_path)

    # Get target column from config
    target_column = project_ctx["config"].TARGET_COLUMN

    return train_df, target_column


def run_optuna_tuning(
    project_name: str,
    model_name: str,
    experiment_id: str | None = None,
    preset: str | None = None,
    n_trials: int | None = None,
    timeout: int | None = None,
    cv_folds: int | None = None,
    use_transformed: bool = False,
    resume: bool = False,
    force: bool = False,
    launch_dashboard: bool = False,
) -> None:
    """
    Run Optuna hyperparameter tuning.

    Args:
        project_name: Competition project name
        model_name: Model to tune (xgboost, lightgbm, catboost)
        experiment_id: Existing experiment ID (or None to create new)
        preset: Preset name (quick, thorough, extreme)
        n_trials: Override number of trials
        timeout: Override timeout (seconds)
        cv_folds: Override CV folds
        use_transformed: Use transformed features from feat module
        resume: Resume existing study
        force: Force re-run if already completed
        launch_dashboard: Launch optuna-dashboard web UI
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
            module="tune",
            notes=f"Optuna tuning: {model_name} ({preset or 'custom'})",
        )

    # Check module state
    if not resume:
        try:
            exp_manager.start_module("tune", allow_restart=force)
        except ModuleStateError as e:
            console.print(f"[red]✗[/red] {e}")
            sys.exit(1)

    # Load config with overrides
    overrides = {}
    if n_trials is not None:
        overrides["n_trials"] = n_trials
    if timeout is not None:
        overrides["timeout"] = timeout
    if cv_folds is not None:
        overrides["cv_folds"] = cv_folds

    optuna_config = load_optuna_config(project_ctx, model_name, preset, overrides)

    # Display configuration
    console.print(Panel(
        f"[bold]Optuna Hyperparameter Tuning[/bold]\n"
        f"Project: {project_name}\n"
        f"Experiment: {experiment_id}\n"
        f"Model: {model_name}\n"
        f"Preset: {preset or 'custom'}\n"
        f"Trials: {optuna_config.n_trials}\n"
        f"Timeout: {optuna_config.timeout or 'None'}s\n"
        f"CV Folds: {optuna_config.cv_folds}\n"
        f"Storage: {optuna_config.storage}",
        title="Configuration",
        border_style="blue"
    ))

    try:
        # Load data
        train_df, target_column = load_data(project_ctx, experiment_id, use_transformed)
        console.print(f"[green]✓[/green] Train shape: {train_df.shape}")

        # Prepare X, y
        X = train_df.drop(columns=[target_column])
        y = train_df[target_column]

        # Drop ID column if present
        id_column = getattr(project_ctx["config"], "ID_COLUMN", "id")
        if id_column in train_df.columns:
            train_df = train_df.drop(columns=[id_column])

        # For XGBoost: convert object columns to 'category' dtype (required for enable_categorical=True)
        if model_name == "xgboost":
            categorical_cols = list(train_df.select_dtypes(include=['object']).columns)
            if target_column in categorical_cols:
                categorical_cols.remove(target_column)
            if categorical_cols:
                console.print(f"[cyan]Converting {len(categorical_cols)} columns to category dtype for XGBoost: {categorical_cols}[/cyan]")
                for col in categorical_cols:
                    train_df[col] = train_df[col].astype('category')

        # For LightGBM: drop categorical columns (doesn't have good native categorical support)
        elif model_name == "lightgbm":
            categorical_cols = list(train_df.select_dtypes(include=['object', 'category']).columns)
            if target_column in categorical_cols:
                categorical_cols.remove(target_column)
            if categorical_cols:
                console.print(f"[yellow]Dropping {len(categorical_cols)} categorical columns for LightGBM (use freq encoding instead)[/yellow]")
                train_df = train_df.drop(columns=categorical_cols)

        # Get model class and param space
        model_class = get_model_class(model_name)
        param_space_fn = get_param_space_fn(model_name)

        # Determine metric
        problem_type = getattr(project_ctx["config"], "AUTOGLUON_PROBLEM_TYPE", "binary")
        if problem_type == "regression":
            from sklearn.metrics import mean_squared_error
            metric_fn = lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)  # Negative for maximization
            metric_name = "neg_mse"
        else:
            from sklearn.metrics import roc_auc_score
            metric_fn = roc_auc_score
            metric_name = "roc_auc"

        # Create study manager
        study_manager = StudyManager(optuna_config)
        study = study_manager.create_or_load_study()

        # Launch dashboard if requested
        if launch_dashboard:
            import subprocess
            console.print(f"[cyan]Launching optuna-dashboard...[/cyan]")
            console.print(f"[dim]Visit http://localhost:8080[/dim]")
            subprocess.Popen([
                "optuna-dashboard",
                optuna_config.storage,
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Check if already has trials
        if resume and len(study.trials) > 0:
            console.print(f"[yellow]⚠[/yellow] Resuming study with {len(study.trials)} existing trials")

        # Create CV objective
        console.print(f"\n[cyan]Creating CV objective...[/cyan]")
        objective = CVObjective(
            model_class=model_class,
            train_df=train_df,
            target_col=target_column,
            param_space_fn=lambda trial: param_space_fn(
                trial,
                optuna_config.param_space.get(model_name, {})
            ),
            metric_fn=metric_fn,
            cv_folds=optuna_config.cv_folds,
            early_stopping_rounds=optuna_config.early_stopping_rounds,
            random_seed=getattr(project_ctx["config"], "RANDOM_SEED", 42),
        )

        # Run optimization
        console.print(f"\n[bold green]Starting optimization...[/bold green]")
        study.optimize(
            objective,
            n_trials=optuna_config.n_trials,
            timeout=optuna_config.timeout,
            n_jobs=optuna_config.n_jobs,
            show_progress_bar=True,
        )

        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value

        console.print(f"\n[bold green]✓ Optimization completed![/bold green]")
        console.print(f"Best {metric_name}: {best_value:.5f}")

        # Display best parameters
        param_table = Table(title="Best Parameters")
        param_table.add_column("Parameter", style="cyan")
        param_table.add_column("Value", style="green")

        for param, value in best_params.items():
            if isinstance(value, float):
                param_table.add_row(param, f"{value:.6f}")
            else:
                param_table.add_row(param, str(value))

        console.print(param_table)

        # Save outputs
        output_dir = project_ctx["experiments_dir"] / experiment_id / "optuna"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save best parameters
        best_params_file = output_dir / f"best_params_{model_name}.json"
        with open(best_params_file, "w") as f:
            json.dump({
                "model": model_name,
                "best_params": best_params,
                "best_value": best_value,
                "metric": metric_name,
                "n_trials": len(study.trials),
            }, f, indent=2)

        console.print(f"[green]✓[/green] Saved best params: {best_params_file}")

        # Export trials
        trials_file = output_dir / f"trials_{model_name}.csv"
        study_manager.export_trials_dataframe(trials_file)
        console.print(f"[green]✓[/green] Exported trials: {trials_file}")

        # Complete module
        exp_manager.complete_module("tune", {
            "model": model_name,
            "preset": preset,
            "best_value": best_value,
            "n_trials": len(study.trials),
            "best_params": best_params,
        })

        console.print(f"\n[green]✓[/green] Hyperparameter tuning completed!")
        console.print(f"[dim]Experiment ID: {experiment_id}[/dim]")

        if launch_dashboard:
            console.print(f"\n[cyan]Dashboard running at http://localhost:8080[/cyan]")
            console.print(f"[dim]Press Ctrl+C to continue[/dim]")
            try:
                import time
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                console.print("\n[yellow]Dashboard closed[/yellow]")

    except Exception as e:
        exp_manager.fail_module("tune", str(e))
        console.print(f"\n[red]✗ Error:[/red] {e}")
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for XGBoost/LightGBM/CatBoost"
    )
    parser.add_argument(
        "--project",
        required=True,
        help="Competition project name (e.g., playground-series-s5e11)"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["xgboost", "lightgbm", "catboost"],
        help="Model to tune"
    )
    parser.add_argument(
        "--experiment-id",
        help="Existing experiment ID (auto-generated if omitted)"
    )
    parser.add_argument(
        "--preset",
        choices=["quick", "thorough", "extreme"],
        help="Preset configuration (quick: 20 trials/30min, thorough: 100 trials/2h, extreme: 500 trials/24h)"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        help="Override number of trials"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Override timeout in seconds"
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        help="Override number of CV folds"
    )
    parser.add_argument(
        "--use-transformed",
        action="store_true",
        help="Use transformed features from feat module"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume existing study"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run if module already completed"
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch optuna-dashboard web UI (http://localhost:8080)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        run_optuna_tuning(
            project_name=args.project,
            model_name=args.model,
            experiment_id=args.experiment_id,
            preset=args.preset,
            n_trials=args.n_trials,
            timeout=args.timeout,
            cv_folds=args.cv_folds,
            use_transformed=args.use_transformed,
            resume=args.resume,
            force=args.force,
            launch_dashboard=args.dashboard,
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
