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
import optuna
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from experiment_manager import ExperimentManager, ModuleStateError

TOOLS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_ROOT.parent

# Add src to path
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from kaggle_tools.stacking import WeightedBlender, RankBlender, PowerBlender, MetaLearner
from kaggle_tools.optuna import CVObjective, xgboost_param_space, lightgbm_param_space, catboost_param_space

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


def load_oof_predictions(
    project_ctx: Dict[str, Any],
    oof_files: List[str],
) -> pd.DataFrame:
    """
    Load out-of-fold predictions for meta-learning.

    Expects each OOF file to have ID column and target prediction column.
    """
    id_column = project_ctx["config"].ID_COLUMN
    target_column = project_ctx["config"].TARGET_COLUMN

    if len(oof_files) == 0:
        raise ValueError("No OOF files provided.")

    oof_frames = []
    for file_path in oof_files:
        path = Path(file_path)
        if not path.exists():
            path = project_ctx["submissions_dir"] / file_path
        if not path.exists():
            raise FileNotFoundError(f"OOF file not found: {file_path}")
        df = pd.read_csv(path)
        missing = [col for col in [id_column, target_column] if col not in df.columns]
        if missing:
            raise ValueError(f"OOF file {file_path} missing columns: {missing}")
        oof_frames.append(df[[id_column, target_column]].rename(columns={target_column: path.stem}))
    # Merge on id
    result = oof_frames[0]
    for frame in oof_frames[1:]:
        result = result.merge(frame, on=id_column, how="inner")
    return result


def load_tracker_scores(project_ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    tracker_path = project_ctx["submissions_dir"] / "submissions.json"
    if not tracker_path.exists():
        raise FileNotFoundError(f"Tracker not found at {tracker_path}")
    return json.loads(tracker_path.read_text())


def select_models_from_tracker(project_ctx: Dict[str, Any], source: str, top_n: int) -> List[str]:
    tracker = load_tracker_scores(project_ctx)
    key = "public_score" if source == "public" else "local_cv_score"
    scored = [s for s in tracker if s.get(key) is not None and s.get("filename")]
    if not scored:
        raise RuntimeError(f"No submissions in tracker with '{key}' available.")
    scored = sorted(scored, key=lambda x: x[key], reverse=True)
    if top_n > len(scored):
        top_n = len(scored)
    selected = scored[:top_n]
    files = [s["filename"] for s in selected]
    console.print(f"[green]✓[/green] Selected top {len(files)} models by {source}: {files}")
    return files


def build_weights_from_tracker(
    project_ctx: Dict[str, Any],
    model_files: List[str],
    source: str,
) -> List[float]:
    """
    Build weights from submissions tracker using public or local scores.
    """
    tracker = load_tracker_scores(project_ctx)
    key = "public_score" if source == "public" else "local_cv_score"
    weights = []
    for model_file in model_files:
        name = Path(model_file).name
        entry = next((s for s in tracker if s.get("filename") == name), None)
        score = entry.get(key) if entry else None
        weights.append(score if score is not None else 0.0)
    arr = np.array(weights, dtype=float)
    if np.all(arr == 0):
        console.print(f"[yellow]Tracker has no {source} scores for provided models; falling back to equal weights.[/yellow]")
        return [1.0 / len(model_files)] * len(model_files)
    arr = arr / arr.sum()
    console.print(f"[green]✓[/green] Weights from {source} scores: {arr}")
    return arr.tolist()


def _param_space_fn(name: str):
    if name == "xgboost":
        return xgboost_param_space
    if name == "lightgbm":
        return lightgbm_param_space
    if name == "catboost":
        return catboost_param_space
    raise ValueError(f"Unknown base model: {name}")


def _model_class(name: str):
    if name == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier
    if name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier
    if name == "catboost":
        import catboost as cb
        return cb.CatBoostClassifier
    raise ValueError(f"Unknown base model: {name}")


def _load_optuna_param_space(project_root: Path) -> Dict[str, Any]:
    cfg_path = project_root / "configs" / "project.yaml"
    if not cfg_path.exists():
        return {}
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    return cfg.get("optuna", {}).get("param_space", {})


def tune_best_params(model_name: str, X: pd.DataFrame, y: pd.Series, param_space_cfg: Dict[str, Any], n_trials: int, timeout: int, cv_folds: int) -> Dict[str, Any]:
    param_space_fn = _param_space_fn(model_name)
    model_cls = _model_class(model_name)
    model_kwargs: Dict[str, Any] = {}
    if model_name == "catboost":
        cat_features = [i for i, col in enumerate(X.columns) if str(X[col].dtype) == "category"]
        model_kwargs = {"cat_features": cat_features, "verbose": False}

    def objective(trial: optuna.Trial) -> float:
        params = param_space_fn(trial, param_space_cfg)
        # Ensure categorical flags for XGBoost
        if model_name == "xgboost":
            params.setdefault("enable_categorical", True)
            params.setdefault("tree_method", "hist")
        if model_name == "catboost":
            params.pop("verbose", None)
        obj = CVObjective(
            model_class=model_cls,
            X=X,
            y=y,
            param_space_fn=lambda t: params,
            metric_fn=roc_auc_score,
            cv_folds=cv_folds,
            early_stopping_rounds=50,
            random_seed=42,
            model_kwargs=model_kwargs,
        )
        return obj(trial)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    return study.best_params


def generate_oof_and_preds(model_name: str, best_params: Dict[str, Any], X: pd.DataFrame, y: pd.Series, test_df: pd.DataFrame, cv_folds: int, id_column: str, target_column: str) -> tuple[pd.Series, pd.Series]:
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    oof = pd.Series(0.0, index=X.index)
    test_preds = np.zeros(len(test_df))

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model_cls = _model_class(model_name)
        params = dict(best_params)
        if model_name == "xgboost":
            params.setdefault("enable_categorical", True)
            params.setdefault("tree_method", "hist")
            model = model_cls(**params, n_estimators=300, random_state=42)
        elif model_name == "lightgbm":
            params.setdefault("objective", "binary")
            params.setdefault("metric", "auc")
            model = model_cls(**params, n_estimators=300, random_state=42)
        else:  # catboost
            params.pop("verbose", None)
            cat_features = [i for i, col in enumerate(X_train.columns) if str(X_train[col].dtype) == "category"]
            model = model_cls(
                **params,
                iterations=300,
                random_state=42,
                cat_features=cat_features,
                verbose=False,
            )

        model.fit(X_train, y_train)
        if hasattr(model, "predict_proba"):
            val_pred = model.predict_proba(X_val)[:, 1]
            test_pred = model.predict_proba(test_df)[:, 1]
        else:
            val_pred = model.predict(X_val)
            test_pred = model.predict(test_df)

        oof.iloc[val_idx] = val_pred
        test_preds += test_pred / cv_folds

    test_series = pd.Series(test_preds, index=test_df.index)
    return oof, test_series


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
    model_files: List[str] | None,
    experiment_id: str | None = None,
    strategy: str = "blend",
    blend_method: str = "weighted",
    blend_weights: List[float] | None = None,
    blend_power: float = 2.0,
    auto_weights: str | None = None,
    top_n: int | None = None,
    meta_model: str = "logistic",
    oof_files: List[str] | None = None,
    meta_base_models: List[str] | None = None,
    meta_n_trials: int = 10,
    meta_cv_folds: int = 3,
    meta_timeout: int = 600,
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
        auto_weights: Auto weights source (public/local)
        top_n: When using auto-weights, optionally pick top-N models from tracker by that metric
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
        f"Models: {len(model_files) if model_files else 0}\n"
        f"Method: {blend_method if strategy == 'blend' else meta_model}\n"
        f"Optimize: {optimize}",
        title="Configuration",
        border_style="blue"
    ))

    try:
        predictions: List[pd.Series] = []
        model_names: List[str] = []

        # For blend or meta with provided models/OOF, load predictions
        if not model_files and auto_weights and top_n:
            model_files = select_models_from_tracker(project_ctx, auto_weights, top_n)

        if model_files:
            predictions, model_names = load_model_predictions(project_ctx, model_files)
            if len(set(len(p) for p in predictions)) > 1:
                raise ValueError("All model predictions must have the same length")
            n_predictions = len(predictions[0])
            console.print(f"[green]✓[/green] Loaded {len(predictions)} models with {n_predictions} predictions each")

        # Run ensemble strategy
        if strategy == "blend":
            # Blending strategy
            if auto_weights:
                if not model_files:
                    raise ValueError("--auto-weights requires --models files")
                if blend_method not in {"weighted", "power"}:
                    console.print("[yellow]Auto-weights require weighted/power blend; switching blend_method to weighted.[/yellow]")
                    blend_method = "weighted"
                blend_weights = build_weights_from_tracker(project_ctx, model_files, auto_weights)

            if optimize:
                # Optimize weights (requires validation data)
                # For now, use equal weights - TODO: implement validation split
                console.print("[yellow]⚠[/yellow] Weight optimization requires validation data (not implemented yet)")
                console.print("[yellow]⚠[/yellow] Using equal weights")
                blend_weights = [1.0 / len(predictions)] * len(predictions)

            elif blend_weights is None:
                # Equal weights
                blend_weights = [1.0 / len(predictions)] * len(predictions)

            # Apply blending method
            if blend_method == "weighted":
                blend_weights = np.array(blend_weights) / np.sum(blend_weights)
                blender = WeightedBlender()
                ensemble_pred = blender.blend(predictions, blend_weights.tolist())
                console.print(f"[green]✓[/green] Weighted blending with weights: {blend_weights}")

            elif blend_method == "rank":
                blender = RankBlender()
                ensemble_pred = blender.blend(predictions)
                console.print(f"[green]✓[/green] Rank averaging (robust to outliers)")

            elif blend_method == "power":
                blend_weights = np.array(blend_weights) / np.sum(blend_weights)
                blender = PowerBlender()
                ensemble_pred = blender.blend(predictions, power=blend_power, weights=blend_weights.tolist())
                console.print(f"[green]✓[/green] Power averaging (power={blend_power})")

            else:
                raise ValueError(f"Unknown blend method: {blend_method}")

        elif strategy == "meta":
            target_column = project_ctx["config"].TARGET_COLUMN
            id_column = project_ctx["config"].ID_COLUMN
            train_df = pd.read_csv(project_ctx["root"] / "data" / "train.csv")
            test_df = pd.read_csv(project_ctx["root"] / "data" / "test.csv")

            if oof_files:
                if len(oof_files) != len(predictions):
                    raise ValueError("--oof-files must match number of models when provided.")
                # Use provided OOF + provided test predictions
                oof_df = load_oof_predictions(project_ctx, oof_files)
                merged = oof_df.merge(train_df[[id_column, target_column]], on=id_column, how="inner")
                X_meta = merged[oof_df.columns.drop(id_column)]
                y_meta = merged[target_column]
                test_meta = pd.concat(predictions, axis=1)
                test_meta.columns = model_names
            else:
                # Auto-generate OOF + test preds for base models
                optuna_param_space = _load_optuna_param_space(project_ctx["root"])
                auto_models = meta_base_models or ["xgboost", "lightgbm", "catboost"]
                cv_folds = meta_cv_folds
                n_trials = meta_n_trials
                timeout = meta_timeout

                # Prepare features (drop id/target, cast categoricals)
                feature_drop = [target_column]
                if id_column in train_df.columns:
                    feature_drop.append(id_column)
                X_full = train_df.drop(columns=feature_drop)
                test_features = test_df.drop(columns=[id_column], errors="ignore")
                cat_cols = X_full.select_dtypes(include=["object"]).columns
                if len(cat_cols) > 0:
                    X_full[cat_cols] = X_full[cat_cols].astype("category")
                    test_features[cat_cols] = test_features[cat_cols].astype("category")
                y_full = train_df[target_column]

                auto_oof = []
                auto_test_preds = []
                auto_names = []

                for base_model in auto_models:
                    if base_model not in optuna_param_space:
                        raise ValueError(f"Param space for {base_model} not found in project optuna config.")
                    console.print(f"[cyan]Auto-generating OOF for {base_model} (trials={n_trials}, cv={cv_folds})[/cyan]")
                    best_params = tune_best_params(
                        base_model,
                        X_full,
                        y_full,
                        optuna_param_space[base_model],
                        n_trials=n_trials,
                        timeout=timeout,
                        cv_folds=cv_folds,
                    )
                    oof_series, test_series = generate_oof_and_preds(
                        base_model,
                        best_params,
                        X_full,
                        y_full,
                        test_features,
                        cv_folds,
                        id_column,
                        target_column,
                    )
                    auto_oof.append(oof_series)
                    auto_test_preds.append(test_series)
                    auto_names.append(base_model)

                # Build meta matrices
                X_meta = pd.concat(auto_oof, axis=1)
                X_meta.columns = auto_names
                X_meta[target_column] = y_full
                y_meta = X_meta[target_column]
                X_meta = X_meta.drop(columns=[target_column])

                test_meta = pd.concat(auto_test_preds, axis=1)
                test_meta.columns = auto_names
                test_meta.index = test_df.index
                model_names = auto_names
                predictions = [test_meta[col] for col in test_meta.columns]

            # Train meta-model
            if meta_model == "logistic":
                from sklearn.linear_model import LogisticRegression
                meta = LogisticRegression(max_iter=1000)
                meta.fit(X_meta, y_meta)
                ensemble_pred = pd.Series(meta.predict_proba(test_meta)[:, 1], index=test_meta.index)
            elif meta_model == "ridge":
                from sklearn.linear_model import Ridge
                meta = Ridge()
                meta.fit(X_meta, y_meta)
                ensemble_pred = pd.Series(meta.predict(test_meta), index=test_meta.index)
            else:
                raise ValueError(f"Unsupported meta-model: {meta_model}")

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Create submission DataFrame
        target_column = project_ctx["config"].TARGET_COLUMN
        id_column = project_ctx["config"].ID_COLUMN

        # Load ID column
        if strategy == "meta" and not oof_files:
            id_df = pd.DataFrame({id_column: test_df[id_column].values})
        else:
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
        tracker_map = None
        if model_files:
            try:
                tracker_entries = load_tracker_scores(project_ctx)
                tracker_map = {}
                for mf, stem in zip(model_files, model_names):
                    entry = next((e for e in tracker_entries if e.get("filename") == Path(mf).name), None)
                    if entry:
                        tracker_map[stem] = entry.get("local_cv_score")
            except Exception:
                tracker_map = None
        display_ensemble_stats(predictions, ensemble_pred, model_names, project_ctx["name"], output_path, tracker_map)

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
        else:
            metadata["meta_model"] = meta_model
            metadata["meta_base_models"] = model_names

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
    model_names: List[str],
    project_name: str,
    output_path: Path,
    local_cv_map: Dict[str, Optional[float]] | None = None,
) -> None:
    """Display ensemble statistics."""
    table = Table(title="Ensemble Statistics")
    table.add_column("Model", style="cyan")
    table.add_column("Local CV", style="yellow")
    table.add_column("Mean", style="yellow")
    table.add_column("Std", style="yellow")
    table.add_column("Correlation", style="green")

    # Individual models
    for i, (pred, name) in enumerate(zip(predictions, model_names)):
        corr = pred.corr(ensemble_pred)
        lcv = "-"
        if local_cv_map is not None:
            lval = local_cv_map.get(name)
            if isinstance(lval, (int, float)):
                lcv = f"{lval:.5f}"
        table.add_row(
            name,
            lcv,
            f"{pred.mean():.5f}",
            f"{pred.std():.5f}",
            f"{corr:.5f}"
        )

    # Ensemble
    table.add_row(
        "[bold]Ensemble[/bold]",
        "-",
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

    console.print(
        f"\n[yellow]Next steps:[/yellow] "
        f"Submit with "
        f"`python scripts/experiment_manager.py submit --project {project_name} "
        f"--filename {output_path.name}` "
        f"or rerun stack with `--output-name` of your choice."
    )


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
        help="Submission CSV files (required for blend, optional for meta when auto-generating OOF)"
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
        "--auto-weights",
        choices=["public", "local"],
        help="Auto-build weights from submissions tracker using public or local scores (blend strategy)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        help="When using --auto-weights, optionally select top-N models by that metric from tracker (defaults to all provided)"
    )
    parser.add_argument(
        "--meta-model",
        choices=["logistic", "ridge"],
        default="logistic",
        help="Meta-learner model (for meta strategy)"
    )
    parser.add_argument(
        "--meta-base-models",
        nargs="+",
        default=["xgboost", "lightgbm", "catboost"],
        help="Base models to auto-generate OOF/preds for meta strategy (ignored if --oof-files provided)"
    )
    parser.add_argument(
        "--meta-n-trials",
        type=int,
        default=10,
        help="Optuna trials per base model when auto-generating OOF"
    )
    parser.add_argument(
        "--meta-cv-folds",
        type=int,
        default=3,
        help="CV folds for OOF/meta generation"
    )
    parser.add_argument(
        "--meta-timeout",
        type=int,
        default=600,
        help="Timeout (seconds) per base model tuning when auto-generating OOF"
    )
    parser.add_argument(
        "--oof-files",
        nargs="+",
        help="OOF prediction CSVs aligned with --models for meta strategy (must include ID column and target column)"
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
    if args.blend_weights and args.models and len(args.blend_weights) != len(args.models):
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
            auto_weights=args.auto_weights,
            top_n=args.top_n,
            meta_model=args.meta_model,
            oof_files=args.oof_files,
            meta_base_models=args.meta_base_models,
            meta_n_trials=args.meta_n_trials,
            meta_cv_folds=args.meta_cv_folds,
            meta_timeout=args.meta_timeout,
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
