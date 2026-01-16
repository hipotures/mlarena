from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from rich.console import Console
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight

from kaggle_tools.config_models import ModelConfig


@dataclass
class PreparedData:
    train_data: pd.DataFrame
    tuning_data: Optional[pd.DataFrame]
    eval_data: Optional[pd.DataFrame]
    sample_weight: Optional[np.ndarray]
    target_column: str
    meta: Dict[str, Any]


def _infer_problem_type(config: ModelConfig, y: pd.Series) -> str:
    problem_type = str(config.dataset.problem_type or "").strip().lower()
    if problem_type:
        return problem_type

    unique_count = y.nunique(dropna=True)
    if unique_count <= 2:
        return "binary"
    if unique_count <= 20:
        return "multiclass"
    return "regression"


def _default_metric(problem_type: str) -> str:
    if problem_type == "regression":
        return "mean_absolute_error"
    if problem_type == "multiclass":
        return "accuracy"
    return "roc_auc"


def resolve_metric(config: ModelConfig, y: pd.Series) -> Tuple[str, Callable[[Any, Any], float], bool, bool]:
    metric_name = (config.dataset.metric or "").strip().lower()
    problem_type = _infer_problem_type(config, y)
    if not metric_name:
        metric_name = _default_metric(problem_type)

    metric_name = metric_name.replace("-", "_")
    needs_proba = False
    greater_is_better = True

    if metric_name in {"roc_auc", "auc"}:
        needs_proba = True
        greater_is_better = True
        if problem_type == "multiclass":
            base_fn = lambda yt, yp: roc_auc_score(yt, yp, multi_class="ovr", average="macro")
        else:
            base_fn = roc_auc_score
    elif metric_name in {"accuracy", "acc"}:
        base_fn = accuracy_score
    elif metric_name in {"log_loss", "logloss"}:
        needs_proba = True
        greater_is_better = False
        base_fn = log_loss
    elif metric_name in {"rmse", "root_mean_squared_error"}:
        greater_is_better = False
        base_fn = lambda yt, yp: mean_squared_error(yt, yp, squared=False)
    elif metric_name in {"mse", "mean_squared_error"}:
        greater_is_better = False
        base_fn = mean_squared_error
    elif metric_name in {"mae", "mean_absolute_error"}:
        greater_is_better = False
        base_fn = mean_absolute_error
    elif metric_name in {"rmsle"}:
        greater_is_better = False
        base_fn = mean_squared_log_error
    elif metric_name in {"mape"}:
        greater_is_better = False
        base_fn = mean_absolute_percentage_error
    elif metric_name in {"r2", "r2_score"}:
        base_fn = r2_score
    elif metric_name in {"f1", "f1_score"}:
        if problem_type == "multiclass":
            base_fn = lambda yt, yp: f1_score(yt, yp, average="macro")
        else:
            base_fn = f1_score
    else:
        base_fn = accuracy_score if problem_type != "regression" else mean_absolute_error
        greater_is_better = problem_type != "regression"

    def score_fn(y_true: Any, y_pred: Any) -> float:
        score = float(base_fn(y_true, y_pred))
        return score if greater_is_better else -score

    return metric_name, score_fn, needs_proba, greater_is_better


def prepare_training_data(
    train_df: pd.DataFrame,
    config: ModelConfig,
    artifacts: Optional[Dict[str, Any]],
    console: Optional[Console] = None,
) -> PreparedData:
    target_column = config.dataset.target
    base_train_rows = len(train_df)

    orig_df = None
    sample_weight = None
    tuning_df = None
    eval_df = None
    merged_rows = 0

    if artifacts:
        orig_df = artifacts.get("orig_df")
        sample_weight = artifacts.get("sample_weight")
        tuning_df = artifacts.get("tuning_df")
        eval_df = artifacts.get("eval_df")

    merge_orig = True
    if isinstance(config.model, dict):
        merge_orig = config.model.get("merge_orig", True)
    if not merge_orig:
        orig_df = None

    if orig_df is not None:
        if target_column not in orig_df.columns:
            orig_df = None
        else:
            orig_df = orig_df.dropna(subset=[target_column])

    if orig_df is not None:
        train_df = pd.concat([train_df, orig_df], ignore_index=True)
        merged_rows = int(len(orig_df))

    drop_cols = set((config.dataset.ignored_columns or []) + [config.dataset.id_column])
    drop_cols.discard(target_column)
    train_data = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns], errors="ignore")

    if target_column not in train_data.columns:
        raise ValueError(f"Target column '{target_column}' not found in training data")

    tuning_data = None
    if tuning_df is not None:
        tuning_data = tuning_df.drop(columns=[c for c in drop_cols if c in tuning_df.columns], errors="ignore")
        if target_column not in tuning_data.columns:
            tuning_data = None
            if console:
                console.print(
                    f"[yellow]⚠[/yellow] [bold]Tuning:[/bold] target '{target_column}' missing, ignoring tuning_df"
                )

    eval_data = None
    if eval_df is not None:
        eval_data = eval_df.drop(columns=[c for c in drop_cols if c in eval_df.columns], errors="ignore")
        if target_column not in eval_data.columns:
            eval_data = None
            if console:
                console.print(
                    f"[yellow]⚠[/yellow] [bold]Eval:[/bold] target '{target_column}' missing, ignoring eval_df"
                )

    weights = _resolve_sample_weight(
        train_data=train_data,
        target_column=target_column,
        base_train_rows=base_train_rows,
        merged_rows=merged_rows,
        config=config,
        sample_weight=sample_weight,
        console=console,
    )

    meta = {
        "used_orig": orig_df is not None,
        "orig_rows": merged_rows,
        "total_train_rows": len(train_df),
        "tuning_rows": len(tuning_data) if tuning_data is not None else 0,
        "eval_rows": len(eval_data) if eval_data is not None else 0,
    }

    return PreparedData(
        train_data=train_data,
        tuning_data=tuning_data,
        eval_data=eval_data,
        sample_weight=weights,
        target_column=target_column,
        meta=meta,
    )


def _resolve_sample_weight(
    train_data: pd.DataFrame,
    target_column: str,
    base_train_rows: int,
    merged_rows: int,
    config: ModelConfig,
    sample_weight: Any,
    console: Optional[Console],
) -> Optional[np.ndarray]:
    weights: Optional[pd.Series] = None
    strategy = config.dataset.sample_weight_strategy

    if strategy:
        if strategy in {"auto_weight", "balance_weight"}:
            y = train_data[target_column]
            problem_type = _infer_problem_type(config, y)
            if problem_type in {"binary", "multiclass"}:
                weights = pd.Series(compute_sample_weight(class_weight="balanced", y=y))
                if console:
                    console.print(
                        f"[cyan]i[/cyan] [bold]Sample Weights:[/bold] computed '{strategy}' weights"
                    )
        else:
            if strategy in train_data.columns:
                weights = pd.to_numeric(train_data[strategy], errors="coerce")
                train_data.drop(columns=[strategy], inplace=True)
                if console:
                    console.print(
                        f"[green]✓[/green] [bold]Sample Weights:[/bold] using column '{strategy}'"
                    )
            elif console:
                console.print(
                    f"[yellow]⚠[/yellow] [bold]Sample Weights:[/bold] column '{strategy}' not found"
                )
    elif sample_weight is not None:
        if isinstance(sample_weight, pd.Series):
            weights = sample_weight
        elif isinstance(sample_weight, pd.DataFrame) and not sample_weight.empty:
            if "__sample_weight__" in sample_weight.columns:
                weights = sample_weight["__sample_weight__"]
            elif "sample_weight" in sample_weight.columns:
                weights = sample_weight["sample_weight"]
            else:
                weights = sample_weight.iloc[:, 0]
        else:
            try:
                weights = pd.Series(sample_weight)
            except Exception:
                weights = None

    if weights is None:
        return None

    weights = pd.to_numeric(weights, errors="coerce").reset_index(drop=True).astype(float)
    if weights.isna().any():
        fill_value = float(weights.mean()) if weights.notna().any() else 1.0
        weights = weights.fillna(fill_value)

    expected_rows = len(train_data)
    if merged_rows and len(weights) == base_train_rows:
        fill_value = float(weights.mean()) if weights.notna().any() else 1.0
        weights = pd.concat([weights, pd.Series([fill_value] * merged_rows)], ignore_index=True)

    if len(weights) != expected_rows:
        if console:
            console.print(
                f"[yellow]⚠[/yellow] [bold]Sample Weights:[/bold] ignoring weights "
                f"(expected {expected_rows:,}, got {len(weights):,})"
            )
        return None

    return weights.to_numpy()


def score_dataset(
    model: Any,
    data: pd.DataFrame,
    target_column: str,
    score_fn: Callable[[Any, Any], float],
    needs_proba: bool,
) -> float:
    X = data.drop(columns=[target_column])
    y = data[target_column]

    if needs_proba and hasattr(model, "predict_proba"):
        preds = model.predict_proba(X)
        if preds.ndim == 2 and preds.shape[1] == 2:
            preds = preds[:, 1]
    else:
        preds = model.predict(X)
    return float(score_fn(y, preds))


def cast_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    obj_cols = df.select_dtypes(include=["object"]).columns
    if not obj_cols.empty:
        df = df.copy()
        df[obj_cols] = df[obj_cols].astype("category")
    return df


def detect_categorical_features(df: pd.DataFrame, max_unique_ratio: float = 0.05) -> list[int]:
    cat_indices: list[int] = []
    for idx, col in enumerate(df.columns):
        if df[col].dtype == "object" or df[col].dtype.name == "category":
            cat_indices.append(idx)
            continue
        if df[col].dtype in ["int64", "int32", "float64", "float32"]:
            unique_ratio = df[col].nunique() / max(len(df), 1)
            if unique_ratio < max_unique_ratio:
                cat_indices.append(idx)
    return cat_indices


def build_leaderboard(model_name: str, score: Optional[float], metric_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": model_name,
                "score_val": score,
                "eval_metric": metric_name,
                "stack_level": 0,
            }
        ]
    )

