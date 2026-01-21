"""
Universal utilities for creating and managing competition submissions.

This module is shared across all competition projects. Individual projects
use wrappers in their code/utils/submission.py that import from here.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import inspect
import sys

import pandas as pd
import numpy as np
from rich.console import Console

console = Console()

# Import tools from scripts/ (experiment_logger, submissions_tracker)
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiment_logger import ExperimentLogger  # noqa: E402
from submissions_tracker import SubmissionsTracker  # noqa: E402


@dataclass
class SubmissionArtifact:
    """Container returned by `create_submission` with tracking metadata."""

    path: Path
    filename: str
    project_root: Path
    competition: str
    tracker_entry: Optional[Dict[str, Any]] = None
    experiment: Optional[Dict[str, Any]] = None
    model_name: Optional[str] = None
    local_cv_score: Optional[float] = None
    notes: str = ""
    config: Optional[Dict[str, Any]] = None
    feature_count: Optional[int] = None

    def tracker_id(self) -> Optional[int]:
        if self.tracker_entry:
            return self.tracker_entry.get("id")
        return None


def _safe_relative_code_path(path: Path, project_root: Path) -> Optional[str]:
    """
    Return path relative to project or repo root to avoid tracker warnings.

    Args:
        path: Path to make relative
        project_root: Project root directory

    Returns:
        Relative path string or original path if cannot be made relative
    """
    if not path:
        return None
    bases = [project_root, project_root.parent]
    for base in bases:
        try:
            return str(path.relative_to(base))
        except ValueError:
            continue
    return str(path)


def _infer_target_type(series: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(series) or pd.api.types.is_bool_dtype(series):
        return "int"
    if pd.api.types.is_float_dtype(series):
        return "float"
    return "text"


def _ensure_target_value_types(
    values: pd.Series, target_type: Optional[str], column_name: str
):
    if target_type is None:
        return
    if target_type == "int":
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.isna().any():
            raise ValueError(
                f"Submission column '{column_name}' must contain integer values, but non-numeric values were found."
            )
        if not np.all(np.isclose(numeric, np.round(numeric))):
            raise ValueError(
                f"Submission column '{column_name}' must contain integer class labels (e.g., 0/1). "
                "Detected non-integer values. Use predictor.predict(...) instead of predict_proba(...)."
            )
    elif target_type == "float":
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.isna().any():
            raise ValueError(
                f"Submission column '{column_name}' must contain numeric values (float)."
            )
    elif target_type == "text":
        if not pd.api.types.is_string_dtype(values):
            raise ValueError(
                f"Submission column '{column_name}' must contain text values."
            )


def _validate_prediction_signal(
    predictions: pd.Series, sample_target: Optional[pd.Series], column_name: str
):
    if sample_target is None:
        return
    target_values = sample_target.dropna().unique()
    if len(target_values) <= 10:
        unique_predictions = pd.Series(predictions).dropna().unique()
        if len(unique_predictions) <= 1:
            raise ValueError(
                f"Predictions for '{column_name}' contain a single unique value ({unique_predictions[0] if len(unique_predictions) == 1 else 'NA'}). "
                "This usually indicates a degenerate submission. Ensure you're submitting class labels with variation."
            )


def _build_submission_from_dataframe(
    predictions: pd.DataFrame,
    sample_df: Optional[pd.DataFrame],
    fallback_id_col: str,
    default_target_col: str,
    provided_test_ids,
    force_probabilities: bool = False,
):
    submission = predictions.copy()
    if sample_df is not None:
        expected_cols = sample_df.columns.tolist()
        missing = [col for col in expected_cols if col not in submission.columns]
        if missing:
            raise ValueError(
                "Predictions DataFrame is missing required columns from sample_submission: "
                f"{missing}"
            )
        submission = submission[expected_cols]
        # Validate every target column based on the sample signature.
        for column in expected_cols[1:]:
            sample_series = sample_df[column]
            target_type = _infer_target_type(sample_series)
            if force_probabilities and target_type == "int":
                target_type = "float"
            _ensure_target_value_types(submission[column], target_type, column)
            _validate_prediction_signal(submission[column], sample_series, column)
        return submission

    id_col = fallback_id_col
    if id_col not in submission.columns:
        if provided_test_ids is None:
            raise ValueError(
                "Predictions DataFrame must contain the ID column or provide test_ids."
            )
        submission.insert(0, id_col, provided_test_ids)
    else:
        other_cols = [col for col in submission.columns if col != id_col]
        submission = submission[[id_col] + other_cols]

    if submission.shape[1] < 2:
        raise ValueError(
            "Predictions DataFrame must include at least one target column besides the ID column."
        )

    for column in submission.columns[1:]:
        _ensure_target_value_types(submission[column], None, column)
        _validate_prediction_signal(submission[column], None, column)

    return submission


def _build_submission_from_series(
    predictions,
    test_ids,
    sample_df: Optional[pd.DataFrame],
    fallback_id_col: str,
    fallback_target_col: str,
    explicit_id_column: Optional[str],
    force_probabilities: bool = False,
):
    if test_ids is None:
        raise ValueError(
            "test_ids must be provided when predictions are not a DataFrame."
        )

    if sample_df is not None:
        id_col = sample_df.columns[0]
        if len(sample_df.columns) != 2:
            raise ValueError(
                "Sample submission contains multiple target columns. "
                "Return a DataFrame with all required columns instead of a single Series."
            )
        target_col = sample_df.columns[1]
        sample_target_series = sample_df[target_col]
        target_type = _infer_target_type(sample_target_series)
        if force_probabilities and target_type == "int":
            target_type = "float"
    else:
        id_col = explicit_id_column or fallback_id_col
        target_col = fallback_target_col
        sample_target_series = None
        target_type = None

    predictions_series = pd.Series(predictions)
    _ensure_target_value_types(predictions_series, target_type, target_col)
    _validate_prediction_signal(predictions_series, sample_target_series, target_col)

    submission = pd.DataFrame({id_col: test_ids, target_col: predictions_series})
    return submission


def create_submission(
    predictions,
    project_root: Path,
    competition_name: str,
    submissions_dir: Path,
    sample_submission_path: Path,
    test_ids=None,
    filename_prefix="submission",
    metric_name=None,
    metric_value=None,
    model_name: Optional[str] = None,
    local_cv_score: Optional[float] = None,
    cv_std: Optional[float] = None,
    notes: str = "",
    config: Optional[Dict] = None,
    track: bool = True,
    default_target_col: str = "target",
    id_column: Optional[str] = None,
    default_id_col: str = "id",
    submission_probas: Optional[bool] = None,
    feature_count: Optional[int] = None,
):
    """
    Create a submission file with timestamp and optional metric info.

    This is the universal implementation used by all competition projects.

    Args:
        predictions: Array, Series, or DataFrame of predictions. When passing a
            DataFrame it should already contain the ID column and all required
            target columns (matching sample_submission.csv).
        test_ids: Array or Series of test IDs. Required when `predictions` is not
            a DataFrame.
        project_root: Path to project root directory
        competition_name: Kaggle competition slug
        submissions_dir: Path to submissions directory
        sample_submission_path: Path to sample_submission.csv
        filename_prefix: Prefix for the submission filename
        metric_name: Name of the metric (e.g., 'RMSE', 'MAPE')
        metric_value: Value of the metric
        model_name: Model identifier for tracking
        local_cv_score: Local cross-validation score
        cv_std: Standard deviation of CV score
        notes: Additional notes
        config: Model configuration dictionary
        track: Whether to add to submissions tracker (default: True)
        default_target_col: Default target column name if sample not found
        submission_probas: If True, allow probability outputs even when the sample
            target column is integer-typed (skip int-only enforcement).
        feature_count: Number of features used for training (after drops).

    Returns:
        SubmissionArtifact with path and metadata
    """
    if sample_submission_path.exists():
        sample_df = pd.read_csv(sample_submission_path, compression="infer")
    else:
        sample_df = None
        if test_ids is None and not isinstance(predictions, pd.DataFrame):
            raise ValueError(
                f"test_ids is required when sample submission is missing ({sample_submission_path})."
            )
        console.print(
            f"[yellow]⚠ Warning: sample submission not found at {sample_submission_path}.[/yellow] "
            "[dim]Falling back to default column names.[/dim]"
        )

    fallback_id = id_column or default_id_col
    fallback_target = default_target_col

    # Auto-detect regression to allow floats even if sample uses ints
    should_force_probas = bool(submission_probas)
    if not should_force_probas and config:
        p_type = (
            config.get("AUTOGLUON_PROBLEM_TYPE")
            if isinstance(config, dict)
            else getattr(config, "AUTOGLUON_PROBLEM_TYPE", None)
        )
        if p_type == "regression":
            should_force_probas = True

    if isinstance(predictions, pd.DataFrame):
        submission = _build_submission_from_dataframe(
            predictions,
            sample_df,
            fallback_id,
            fallback_target,
            test_ids,
            force_probabilities=should_force_probas,
        )
    else:
        submission = _build_submission_from_series(
            predictions,
            test_ids,
            sample_df,
            fallback_id,
            fallback_target,
            id_column,
            force_probabilities=should_force_probas,
        )

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename_parts = [filename_prefix, timestamp]

    if metric_name and metric_value is not None:
        filename_parts.append(f"{metric_name}_{metric_value:.5f}")

    filename = "-".join(filename_parts) + ".csv"
    filepath = submissions_dir / filename

    # Ensure submissions directory exists
    submissions_dir.mkdir(parents=True, exist_ok=True)

    # Save submission
    submission.to_csv(filepath, index=False, compression="infer")

    console.print(
        f"[green]✓[/green] [bold]Submission saved:[/bold] [cyan]{filepath}[/cyan]"
    )
    console.print(
        f"  [dim]Shape: {submission.shape[0]:,} rows × {submission.shape[1]} columns[/dim]"
    )

    # Add to tracker if requested
    experiment = None
    tracker_entry = None
    if track:
        try:
            # Get calling code path (skip one more frame since we're in a shared module)
            frame = inspect.stack()[1]
            code_path = Path(frame.filename)
            code_rel = (
                _safe_relative_code_path(code_path, project_root)
                if code_path.exists()
                else None
            )

            exp_logger = ExperimentLogger(project_root)
            experiment = exp_logger.log_experiment(
                model_name=model_name or filename_prefix,
                code_path=code_path if code_path.exists() else None,
                config=config,
                notes=notes,
            )

            tracker = SubmissionsTracker(project_root)
            tracker_entry = tracker.add_submission(
                filename=filename,
                model_name=model_name or filename_prefix,
                local_cv_score=local_cv_score or metric_value,
                cv_std=cv_std,
                notes=notes,
                config=config,
                experiment_id=experiment.get("experiment_id") if experiment else None,
                git_hash=experiment["git"]["hash"] if experiment else None,
                code_path=code_rel,
                feature_count=feature_count,
            )
        except Exception as e:
            console.print(f"[yellow]⚠ Warning: Could not add to tracker: {e}[/yellow]")

    return SubmissionArtifact(
        path=filepath,
        filename=filename,
        project_root=project_root,
        competition=competition_name,
        tracker_entry=tracker_entry,
        experiment=experiment,
        model_name=model_name or filename_prefix,
        local_cv_score=local_cv_score or metric_value,
        notes=notes,
        config=config,
        feature_count=feature_count,
    )


def validate_submission(
    submission_path: Path,
    sample_submission_path: Path,
    submission_probas: bool = False,
) -> bool:
    """
    Validate submission format against sample submission.

    Args:
        submission_path: Path to submission file
        sample_submission_path: Path to sample_submission.csv
        submission_probas: If True, allow probability outputs even when the sample
            uses integer placeholders.

    Returns:
        bool: True if valid, raises exception otherwise
    """
    if not sample_submission_path.exists():
        console.print(
            "[yellow]⚠ Warning: sample_submission.csv not found, skipping validation[/yellow]"
        )
        return True

    sample = pd.read_csv(sample_submission_path, compression="infer")
    submission = pd.read_csv(submission_path, compression="infer")

    # Check shape
    assert submission.shape == sample.shape, (
        f"Shape mismatch: {submission.shape} vs {sample.shape}"
    )

    # Check columns
    assert list(submission.columns) == list(sample.columns), (
        f"Column mismatch: {submission.columns} vs {sample.columns}"
    )

    # Check IDs
    assert (submission.iloc[:, 0] == sample.iloc[:, 0]).all(), "ID column mismatch"

    target_type = _infer_target_type(sample.iloc[:, 1])
    if submission_probas and target_type == "int":
        target_type = "float"
    _ensure_target_value_types(submission.iloc[:, 1], target_type, sample.columns[1])

    console.print("[green]✓[/green] [bold]Submission format is valid[/bold]")
    return True
