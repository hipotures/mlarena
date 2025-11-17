"""
Universal utilities for creating and managing competition submissions.

This module is shared across all competition projects. Individual projects
use wrappers in their code/utils/submission.py that import from here.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import inspect
import sys

import pandas as pd
import numpy as np

# Import tools from scripts/ (experiment_logger, submissions_tracker, submission_workflow)
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiment_logger import ExperimentLogger  # noqa: E402
from submission_workflow import SubmissionArtifact  # noqa: E402
from submissions_tracker import SubmissionsTracker  # noqa: E402


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


def _ensure_target_value_types(values: pd.Series, target_type: Optional[str], column_name: str):
    if target_type is None:
        return
    if target_type == "int":
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.isna().any():
            raise ValueError(f"Submission column '{column_name}' must contain integer values, but non-numeric values were found.")
        if not np.all(np.isclose(numeric, np.round(numeric))):
            raise ValueError(
                f"Submission column '{column_name}' must contain integer class labels (e.g., 0/1). "
                "Detected non-integer values. Use predictor.predict(...) instead of predict_proba(...)."
            )
    elif target_type == "float":
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.isna().any():
            raise ValueError(f"Submission column '{column_name}' must contain numeric values (float).")
    elif target_type == "text":
        if not pd.api.types.is_string_dtype(values):
            raise ValueError(f"Submission column '{column_name}' must contain text values.")


def _validate_prediction_signal(predictions: pd.Series, sample_target: Optional[pd.Series], column_name: str):
    if sample_target is None:
        return
    target_values = sample_target.dropna().unique()
    if len(target_values) <= 10:
        unique_predictions = pd.Series(predictions).dropna().unique()
        if len(unique_predictions) <= 1:
            raise ValueError(
                f"Predictions for '{column_name}' contain a single unique value ({unique_predictions[0] if len(unique_predictions)==1 else 'NA'}). "
                "This usually indicates a degenerate submission. Ensure you're submitting class labels with variation."
            )


def create_submission(
    predictions,
    test_ids,
    project_root: Path,
    competition_name: str,
    submissions_dir: Path,
    sample_submission_path: Path,
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
):
    """
    Create a submission file with timestamp and optional metric info.

    This is the universal implementation used by all competition projects.

    Args:
        predictions: Array or Series of predictions
        test_ids: Array or Series of test IDs
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

    Returns:
        SubmissionArtifact with path and metadata
    """
    # Read sample submission to get correct column names
    target_type = None
    if sample_submission_path.exists():
        sample = pd.read_csv(sample_submission_path)
        id_col = sample.columns[0]
        target_col = sample.columns[1]
        sample_target_series = sample[target_col]
        target_type = _infer_target_type(sample_target_series)
    else:
        id_col = id_column or default_id_col
        target_col = default_target_col
        print(
            f"Warning: sample submission not found at {sample_submission_path}. "
            f"Using columns [{id_col}, {target_col}]"
        )
        sample_target_series = None
    predictions_series = pd.Series(predictions)
    _ensure_target_value_types(predictions_series, target_type, target_col)
    _validate_prediction_signal(predictions_series, sample_target_series, target_col)

    # Create submission DataFrame
    submission = pd.DataFrame({
        id_col: test_ids,
        target_col: predictions_series
    })

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
    submission.to_csv(filepath, index=False)

    print(f"✓ Submission saved: {filepath}")
    print(f"  Shape: {submission.shape}")

    # Add to tracker if requested
    experiment = None
    tracker_entry = None
    if track:
        try:
            # Get calling code path (skip one more frame since we're in a shared module)
            frame = inspect.stack()[1]
            code_path = Path(frame.filename)
            code_rel = _safe_relative_code_path(code_path, project_root) if code_path.exists() else None

            exp_logger = ExperimentLogger(project_root)
            experiment = exp_logger.log_experiment(
                model_name=model_name or filename_prefix,
                code_path=code_path if code_path.exists() else None,
                config=config,
                notes=notes
            )

            tracker = SubmissionsTracker(project_root)
            tracker_entry = tracker.add_submission(
                filename=filename,
                model_name=model_name or filename_prefix,
                local_cv_score=local_cv_score or metric_value,
                cv_std=cv_std,
                notes=notes,
                config=config,
                experiment_id=experiment.get('experiment_id') if experiment else None,
                git_hash=experiment['git']['hash'] if experiment else None,
                code_path=code_rel
            )
        except Exception as e:
            print(f"Warning: Could not add to tracker: {e}")

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
        config=config
    )


def validate_submission(submission_path: Path, sample_submission_path: Path) -> bool:
    """
    Validate submission format against sample submission.

    Args:
        submission_path: Path to submission file
        sample_submission_path: Path to sample_submission.csv

    Returns:
        bool: True if valid, raises exception otherwise
    """
    if not sample_submission_path.exists():
        print("Warning: sample_submission.csv not found, skipping validation")
        return True

    sample = pd.read_csv(sample_submission_path)
    submission = pd.read_csv(submission_path)

    # Check shape
    assert submission.shape == sample.shape, \
        f"Shape mismatch: {submission.shape} vs {sample.shape}"

    # Check columns
    assert list(submission.columns) == list(sample.columns), \
        f"Column mismatch: {submission.columns} vs {sample.columns}"

    # Check IDs
    assert (submission.iloc[:, 0] == sample.iloc[:, 0]).all(), \
        "ID column mismatch"

    target_type = _infer_target_type(sample.iloc[:, 1])
    _ensure_target_value_types(submission.iloc[:, 1], target_type, sample.columns[1])

    print("✓ Submission format is valid")
    return True
