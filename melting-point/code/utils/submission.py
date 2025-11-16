"""
Utilities for creating and managing competition submissions
"""

import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict
from .config import SUBMISSIONS_DIR, SAMPLE_SUBMISSION_PATH, PROJECT_ROOT


def create_submission(
    predictions,
    test_ids,
    filename_prefix="submission",
    metric_name=None,
    metric_value=None,
    model_name: Optional[str] = None,
    local_cv_score: Optional[float] = None,
    cv_std: Optional[float] = None,
    notes: str = "",
    config: Optional[Dict] = None,
    track: bool = True
):
    """
    Create a submission file with timestamp and optional metric info

    Args:
        predictions: Array or Series of predictions
        test_ids: Array or Series of test IDs
        filename_prefix: Prefix for the submission filename
        metric_name: Name of the metric (e.g., 'RMSE', 'MAPE')
        metric_value: Value of the metric
        model_name: Model identifier for tracking
        local_cv_score: Local cross-validation score
        cv_std: Standard deviation of CV score
        notes: Additional notes
        config: Model configuration dictionary
        track: Whether to add to submissions tracker (default: True)

    Returns:
        Path to the created submission file
    """
    # Read sample submission to get correct column names
    if SAMPLE_SUBMISSION_PATH.exists():
        sample = pd.read_csv(SAMPLE_SUBMISSION_PATH)
        target_col = sample.columns[1]  # Second column is the target
    else:
        target_col = 'loan_paid_back'  # Default for this competition

    # Create submission DataFrame
    submission = pd.DataFrame({
        'id': test_ids,
        target_col: predictions
    })

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename_parts = [filename_prefix, timestamp]

    if metric_name and metric_value is not None:
        filename_parts.append(f"{metric_name}_{metric_value:.5f}")

    filename = "-".join(filename_parts) + ".csv"
    filepath = SUBMISSIONS_DIR / filename

    # Save submission
    submission.to_csv(filepath, index=False)

    print(f"✓ Submission saved: {filepath}")
    print(f"  Shape: {submission.shape}")

    # Add to tracker if requested
    if track:
        try:
            import sys
            import inspect
            tools_path = PROJECT_ROOT.parent / "tools"
            if str(tools_path) not in sys.path:
                sys.path.insert(0, str(tools_path))

            from submissions_tracker import SubmissionsTracker
            from experiment_logger import ExperimentLogger

            # Get calling code path
            frame = inspect.stack()[1]
            code_path = Path(frame.filename)

            # Log experiment
            exp_logger = ExperimentLogger(PROJECT_ROOT)
            experiment = exp_logger.log_experiment(
                model_name=model_name or filename_prefix,
                code_path=code_path if code_path.exists() else None,
                config=config,
                notes=notes
            )

            # Add to submissions tracker with experiment info
            tracker = SubmissionsTracker(PROJECT_ROOT)
            tracker.add_submission(
                filename=filename,
                model_name=model_name or filename_prefix,
                local_cv_score=local_cv_score or metric_value,
                cv_std=cv_std,
                notes=notes,
                config=config,
                experiment_id=experiment.get('experiment_id'),
                git_hash=experiment['git']['hash'],
                code_path=str(code_path.relative_to(PROJECT_ROOT)) if code_path.exists() else None
            )
        except Exception as e:
            print(f"Warning: Could not add to tracker: {e}")

    return filepath


def validate_submission(submission_path):
    """
    Validate submission format against sample submission

    Args:
        submission_path: Path to submission file

    Returns:
        bool: True if valid, raises exception otherwise
    """
    if not SAMPLE_SUBMISSION_PATH.exists():
        print("Warning: sample_submission.csv not found, skipping validation")
        return True

    sample = pd.read_csv(SAMPLE_SUBMISSION_PATH)
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

    print("✓ Submission format is valid")
    return True
