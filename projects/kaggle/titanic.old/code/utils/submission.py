"""
Utilities for creating and managing competition submissions.

This is a lightweight wrapper around src/kaggle_tools that injects
competition-specific configuration from this project's config.py.
"""

from pathlib import Path
from typing import Dict, Optional
import sys

# Import competition-specific configuration
from .config import (
    COMPETITION_NAME,
    PROJECT_ROOT,
    SAMPLE_SUBMISSION_PATH,
    SUBMISSIONS_DIR,
    TARGET_COLUMN
)

try:
    from .config import ID_COLUMN
except ImportError:
    ID_COLUMN = "id"

# Add src to path and import shared utilities
SRC_PATH = PROJECT_ROOT.parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from kaggle_tools import submission as submission_utils  # noqa: E402


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
    Create a submission file with timestamp and optional metric info.

    This wrapper delegates to tools/submission_utils.py with competition-specific config.

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
        SubmissionArtifact with path and metadata
    """
    return submission_utils.create_submission(
        predictions=predictions,
        test_ids=test_ids,
        project_root=PROJECT_ROOT,
        competition_name=COMPETITION_NAME,
        submissions_dir=SUBMISSIONS_DIR,
        sample_submission_path=SAMPLE_SUBMISSION_PATH,
        filename_prefix=filename_prefix,
        metric_name=metric_name,
        metric_value=metric_value,
        model_name=model_name,
        local_cv_score=local_cv_score,
        cv_std=cv_std,
        notes=notes,
        config=config,
        track=track,
        default_target_col=TARGET_COLUMN,
        id_column=ID_COLUMN
    )


def validate_submission(submission_path):
    """
    Validate submission format against sample submission.

    Args:
        submission_path: Path to submission file

    Returns:
        bool: True if valid, raises exception otherwise
    """
    return submission_utils.validate_submission(
        submission_path=Path(submission_path),
        sample_submission_path=SAMPLE_SUBMISSION_PATH
    )
