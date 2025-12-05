"""Configuration file generation for new projects."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from rich.console import Console


def generate_config_py(
    project_root: Path,
    competition_slug: str,
    target_column: str,
    id_column: str,
    problem_type: str,
    metric: str,
    submit_probabilities: bool,
    ignored_columns: list[str],
    sample_submission_name: str,
    console: Console,
) -> None:
    """Generate code/utils/config.py with competition-specific settings."""

    ignored_columns_repr = repr(ignored_columns)

    config_content = f'''"""Configuration for {competition_slug}"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CODE_DIR = PROJECT_ROOT / "code"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Data paths
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "{sample_submission_name}"

# Model settings
RANDOM_SEED = 42
N_FOLDS = 5

# Target column
TARGET_COLUMN = "{target_column}"

# Row identifier column (ignored when training)
ID_COLUMN = "{id_column}"

# Columns to ignore during training
IGNORED_COLUMNS = {ignored_columns_repr}

# AutoGluon settings
AUTOGLUON_TIME_LIMIT = 600  # seconds (10 minutes)
AUTOGLUON_PRESET = "medium"  # best, high, medium, deployment
AUTOGLUON_PROBLEM_TYPE = "{problem_type}"  # binary, regression, multiclass
AUTOGLUON_EVAL_METRIC = "{metric}"  # evaluation metric

# Competition details
COMPETITION_NAME = "{competition_slug}"
METRIC = "{metric}"

# Submission format
SUBMISSION_PROBAS = {str(bool(submit_probabilities))}
'''

    config_path = project_root / "code/utils/config.py"
    config_path.write_text(config_content)
    console.print(f"\n[green]✓[/green] Customized code/utils/config.py")
