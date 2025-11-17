"""
Configuration and constants for the competition
"""

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

# Auto-detect submission file (sample_submission.csv, gender_submission.csv, etc.)
_submission_files = list(DATA_DIR.glob("*submission*.csv")) if DATA_DIR.exists() else []
SAMPLE_SUBMISSION_PATH = _submission_files[0] if _submission_files else DATA_DIR / "sample_submission.csv"

# Model settings
RANDOM_SEED = 42
N_FOLDS = 5

# Target column
TARGET_COLUMN = "Survived"

# AutoGluon settings
AUTOGLUON_TIME_LIMIT = 600  # seconds (10 minutes)
AUTOGLUON_PRESET = "medium_quality"  # best_quality, high_quality, medium_quality, optimize_for_deployment
AUTOGLUON_PROBLEM_TYPE = "binary"  # binary, regression, multiclass
AUTOGLUON_EVAL_METRIC = "roc_auc"  # evaluation metric

# Competition details
COMPETITION_NAME = "titanic"
METRIC = "roc_auc"
