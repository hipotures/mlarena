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
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"

# Model settings
RANDOM_SEED = 42
N_FOLDS = 5

# Target column
TARGET_COLUMN = "loan_paid_back"

# AutoGluon settings
AUTOGLUON_TIME_LIMIT = 600  # seconds (10 minutes)
AUTOGLUON_PRESET = "medium_quality"  # best_quality, high_quality, medium_quality, optimize_for_deployment
AUTOGLUON_PROBLEM_TYPE = "binary"  # binary classification
AUTOGLUON_EVAL_METRIC = "roc_auc"  # ROC AUC Score

# Competition details
COMPETITION_NAME = "playground-series-s5e11"
METRIC = "roc_auc"  # Area under the ROC curve
DEADLINE = "2025-11-30 23:59 UTC"
