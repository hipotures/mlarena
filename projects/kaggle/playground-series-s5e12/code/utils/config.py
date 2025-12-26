"""Configuration for playground-series-s5e12"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CODE_DIR = PROJECT_ROOT / "code"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Data paths
TRAIN_PATH = DATA_DIR / "train.csv.gz"
TEST_PATH = DATA_DIR / "test.csv.gz"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv.gz"

# Model settings
RANDOM_SEED = 42
N_FOLDS = 5

# Target column
TARGET_COLUMN = "diagnosed_diabetes"

# Row identifier column (ignored when training)
ID_COLUMN = "id"

# Columns to ignore during training
IGNORED_COLUMNS = ['id']

# AutoGluon settings
AUTOGLUON_TIME_LIMIT = 600  # seconds (10 minutes)
AUTOGLUON_PRESET = "medium"  # best, high, medium, deployment
AUTOGLUON_PROBLEM_TYPE = "binary"  # binary, regression, multiclass
AUTOGLUON_EVAL_METRIC = "roc_auc"  # AutoGluon metric (approximates Kaggle metric if different)

# Competition details
COMPETITION_NAME = "playground-series-s5e12"
METRIC = "roc_auc"  # Kaggle evaluation metric

# Submission format
SUBMISSION_PROBAS = True
