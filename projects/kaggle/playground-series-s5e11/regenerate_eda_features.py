#!/usr/bin/env python3
"""
Regenerate predictions from autogluon_eda_features model with probabilities.
"""

import sys
from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor

# Add code to path
CODE_DIR = Path(__file__).parent / "code"
sys.path.insert(0, str(CODE_DIR))

from utils import config
from utils.submission import create_submission

# Import model with feature engineering
sys.path.insert(0, str(CODE_DIR / "models"))
import autogluon_eda_features

# Load test data
test_df = pd.read_csv(config.TEST_PATH)
test_ids = test_df[config.ID_COLUMN if hasattr(config, 'ID_COLUMN') else 'id']

# Apply feature engineering
print("Applying feature engineering...")
from kaggle_tools.config_models import ModelConfig, DatasetConfig, SystemConfig, Hyperparameters

# Create minimal config for feature engineering
dataset_cfg = DatasetConfig(
    train_path=config.TRAIN_PATH,
    test_path=config.TEST_PATH,
    target=config.TARGET_COLUMN,
    id_column='id',
    ignored_columns=[],
    sample_submission_path=config.SAMPLE_SUBMISSION_PATH,
)
system_cfg = SystemConfig(
    project_root=config.PROJECT_ROOT,
    code_dir=config.CODE_DIR,
    experiment_dir=config.PROJECT_ROOT / "experiments",
    artifact_dir=config.PROJECT_ROOT / "experiments",
    model_path=config.PROJECT_ROOT / "experiments",
    template="eda-cpu",
    experiment_id="exp-20251118-010924",
    random_seed=42,
)
model_cfg = ModelConfig(
    system=system_cfg,
    dataset=dataset_cfg,
    hyperparameters=Hyperparameters(),
)

test_engineered = autogluon_eda_features.preprocess(test_df, model_cfg, is_train=False)
print(f"Features after engineering: {test_engineered.shape[1]} (was {test_df.shape[1]})")

# Load trained model
model_path = Path(__file__).parent / "experiments/exp-20251118-010924/artifacts/eda-cpu-autogluon_eda_features"
print(f"Loading model from: {model_path}")
predictor = TabularPredictor.load(str(model_path))

# Prepare features (drop ID column)
id_col = 'id'
features = test_engineered.drop(columns=[id_col], errors='ignore')

# Generate predictions with probabilities
print(f"Generating predictions (probas={config.SUBMISSION_PROBAS})...")
if config.SUBMISSION_PROBAS:
    # Get probabilities for class 1
    predictions = predictor.predict_proba(features, as_multiclass=False)
    if isinstance(predictions, pd.DataFrame):
        predictions = predictions.iloc[:, 1]
else:
    # Get hard predictions
    predictions = predictor.predict(features)

print(f"Predictions shape: {len(predictions)}")
print(f"Sample predictions: {predictions[:5].tolist()}")
print(f"Min: {predictions.min():.4f}, Max: {predictions.max():.4f}")

# Get model performance from leaderboard
leaderboard = predictor.leaderboard(silent=True)
best_score = float(leaderboard.iloc[0]["score_val"]) if not leaderboard.empty else None

print(f"\nBest model: {leaderboard.iloc[0]['model'] if not leaderboard.empty else 'N/A'}")
print(f"Validation score: {best_score}")

# Create submission
artifact = create_submission(
    predictions=predictions,
    test_ids=test_ids,
    model_name="autogluon_eda_features_fixed",
    local_cv_score=best_score,
    notes="Regenerated with SUBMISSION_PROBAS=True + feature engineering (fixed)",
    config={
        "preset": "medium_quality",
        "time_limit": 600,
        "use_gpu": False,
        "template": "eda-cpu",
        "note": "Regenerated from exp-20251118-010924 with feature engineering",
    },
    track=True,
)

print(f"\n✓ Submission created: {artifact.path}")
print(f"  Ready to submit to Kaggle")
