#!/usr/bin/env python3
"""
Regenerate predictions from existing trained model with correct config.
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

# Load test data
test_df = pd.read_csv(config.TEST_PATH)
test_ids = test_df[config.ID_COLUMN if hasattr(config, 'ID_COLUMN') else 'id']

# Load trained model
model_path = Path(__file__).parent / "experiments/exp-20251118-021330/artifacts/tuned-cpu-autogluon_tuned"
print(f"Loading model from: {model_path}")
predictor = TabularPredictor.load(str(model_path))

# Prepare features (drop ID column)
id_col = config.ID_COLUMN if hasattr(config, 'ID_COLUMN') else 'id'
features = test_df.drop(columns=[id_col], errors='ignore')

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
    model_name="autogluon_tuned_fixed",
    local_cv_score=best_score,
    notes="Regenerated with SUBMISSION_PROBAS=True (fixed)",
    config={
        "preset": "medium_quality",
        "time_limit": 600,
        "use_gpu": False,
        "template": "tuned-cpu",
        "note": "Regenerated from exp-20251118-021330",
    },
    track=True,
)

print(f"\n✓ Submission created: {artifact.path}")
print(f"  Ready to submit to Kaggle")
