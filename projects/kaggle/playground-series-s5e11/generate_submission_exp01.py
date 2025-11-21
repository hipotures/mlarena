#!/usr/bin/env python3
"""
Generate submission for already trained exp01 model.
Usage: uv run python generate_submission_exp01.py
"""

import sys
from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor

# Add project to path
PROJECT_ROOT = Path(__file__).parent
CODE_DIR = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))

# Import preprocessing
from preprocessing.fe_tier1 import add_tier1_features

# Paths
DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = PROJECT_ROOT / "AutogluonModels" / "exp01_tier1_features"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

print("="*60)
print("GENERATE SUBMISSION FOR EXP01 (Tier 1 Features)")
print("="*60)
print()

# 1. Load test data
print("📂 Loading test data...")
test_df = pd.read_csv(DATA_DIR / "test.csv")
print(f"   Test samples: {len(test_df)}")
print(f"   Columns: {list(test_df.columns)}")

# 2. Apply preprocessing (Tier 1 features)
print()
print("⚙️  Applying Tier 1 feature engineering...")
test_processed = add_tier1_features(test_df)
print(f"   Original features: {len(test_df.columns)}")
print(f"   After FE: {len(test_processed.columns)}")

# 3. Load trained model
print()
print(f"📦 Loading trained model from: {MODEL_PATH}")
if not MODEL_PATH.exists():
    print(f"❌ ERROR: Model not found at {MODEL_PATH}")
    sys.exit(1)

predictor = TabularPredictor.load(str(MODEL_PATH))
print(f"   ✅ Model loaded successfully")
print(f"   Model info: {predictor.model_names()[:3]}...")

# 4. Generate predictions
print()
print("🔮 Generating predictions...")
predictions = predictor.predict_proba(test_processed, as_multiclass=False)
print(f"   Predictions shape: {predictions.shape}")
print(f"   Mean prediction: {predictions.mean():.4f}")
print(f"   Min: {predictions.min():.4f}, Max: {predictions.max():.4f}")

# 5. Create submission file
print()
print("💾 Creating submission file...")
submission = pd.DataFrame({
    'id': test_df['id'],
    'loan_paid_back': predictions
})

# Generate filename
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
submission_filename = f"submission_exp01_{timestamp}.csv"
submission_path = SUBMISSIONS_DIR / submission_filename

submission.to_csv(submission_path, index=False)
print(f"   ✅ Submission saved: {submission_path}")
print(f"   Rows: {len(submission)}")

print()
print("="*60)
print("SUCCESS!")
print("="*60)
print()
print(f"Submission file: {submission_path}")
print()
print("Next steps:")
print(f"1. Submit to Kaggle:")
print(f"   kaggle competitions submit -c playground-series-s5e11 -f {submission_path} -m 'exp01 tier1 features'")
print()
print(f"2. Or use submission workflow:")
print(f"   uv run python ../../scripts/submission_workflow.py submit \\")
print(f"     --project playground-series-s5e11 \\")
print(f"     --filename {submission_filename}")
print()
