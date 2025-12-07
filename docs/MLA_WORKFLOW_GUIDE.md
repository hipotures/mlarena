# Manual MLArena Pipeline Guide

Complete step-by-step guide for running ML experiments manually using MLArena CLI. Uses Titanic competition as reference example.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Walkthrough](#detailed-walkthrough)
- [One-Command Pipeline](#one-command-pipeline)
- [Advanced Workflows](#advanced-workflows)
- [Verification Steps](#verification-steps)
- [Troubleshooting](#troubleshooting)
- [Expected Results](#expected-results)

---

## Prerequisites

### 1. Install Dependencies
```bash
cd /home/xai/ml/kaggle
uv sync
```

### 2. Configure Kaggle API (one-time)
```bash
# Download kaggle.json from https://www.kaggle.com/settings → API
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Install Playwright (optional, for score scraping)
```bash
uv run playwright install chromium
```

### 4. Start Chrome with CDP (optional, leave running)
```bash
google-chrome --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-debug
# Login to Kaggle in this window
```

---

## Quick Start

**Fastest way to run complete pipeline:**
```bash
# Initialize project (downloads data)
uv run python scripts/mla.py init --project titanic

# Run full pipeline with auto-submit
uv run python scripts/mla.py model \
  --project titanic \
  --model-template cpu-fast-1m \
  --auto-submit \
  --wait-seconds 45

# Check results
uv run python scripts/submissions_tracker.py --project titanic list
```

---

## Detailed Walkthrough

### Step 1: Initialize Project

Creates directory structure and downloads competition data.

```bash
uv run python scripts/mla.py init --project titanic
```

**What it does:**
- Creates `projects/kaggle/titanic/` structure
- Downloads train.csv (891 rows), test.csv (418 rows)
- Copies template files (model.yaml, preprocess.yaml)
- Creates config.py with TARGET_COLUMN="Survived"

**Verify:**
```bash
ls projects/kaggle/titanic/data/
# Expected: train.csv, test.csv, gender_submission.csv

cat projects/kaggle/titanic/code/utils/config.py | grep TARGET
# Expected: TARGET_COLUMN = "Survived"
```

---

### Step 2: Exploratory Data Analysis (EDA)

Generates data profiling reports using ydata-profiling.

```bash
uv run python scripts/mla.py eda \
  --project titanic \
  --eda-notes "Initial exploration"
```

**What it does:**
- Creates experiment with ID "eda"
- Generates HTML profiles for train/test data
- Saves summary to `experiments/eda/artifacts/eda/`

**Verify:**
```bash
ls projects/kaggle/titanic/experiments/eda/artifacts/eda/
# Expected: eda_summary.json, train_profile.html, test_profile.html

cat projects/kaggle/titanic/experiments/eda/state.json | jq '.modules.eda.status'
# Expected: "completed"
```

**View profile:**
```bash
open projects/kaggle/titanic/experiments/eda/artifacts/eda/train_profile.html
# Or: xdg-open on Linux
```

---

### Step 3: Train Model

Trains AutoGluon model with specified template.

```bash
uv run python scripts/mla.py model \
  --project titanic \
  --experiment-id eda \
  --model-template cpu-fast-1m \
  --skip-submit
```

**Available templates:**
- `cpu-fast-1m`: 60s limit, quick test
- `cpu-dev-5m`: 300s limit, development
- `cpu-best-1h`: 3600s limit, production
- `gpu-dev-5m`: 300s with GPU
- `gpu-best-1h`: 3600s with GPU

**What it does:**
- Loads template configuration
- Trains AutoGluon TabularPredictor
- Saves model to `AutogluonModels/`
- Records local CV score in state.json

**Verify:**
```bash
# Check model directory
ls projects/kaggle/titanic/AutogluonModels/models/
# Expected: LightGBM/, CatBoost/, WeightedEnsemble_L2/, etc.

# Check CV score
cat projects/kaggle/titanic/experiments/eda/state.json | jq '.modules.model.payload.local_cv_score'
# Expected: 0.80-0.85 for Titanic

# Check leaderboard
cat projects/kaggle/titanic/experiments/eda/artifacts/model/leaderboard.csv
```

---

### Step 4: Generate Predictions

Creates submission CSV from trained model.

```bash
uv run python scripts/mla.py predict \
  --project titanic \
  --experiment-id eda
```

**What it does:**
- Loads trained model
- Predicts on test.csv
- Creates submission.csv with PassengerId + Survived

**Verify:**
```bash
# Check submission file
cat projects/kaggle/titanic/experiments/eda/artifacts/predict/submission.csv | head -5
# Expected:
# PassengerId,Survived
# 892,0
# 893,1
# 894,0
# 895,0

# Count rows
wc -l projects/kaggle/titanic/experiments/eda/artifacts/predict/submission.csv
# Expected: 419 (1 header + 418 predictions)
```

---

### Step 5: Submit to Kaggle

Uploads submission via Kaggle CLI.

```bash
uv run python scripts/mla.py submit \
  --project titanic \
  --experiment-id eda \
  --auto-submit
```

**What it does:**
- Calls `kaggle competitions submit`
- Records submission in `submissions/submissions.json`
- Links experiment_id to submission

**Verify:**
```bash
# Check submission recorded
cat projects/kaggle/titanic/submissions/submissions.json | jq '.[-1]'
# Expected: {experiment_id, timestamp, local_cv_score, message}

# Manual verification on Kaggle
open https://www.kaggle.com/competitions/titanic/submissions
```

---

### Step 6: Fetch Public Score

Scrapes public score from Kaggle submissions page.

```bash
# Ensure Chrome with CDP is running first!
uv run python scripts/mla.py fetch-score \
  --project titanic \
  --experiment-id eda \
  --wait-seconds 45
```

**What it does:**
- Connects to Chrome via CDP (port 9222)
- Navigates to competition submissions
- Scrapes latest public score
- Updates submissions.json

**Verify:**
```bash
cat projects/kaggle/titanic/submissions/submissions.json | jq '.[-1].public_score'
# Expected: 0.77-0.80 for Titanic baseline
```

---

## One-Command Pipeline

Runs EDA → Model → Predict → Submit → Fetch Score automatically:

```bash
uv run python scripts/mla.py model \
  --project titanic \
  --model-template cpu-fast-1m \
  --auto-submit \
  --wait-seconds 45
```

**With preprocessing:**
```bash
uv run python scripts/mla.py model \
  --project titanic \
  --preprocess-template identity \
  --model-template cpu-dev-5m \
  --auto-submit
```

---

## Advanced Workflows

### With Custom Preprocessing

```bash
# Step 1: EDA
uv run python scripts/mla.py eda --project titanic

# Step 2: Preprocess
uv run python scripts/mla.py preprocess \
  --project titanic \
  --experiment-id eda \
  --preprocess-template my-preprocess

# Step 3: Model (uses preprocessed data)
uv run python scripts/mla.py model \
  --project titanic \
  --experiment-id eda \
  --preprocess-template my-preprocess \
  --model-template cpu-dev-5m
```

### Re-run with Different Template

```bash
# Force re-run model with different settings
uv run python scripts/mla.py model \
  --project titanic \
  --experiment-id eda \
  --model-template cpu-best-1h \
  --force
```

### Multiple Experiments

```bash
# Experiment 1: Fast baseline
uv run python scripts/mla.py eda --project titanic --eda-notes "exp1"
uv run python scripts/mla.py model --project titanic --experiment-id eda --model-template cpu-fast-1m

# Experiment 2: Best model (creates new exp)
uv run python scripts/mla.py eda --project titanic --eda-notes "exp2"
uv run python scripts/mla.py model --project titanic --experiment-id eda --model-template cpu-best-1h

# Compare results
uv run python scripts/submissions_tracker.py --project titanic list
```

---

## Verification Steps

### After Each Step

**After init:**
```bash
[ -d projects/kaggle/titanic/data ] && echo "✓ Data dir exists"
[ -f projects/kaggle/titanic/data/train.csv ] && echo "✓ Train data downloaded"
```

**After EDA:**
```bash
[ -f projects/kaggle/titanic/experiments/eda/state.json ] && echo "✓ Experiment created"
cat projects/kaggle/titanic/experiments/eda/state.json | jq -r '.modules.eda.status' | grep -q "completed" && echo "✓ EDA completed"
```

**After model:**
```bash
[ -d projects/kaggle/titanic/AutogluonModels ] && echo "✓ Model saved"
cat projects/kaggle/titanic/experiments/eda/state.json | jq -r '.modules.model.status' | grep -q "completed" && echo "✓ Model trained"
```

**After predict:**
```bash
ls projects/kaggle/titanic/experiments/eda/artifacts/predict/submission.csv >/dev/null 2>&1 && echo "✓ Submission created"
```

**After submit:**
```bash
cat projects/kaggle/titanic/submissions/submissions.json | jq -e '.[-1].experiment_id' >/dev/null && echo "✓ Submission recorded"
```

**After fetch-score:**
```bash
cat projects/kaggle/titanic/submissions/submissions.json | jq -e '.[-1].public_score' >/dev/null && echo "✓ Score fetched"
```

---

## Troubleshooting

### "Project not initialized"
```
[error] Project 'titanic' not initialized. Run: mla init --project titanic
```
**Fix:** Run init command first.

### "Template not found"
```
✗ Template 'my-template' not found
```
**Fix:** Use standard template (cpu-fast-1m, cpu-dev-5m, etc.) or create custom in `templates/model.yaml`

### "Score fetch failed - CDP not available"
```
[error] Failed to connect to Chrome CDP
```
**Fix:**
```bash
# Start Chrome with remote debugging
google-chrome --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-debug

# Verify connection
curl http://localhost:9222/json/version
```

### "Kaggle API authentication failed"
```
403 - Forbidden
```
**Fix:**
```bash
chmod 600 ~/.kaggle/kaggle.json
cat ~/.kaggle/kaggle.json  # Verify valid JSON
```

### "Module already completed"
```
[warn] Module 'model' already completed. Use --force to re-run
```
**Fix:** Add `--force` flag or create new experiment.

---

## Expected Results

**Titanic Competition:**
- Dataset: 891 train, 418 test
- Target: Binary (Survived 0/1)
- Metric: Accuracy

**Typical Performance:**

| Template | Time | Local CV | Public Score |
|----------|------|----------|--------------|
| cpu-fast-1m | 60s | 0.80-0.82 | 0.77-0.79 |
| cpu-dev-5m | 300s | 0.82-0.84 | 0.78-0.80 |
| cpu-best-1h | 3600s | 0.83-0.85 | 0.79-0.81 |

**Note:** Local CV typically 2-4% higher than public due to distribution shift.

---

## See Also

- [CLAUDE.md](../CLAUDE.md) - Repository overview
- [scripts/README.md](../scripts/README.md) - All scripts reference
- [configs.md](./configs.md) - Template configuration
- [mlarena_architecture.md](./mlarena_architecture.md) - System design
