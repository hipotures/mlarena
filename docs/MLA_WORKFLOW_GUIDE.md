# MLArena Workflow Guide

MLArena uses a modular pipeline where each step can be run individually or as part of an automated flow.

## Module Categories

### 1. Setup Modules (Prerequisites)
These modules prepare the environment and data. They must be run once per competition before any modeling.
- `init`: Project scaffolding and data download.
- `eda`: Exploratory data analysis and profiling.

### 2. Auto-Flow Pipeline
The core modeling loop. Can be run with a single command or step-by-step.
- `preprocess` → `model` → `predict` → `submit` → `fetch-score`

### 3. Utility & Admin Modules
Helper modules for managing the project and advanced engineering.
- `experiments`: View experiment history and leaderboards.
- `submissions`: Track Kaggle submissions and scores.
- `queue`: Sequential task management for batch runs.
- `feat`: Quick feature transformations.
- `tune`/`stack`: Advanced modeling utilities.

---

## Detailed Walkthrough

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
# Step 1: Initialize project (downloads data)
uv run python scripts/mla.py init --project titanic

# Step 2: Run EDA
uv run python scripts/mla.py eda --project titanic

# Step 3: Run auto-flow pipeline
uv run python scripts/mla.py --project titanic model_template=cpu-fast-1m wait_seconds=45

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

# Step 2: Exploratory Data Analysis (EDA)

Generates data profiling reports using ydata-profiling.

```bash
uv run python scripts/mla.py eda --project titanic
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
  model_template=cpu-fast-1m \
  skip_submit=true
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
# Check model directory (replace 'eda' with your experiment-id if different)
ls projects/kaggle/titanic/experiments/eda/artifacts/model/model/models/
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
# Default: 60s countdown before auto-submit
uv run python scripts/mla.py submit \
  --project titanic \
  --experiment-id eda

# Disable confirmation (submit immediately)
uv run python scripts/mla.py submit \
  --project titanic \
  --experiment-id eda \
  submit.confirm_timeout=0

# Skip submission (save CSV only)
uv run python scripts/mla.py submit \
  --project titanic \
  --experiment-id eda \
  skip_submit=true

# Add to submission queue (for batch processing later)
uv run python scripts/mla.py submit \
  --project titanic \
  --experiment-id eda \
  submit.queue_submit=true
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

#### Submission Queue (Batch Processing)

Queue submissions for later batch upload with duplicate detection.

**Add to queue:**
```bash
uv run python scripts/mla.py submit \
  --project titanic \
  --experiment-id exp-20251226-103504 \
  submit.queue_submit=true
```

**Manage queue:**
```bash
# List queued submissions
python scripts/submission_queue.py --project titanic list

# Submit from queue (by queue number, experiment-id, or filename)
python scripts/submission_queue.py --project titanic submit 1
python scripts/submission_queue.py --project titanic submit exp-20251226-103504
python scripts/submission_queue.py --project titanic submit submission.csv

# Submit with auto fetch-score (waits 30s, then fetches score, removes on success)
python scripts/submission_queue.py --project titanic submit 1 --continue-flow

# Remove from queue
python scripts/submission_queue.py --project titanic remove 1
```

**Features:**
- **Duplicate detection**: Checks Kaggle API before upload to prevent re-submission
- **Error tracking**: Logs all submission attempts with timestamps
- **Status tracking**: pending → submitted → completed (or failed)
- **Auto-cleanup**: Removes from queue on successful fetch-score (with --continue-flow)
- **Thread-safe**: Uses file locking for concurrent access

**Queue file location:** `projects/kaggle/{project}/submissions/queue.json`

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

## Viewing History

You can list all tracked submissions and experiments for a project using built-in modules:

```bash
# View all submissions for a project
uv run python scripts/mla.py submissions --project titanic list

# View all experiments for a project
uv run python scripts/mla.py experiments --project titanic list

# Filter experiments by status
uv run python scripts/mla.py experiments --project titanic list --status failed

# Sort experiments by public score
uv run python scripts/mla.py experiments --project titanic list --sort-by public
```

---

## Auto-Flow Pipeline

**Prerequisites** (one-time setup):

```bash
# Step 1: Initialize project structure and download data
uv run python scripts/mla.py init --project titanic

# Step 2: Run exploratory data analysis
uv run python scripts/mla.py eda --project titanic
```

### Auto-Flow Validation

Before executing auto-flow, the system validates:

1. **Init module completed**: Checks `experiments/init/state.json` status
2. **EDA module completed**: Checks `experiments/eda/state.json` status

If either validation fails, you'll see:
```
✗ Prerequisites validation failed:

Project initialization not found.
Run: mla init --project {project}
```

**Note**: These modules must be run manually before auto-flow.

**Auto-Flow**: Runs Preprocess → Model → Predict → Submit → Fetch Score automatically. Note that `init` and `eda` are manual prerequisites that must be run once per project.

```bash
# Default: 30s countdown before auto-submit
uv run python scripts/mla.py --project titanic model_template=cpu-fast-1m wait_seconds=45
```

**With custom preprocessing:**
```bash
# Specify preprocessing template
uv run python scripts/mla.py --project titanic \
  preprocess_template=identity \
  model_template=cpu-dev-5m
```

**Note**: Auto-flow validates that init and eda are completed before starting. If missing, you'll get a clear error message with instructions.

---

## Advanced Workflows

### Hierarchical Configuration System

MLArena now uses a unified configuration system based on **OmegaConf** and **Pydantic**. This allows for hierarchical merging and precise CLI overrides using dotted paths.

**Merging Order (lowest to highest priority):**
1. **Hardcoded Defaults**: Base values in `GlobalConfig`.
2. **Profiles**: Sets of values (e.g., `--profile smoke` sets short time limits).
3. **Project Config**: `projects/kaggle/<name>/config.yaml`.
4. **CLI Overrides**: Any `key=value` pair provided in the command line.

**Example with Dotted Overrides:**
```bash
uv run python scripts/mla.py model \
  --project titanic \
  model_template=cpu-dev-5m \
  common.time_limit=120 \
  model.hyperparameters.GBM.max_depth=5
```

**Common Global Overrides:**
- `model_template=name`: Set model template (replaces `--model-template`)
- `preprocess_template=name`: Set preprocessing template
- `force=true`: Force re-run (same as `-f`)
- `common.seed=123`: Set global random seed
- `common.use_gpu=true`: Enable GPU globally

**Using Profiles:**
```bash
# Run a quick smoke test
uv run python scripts/mla.py model --project titanic --profile smoke

# Use a development profile with custom overrides
uv run python scripts/mla.py model --project titanic --profile dev model.time_limit=600
```

### With Custom Preprocessing

```bash
# Step 1: EDA
uv run python scripts/mla.py eda --project titanic

# Step 2: Preprocess
uv run python scripts/mla.py preprocess \
  --project titanic \
  --experiment-id eda \
  preprocess_template=my-preprocess

# Step 3: Model (uses preprocessed data)
uv run python scripts/mla.py model \
  --project titanic \
  --experiment-id eda \
  preprocess_template=my-preprocess \
  model_template=cpu-dev-5m
```

### Re-run with Different Template

```bash
# Force re-run model with different settings
uv run python scripts/mla.py model \
  --project titanic \
  --experiment-id eda \
  model_template=cpu-best-1h \
  --force
```

### Multiple Experiments

```bash
# Experiment 1: Fast baseline
uv run python scripts/mla.py eda --project titanic
uv run python scripts/mla.py model --project titanic --experiment-id eda model_template=cpu-fast-1m

# Experiment 2: Best model (creates new exp)
uv run python scripts/mla.py eda --project titanic
uv run python scripts/mla.py model --project titanic --experiment-id eda model_template=cpu-best-1h

# Compare results
uv run python scripts/submissions_tracker.py --project titanic list
```

### Hyperparameter Optimization (HPO)

Run AutoGluon native hyperparameter tuning with presets:

```bash
# Quick HPO test (50 trials, ~1-2h)
uv run python scripts/mla.py model \
  --project titanic \
  model_template=hpo_boost_medium

# Serious tuning (100 trials, ~4-6h)
uv run python scripts/mla.py model \
  --project titanic \
  model_template=hpo_boost_high

# Final push (200 trials, ~8-12h)
uv run python scripts/mla.py model \
  --project titanic \
  model_template=hpo_boost_best
```

**Available HPO presets:**
- `hpo_boost_medium` - 50 trials, conservative search spaces
- `hpo_boost_high` - 100 trials, broader search spaces
- `hpo_boost_best` - 200 trials, exhaustive search spaces

**Create custom HPO template:**

```yaml
# projects/kaggle/titanic/templates/model/my_hpo.yaml
model: autogluon_baseline
hpo_preset: hpo_boost_high
config:
  preset: best
  time_limit: 7200
  use_gpu: false
  included_model_types: [GBM, XGB, CAT]

  # Override preset defaults
  num_trials: 150      # Increase from 100
  searcher: bayesian   # Change from auto
```

**What you'll see:**

```
HPO Preset: hpo_boost_medium

HPO Configuration:
  Preset: hpo_boost_medium
  Trials: 50
  Scheduler: local
  Searcher: auto
  Models: ['GBM', 'XGB', 'CAT']

[AutoGluon HPO] Enabled with 50 trials
[AutoGluon HPO] Search spaces defined for: ['GBM', 'XGB', 'CAT']
[AutoGluon HPO]   GBM: 8 parameters
[AutoGluon HPO]   XGB: 8 parameters
[AutoGluon HPO]   CAT: 5 parameters

Fitted model: LightGBM/T1 ...
Fitted model: LightGBM/T2 ...
...
```

**Expected improvements:**
- Medium preset: +0.5-1% over baseline
- High preset: +1-2% over baseline
- Best preset: +2-3% over baseline (diminishing returns)

### Experiment Locking (`lock=true`)

To prevent accidental overwriting of valuable experiment results (especially long-running production models), you can use the `lock` parameter.

```bash
# Run and lock upon success
uv run python scripts/mla.py --project titanic model.time_limit=3600 lock=true
```

**Behavior:**
- Creates an `overwrite.lock` file in the experiment directory after successful completion.
- Prevents any future execution of this experiment (even with `--force`).
- **To Unlock:** Manually delete the `overwrite.lock` file from the experiment directory.

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
curl http://127.0.0.1:9222/json/version
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

- [README.md](../README.md) - Repository overview and quick start
- [ARCHITECTURE.md](./ARCHITECTURE.md) - MLArena architecture and design
- [configs.md](./configs.md) - Template configuration reference
- [MIGRATION_GUIDE.md](./MIGRATION_GUIDE.md) - Migration from legacy scripts
