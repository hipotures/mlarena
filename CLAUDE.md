# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Kaggle competitions repository with standardized structure, experiment tracking, and reproducibility system. Uses `uv` for dependency management, AutoGluon for baseline models, and custom tracking tools for submissions/experiments.

## Architecture

### Three-Layer System

1. **Tools Layer** (`tools/`): Universal utilities for all competitions
   - `submissions_tracker.py`: Tracks local CV, public/private scores with git integration
   - `experiment_logger.py`: Logs experiments with git hash, code snapshots, config
   - `kaggle_scraper.py`: Scrapes Kaggle leaderboard/submissions via CDP

2. **Project Layer** (`[competition-name]/`): Individual competition directories
   - `code/utils/config.py`: Competition-specific constants (target, metric, AutoGluon settings)
   - `code/utils/submission.py`: Auto-integrates with tracking tools
   - `code/models/`: Model implementations
   - `code/exploration/`: EDA scripts

3. **Tracking Layer**: Automatic experiment → submission → git linkage
   - Every `create_submission()` call captures git hash, creates code snapshot
   - Stored in `experiments/*.json` and `submissions/submissions.json`
   - Enables full reproducibility via experiment ID or git hash

### Key Integration Points

**`code/utils/submission.py`** is the integration hub:
```python
create_submission(predictions, test_ids, model_name, local_cv_score, ...)
```
Automatically:
1. Detects calling code path via `inspect.stack()`
2. Calls `ExperimentLogger` → saves snapshot to `experiments/`
3. Calls `SubmissionsTracker` → links submission to experiment + git hash
4. Warns if uncommitted changes exist

**`code/utils/config.py`** per competition defines:
- `TARGET_COLUMN`, `AUTOGLUON_PROBLEM_TYPE`, `AUTOGLUON_EVAL_METRIC`
- All paths relative to `PROJECT_ROOT = Path(__file__).parent.parent.parent`

## Common Commands

### Setup
```bash
# Install dependencies
uv sync

# Configure Kaggle API (one-time)
mkdir -p ~/.kaggle
# Copy kaggle.json to ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Competition Workflow

**1. Download data:**
```bash
cd [competition-name]/data
kaggle competitions download -c [competition-name]
unzip [competition-name].zip
cd ../..
```

**2. Run EDA:**
```bash
python [competition-name]/code/exploration/01_initial_eda.py
```

**3. Train model (ALWAYS commit first!):**
```bash
git status
git add code/
git commit -m "feat: baseline model"
python [competition-name]/code/models/baseline_autogluon.py
```

**4. Submit to Kaggle:**
```bash
cd [competition-name]/submissions
kaggle competitions submit -c [competition-name] \
    -f submission-TIMESTAMP.csv \
    -m "Model description with local CV"
cd ../..
```

**5. Update public score:**
```bash
python tools/submissions_tracker.py --project [competition-name] \
    update 1 --public 0.85123
```

### Tracking & Reproducibility

**View submissions:**
```bash
python tools/submissions_tracker.py --project [competition-name] list
```

**View experiments:**
```bash
python tools/experiment_logger.py --project [competition-name] list
```

**Reproduce submission:**
```bash
# Method 1: Git checkout
python tools/submissions_tracker.py --project [competition-name] list  # get git hash
git checkout <GIT_HASH>

# Method 2: Code snapshot restore
python tools/experiment_logger.py --project [competition-name] restore <EXPERIMENT_ID>
```

## Critical Workflows

### Experiment Tracking System

**ALWAYS commit before running experiments** - system captures git hash automatically.

When `create_submission()` is called:
1. ⚠️ Warns if `git status` shows uncommitted changes
2. Creates `experiments/TIMESTAMP_MODELNAME.json` with:
   - Full git info (hash, branch, commit message, uncommitted files)
   - Code path + MD5 hash of code
   - Full config dictionary
3. Creates `experiments/TIMESTAMP_MODELNAME.py` (code snapshot)
4. Adds to `submissions/submissions.json` with experiment_id + git_hash

**To reproduce ANY submission:**
- Find experiment_id from submissions tracker
- Either checkout git hash OR restore code snapshot
- Config is in experiment JSON

### Creating New Competition

```bash
COMP_NAME="competition-slug"
mkdir -p ${COMP_NAME}/{data,code/{exploration,models,utils},submissions,experiments,docs}
touch ${COMP_NAME}/{data,submissions,experiments}/.gitkeep

# Copy templates from existing competition:
cp playground-series-s5e11/.gitignore ${COMP_NAME}/
cp playground-series-s5e11/README.md ${COMP_NAME}/  # edit competition details
cp playground-series-s5e11/code/utils/*.py ${COMP_NAME}/code/utils/
cp playground-series-s5e11/code/exploration/01_initial_eda.py ${COMP_NAME}/code/exploration/
cp playground-series-s5e11/code/models/baseline_autogluon.py ${COMP_NAME}/code/models/

# Edit config.py with competition-specific settings:
# - TARGET_COLUMN
# - AUTOGLUON_PROBLEM_TYPE (binary/regression/multiclass)
# - AUTOGLUON_EVAL_METRIC
# - COMPETITION_NAME
```

## Project-Specific Configuration

Each competition's `code/utils/config.py` must define:

```python
# Paths (auto-derived from PROJECT_ROOT)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Competition specifics
TARGET_COLUMN = "target_name"  # CRITICAL: must match actual column
AUTOGLUON_PROBLEM_TYPE = "binary"  # or "regression", "multiclass"
AUTOGLUON_EVAL_METRIC = "roc_auc"  # or "mean_absolute_error", etc.
AUTOGLUON_TIME_LIMIT = 600  # seconds
AUTOGLUON_PRESET = "medium_quality"

# Model settings
RANDOM_SEED = 42
N_FOLDS = 5
```

## File Path Conventions

- All scripts use **absolute imports from project root**
- `sys.path.insert(0, str(Path(__file__).parent.parent))` to import from `utils/`
- Tools accessed via `sys.path.insert(0, str(PROJECT_ROOT.parent / "tools"))`
- Data paths via `config.py` constants, never hardcoded

## AutoGluon Baseline Pattern

Standard baseline model structure:
```python
from utils.config import (TRAIN_PATH, TEST_PATH, TARGET_COLUMN,
                          AUTOGLUON_*, PROJECT_ROOT)
from utils.submission import create_submission

# Load data, drop 'id' column for training
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# Train predictor
predictor = TabularPredictor(
    label=TARGET_COLUMN,
    problem_type=AUTOGLUON_PROBLEM_TYPE,
    eval_metric=AUTOGLUON_EVAL_METRIC,
    path=str(PROJECT_ROOT / "AutogluonModels")
)
predictor.fit(train.drop('id', axis=1), presets=AUTOGLUON_PRESET,
              time_limit=AUTOGLUON_TIME_LIMIT, num_gpus=1)

# Predict (use predict_proba for classification, predict for regression)
predictions = predictor.predict_proba(test.drop('id', axis=1), as_multiclass=False)

# Create submission (auto-tracks everything)
create_submission(predictions, test['id'], model_name="autogluon-baseline",
                  local_cv_score=best_score, notes="...", config={...})
```

## Data Management

**NEVER commit:**
- Data files (`data/*.csv`, `*.zip`)
- Model outputs (`AutogluonModels/`, `*.pkl`, `*.h5`)
- Experiment logs (`experiments/*.params`, `*.journal`)

**ALWAYS commit:**
- Code changes
- Config updates
- Documentation

Each competition has `.gitignore` with patterns for above.

## Kaggle Scraping (Optional)

To auto-fetch leaderboard/submissions (requires manual Chrome setup):

```bash
# 1. User manually starts Chrome with CDP
google-chrome --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-debug

# 2. User logs into Kaggle in that Chrome instance

# 3. Run scraper (only connects, doesn't launch browser)
python tools/kaggle_scraper.py [competition-name]

# Output: JSON files in [competition-name]/data/kaggle_scrapes/
```

See `tools/README_KAGGLE.md` for details.

## Common Pitfalls

1. **Running experiments without committing** → Missing git hash, can't reproduce
   - System warns but doesn't block
   - Always `git commit` before `python code/models/...`

2. **Wrong TARGET_COLUMN in config.py** → AutoGluon fails
   - Check `sample_submission.csv` column names
   - Binary classification: use `predict_proba(..., as_multiclass=False)`
   - Regression: use `predict()`

3. **Calling tools from wrong directory** → File not found
   - Tools must be called from repo root: `python tools/...`
   - Or use `cd .. && python tools/...` from competition dir

4. **Missing submission column name** → Kaggle rejects
   - `submission.py` auto-reads from `sample_submission.csv`
   - Fallback to competition-specific default in config

## Dependencies

Managed via `pyproject.toml`:
- `kaggle` - CLI for downloads/submissions
- `autogluon` - Primary modeling framework
- `pandas`, `numpy`, `scikit-learn` - Data processing
- `rich` - Console output formatting
- `playwright` - Optional, for Kaggle scraping

Install: `uv sync`
