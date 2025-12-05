# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Table of Contents

- [Repository Overview](#repository-overview)
- [Quick Reference](#quick-reference)
- [Architecture](#architecture)
  - [Four-Layer System](#four-layer-system)
  - [Key Integration Points](#key-integration-points)
- [Common Commands](#common-commands)
  - [Setup](#setup)
  - [Metric Detection (CDP)](#metric-detection-cdp)
  - [Modern Workflow (Recommended)](#modern-workflow-recommended)
  - [Alternative: Direct Runner](#alternative-direct-runner-without-experiment-manager)
  - [Tracking & Reproducibility](#tracking--reproducibility)
- [Critical Workflows](#critical-workflows)
  - [Modular Experiment Pipeline](#modular-experiment-pipeline)
  - [Legacy Experiment Tracking](#legacy-experiment-tracking)
  - [Creating New Competition](#creating-new-competition)
- [Project-Specific Configuration](#project-specific-configuration)
- [File Path Conventions](#file-path-conventions)
- [AutoGluon Baseline Pattern](#autogluon-baseline-pattern)
- [Data Management](#data-management)
- [Automated Submission & Score Fetching](#automated-submission--score-fetching)
  - [Setup Chrome Debugging](#setup-chrome-debugging-one-time)
  - [Automatic Workflow](#automatic-workflow)
  - [Control Flags](#control-flags)
  - [Manual Scraping](#manual-scraping-standalone)
- [Common Pitfalls](#common-pitfalls)
- [Development Guidelines](#development-guidelines)
  - [Coding Style & Naming Conventions](#coding-style--naming-conventions)
  - [Testing Guidelines](#testing-guidelines)
  - [Commit & Pull Request Guidelines](#commit--pull-request-guidelines)
  - [Security & Configuration](#security--configuration)
- [Dependencies](#dependencies)

---

## Repository Overview

Kaggle competitions repository with standardized structure, experiment tracking, and reproducibility system. Uses `uv` for dependency management, AutoGluon for baseline models, and custom tracking tools for submissions/experiments.

## Quick Reference

**Most common workflow (recommended):**
```bash
# 1. Setup (once)
uv sync
uv run playwright install chromium

# 2. Run pipeline with MLArena
uv run python scripts/mla.py eda --project [competition-name]

uv run python scripts/mla.py model --project [competition-name] \
    --model-template dev-gpu \
    --auto-submit \
    --wait-seconds 45

# 3. Fetch score later if needed
uv run python scripts/mla.py fetch-score --project [competition-name] \
    --experiment-id exp-YYYYMMDD-HHMMSS

# 4. Check submissions/experiments
uv run python scripts/submissions_tracker.py --project [competition-name] list
```

**Key tools:**
- `mla.py` - Single CLI entry point (EDA → preprocess → model → predict → submit → fetch-score)
- `submission_workflow.py` - Kaggle upload + score scraping automation
- `submissions_tracker.py` - Track local CV, public, private scores
- `experiment_logger.py` - Git-based reproducibility system

## Architecture

### Four-Layer System

1. **Scripts Layer** (`scripts/`): CLI entry points and orchestration
   - `mla.py`: Orchestrates modular pipeline (EDA → preprocess → model → predict → submit → fetch-score)
   - `submission_workflow.py`: Kaggle upload + Playwright score scraping automation
   - `submissions_tracker.py`: Tracks local CV, public/private scores with git integration
   - `experiment_logger.py`: Logs experiments with git hash, code snapshots, config
   - `kaggle_scraper.py`: Scrapes Kaggle leaderboard/submissions via CDP

2. **Core Package** (`src/kaggle_tools/`): Shared Python package (installed with `uv pip install -e .`)
   - `submission.py`: Universal submission creation/validation logic
   - Importable as `from kaggle_tools import create_submission, validate_submission`
   - Used by all project wrappers

3. **Project Layer** (`projects/kaggle/[competition-name]/`): Individual competition directories
  - `code/utils/config.py`: Competition-specific constants (target, metric, AutoGluon settings)
  - `code/utils/submission.py`: Optional; when absent, runner falls back to the global helper in `src/kaggle_tools/submission.py`
  - `code/models/`: Model implementations
  - `code/exploration/`: EDA scripts
  - `templates/model.yaml`: Experiment templates (model + hyperparameters) - see `docs/configs.md` for detailed structure documentation

4. **Tracking Layer**: Automatic experiment → submission → git linkage
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

# Install Playwright for score scraping (one-time)
uv run playwright install chromium
```

### Modern Workflow (Recommended)

MLArena (`scripts/mla.py`) is the single pipeline entry point. Modules: EDA → preprocess → model → predict → submit → fetch-score.

**1. Download data:**
```bash
cd projects/kaggle/[competition-name]/data
kaggle competitions download -c [competition-name]
unzip [competition-name].zip
cd ../..
```

**2. EDA:**
```bash
uv run python scripts/mla.py eda --project [competition-name] --eda-notes "baseline"
```

**3. Model (AutoGluon via template):**
```bash
uv run python scripts/mla.py model --project [competition-name]     --model-template dev-gpu     --auto-submit     --wait-seconds 45
```
Templates: `fast-cpu`, `dev-cpu`, `dev-gpu`, `best-cpu`, `best-gpu`, `extreme-gpu` (overrides: `--time-limit`, `--preset`, `--use-gpu`).

**4. Predict (if needed):**
```bash
uv run python scripts/mla.py predict --project [competition-name] --experiment-id <exp>
```

**5. Submit / Fetch score:**
```bash
uv run python scripts/mla.py fetch-score --project [competition-name]     --experiment-id <exp>
```

List modules: `uv run python scripts/mla.py modules --project [competition-name]`

### Tracking & Reproducibility

**View submissions:**
```bash
python scripts/submissions_tracker.py --project [competition-name] list
```

**View experiments:**
```bash
python scripts/experiment_logger.py --project [competition-name] list
```

**Reproduce submission:**
```bash
# Method 1: Git checkout
python scripts/submissions_tracker.py --project [competition-name] list  # get git hash
git checkout <GIT_HASH>

# Method 2: Code snapshot restore
python scripts/experiment_logger.py --project [competition-name] restore <EXPERIMENT_ID>
```

## Critical Workflows

### Modular Experiment Pipeline

**Key Concept:** Each experiment is tracked in `experiments/<experiment_id>/state.json` with module-level granularity. Modules can be run independently or resumed from any step.

**Module Lifecycle:**
1. **EDA** - Basic data exploration, generates experiment_id
2. **Model** - Training with AutoGluon, creates submission CSV
3. **Submit** - Uploads to Kaggle via CLI
4. **Fetch-score** - Scrapes public score via Playwright/CDP

**Module States:** `pending` → `running` → `completed` or `failed`

**ALWAYS commit before running experiments** - system captures git hash automatically.

**Typical Flow (MLArena):**
```bash
# 1. EDA creates experiment ID
uv run python scripts/mla.py eda --project playground-series-s5e11

# 2. Model
uv run python scripts/mla.py model --project playground-series-s5e11 \
    --model-template dev-gpu \
    --auto-submit

# 3. Submit/fetch can be run later if needed
uv run python scripts/mla.py fetch-score --project playground-series-s5e11 \
    --experiment-id exp-20251117-020830
```

**Module Safety:**
- Module won't start if already `completed` (use `--force` to override)
- Module won't start if already `running` (prevents parallel runs)
- Failed modules can be retried with same experiment_id

### Legacy Experiment Tracking

When `create_submission()` is called directly (outside pipeline):
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

Preferred: `uv run python scripts/mla.py init --project <competition-slug>` (creates structure, copies templates, downloads data).

If you need a manual fallback:

```bash
COMP_NAME="competition-slug"
mkdir -p projects/kaggle/${COMP_NAME}/{data,code/{exploration,models,utils},submissions,experiments,docs}
touch projects/kaggle/${COMP_NAME}/{data,submissions,experiments}/.gitkeep

# Copy templates:
cp config/templates/kaggle_competition/.gitignore projects/kaggle/${COMP_NAME}/
cp config/templates/kaggle_competition/README.md projects/kaggle/${COMP_NAME}/  # edit competition details
cp config/templates/kaggle_competition/code/utils/*.py projects/kaggle/${COMP_NAME}/code/utils/
cp config/templates/kaggle_competition/code/exploration/01_initial_eda.py projects/kaggle/${COMP_NAME}/code/exploration/

# Edit config.py with competition-specific settings:
# - TARGET_COLUMN
# - AUTOGLUON_PROBLEM_TYPE (binary/regression/multiclass)
# - AUTOGLUON_EVAL_METRIC
# - COMPETITION_NAME
```

Download data:
```bash
cd projects/kaggle/${COMP_NAME}/data
kaggle competitions download -c ${COMP_NAME}
unzip ${COMP_NAME}.zip
cd ../..
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
AUTOGLUON_PRESET = "medium"

# Model settings
RANDOM_SEED = 42
N_FOLDS = 5

# Submission settings
ID_COLUMN = "PassengerId"          # Detected automatically from sample_submission
IGNORED_COLUMNS = ["PassengerId"]  # Always dropped before training
SUBMISSION_PROBAS = False          # False → send class labels (e.g., accuracy), True → send probabilities (e.g., ROC AUC)
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

## Automated Submission & Score Fetching

The pipeline can automatically upload submissions and fetch public scores using Playwright/CDP.

### Setup Chrome Debugging (One-time)
```bash
# Start Chrome with remote debugging
google-chrome --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-debug

# Login to Kaggle in this Chrome instance (stays logged in)
# Keep this Chrome window open when running experiments
```

### Automatic Workflow
```bash
# Model command with auto-submit fetches score automatically
uv run python scripts/mla.py model --project playground-series-s5e11 \
    --model-template dev-gpu \
    --auto-submit \
    --wait-seconds 45
```

**What happens:**
1. Model trains and creates submission CSV
2. CSV uploaded to Kaggle via `kaggle competitions submit`
3. Script waits (default 30s, configurable with `--wait-seconds`)
4. Playwright connects to Chrome and navigates to submissions page
5. Public score scraped from latest submission
6. `submissions/submissions.json` updated with score
7. Git commit created: `submission(project): model | local 0.923 | public 0.922`

### Control Flags
- `--auto-submit` - Skip interactive prompt, submit immediately
- `--skip-submit` - Train only, don't upload to Kaggle
- `--skip-score-fetch` - Upload but don't scrape score (when browser offline)
- `--skip-git` - Don't auto-commit, review changes manually
- `--wait-seconds N` - Wait N seconds before scraping (default: 30)
- `--cdp-url URL` - Optional CDP endpoint (defaults to http://127.0.0.1:9222); only needed if Chrome listens elsewhere.

### Manual Scraping (Standalone)

For existing submissions or manual score updates:

```bash
# Scrape leaderboard/submissions (requires Chrome with CDP)
python scripts/kaggle_scraper.py [competition-name]

# Output: JSON files in projects/kaggle/[competition-name]/data/kaggle_scrapes/
```

See `scripts/README_KAGGLE.md` for details.

## Common Pitfalls

1. **Running experiments without committing** → Missing git hash, can't reproduce
   - System warns but doesn't block
   - Always `git commit` before `python code/models/...`

2. **Wrong TARGET_COLUMN in config.py** → AutoGluon fails
   - Check `sample_submission.csv` column names
   - Binary classification: use `predict_proba(..., as_multiclass=False)`
   - Regression: use `predict()`

3. **Calling tools from wrong directory** → File not found
   - Tools must be called from repo root: `uv run python scripts/...`
   - Or use `cd .. && uv run python scripts/...` from competition dir

4. **Missing submission column name** → Kaggle rejects
   - `submission.py` auto-reads from `sample_submission.csv`
   - Fallback to competition-specific default in config

5. **Skipping EDA when data changed** → Hidden schema drifts
   - `mla init` does not run EDA automatically; run it manually when ready
   - Per-experiment EDA is optional; add `--require-eda` when launching the model module if you want it enforced
   - Re-run `uv run python scripts/mla.py eda --project ...` whenever you materially change the data

6. **Module already completed** → Won't re-run
   - Modules won't execute if already marked `completed`
   - Use `--force` flag to override safety check
   - Or create new experiment_id for clean run

7. **Score fetch fails** → Chrome not running or not logged in
   - Ensure Chrome started with `--remote-debugging-port=9222`
   - Verify logged into Kaggle in that Chrome instance
   - Use `--skip-score-fetch` to skip automation
   - Can resume later with `submission_workflow.py pull-score`

8. **Template confusion** → Wrong compute resources
   - `fast-cpu` is XGBoost-only smoke test (60s), not for final submissions
   - Use `dev-{cpu,gpu}` for iteration, `best-{cpu,gpu}` for serious runs
   - `extreme-gpu` requires confirmation if dataset >30k rows

9. **mla init without Kaggle API** → Download fails
   - Ensure `~/.kaggle/kaggle.json` exists and has correct permissions (chmod 600)
   - Check competition slug is correct (use exact name from Kaggle URL)

10. **Migrating project with .old/ already existing** → Name collision
   - Remove or rename existing `.old/` directory before migration
   - Or manually merge contents if needed

## Development Guidelines

### Coding Style & Naming Conventions

- Use 4-space indentation
- `snake_case` for modules/functions
- `CamelCase` for classes
- Zero-pad exploration scripts (`01_initial_eda.py`) for chronological ordering
- Name outputs `submission-YYYYMMDDHHMM-model.csv` for automatic ingestion by `submissions_tracker.py`
- Centralize constants and random seeds in `code/utils/config.py`
- Keep logging helpers and datapath utilities in `code/utils/` so every project shares conventions

### Testing Guidelines

Rapid validation relies on shared templates:

```bash
# Smoke test in ~60s (XGBoost only) before committing heavier compute
uv run python scripts/mla.py model --project <proj> \
    --model-template fast-cpu --skip-submit
```

**Quality gates:**
- Use `dev-*` or `best-*` templates for longer jobs
- Compare local CV in experiment's `state.json`
- Block merges when metrics deviate by >±0.002 ROC-AUC/RMSE
- Block when leaderboard trend in `submissions/submissions.json` regresses
- All preprocessing/modeling scripts must be deterministic (respect seeds in `config.py`)

### Commit & Pull Request Guidelines

**Commit message format:**
- Prefix with `feat:`, `fix:`, or `experiment:`
- Describe observable change: `"feat: add AutoGluon medium baseline"`
- Example: `"experiment: tune XGBoost hyperparams for s5e11"`

**Pull Request requirements:**
- Link tracked Kaggle issue or discussion
- Include reproduction commands
- Mention relevant tracker entry
- Attach leaderboard evidence for new submissions
- Never include dataset files
- Refresh `uv.lock` when dependencies change

### Security & Configuration

**Kaggle credentials:**
- Keep `~/.kaggle/kaggle.json` local with `chmod 600`
- Reference via environment variables, not hard-coded paths
- Never commit credentials

**Best practices:**
- Document sensitive configs in project-level READMEs
- Scrub notebooks before committing
- Share only minimum per-competition folder with external agents
- Runner only stages active competition directory

## Dependencies

Managed via `pyproject.toml`:
- `kaggle` - CLI for downloads/submissions
- `autogluon` - Primary modeling framework
- `pandas`, `numpy`, `scikit-learn` - Data processing
- `rich` - Console output formatting
- `playwright` - Optional, for Kaggle scraping

Install: `uv sync`

---

## See Also

**Related Documentation:**
- [docs/OPTUNA_GUIDE.md](docs/OPTUNA_GUIDE.md) - Complete guide to hyperparameter tuning, feature engineering, and ensembling
- [docs/configs.md](docs/configs.md) - Detailed template configuration reference (model.yaml, preprocess.yaml structure)
- [scripts/README.md](scripts/README.md) - Comprehensive scripts documentation with usage examples
- [README.md](README.md) - Main repository overview and quick start guide

**Design Documents:**
- [docs/ml_code_separation_design_v2.md](docs/ml_code_separation_design_v2.md) - ML code architecture and separation patterns
- [docs/template_system_redesign.md](docs/template_system_redesign.md) - Template system design and roadmap
- [docs/template_merge_guidelines.md](docs/template_merge_guidelines.md) - Global vs project template resolution
