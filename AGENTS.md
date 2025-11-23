# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Kaggle competitions repository with standardized structure, experiment tracking, and reproducibility system. Uses `uv` for dependency management, AutoGluon for baseline models, and custom tracking tools for submissions/experiments.

## Quick Reference

**Most common workflow (recommended):**
```bash
# 1. Setup (once)
uv sync
uv run playwright install chromium

# 2. Initialize new competition (creates structure + downloads data)
uv run python scripts/experiment_manager.py init-project --project [competition-name]
# Auto-detects target column and prompts for problem type/metric

# 3. Run complete pipeline
uv run python scripts/experiment_manager.py eda --project [competition-name]
# Note the experiment_id from output

uv run python scripts/experiment_manager.py model \
    --project [competition-name] \
    --experiment-id exp-YYYYMMDD-HHMMSS \
    --template dev-gpu \
    --auto-submit \
    --wait-seconds 45

# 4. Check results
uv run python scripts/experiment_manager.py list --project [competition-name]
uv run python scripts/submissions_tracker.py --project [competition-name] list
```

**Key tools:**
- `experiment_manager.py` - Modular pipeline orchestration (EDA → Model → Submit → Fetch)
- `autogluon_runner.py` - Template-based AutoGluon training
- `submission_workflow.py` - Kaggle upload + score scraping automation
- `submissions_tracker.py` - Track local CV, public, private scores
- `experiment_logger.py` - Git-based reproducibility system

## Architecture

### Four-Layer System

1. **Scripts Layer** (`scripts/`): CLI entry points and orchestration
   - `experiment_manager.py`: Orchestrates modular pipeline (EDA → Model → Submit), project initialization
   - `autogluon_runner.py`: Template-based AutoGluon training (fast-cpu, dev-gpu, best-gpu, etc.)
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
   - `code/utils/submission.py`: Lightweight wrapper that injects config into `src/kaggle_tools`
   - `code/models/`: Model implementations
   - `code/exploration/`: EDA scripts
   - `configs/templates.yaml`: Experiment templates (model + hyperparameters) - see [configs/README.md](projects/kaggle/playground-series-s5e11/configs/README.md) for detailed structure documentation

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

### Metric Detection (CDP)
- AI-based metric detection (`init-project`, `detect-metric`) now scrapes the Evaluation section directly from Kaggle via Playwright + Chrome CDP.
- Start Chrome with `--remote-debugging-port=9222` (and stay logged into Kaggle) before running those commands.
- Configure the endpoint via `KAGGLE_CDP_URL` or pass `--cdp-url http://127.0.0.1:9222`; without a reachable CDP session the detection step aborts (no sample_submission guessing fallback).

### Modern Workflow (Recommended)

The repository uses a **modular experiment pipeline** managed by `scripts/experiment_manager.py`. Each experiment progresses through modules: EDA → Model → Submit → Fetch-score.

**1. Download data:**
```bash
cd projects/kaggle/[competition-name]/data
kaggle competitions download -c [competition-name]
unzip [competition-name].zip
cd ../..
```

**2. Start experiment with EDA:**
```bash
uv run python scripts/experiment_manager.py eda \
    --project [competition-name] \
    --notes "baseline exploration"
# Outputs: experiment_id (e.g., exp-20251117-020830)
```

**3. Train model using templates:**
```bash
uv run python scripts/experiment_manager.py model \
    --project [competition-name] \
    --experiment-id exp-20251117-020830 \
    --template dev-gpu \
    --auto-submit \
    --wait-seconds 45
```

**AutoGluon Templates** (replaces manual time-limit/preset configuration):
| Template       | Time  | Preset           | GPU | Use Case |
|----------------|------:|------------------|-----|----------|
| `fast-cpu`     | 60s   | `medium_quality` | ❌  | XGBoost-only smoke test |
| `dev-cpu`      | 300s  | `medium_quality` | ❌  | Quick iteration |
| `dev-gpu`      | 300s  | `medium_quality` | ✅  | Quick iteration (GPU) |
| `best-cpu`     | 1h    | `best_quality`   | ❌  | High-quality ensemble |
| `best-gpu`     | 1h    | `best_quality`   | ✅  | High-quality ensemble (GPU) |
| `extreme-gpu`  | 24h   | `extreme_quality`| ✅  | Max quality (prompts if >30k rows) |

Template overrides available: `--time-limit`, `--preset`, `--use-gpu 0/1`

**Custom Templates**: Projects can define custom templates in `configs/templates.yaml` combining specific models with hyperparameters. See [configs/README.md](projects/kaggle/playground-series-s5e11/configs/README.md) for complete structure and examples (feature engineering variants, excluded models, custom parameters, etc.)

**4. Resume/fetch score later (if browser offline):**
```bash
# Start Chrome with debugging port
google-chrome --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-debug

# Resume and fetch score
uv run python scripts/submission_workflow.py resume \
    --project [competition-name] \
    --filename submission-20251117015359.csv \
    --experiment-id exp-20251117-020830 \
    --cdp-url http://127.0.0.1:9222
```

**5. View experiment status:**
```bash
# List all experiments with module status
uv run python scripts/experiment_manager.py list --project [competition-name]

# Show available modules
uv run python scripts/experiment_manager.py modules
```

### Alternative: Direct Runner (Without Experiment Manager)

Use `scripts/autogluon_runner.py` directly for quick iterations:

```bash
# Direct runner with template
uv run python scripts/autogluon_runner.py \
    --project [competition-name] \
    --template best-gpu \
    --auto-submit \
    --wait-seconds 45

# Or with manual parameters (overrides template)
uv run python scripts/autogluon_runner.py \
    --project [competition-name] \
    --time-limit 1800 \
    --preset high_quality \
    --use-gpu 1 \
    --skip-submit
```

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

**Typical Flow:**
```bash
# 1. EDA creates experiment ID
uv run python scripts/experiment_manager.py eda --project playground-series-s5e11

# 2. Model (optionally enforce EDA with --require-eda)
uv run python scripts/experiment_manager.py model \
    --project playground-series-s5e11 \
    --experiment-id exp-20251117-020830 \
    --template dev-gpu

# 3. Submit/fetch can be run later if needed
uv run python scripts/submission_workflow.py resume \
    --project playground-series-s5e11 \
    --filename submission-20251117015359.csv \
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

**Recommended: Use init-project command**

```bash
# New competition (downloads data automatically)
uv run python scripts/experiment_manager.py init-project \
    --project competition-slug \
    --target-column target_name \
    --problem-type binary \
    --metric roc_auc

# Or let it auto-detect and prompt interactively
uv run python scripts/experiment_manager.py init-project \
    --project competition-slug

# Migrate old project to new structure
uv run python scripts/experiment_manager.py init-project \
    --project playground-series-s4e6 \
    --migrate
```

**What it does:**
1. Creates standard directory structure (code/, data/, docs/, experiments/, submissions/)
2. Copies template files from `config/templates/kaggle_competition/`
3. Downloads and extracts data from Kaggle (unless `--skip-download`)
4. Auto-detects target column from sample_submission.csv
5. Customizes config.py and README.md
6. Creates .gitignore and .gitkeep files

**Flags:**
- `--migrate`: Move existing project files to `.old/` before initialization
- `--target-column NAME`: Specify target column (auto-detected if sample_submission.csv exists)
- `--problem-type {binary,regression,multiclass}`: Problem type (prompted if not provided)
- `--metric NAME`: Evaluation metric (defaults based on problem type)
- `--skip-download`: Don't download data from Kaggle
- `--keep-zip`: Keep zip file after extraction

**Manual method (legacy):**

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
uv run python scripts/experiment_manager.py model \
    --project playground-series-s5e11 \
    --experiment-id exp-20251117-020830 \
    --template dev-gpu \
    --auto-submit \
    --wait-seconds 45 \
    --cdp-url http://127.0.0.1:9222
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
- `--cdp-url URL` - Custom Chrome debug endpoint (default: http://127.0.0.1:9222)

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
   - `init-project` automatically generates baseline train/test reports under `experiments/init`
   - Per-experiment EDA is optional; add `--require-eda` when launching the model module if you want it enforced
   - Re-run `uv run python scripts/experiment_manager.py eda --project ...` whenever you materially change the data

6. **Module already completed** → Won't re-run
   - Modules won't execute if already marked `completed`
   - Use `--force` flag to override safety check
   - Or create new experiment_id for clean run

7. **Score fetch fails** → Chrome not running or not logged in
   - Ensure Chrome started with `--remote-debugging-port=9222`
   - Verify logged into Kaggle in that Chrome instance
   - Use `--skip-score-fetch` to skip automation
   - Can resume later with `submission_workflow.py resume`

8. **Template confusion** → Wrong compute resources
   - `fast-cpu` is XGBoost-only smoke test (60s), not for final submissions
   - Use `dev-{cpu,gpu}` for iteration, `best-{cpu,gpu}` for serious runs
   - `extreme-gpu` requires confirmation if dataset >30k rows

9. **init-project without Kaggle API** → Download fails
   - Ensure `~/.kaggle/kaggle.json` exists and has correct permissions (chmod 600)
   - Use `--skip-download` to create structure only, download data manually later
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
uv run python scripts/experiment_manager.py model --project <proj> \
    --template fast-cpu --skip-submit
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
