# Kaggle Competitions Repository

Repository for managing Kaggle competition projects with standardized structure and workflows.

## Table of Contents
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Project Structure](#project-structure)
- [Creating a New Competition Project](#creating-a-new-competition-project)
- [Workflow](#workflow)
- [Optuna Hyperparameter Tuning System](#optuna-hyperparameter-tuning-system)
- [Submission Tracking](#submission-tracking)
- [Best Practices](#best-practices)
- [Utilities](#utilities)

## Quick Start

### Prerequisites
- Python environment with `uv` package manager
- Kaggle CLI configured with API credentials
- Required packages: pandas, numpy, scikit-learn, autogluon, rich

### Setup Kaggle API
```bash
# Kaggle API credentials should be in ~/.kaggle/kaggle.json
# If not, create it with your credentials from kaggle.com/account

mkdir -p ~/.kaggle
# Copy your kaggle.json to ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

## Repository Structure

```
competitions/
├── scripts/                     # Universal tools for all competitions
│   ├── submissions_tracker.py  # Tracks local CV, public, private scores
│   └── README.md               # Tools documentation
├── [competition-name]/         # Individual competition directories
│   └── ...                     # (see Project Structure below)
├── README.md                   # This file
└── pyproject.toml             # Python dependencies
```

## Project Structure

Each competition directory mirrors the same minimal layout so the shared tooling works everywhere:

```
competition-name/
├── README.md                # Competition-specific notes
├── data/                    # Raw Kaggle files (train.csv, test.csv, sample_submission.csv)
├── code/
│   └── utils/
│       ├── config.py        # Declares paths, metric, target column, Kaggle slug
│       └── submission.py    # Thin wrapper around the shared submission helpers
├── experiments/
│   └── exp-*/               # One folder per experiment with state.json + module artefacts
├── submissions/             # Generated CSVs + submissions.json tracker
└── docs/                    # Optional notebooks/notes
```

`scripts/` at the repository root contains the reusable tools. **MLArena** is the single entry point (`mla.py`) that replaces the legacy `experiment_manager.py` / `ml_runner.py` / `autogluon_runner.py`. Competition folders only hold lightweight configuration and any extra scripts that are truly competition-specific. Large artefacts (models, downloaded zips) stay ignored via each project's `.gitignore`.

`code/utils/config.py` is the only file you normally edit per competition. The init script fills in:
- Data paths (`TRAIN_PATH`, `TEST_PATH`, `SAMPLE_SUBMISSION_PATH`) and Kaggle metadata (`COMPETITION_NAME`, `METRIC`).
- Experiment knobs (`RANDOM_SEED`, `N_FOLDS`, AutoGluon preset/problem type/time limit).
- Submission wiring:
  - `ID_COLUMN` and `IGNORED_COLUMNS` are inferred from `*submission*.csv` so every model drops the identifier before training but reuses it for Kaggle uploads.
  - `SUBMISSION_PROBAS` signals whether the submission should contain probabilities (`True`, e.g., ROC AUC) or discrete labels (`False`, e.g., accuracy). This is auto-detected from the Evaluation text but can be overridden manually.

All shared tools (runner, submissions tracker, validation) rely on these fields, so keep them in sync with the competition rules.

## Global vs. project templates and code
- Templates merge global files in `config/templates/*.yaml` with project files in `projects/kaggle/<proj>/templates/*.yaml`; when names collide, the project entry wins (a warning is printed).
- Model modules load from `projects/kaggle/<proj>/code/models/` or `config/code/models/`. If the same filename exists in both, the runner fails (no shadowing). Use unique names for project-only variants; edit globals only for cross-project changes.
- Preprocessing modules load from `projects/kaggle/<proj>/code/preprocessing/` or `config/code/preprocessing/` (e.g., global `identity.py`). Same-name clashes also fail; keep shared utilities global and add custom pipelines locally under a distinct name.

## Creating a New Competition Project

1. Accept the rules on Kaggle and grab the competition slug.
2. Create a project with the init script or copy any existing one, then remove its submissions/experiments.
3. Edit `code/utils/config.py` to set `COMPETITION_NAME`, metric, and target column; drop the downloaded CSV files into `data/`.
4. Run `uv run python scripts/mla.py eda --project <project>` to confirm the loader works. From now on, rely on the shared MLArena runner/templates instead of bespoke scripts.

`uv sync` installs all dependencies; no per-project virtualenvs are needed.

## Workflow

Every experiment is broken into modules tracked in `experiments/<id>/state.json`. Use **MLArena** (`scripts/mla.py`) to orchestrate the flow:

1. **EDA** – `uv run python scripts/mla.py eda --project <project>`. Prints the shapes/target distribution and records the ID (e.g., `exp-20251117-020830`).
2. **Model (train)** – `uv run python scripts/mla.py model --project <project> --experiment-id exp-20251117-020830 --model-template dev-gpu [--auto-submit]`. Templates map to AutoGluon presets/time limits; overrides like `--time-limit`, `--preset`, `--use-gpu`, or `--skip-submit` are available per module.
3. **Predict** – `uv run python scripts/mla.py predict --project <project> --experiment-id exp-... [--predict-suffix foo]` reuses the trained model to generate a submission. The `model` command runs this automatically unless you stop at training.
4. **Submit / Fetch Score** – either let the runner auto-submit, or call `uv run python scripts/mla.py submit --project <project> --experiment-id exp-...` to upload an existing CSV. `fetch-score` re-scrapes Kaggle later via Playwright/CDP if the browser was offline during training.

Use `uv run python scripts/mla.py modules --project <project>` to list modules.

### Submission Automation

`scripts/submission_workflow.py` handles Kaggle CLI uploads, waits (`--wait-seconds`), connects to the already-running Chrome via `--cdp-url`, grabs the leaderboard row, updates `submissions/submissions.json`, and commits the code/artefacts. Flags such as `--auto-submit`, `--skip-submit`, `--skip-score-fetch`, and `--skip-git` give fine-grained control over each run.

### Troubleshooting Templates

`fast-cpu` is a 60-second XGBoost-only smoke test that is ideal for verifying code paths before launching a long run. `extreme-gpu` enforces a confirmation prompt if the training set exceeds 30k rows, preventing accidental day-long jobs on gigantic data. When AutoGluon raises (e.g., invalid hyperparameter), MLArena records the module as `failed`, allowing you to rerun the step with the same ID once the issue is fixed.

## Optuna Hyperparameter Tuning System

The repository includes a comprehensive Optuna-based system for hyperparameter tuning, feature engineering, and model ensembling. This system provides:

- **Feature Engineering** - Transform features with data leakage protection (two-stage pipeline)
- **Hyperparameter Tuning** - Optimize XGBoost/LightGBM/CatBoost with Optuna TPE sampler
- **Model Ensembling** - Blend predictions using weighted/rank/power averaging or meta-learning

### Quick Start

```bash
# 1. Tune XGBoost (100 trials, 2h)
uv run python scripts/optuna_runner.py \
    --project <project> \
    --model xgboost \
    --preset thorough

# 2. Train with best params via MLArena
uv run python scripts/mla.py model --project <project> \
    --model-template dev-gpu \
    --time-limit 7200 \
    --preset high

# 3. Ensemble multiple models
uv run python scripts/stacking_runner.py \
    --project <project> \
    --models xgb.csv lgb.csv cat.csv \
    --blend-method weighted \
    --blend-weights 0.5 0.3 0.2
```

### Key Features

**Data Leakage Protection:**
- `feat_stage` - Global transformers (fitted on full train, safe)
- `cv_stage` - Per-fold transformers (fitted per CV fold, prevents leakage)

**Hyperparameter Tuning:**
- Three presets: `quick` (20 trials, 30min), `thorough` (100 trials, 2h), `extreme` (500 trials, 24h)
- SQLite persistence for resume support
- Optuna dashboard for visualization (http://localhost:8080)
- Early stopping and pruning for efficiency

**Model Ensembling:**
- Weighted blending (manual or optimized weights)
- Rank averaging (robust to outliers)
- Power averaging (emphasizes confident predictions)
- Meta-learning stacking (future enhancement)

### Documentation

- **Complete Guide:** [docs/OPTUNA_GUIDE.md](docs/OPTUNA_GUIDE.md) - Full documentation with quick start and examples
- **Unit Tests:** `tests/test_data_leakage.py`, `tests/test_optuna_e2e.py`

### Architecture

```
ExperimentManager
  ├── feat   → feature_runner.py → FeaturePipeline
  ├── tune   → optuna_runner.py → StudyManager + CVObjective
  ├── model  → Train with best params (xgboost_optuna.py, etc.)
  └── stack  → stacking_runner.py → Blenders + MetaLearner
```

## Submission Tracking

`submissions/submissions.json` stores every Kaggle upload along with local CV, public score, experiment ID, git hash, and optional notes. Normally you do not add entries manually: `code/utils/submission.py` logs the experiment + tracker entry whenever a CSV is created, and `submission_workflow.py` updates the public score/commit message after scraping Kaggle. When a fix is required, the CLI mirror still exists:

```bash
uv run python scripts/submissions_tracker.py --project <project> list
uv run python scripts/submissions_tracker.py --project <project> add submission.csv autogluon-medium --local-cv 0.92
uv run python scripts/submissions_tracker.py --project <project> update 3 --public 0.9213
```

Pair this with `scripts/experiment_logger.py` if you need to inspect the git state or restore a code snapshot referenced by a tracker entry.

## Utilities

### Experiment Tracking Tools

**scripts/experiment_logger.py** - Complete experiment tracking system

```bash
# List experiments
python scripts/experiment_logger.py --project PROJECT_NAME list [--limit N]

# Show experiment details (git, config, code)
python scripts/experiment_logger.py --project PROJECT_NAME show EXPERIMENT_ID

# Restore code from experiment
python scripts/experiment_logger.py --project PROJECT_NAME restore EXPERIMENT_ID [--output PATH]

# Get git checkout instructions
python scripts/experiment_logger.py --project PROJECT_NAME checkout EXPERIMENT_ID
```

**scripts/submissions_tracker.py** - Track submissions with scores

```bash
# Add submission manually (usually automatic via create_submission())
python scripts/submissions_tracker.py --project PROJECT_NAME add \
    submission.csv model-name \
    --local-cv 0.85 --notes "baseline"

# Update public/private scores
python scripts/submissions_tracker.py --project PROJECT_NAME update SUBMISSION_ID \
    --public 0.84 --private 0.83

# List submissions (with git & experiment info)
python scripts/submissions_tracker.py --project PROJECT_NAME list \
    [--sort-by public_score] [--limit N]

# Export to CSV
python scripts/submissions_tracker.py --project PROJECT_NAME export
```

### Kaggle CLI Common Commands

```bash
# List competitions
kaggle competitions list

# Download competition data
kaggle competitions download -c competition-name

# Submit to competition
kaggle competitions submit -c competition-name -f submission.csv -m "Message"

# View leaderboard
kaggle competitions leaderboard competition-name

# View submissions
kaggle competitions submissions competition-name
```

### AutoGluon Quick Start

```python
from autogluon.tabular import TabularPredictor

# Train model
predictor = TabularPredictor(
    label='target_column',
    eval_metric='rmse',  # or other metric
    problem_type='regression'  # or 'binary', 'multiclass'
)

predictor.fit(
    train_data=train_df,
    time_limit=3600,  # seconds
    presets='medium'  # or 'good', 'high', 'best', 'extreme'
)

# Make predictions
predictions = predictor.predict(test_df)
```

### Git Workflow

```bash
# Before starting new work
git status  # Check for uncommitted changes
git add .
git commit -m "Description of changes"

# After experiments
git add code/  # Only commit code, not data
git commit -m "Add feature engineering for XYZ"
git push
```

## MLArena Templates

MLArena uses simple template names aligned with AutoGluon presets/time limits (see `config/templates/model*.json` for specifics). Common presets:

- `fast-cpu` (≈60s, smoke test)
- `dev-cpu` / `dev-gpu` (≈5 min iteration)
- `best-cpu` / `best-gpu` (≈1h quality ensemble)
- `extreme-gpu` (24h foundation-model stack; requires GPU)

Override per run: `--time-limit`, `--preset`, `--use-gpu`, `--model-template <name>`.

## MLArena Experiment Workflow

1) **EDA**: `uv run python scripts/mla.py eda --project <project>`

2) **Model**: `uv run python scripts/mla.py model --project <project> --model-template dev-gpu --auto-submit`

3) **Predict** (if needed): `uv run python scripts/mla.py predict --project <project> --experiment-id <exp>`

4) **Submit / Fetch score**: `uv run python scripts/mla.py fetch-score --project <project> --experiment-id <exp> --cdp-url http://127.0.0.1:9222`

State lives in `projects/kaggle/<project>/experiments/<exp>/state.json`; rerun modules with `--force` to overwrite completed steps.

## Active Competitions

| Competition | Deadline | Status | Best Score | Notes |
|-------------|----------|--------|------------|-------|
| <project> | TBD | In Progress | - | - |
| melting-point | TBD | In Progress | - | - |

## Resources

- [Kaggle API Documentation](https://github.com/Kaggle/kaggle-api)
- [AutoGluon Documentation](https://auto.gluon.ai/)
- [Rich Documentation](https://rich.readthedocs.io/)
- [Pandas Documentation](https://pandas.pydata.org/)
