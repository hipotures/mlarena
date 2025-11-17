# Kaggle Competitions Repository

Repository for managing Kaggle competition projects with standardized structure and workflows.

## Table of Contents
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Project Structure](#project-structure)
- [Creating a New Competition Project](#creating-a-new-competition-project)
- [Workflow](#workflow)
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
├── tools/                       # Universal tools for all competitions
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

`tools/` at the repository root contains the reusable runners (`autogluon_runner.py`, `experiment_manager.py`, `submission_workflow.py`, etc.). Competition folders only hold lightweight configuration and any extra scripts that are truly competition-specific. Large artefacts (models, downloaded zips) stay ignored via each project's `.gitignore`.

`code/utils/config.py` is the only file you normally edit per competition. The init script fills in:
- Data paths (`TRAIN_PATH`, `TEST_PATH`, `SAMPLE_SUBMISSION_PATH`) and Kaggle metadata (`COMPETITION_NAME`, `METRIC`).
- Experiment knobs (`RANDOM_SEED`, `N_FOLDS`, AutoGluon preset/problem type/time limit).
- Submission wiring:
  - `ID_COLUMN` and `IGNORED_COLUMNS` are inferred from `*submission*.csv` so every model drops the identifier before training but reuses it for Kaggle uploads.
  - `SUBMISSION_PROBAS` signals whether the submission should contain probabilities (`True`, e.g., ROC AUC) or discrete labels (`False`, e.g., accuracy). This is auto-detected from the Evaluation text but can be overridden manually.

All shared tools (runner, submissions tracker, validation) rely on these fields, so keep them in sync with the competition rules.

## Creating a New Competition Project

1. Accept the rules on Kaggle and grab the competition slug.
2. Copy the skeleton from an existing project (`cp -R playground-series-s5e11 NEW-COMP && rm -rf NEW-COMP/submissions/*.csv NEW-COMP/experiments`).
3. Edit `code/utils/config.py` to set `COMPETITION_NAME`, metric, and target column; drop the downloaded CSV files into `data/`.
4. Run `uv run python tools/experiment_manager.py eda --project NEW-COMP` to confirm the loader works. From now on, rely on the shared runner/templates instead of bespoke scripts.

`uv sync` installs all dependencies; no per-project virtualenvs are needed.

## Workflow

Every experiment is broken into modules tracked in `experiments/<id>/state.json`. Use `tools/experiment_manager.py` to orchestrate the flow:

1. **EDA** – `uv run python tools/experiment_manager.py eda --project playground-series-s5e11`. Prints the shapes/target distribution and records the ID (e.g., `exp-20251117-020830`).
2. **Model** – `uv run python tools/experiment_manager.py model --project playground-series-s5e11 --experiment-id exp-20251117-020830 --template dev-gpu [--auto-submit]`. This wraps `tools/autogluon_runner.py`, which supports templates such as `fast-cpu`, `dev-{cpu,gpu}`, `best-{cpu,gpu}`, and `extreme-gpu` (24h, prompts if >30k rows). Overrides like `--time-limit`, `--preset`, `--use-gpu`, or `--skip-submit` are available when needed.
3. **Submit / Fetch Score** – either let the runner auto-submit, or call `uv run python tools/experiment_manager.py submit --project playground-series-s5e11 --experiment-id exp-...` to upload an existing CSV. `fetch-score` re-scrapes Kaggle later via Playwright/CDP if the browser was offline during training.

Use `uv run python tools/experiment_manager.py list --project playground-series-s5e11` to inspect module statuses, and `uv run python tools/experiment_manager.py modules` to show the available module names.

### Submission Automation

`tools/submission_workflow.py` handles Kaggle CLI uploads, waits (`--wait-seconds`), connects to the already-running Chrome via `--cdp-url`, grabs the leaderboard row, updates `submissions/submissions.json`, and commits the code/artefacts. Flags such as `--auto-submit`, `--skip-submit`, `--skip-score-fetch`, and `--skip-git` give fine-grained control over each run.

### Troubleshooting Templates

`fast-cpu` is a 60-second XGBoost-only smoke test that is ideal for verifying code paths before launching a long run. `extreme-gpu` enforces a confirmation prompt if the training set exceeds 30k rows, preventing accidental day-long jobs on gigantic data. When AutoGluon raises (e.g., invalid hyperparameter), the experiment manager records the module as `failed`, allowing you to rerun the step with the same ID once the issue is fixed.

## Submission Tracking

`submissions/submissions.json` stores every Kaggle upload along with local CV, public score, experiment ID, git hash, and optional notes. Normally you do not add entries manually: `code/utils/submission.py` logs the experiment + tracker entry whenever a CSV is created, and `submission_workflow.py` updates the public score/commit message after scraping Kaggle. When a fix is required, the CLI mirror still exists:

```bash
uv run python tools/submissions_tracker.py --project playground-series-s5e11 list
uv run python tools/submissions_tracker.py --project playground-series-s5e11 add submission.csv autogluon-medium --local-cv 0.92
uv run python tools/submissions_tracker.py --project playground-series-s5e11 update 3 --public 0.9213
```

Pair this with `tools/experiment_logger.py` if you need to inspect the git state or restore a code snapshot referenced by a tracker entry.

## Utilities

### Experiment Tracking Tools

**tools/experiment_logger.py** - Complete experiment tracking system

```bash
# List experiments
python tools/experiment_logger.py --project PROJECT_NAME list [--limit N]

# Show experiment details (git, config, code)
python tools/experiment_logger.py --project PROJECT_NAME show EXPERIMENT_ID

# Restore code from experiment
python tools/experiment_logger.py --project PROJECT_NAME restore EXPERIMENT_ID [--output PATH]

# Get git checkout instructions
python tools/experiment_logger.py --project PROJECT_NAME checkout EXPERIMENT_ID
```

**tools/submissions_tracker.py** - Track submissions with scores

```bash
# Add submission manually (usually automatic via create_submission())
python tools/submissions_tracker.py --project PROJECT_NAME add \
    submission.csv model-name \
    --local-cv 0.85 --notes "baseline"

# Update public/private scores
python tools/submissions_tracker.py --project PROJECT_NAME update SUBMISSION_ID \
    --public 0.84 --private 0.83

# List submissions (with git & experiment info)
python tools/submissions_tracker.py --project PROJECT_NAME list \
    [--sort-by public_score] [--limit N]

# Export to CSV
python tools/submissions_tracker.py --project PROJECT_NAME export
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

### Metric Detection via CDP
- Commands such as `uv run python scripts/experiment_manager.py detect-metric` or `init-project` scrape the Evaluation section directly from the Kaggle overview page.
- Keep Chrome running with `--remote-debugging-port=9222` (and logged into Kaggle) before invoking them.
- Configure the connection through `KAGGLE_CDP_URL` or `--cdp-url http://127.0.0.1:9222`; without a reachable CDP endpoint the detection step will exit instead of guessing from `sample_submission.csv`.

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
    presets='medium_quality'  # or 'best_quality', 'high_quality'
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

### AutoGluon Runner Templates

Every competition now shares a single runner: `tools/autogluon_runner.py`. Call it directly or via the thin wrappers in each project (e.g., `uv run python playground-series-s5e11/code/models/baseline_autogluon.py`), and pass a compute template instead of memorising raw parameters:

| Template     | Time Limit | Preset           | GPU | Notes |
|--------------|-----------:|------------------|-----|-------|
| `fast-cpu`   | 60 s       | `medium_quality` | ❌  | XGBoost only, smoke tests |
| `dev-cpu`    | 300 s      | `medium_quality` | ❌  | default stack |
| `dev-gpu`    | 300 s      | `medium_quality` | ✅  | default stack |
| `best-cpu`   | 3600 s     | `best_quality`   | ❌  | high-quality ensemble |
| `best-gpu`   | 3600 s     | `best_quality`   | ✅  | high-quality ensemble |
| `extreme-gpu`| 24 h       | `extreme_quality`| ✅  | ≤30k rows, prompts before run |

Example:

```bash
uv run python tools/autogluon_runner.py \
    --project playground-series-s5e11 \
    --template best-gpu \
    --auto-submit \
    --wait-seconds 45
```

`fast-cpu` is intended purely for smoke testing—it limits AutoGluon to a single XGBoost learner for ~60 seconds. `extreme-gpu` prompts with the training-row count if the dataset exceeds 30k rows so you can abort before launching a marathon job. Overrides such as `--time-limit`, `--preset`, or `--use-gpu 0/1` are available when needed.

## Experiment Workflow

Each run is tracked in `competitions/<project>/experiments/<experiment_id>.json`. Modules append their own sections, so możesz odpalać je niezależnie lub wznawiać od dowolnego miejsca:

1. **EDA** – uruchamia podstawową analizę i rejestruje identyfikator:
   ```bash
   uv run python tools/experiment_manager.py eda \
       --project playground-series-s5e11 \
       --notes "baseline sweep"
   ```
   Komunikat pokaże `experiment_id` w formacie `exp-YYYYMMDD-HHMMSS`.

2. **Model** – przekazujesz ten sam identyfikator:
   ```bash
   uv run python tools/autogluon_runner.py \
       --project playground-series-s5e11 \
       --template dev-gpu \
       --experiment-id exp-20251117-011230 \
       --auto-submit \
       --wait-seconds 45
   ```
   Runner zweryfikuje moduł EDA tylko wtedy, gdy dodasz `--require-eda` (domyślnie pomija ten krok).

3. **Submit / Resume** – gdy CSV jest już wysłany, pobierasz skorę i aktualizujesz zarówno tracker, jak i eksperyment:
   ```bash
   uv run python tools/submission_workflow.py resume \
       --project playground-series-s5e11 \
       --filename submission-20251117015359.csv \
       --experiment-id exp-20251117-011230 \
       --cdp-url http://127.0.0.1:9222
   ```

Podgląd stanu:

```bash
uv run python tools/experiment_manager.py list --project playground-series-s5e11
```
Need a reminder of available stages? `uv run python tools/experiment_manager.py modules`.

## Automated Submission Workflow

Model scripts now embed an optional end-to-end pipeline that creates the Kaggle submission, waits for scoring, fetches the public score via Playwright, updates tracking files, and commits the result.

```bash
uv run python playground-series-s5e11/code/models/baseline_autogluon.py \
    --auto-submit \
    --kaggle-message "autogluon-medium exp-2" \
    --wait-seconds 45
```

What happens:
1. The CSV emitted by `create_submission()` is uploaded with the Kaggle CLI.
2. The runner sleeps (default 30s) so Kaggle can evaluate the submission.
3. Playwright connects to your existing Chrome session (`google-chrome --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-debug`) and reads the newest entry on `/submissions`. Install the browser driver once with `uv run playwright install chromium`.
4. `SubmissionsTracker` links the fetched public score to the tracker entry/experiment ID.
5. A git commit is created inside `competitions/` tying the code + local CV + public score (`submission(playground-series-s5e11): autogluon-medium | local 0.92379 | public 0.92227`).

Flags to know: `--skip-submit` (train only), `--auto-submit` (skip the confirmation prompt), `--skip-score-fetch` (useful when Chrome/CDP isn't running), `--skip-git` (review and commit manually), `--cdp-url` (point to a custom debug endpoint).

## Active Competitions

| Competition | Deadline | Status | Best Score | Notes |
|-------------|----------|--------|------------|-------|
| playground-series-s5e11 | TBD | In Progress | - | - |
| melting-point | TBD | In Progress | - | - |

## Resources

- [Kaggle API Documentation](https://github.com/Kaggle/kaggle-api)
- [AutoGluon Documentation](https://auto.gluon.ai/)
- [Rich Documentation](https://rich.readthedocs.io/)
- [Pandas Documentation](https://pandas.pydata.org/)
