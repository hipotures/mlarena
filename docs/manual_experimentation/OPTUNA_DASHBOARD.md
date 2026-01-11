# Optuna Live Dashboard & Trial Structure

## Overview
The Optuna Live Dashboard (`mla pre tune --optuna-live`) provides a real-time view of running hyperparameter optimization trials. It visualizes the preprocessing and modeling pipeline for each trial, allowing deep inspection of intermediate states, data shapes, and artifacts.

## Directory Structure
An Optuna study is stored in an experiment directory (e.g., `experiments/optuna_study_v1/`). Each trial within the study has its own subdirectory (e.g., `trial_0001`, `trial_0002`).

Inside a trial directory:
```
trial_XXXX/
├── trial_payload.json      # Initial configuration and parameters for the trial
├── trial_pipeline.yaml     # Defines the sequence of steps (modules) to run
├── metrics.json            # Final metrics (e.g., AUC, RMSE) after modeling
├── 0-sanity_check/         # Step 0: Sanity Check
│   ├── state.json          # State metadata for this step
│   └── artifacts/          # Generated files (logs, summaries)
├── 1-train_fraction/       # Step 1: Train/Validation Split
│   ├── state.json
│   └── artifacts/
├── 2-target_transformer/   # Step 2: Target Transformation
│   ├── state.json
│   └── artifacts/
├── ...
└── model/                  # Modeling step
    ├── state.json
    └── artifacts/
```

## `state.json` Schema
Each step's `state.json` contains critical metadata about the execution of that specific module.

### Key Fields
- **`name`**: Module name (e.g., `preprocess`, `model`).
- **`status`**: Execution status (`completed`, `failed`, `running`).
- **`started_at`**, **`finished_at`**: Timestamps.
- **`payload`**: Module-specific output data.
    - **`shapes`**: Dimensions of datasets *before* and *after* transformation.
        - `train_before`, `train_after`: `[rows, cols]`
        - `test_before`, `test_after`: `[rows, cols]`
    - **`files`**: Paths to processed artifacts (e.g., `train_processed.csv.gz`).
    - **`config`**: Effective configuration used for this step.

## Dashboard Features

### 1. Study Overview (Dashboard Tab)
- **Top Trials**: Table of best-performing trials with their params and scores.
- **Running Trials**: List of currently active trials.
- **Telegram Integration**: Notifications on new best scores.

### 2. Trial Inspector (Drill-down)
Clicking a trial opens a detailed tree view of its pipeline:
- **Visual Tree**: Represents the sequence of preprocessing steps.
- **Status Indicators**: ✅ Completed, ⏳ Running, ❌ Failed.
- **Data Shapes**: Shows how dataset dimensions change at each step (e.g., `(1000, 20) -> (1000, 25)`).
- **Artifacts**: Lists generated files.

## Cleaning Zombie Trials

If an optimization process is interrupted (e.g., system crash, worker killed), some trials may remain stuck in the `RUNNING` state in the database. These "zombie" trials can interfere with dashboard statistics and future runs.

You can use the helper script `scripts/optuna_clean_zombie_running.py` to mark these stale trials as `FAIL`.

### Usage Example

```bash
# Preview trials that haven't finished for more than 60 minutes
python scripts/optuna_clean_zombie_running.py \
  --db projects/kaggle/<PROJECT>/experiments/db/mla.db \
  --study <STUDY_NAME> \
  --cutoff-minutes 60 \
  --dry-run

# Apply changes (remove --dry-run)
python scripts/optuna_clean_zombie_running.py \
  --db projects/kaggle/<PROJECT>/experiments/db/mla.db \
  --study <STUDY_NAME> \
  --cutoff-minutes 60
```

- **--cutoff-minutes**: Defines how long a trial must be in `RUNNING` state to be considered "dead" (default: 60).
- **--dry-run**: Always run with this flag first to see which trials will be affected without modifying the database.

---

## Technical Details
- **Library**: `textual` for the TUI (Terminal User Interface).
- **Data Source**: 
    - `optuna.db` (SQLite) for study-level stats.
    - Filesystem (`experiments/.../trial_XXXX/*/state.json`) for detailed trial flow.
- **Updates**: Polls the filesystem and DB every few seconds to refresh the view.

## Usage
```bash
uv run python scripts/mla.py pre tune --optuna-live
```
This command launches the TUI. Use mouse or keyboard to navigate.
