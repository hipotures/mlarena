# Kaggle ML Arena

This repository provides a standardized and powerful workflow for participating in Kaggle competitions. It is built around **MLArena (`mla.py`)**, a centralized command-line interface that streamlines the entire machine learning pipeline from initialization and EDA to model training, submission, and score tracking.

The system is designed for rapid iteration, reproducibility, and modularity, allowing you to focus on experiment design rather than boilerplate code.

## Core Features

-   **Centralized Workflow:** All actions are performed through the `mla.py` script, providing a single, consistent interface.
-   **Modular Pipeline:** The ML pipeline is composed of independent modules (e.g., `init`, `eda`, `preprocess`, `model`, `tune`) that can be run individually or as part of a larger workflow.
-   **Automated Experiment Tracking:** Every experiment and submission is tracked, capturing the git hash and a snapshot of the code for full reproducibility.
-   **Templating System:** Standardize model and preprocessing configurations using YAML templates for easy reuse and modification.
-   **Automated Submission & Score Fetching:** The pipeline can automatically submit to Kaggle and fetch the public score, closing the feedback loop quickly.

## Quick Start

### 1. Setup

First-time setup is required to install dependencies and configure the Kaggle API.

```bash
# Install Python dependencies
uv sync

# Install Playwright browser for score scraping
uv run playwright install chromium

# Configure Kaggle API (one-time)
mkdir -p ~/.kaggle
# Copy your kaggle.json to ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 2. Auto-Flow (Recommended)

The recommended way to run a full experiment is using the auto-flow, which orchestrates the entire pipeline.

```bash
# Run the complete pipeline: init → eda → preprocess → model → submit → fetch-score
# This uses the "baseline" templates by default.
uv run python scripts/mla.py --project <competition-slug>

# Override the model template using dotted syntax
uv run python scripts/mla.py --project <competition-slug> model_template=gpu-dev-5m

# Run a quick smoke test using a profile
uv run python scripts/mla.py --project <competition-slug> --profile smoke
```

## MLArena (`mla.py`) Workflow

`mla.py` is the single entry point for the entire ML pipeline. You can use standard CLI flags for control and **dotted paths** for configuration data.

### CLI Flags vs. Dotted Overrides

MLArena uses a hybrid approach for maximum flexibility:

1.  **CLI Flags** (`--project`, `--profile`, `--force`): Fundamental controls visible in `--help`.
    - Example: `--profile smoke` is a shortcut for loading a set of fast defaults.
2.  **Dotted Overrides** (`key=value`): Precise control over any configuration parameter.
    - Example: `model.time_limit=100` targets a specific module's setting.
    - Example: `profile=smoke` is functionally identical to the flag but uses the data engine.

---

### Modules

The pipeline consists of the following modules:

**Core modules (part of auto-flow):**
-   `init`: Initializes a new competition project structure.
-   `eda`: Performs exploratory data analysis.
-   `preprocess`: Applies data preprocessing steps.
-   `model`: Trains a model using a specified template.
-   `predict`: Generates predictions from a trained model.
-   `submit`: Submits predictions to Kaggle.
-   `fetch-score`: Fetches the public score for a submission.

**Optional modules (manual use only):**
-   `feat`: Applies lightweight feature transformations (log1p, ratios, column drops) defined in feature templates.
-   `tune`: Optuna-based hyperparameter search on a sampled training subset using AutoGluon.
-   `stack`: Averages multiple prediction files to produce an ensemble submission.

**Note**: AutoGluon native HPO is available through model templates with `hpo_preset`. See [HPO Guide](docs/MLA_WORKFLOW_GUIDE.md#hyperparameter-optimization-hpo) for details.

### Manual Workflow Example

If you prefer more granular control, you can run each module individually.

**1. Initialize a new competition project:**
```bash
uv run python scripts/mla.py init --project <competition-slug>
```
This creates a new directory under `projects/kaggle/<competition-slug>` with the required structure and downloads the competition data.

**2. Run Exploratory Data Analysis (EDA):**
```bash
uv run python scripts/mla.py eda --project <competition-slug>
```

**3. Preprocess Data:**
```bash
uv run python scripts/mla.py preprocess --project <competition-slug> preprocess_template=<template-name>
```

**4. Train a Model:**
```bash
uv run python scripts/mla.py model --project <competition-slug> model_template=<template-name>
```

**5. Train with Hyperparameter Optimization (HPO):**
```bash
# Quick HPO (50 trials, 1-2h)
uv run python scripts/mla.py model --project <competition-slug> model_template=test_hpo_medium

# Advanced HPO (100 trials, 4-6h)
uv run python scripts/mla.py model --project <competition-slug> model_template=test_hpo_high
```

### Common Flags

-   `--project <name>` or `-p <name>`: Specifies the competition project.
-   `--experiment-id <id>` or `-e <id>`: Resumes or targets an existing experiment.
-   `--profile <name>` or `-s <name>`: Loads a config profile (e.g., `smoke`, `dev`).
-   `--force` or `-f`: Forces re-execution of completed modules.

### Configuration Overrides

You can override any configuration parameter using dotted paths:
```bash
# Set model time limit
uv run python scripts/mla.py model -p titanic model.time_limit=600

# Set global seed and preprocess option
uv run python scripts/mla.py -p titanic common.seed=42 preprocess.cache=true
```

## Architecture

The framework is designed with a four-layer architecture:

1.  **Scripts Layer (`scripts/`):** The main entry point, `mla.py`, which orchestrates the pipeline.
2.  **Core Package (`src/mlarena/`):** The heart of the CLI, containing the module registry, pipeline executor, and experiment tracking logic.
3.  **Project Layer (`projects/kaggle/`):** Each competition has its own isolated directory containing its data, code, experiments, and submissions.
4.  **Tracking Layer:** A system that automatically links experiments, submissions, and git commits for reproducibility.

## Repository Structure

### Full Directory Layout

```
mlarena/                         # Repository root
├── scripts/                     # Entry points and utilities
│   ├── mla.py                  # ⭐ Main CLI entry point
│   ├── submissions_tracker.py  # Submission tracking (CLI + library)
│   ├── experiment_logger.py    # Experiment tracking (CLI + library)
│   ├── template_loader.py      # YAML template loader (internal)
│   ├── ai_helper.py            # AI code generation (internal)
│   └── utils/                  # Standalone utilities
│       ├── clean.py           # Artifact cleanup
│       ├── sync.py            # Project synchronization
│       └── av_weights_mix.py  # Adversarial validation
│
├── src/                        # Core framework code
│   ├── mlarena/               # Main package
│   │   ├── cli/              # CLI orchestration
│   │   │   └── main.py       # Command parser + config builder
│   │   ├── core/             # Core infrastructure
│   │   │   ├── conf.py       # OmegaConf + Pydantic config system
│   │   │   ├── registry.py   # Module discovery
│   │   │   ├── pipeline.py   # Execution engine
│   │   │   ├── experiment.py # State management
│   │   │   └── module.py     # Base module class
│   │   ├── modules/          # Pipeline modules
│   │   │   ├── init.py
│   │   │   ├── eda.py
│   │   │   ├── preprocess.py
│   │   │   ├── model.py
│   │   │   ├── predict.py
│   │   │   ├── submit.py
│   │   │   └── fetch_score.py
│   │   ├── defaults/         # Global implementations
│   │   │   ├── models/       # Default model trainers
│   │   │   └── preprocessing/ # Default preprocessing steps
│   │   ├── templates/        # Global templates
│   │   │   ├── profiles/     # Config profiles (smoke, dev)
│   │   │   ├── model/        # Model templates (*.yaml)
│   │   │   └── preprocess/   # Preprocessing templates (*.yaml)
│   │   └── utils/            # Shared utilities
│   └── kaggle_tools/         # Competition utilities
│       └── submission.py     # Submission creation
│
└── projects/kaggle/           # Competition projects
    └── <competition-slug>/    # Individual competition
        ├── README.md          # Competition notes
        ├── config.yaml        # Project-level config (optional)
        ├── data/              # Raw competition data
        │   ├── train.csv
        │   ├── test.csv
        │   └── sample_submission.csv
        ├── code/              # Competition-specific code
        │   ├── models/        # Custom model implementations
        │   │   └── my_model.py
        │   ├── preprocessing/ # Custom preprocessing modules
        │   │   └── my_preprocess.py
        │   └── utils/
        │       └── config.py  # ⚙️ Competition constants (TARGET_COLUMN, etc.)
        ├── templates/         # Project template overrides
        │   ├── profiles/      # Custom profiles
        │   ├── model/         # Model templates (override globals)
        │   └── preprocess/    # Preprocess templates (override globals)
        ├── experiments/       # Experiment history (see below)
        └── submissions/       # Submission tracking
            ├── submissions.json          # Submission history
            └── submission-*.csv          # Generated submissions
```

### Experiments Directory Structure

The `experiments/` folder contains all experiment artifacts, organized by type:

#### **1. Fixed Experiments** (Setup modules, always overwrite)

```
experiments/
├── init/                      # Project initialization
│   ├── state.json            # Execution status
│   └── artifacts/
│       └── init/
│           └── config_snapshot.json
│
└── eda/                       # Exploratory data analysis
    ├── state.json
    └── artifacts/
        └── eda/
            ├── train_profile.html
            ├── test_profile.html
            └── eda_summary.json
```

#### **2. Named Experiments** (Preprocessing, cached if input unchanged)

```
experiments/
└── pre-{template}/            # Preprocessing chain experiment
    └── {step_index}-{template_name}/    # Each step in chain
        ├── state.json
        └── artifacts/
            └── preprocess/
                ├── train_processed.csv    # Transformed training data
                ├── test_processed.csv     # Transformed test data
                ├── orig_processed.csv     # External dataset (optional)
                └── preprocess_state.pkl   # Preprocessing artifacts

Example:
  pre-baseline/
    └── 0-baseline/           # Single-step preprocessing

  pre-full-pipeline/          # Multi-step chain
    ├── 0-imputer/           # Step 1
    ├── 1-encoder/           # Step 2
    └── 2-feature_selector/  # Step 3
```

#### **3. Timestamped Experiments** (Pipeline runs, one per model training)

```
experiments/
└── exp-YYYYMMDD-HHMMSS/      # Full pipeline experiment
    ├── state.json             # Execution state for all modules
    └── artifacts/
        ├── model/
        │   ├── model/         # AutoGluon model directory
        │   │   ├── models/
        │   │   ├── predictor.pkl
        │   │   └── learner.pkl
        │   └── leaderboard.csv    # Model performance
        ├── predict/
        │   └── submission-*.csv   # Raw predictions
        ├── submit/
        │   └── submit_success.txt # Submission confirmation
        └── fetch-score/
            └── fetch_score.txt    # Public score from Kaggle

Example experiments:
  exp-20251217-152730/        # Latest experiment
  exp-20251216-144148/        # Previous experiment
  exp-20251215-143822/        # Older experiment
```

### Example: Full Titanic Project

```
projects/kaggle/Titanic/
├── README.md
├── data/
│   ├── train.csv              (891 rows)
│   ├── test.csv               (418 rows)
│   └── sample_submission.csv  (418 rows)
├── code/
│   └── utils/
│       └── config.py          # TARGET_COLUMN = "Survived"
├── experiments/
│   ├── init/                  # Setup
│   ├── eda/                   # Data profiling
│   ├── pre-baseline/          # Preprocessing
│   │   └── 0-baseline/
│   │       ├── state.json
│   │       └── artifacts/preprocess/
│   │           ├── train_processed.csv
│   │           └── test_processed.csv
│   ├── exp-20251217-152730/   # Latest model run
│   │   ├── state.json
│   │   └── artifacts/
│   │       ├── model/
│   │       │   ├── model/     # AutoGluon models (~200MB)
│   │       │   └── leaderboard.csv
│   │       ├── predict/
│   │       │   └── submission-20251217152745.csv
│   │       ├── submit/
│   │       │   └── submit_success.txt
│   │       └── fetch-score/
│   │           └── fetch_score.txt  # "0.7987"
│   └── exp-20251216-144148/   # Previous run
└── submissions/
    ├── submissions.json       # Full history
    └── submission-*.csv       # Archived submissions
```

### State File Format

Each experiment's `state.json` tracks module execution:

```json
{
  "experiment_id": "exp-20251217-152730",
  "modules": {
    "model": {
      "status": "completed",
      "payload": {
        "local_cv": 0.8234,
        "model_path": "artifacts/model/model"
      },
      "invocation": {
        "model_template": "cpu-dev-5m",
        "time_limit": 300
      },
      "error": null
    },
    "predict": {
      "status": "completed",
      "payload": {
        "submission_file": "artifacts/predict/submission-20251217152745.csv"
      }
    }
  },
  "git": {
    "hash": "a475f26",
    "dirty": false
  }
}
```

### Project Configuration

Each project's configuration is defined in `projects/kaggle/<competition-slug>/code/utils/config.py`. This file contains critical constants such as:

-   `TARGET_COLUMN`: The name of the target variable.
-   `AUTOGLUON_PROBLEM_TYPE`: The problem type for AutoGluon (`binary`, `multiclass`, `regression`).
-   `AUTOGLUON_EVAL_METRIC`: The evaluation metric for the competition.

## Experiment Tracking and Reproducibility

The MLArena framework is built for reproducibility.

-   **Experiment State:** Each experiment's state, parameters, and results are tracked in `experiments/<experiment_id>/state.json`.
-   **Git Integration:** The git hash is captured for every experiment, allowing you to check out the exact code version used.
-   **Code Snapshots:** A snapshot of the executed code is saved with each experiment.

### Viewing History

You can list all tracked submissions and experiments for a project:

```bash
# View all submissions for a project
uv run python scripts/mla.py submissions --project <competition-name> list

# View all experiments for a project
uv run python scripts/mla.py experiments --project <competition-name> list
```

### Reproducing a Submission

To reproduce a past submission, you can either check out the corresponding git commit or restore the code snapshot:

```bash
# Method 1: Git Checkout
git checkout <GIT_HASH_FROM_SUBMISSION_LIST>

# Method 2: Restore Code Snapshot
uv run python scripts/mla.py experiments --project <competition-name> restore <EXPERIMENT_ID>
```

## Automated Submission & Score Fetching

The pipeline can automatically upload submissions and fetch public scores using Playwright and the Chrome DevTools Protocol (CDP).

### One-Time Setup

You must have a Chrome browser instance running with remote debugging enabled.

```bash
# Start Chrome with a dedicated user profile and remote debugging enabled
google-chrome --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-debug
```
In this browser window, log in to your Kaggle account. Keep this window open when running experiments that include submission and score fetching.

### Control Flags

-   `--wait-seconds <N>`: Sets the delay (in seconds) between submission and score fetching to allow for Kaggle processing (default: 30).
-   `--skip-score-fetch`: Submits to Kaggle but skips the automated score fetching step.
-   `--cdp-url <URL>`: Specifies a custom CDP endpoint if Chrome is running on a different port or host.
