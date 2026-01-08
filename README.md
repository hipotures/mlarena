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

### Optional: Setup 'mla' Alias

To use the shorter `mla` command as seen in some guides:
```bash
alias mla="uv run python scripts/mla.py"
```

### 2. Auto-Flow (Recommended)

The recommended way to run a full experiment is using the auto-flow, which orchestrates the preprocessing and modeling pipeline.

**Prerequisites** (one-time setup):
```bash
# Initialize project structure and download data
uv run python scripts/mla.py init --project <competition-slug>

# Run exploratory data analysis
uv run python scripts/mla.py eda --project <competition-slug>
```

**Run auto-flow pipeline**:
```bash
# Run the pipeline: preprocess → model → predict → submit → fetch-score
uv run python scripts/mla.py --project <competition-slug> model.mla_retention=true
```

---

## 🛠️ Key Features

### Execution Profiles
Profiles allow you to switch between predefined sets of parameters quickly:
- `--profile smoke`: Fast verification (short time limits, medium quality).
- `--profile dev`: Standard development settings.

Usage:
```bash
uv run python scripts/mla.py --project titanic --profile smoke
```

### Magic Flags (CLI Shortcuts)
Common parameters are automatically mapped for convenience:
- `--seed 123` → `common.seed=123`
- `--time-limit 300` → `common.time_limit=300`
- `--use-gpu` → `common.use_gpu=true`

### Disk Space Management
Use `model.mla_retention=true` to delete intermediate AutoGluon models and save up to 98% of disk space while keeping the best model for predictions.

---

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

The pipeline is organized into three categories:

**1. Setup & Prerequisites (Manual, run once per project):**
-   `init`: Initializes a new competition project structure and downloads data.
-   `eda`: Performs exploratory data analysis (profiles train/test data).

**2. Auto-Flow Pipeline (Core modules executed in sequence):**
-   `preprocess`: Applies data preprocessing steps (supports chains and caching).
-   `model`: Trains a model using a specified template (AutoGluon by default).
-   `predict`: Generates predictions from a trained model for the test set.
-   `submit`: Submits predictions to Kaggle via CLI.
-   `fetch-score`: Scrapes/fetches the latest public score from Kaggle.

**3. Independent Utility Modules (Manual use only):**
-   `experiments`: Lists experiment history, results, and local CV vs. public scores.
-   `submissions`: Manages the local submission database and Kaggle scores.
-   `queue`: Manages task execution order for batch experiments (`mla queue`).
-   `feat`: Applies lightweight feature transformations defined in templates.
-   `tune`: Optuna-based hyperparameter search (experimental).
-   `stack`: Averages multiple prediction files for blending (experimental).

**Note**: AutoGluon native HPO is available through model templates with `hpo_preset`. See [HPO Guide](docs/MLA_WORKFLOW_GUIDE.md#hyperparameter-optimization-hpo) for details.

### Weighted Submission Blending (Utility Script)

For weighted blends based on Kaggle public scores, use `scripts/blend_submissions.py`. It parses a Kaggle CLI output file, selects top-N submissions by public score, maps them to local submission CSVs, and writes a blended submission plus a manifest.

```bash
# Export Kaggle submissions list
kaggle competitions submissions -c <competition-slug> > /tmp/sub.txt

# Blend top 5 by public score (default weighting)
python scripts/blend_submissions.py \
  --project <competition-slug> \
  --kaggle-output /tmp/sub.txt \
  --top-n 5 \
  --weighting public \
  --output-name submission-blend-top5-public.csv
```

Override the submissions directory if needed:
```bash
python scripts/blend_submissions.py \
  --project <competition-slug> \
  --kaggle-output /tmp/sub.txt \
  --submissions-dir /path/to/submissions
```

### Submission Queue (Manual Script)

For managing multiple submissions efficiently, use the submission queue script. This is separate from the CLI Task Queue and specifically handles the upload process.

**Quick start:**
```bash
# Add to queue
uv run python scripts/mla.py submit \
  --project <competition-slug> \
  --exp-id <exp-id> \
  submit.queue_submit=true

# List queued submissions
python scripts/submission_queue.py --project <competition-slug> list

# Submit from queue
python scripts/submission_queue.py --project <competition-slug> submit 1 --continue-flow
```

**Features:** Duplicate detection, error tracking, status tracking, auto-cleanup, thread-safe

**For complete documentation, see:** [Submission Queue Guide](docs/submission_queue.md)

### Task Queue Management

The Task Queue (`mla queue`) manages computation tasks (training, preprocessing) to be executed sequentially. This is ideal for queuing up multiple experiments overnight.

**Note:** The `queue` command is implemented as a separate script (`scripts/task_queue.py`) and is invoked via subprocess delegation for maintainability.

**Basic Commands:**
```bash
# List queued tasks
uv run python scripts/mla.py queue list -p <competition-slug>

# Add a task (e.g., train a model)
uv run python scripts/mla.py queue add -p <competition-slug> model_template=<template-name>

# Add task with high priority (1=highest, 10=default)
uv run python scripts/mla.py queue add -p <competition-slug> model_template=<template-name> --priority 1

# Run the queue
uv run python scripts/mla.py queue run -p <competition-slug>
```

**Features:**
- **Priorities**: Control execution order
- **Templates**: Add tasks using standard model/preprocess templates
- **Logs**: Per-task execution logs in `projects/kaggle/<slug>/queue/logs/`

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
uv run python scripts/mla.py model --project <competition-slug> model_template=hpo_boost_medium

# Advanced HPO (100 trials, 4-6h)
uv run python scripts/mla.py model --project <competition-slug> model_template=hpo_boost_high
```

### Common Flags

-   `--project <name>` or `-p <name>`: Specifies the competition project.
-   `--exp-id <id>` or `-e <id>`: Resumes or targets an existing experiment.
-   `--profile <name>` or `-s <name>`: Loads a config profile (e.g., `smoke`, `dev`).
-   `--force` or `-f`: Forces re-execution of completed modules.
-   `model.mla_retention=true`: Clean up intermediate AutoGluon models after training.

### Configuration Overrides

You can override any configuration parameter using dotted paths:
```bash
# Set model time limit
uv run python scripts/mla.py model -p titanic model.time_limit=600

# Set global seed and preprocess option
uv run python scripts/mla.py -p titanic common.seed=42 preprocess.cache=true
```

**For parameter naming conventions, see:** [Terminology Guide](docs/TERMINOLOGY.md)

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
│   │   │   ├── fetch_score.py
│   │   │   ├── stack.py      # Stacking (Experimental)
│   │   │   └── tune.py       # Tuning (Experimental)
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
                ├── train_processed.csv.gz    # Transformed training data
                ├── test_processed.csv.gz     # Transformed test data
                ├── orig_processed.csv.gz     # External dataset (optional)
                └── preprocess_state.pkl      # Preprocessing artifacts

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
  "project": "titanic",
  "modules": {
    "model": {
      "status": "completed",
      "started_at": "2025-12-17T15:27:31Z",
      "finished_at": "2025-12-17T15:28:15Z",
      "invocation": {
        "model_template": "cpu-fast-1m",
        "time_limit": 60
      },
      "payload": {
        "local_cv_score": 0.8234,
        "model_path": "artifacts/model/model"
      }
    },
    "predict": {
      "status": "completed",
      "invocation": {},
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

### Auto-Flow Git Commits

When auto-flow completes successfully (unless `skip_git=true`), a git commit is created automatically:

**Commit message format:**
```
auto-flow({project}): {module1}→{module2}→... | local {cv_score} | public {score}
```

**Example:**
```
auto-flow(titanic): preprocess→model→predict→submit→fetch-score | local 0.834 | public 0.798
```

**What gets staged:**
- Project directory: `projects/kaggle/{project}/`
- Experiments, submissions, and templates

**Skip auto-commit:**
```bash
uv run python scripts/mla.py --project titanic skip_git=true
```

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
-   `--cdp-url <URL>`: Specifies a custom CDP endpoint if Chrome is running on a different port or host.

**Note**: Score fetching is controlled by the `skip_submit` flag (skips both submit and fetch) and the `wait_seconds` parameter.

### CLI Parsing Behavior

MLArena supports two parameter formats:

1. **Dotted paths** (recommended): `key.subkey=value`
   ```bash
   uv run python scripts/mla.py --project titanic common.time_limit=600
   ```

2. **Flag format** (converted internally): `--flag value`
   ```bash
   uv run python scripts/mla.py --project titanic --time-limit 600
   # Internally converted to: common.time_limit=600
   ```

**Note**: Common parameters (`time_limit`, `use_gpu`, `preset`, `seed`) are automatically prefixed with `common.` when using flag format.

---

## Documentation Index

### Core Documentation
- **[Quick Start Guide](docs/quick_start.md)** - Getting started with MLArena
- **[Architecture](docs/architecture.md)** - System design and execution flow
- **[MLA Workflow Guide](docs/MLA_WORKFLOW_GUIDE.md)** - Complete workflow examples
- **[Configuration System](docs/configs.md)** - Parameter reference and profiles

### Module Documentation
- **[Preprocessing Submodules](docs/submodules/README.md)** - All preprocessing modules and contracts
- **[Model Templates](docs/model_templates.md)** - Available model configurations
- **[Feature Engineering](docs/feat.md)** - Feature transformation module
- **[Hyperparameter Tuning](docs/tune.md)** - Optuna-based HPO
- **[Stacking](docs/stack.md)** - Model ensemble techniques

### Advanced Topics
- **[Terminology Guide](docs/TERMINOLOGY.md)** - Naming conventions (Python, YAML, CLI)
- **[Submission Queue](docs/submission_queue.md)** - Batch upload management
- **[State Payload Formats](docs/state_payload_formats.md)** - Understanding state.json structures
- **[Contributing Guide](docs/contributing.md)** - Development guidelines
- **[FAQ](docs/faq.md)** - Common questions and troubleshooting

### Agent Documentation
- **[AGENTS.md](AGENTS.md)** - AI agent guide for navigating the codebase

---
