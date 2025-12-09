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

# Override the model template for a different experiment
uv run python scripts/mla.py --project <competition-slug> --model-template gpu-dev-5m

# Force re-run all modules from scratch, ignoring cached results
uv run python scripts/mla.py --project <competition-slug> --force
```

## MLArena (`mla.py`) Workflow

`mla.py` is the single entry point for the entire ML pipeline. You can run the full auto-flow or execute each module step-by-step.

### Modules

The pipeline consists of the following modules:
-   `init`: Initializes a new competition project structure.
-   `eda`: Performs exploratory data analysis.
-   `preprocess`: Applies data preprocessing steps.
-   `model`: Trains a model using a specified template.
-   `predict`: Generates predictions from a trained model.
-   `submit`: Submits predictions to Kaggle.
-   `fetch-score`: Fetches the public score for a submission.
-   `tune`: (Coming soon) Hyperparameter tuning.
-   `stack`: (Coming soon) Stacking/ensembling models.

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
uv run python scripts/mla.py preprocess --project <competition-slug> --preprocess-template <template-name>
```

**4. Train a Model:**
```bash
uv run python scripts/mla.py model --project <competition-slug> --model-template <template-name>
```

### Common Flags

-   `--project <name>` or `-p <name>`: Specifies the competition project.
-   `--experiment-id <id>` or `-e <id>`: Resumes or targets an existing experiment.
-   `--force` or `-f`: Forces re-execution of completed modules.
-   `--skip-deps`: Skips automatic execution of module dependencies.
-   `--skip-submit`: Prevents automatic submission to Kaggle (saves the CSV file only).
-   `--skip-git`: Prevents the automatic git commit after a successful auto-flow run.

## Architecture

The framework is designed with a four-layer architecture:

1.  **Scripts Layer (`scripts/`):** The main entry point, `mla.py`, which orchestrates the pipeline.
2.  **Core Package (`src/mlarena/`):** The heart of the CLI, containing the module registry, pipeline executor, and experiment tracking logic.
3.  **Project Layer (`projects/kaggle/`):** Each competition has its own isolated directory containing its data, code, experiments, and submissions.
4.  **Tracking Layer:** A system that automatically links experiments, submissions, and git commits for reproducibility.

## Project Structure

A typical competition project initialized with `mla.py init` has the following structure:

```
projects/kaggle/<competition-slug>/
├── README.md                # Competition-specific notes
├── data/                    # Raw and processed data
├── code/
│   ├── models/              # Model implementation files
│   └── utils/
│       └── config.py        # Project-specific configuration
├── experiments/             # Experiment logs, artifacts, and state
└── submissions/             # Generated submission CSV files
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
