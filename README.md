# Kaggle Competitions Repository

This repository provides a standardized and powerful workflow for participating in Kaggle competitions. It is built around **MLArena (`mla.py`)**, a centralized command-line interface that streamlines the entire machine learning pipeline from EDA to submission.

## Quick Start

1.  **Initialize a new competition project:**
    ```bash
    uv run python scripts/mla.py init --project <competition-slug>
    ```

2.  **Run the end-to-end pipeline:**
    ```bash
    # Run Exploratory Data Analysis
    uv run python scripts/mla.py eda --project <competition-slug>

    # Train a model and auto-submit to Kaggle
    uv run python scripts/mla.py model --project <competition-slug> --model-template dev-gpu --auto-submit
    ```

For a detailed guide on the workflow and available commands, please refer to the [**MLA Workflow Guide**](docs/MLA_WORKFLOW_GUIDE.md).

## Core Concepts

-   **Centralized Workflow:** All actions are performed through the `mla.py` script, providing a single, consistent interface.
-   **Modular Architecture:** The pipeline is composed of independent modules (e.g., `eda`, `model`, `tune`) that can be run individually or as part of a larger workflow.
-   **Reproducibility:** Every experiment and submission is tracked, capturing the git hash and a snapshot of the code.

## Documentation

-   [**MLA Workflow Guide**](docs/MLA_WORKFLOW_GUIDE.md): A comprehensive guide to using the `mla.py` workflow.
-   [**Migration Guide**](docs/MIGRATION_GUIDE.md): For users transitioning from the old, script-based workflow.
-   [**Architecture Overview**](docs/ARCHITECTURE.md): A look into the design and components of the `mla.py` system.

## Project Structure

Each competition project is self-contained within the `projects/kaggle/` directory. The `mla.py init` command will generate the following structure:

```
projects/kaggle/<competition-slug>/
├── README.md                # Competition-specific notes
├── data/                    # Raw Kaggle files
├── code/
│   └── utils/
│       └── config.py        # Project configuration
├── experiments/             # Experiment logs and artifacts
└── submissions/             # Generated submission files
```

The core logic and modules for the `mla.py` workflow are located in the `src/mlarena/` directory.
