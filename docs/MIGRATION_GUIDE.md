# Migration Guide: From Legacy Scripts to `mla.py`

This guide is for users who are familiar with the old, script-based workflow and want to migrate to the new, centralized `mla.py` command-line interface.

## Key Changes

The new `mla.py` workflow replaces the collection of individual scripts in the `scripts/` directory with a single, unified entry point. This provides a more consistent, discoverable, and maintainable way to run experiments.

## Command Mapping

The following table maps the old, legacy script commands to their new `mla.py` equivalents.

| Old Command                                   | New Command (`mla`)                               | Notes                                                              |
| --------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------ |
| `python scripts/ml_runner.py --stage eda`     | `uv run python scripts/mla.py eda --project ...`      | Handles exploratory data analysis.                                 |
| `python scripts/ml_runner.py --stage train`   | `uv run python scripts/mla.py model --project ...`    | For training models.                                               |
| `python scripts/ml_runner.py --stage predict` | `uv run python scripts/mla.py predict --project ...`  | For generating predictions.                                        |
| `python scripts/optuna_runner.py`             | `uv run python scripts/mla.py tune --project ...`     | For hyperparameter tuning with Optuna.                             |
| `python scripts/submission_workflow.py`       | `uv run python scripts/mla.py submit --project ...`   | For submitting to Kaggle and fetching scores.                      |
| `python scripts/experiment_logger.py list`    | `uv run python scripts/mla.py experiments list`     | To list all the experiments.                                       |
| `python scripts/submissions_tracker.py list`  | `uv run python scripts/mla.py submissions list`   | To list all the submissions.                                       |

## Configuration

The new `mla.py` workflow still uses the same `config.py` files within each project for competition-specific settings. No major changes are required to your existing project configurations.

## Benefits of the New Workflow

-   **Centralization:** A single `mla.py` entry point is easier to remember and use.
-   **Modularity:** The new architecture is easier to extend with new functionality.
-   **Discoverability:** You can easily discover commands and options using `mla --help` and `mla <module> --help`.
-   **Orchestration:** The new system intelligently manages dependencies between pipeline stages.
