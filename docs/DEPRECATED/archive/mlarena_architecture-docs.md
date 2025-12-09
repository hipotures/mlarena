# Documentation Update Plan: Migration to `mla.py` Workflow

## 1. Introduction

This document outlines the plan for updating the project's documentation to reflect the migration from the old, script-based workflow to the new, centralized `mla.py` workflow. The goal is to provide clear, up-to-date documentation for users and developers, ensuring a smooth transition and deprecating obsolete information.

## 2. Analysis of Workflows

### Old Workflow (Legacy)

The previous workflow consisted of a collection of standalone Python scripts located in the `scripts/` directory. Each script was responsible for a specific part of the machine learning pipeline.

-   **Key Scripts:** `ml_runner.py`, `optuna_runner.py`, `experiment_manager.py`, `submission_workflow.py`.
-   **Characteristics:**
    -   Decentralized and loosely coupled.
    -   Often monolithic, with scripts like `ml_runner.py` handling multiple stages (e.g., preprocessing, training, prediction) via command-line flags.
    -   Less discoverable and harder to maintain as a cohesive system.

### New Workflow (`mla.py`)

The new workflow is orchestrated by a single, powerful command-line interface (CLI) entry point: `scripts/mla.py`.

-   **Entry Point:** `scripts/mla.py`
-   **Core Logic:** `src/mlarena/cli/main.py`
-   **Modular Architecture:** The functionality is broken down into independent, discoverable modules located in `src/mlarena/modules/`. Each module corresponds to a subcommand of `mla.py` (e.g., `eda`, `model`, `tune`, `submit`).
-   **Characteristics:**
    -   **Centralized:** A single entry point simplifies usage.
    -   **Modular:** Easily extensible by adding new modules.
    -   **Discoverable:** Commands and their options are self-documenting via the CLI (`mla --help`, `mla <module> --help`).
    -   **Orchestrated:** A `PipelineExecutor` manages dependencies and the flow between modules.

## 3. Key Documentation Files for Update

The following files in the `docs/` directory are the primary candidates for updates:

-   `README.md` (root of the project)
-   `docs/README.md`
-   `docs/MANUAL_PIPELINE_GUIDE.md`
-   `GEMINI.md`
-   Any other documents that reference the old scripts.

## 4. Proposed Documentation Structure

To streamline the documentation and make it more user-friendly, the following structure is proposed:

1.  **`README.md` (Root):** Should contain a high-level overview of the project and a "Quick Start" section that directs users to the new `mla.py` workflow.
2.  **`docs/` directory:**
    -   **`README.md`:** Should serve as a table of contents for all documentation.
    -   **`MLA_WORKFLOW_GUIDE.md`:** (Rename from `MANUAL_PIPELINE_GUIDE.md`). This will be the main guide for the new workflow, providing a comprehensive overview and examples.
    -   **`ARCHITECTURE.md`:** A new document describing the architecture of the `mla.py` system (modules, registry, pipeline executor).
    -   **`MIGRATION_GUIDE.md`:** A new document to help users transition from the old scripts to the new `mla.py` commands.
    -   **`DEPRECATED/`:** A new subdirectory to move all obsolete documentation into.

## 5. Content Update Plan

### `README.md` (Root)

-   **Update "Quick Start":** Replace the current quick start guide with instructions on using `mla.py`.
-   **Remove References to Old Scripts:** Remove any mentions of `ml_runner.py`, `optuna_runner.py`, etc.
-   **Add Link to `MLA_WORKFLOW_GUIDE.md`:** Direct users to the main guide for more details.

### `docs/MANUAL_PIPELINE_GUIDE.md` -> `docs/MLA_WORKFLOW_GUIDE.md`

-   **Rename the file.**
-   **Review and Update Content:** Ensure the guide is comprehensive and up-to-date with the latest `mla.py` features.
-   **Add More Examples:** Provide practical examples for common use cases.
-   **Incorporate Migration Information:** Briefly mention the old scripts and point to the `MIGRATION_GUIDE.md`.

### New File: `docs/MIGRATION_GUIDE.md`

-   **Purpose:** To explicitly map old commands to their new `mla.py` equivalents.
-   **Content:**
    -   A table mapping old script commands to new `mla` commands. For example:
        | Old Command                                   | New Command (`mla`)                               |
        | --------------------------------------------- | ------------------------------------------------- |
        | `python scripts/ml_runner.py --stage train`     | `uv run python scripts/mla.py model --project ...`      |
        | `python scripts/optuna_runner.py`             | `uv run python scripts/mla.py tune --project ...`       |
        | `python scripts/submission_workflow.py`       | `uv run python scripts/mla.py submit --project ...`     |
    -   Guidance on configuration changes.

### `GEMINI.md`

- **Update `Quick Reference`:** The "Most common workflow" should be updated to exclusively use `mla.py` commands.
- **Update `Common Commands`:** The "Modern Workflow (Recommended)" section is good, but any conflicting or outdated information should be removed.
- **Deprecate "Alternative: Direct Runner":** This section will become obsolete.
- **Update Critical Workflows:** Refocus this section on the modular experiment pipeline within `mla.py`.

## 6. Handling the Incomplete Migration

Since the migration is not 100% complete, the documentation must address this:

-   **Acknowledge the Transition:** The documentation should clearly state that the project is in a transition period.
-   **Provide a Roadmap (Optional but Recommended):** If possible, outline the plan for completing the migration.
-   **Focus on the New Workflow:** While acknowledging the old scripts, all new documentation and examples should exclusively use the `mla.py` workflow.

## 7. Action Plan

-   [ ] Create a new file `docs/mlarena_architecture-docs.md` with this plan.
-   [ ] Create a new branch for the documentation update.
-   [ ] Rename `docs/MANUAL_PIPELINE_GUIDE.md` to `docs/MLA_WORKFLOW_GUIDE.md`.
-   [ ] Create the new file `docs/MIGRATION_GUIDE.md`.
-   [ ] Create the new file `docs/ARCHITECTURE.md`.
-   [ ] Update the root `README.md`.
-   [ ] Update the `docs/README.md`.
-   [ ] Update `GEMINI.md`.
-   [ ] Move all obsolete documentation to a `docs/DEPRECATED/` directory.
-   [ ] Review all changes and merge the branch.
