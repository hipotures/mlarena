# `mla.py` Architecture

This document provides a high-level overview of the architecture of the `mla.py` command-line interface and its underlying components.

## 1. Overview

The `mla.py` workflow is designed to be a modular, extensible, and centralized system for running machine learning experiments. It replaces the previous collection of disparate scripts with a single, unified CLI entry point.

The architecture can be broken down into the following key components:

-   **CLI Entry Point (`scripts/mla.py`)**
-   **Core CLI Application (`src/mlarena/cli/main.py`)**
-   **Module Registry (`src/mlarena/core/registry.py`)**
-   **Pipeline Executor (`src/mlarena/core/executor.py`)**
-   **Modules (`src/mlarena/modules/`)**

## 2. Component Breakdown

### CLI Entry Point

-   **File:** `scripts/mla.py`
-   **Purpose:** This is the main entry point for the user. It's a thin wrapper that imports and calls the main function from the core CLI application.

### Core CLI Application

-   **File:** `src/mlarena/cli/main.py`
-   **Purpose:** This is the heart of the CLI. It uses the `typer` library to create the command-line interface.
-   **Functionality:**
    -   Initializes the `ModuleRegistry`.
    -   Dynamically discovers and adds all available modules as subcommands to the CLI.
    -   Initializes the `PipelineExecutor`.
    -   Executes the requested module and its dependencies.

### Module Registry

-   **File:** `src/mlarena/core/registry.py`
-   **Purpose:** The registry is responsible for discovering all available modules in the `src/mlarena/modules/` directory.
-   **Functionality:**
    -   It iterates through the `modules` directory.
    -   It imports each module and registers it, making it available to the CLI.

### Pipeline Executor

-   **File:** `src/mlarena/core/executor.py`
-   **Purpose:** The executor is responsible for running the pipeline in the correct order, handling dependencies between modules.
-   **Functionality:**
    -   Takes the requested module from the CLI.
    -   Resolves the dependency graph for the module.
    -   Executes the required modules in the correct sequence.

### Modules

-   **Directory:** `src/mlarena/modules/`
-   **Purpose:** Each file in this directory represents a self-contained stage in the ML pipeline (e.g., `eda.py`, `model.py`, `tune.py`).
-   **Structure:**
    -   Each module is a `typer` application itself.
    -   Each module defines its own command-line arguments.
    -   Modules can declare dependencies on other modules.

## 3. Workflow

1.  A user runs `uv run python scripts/mla.py <module> --project ...`.
2.  The `mla.py` script calls the main function in `src/mlarena/cli/main.py`.
3.  The core CLI application discovers all modules using the `ModuleRegistry`.
4.  `typer` parses the command-line arguments and identifies the requested module.
5.  The `PipelineExecutor` resolves the dependencies for the requested module.
6.  The `PipelineExecutor` executes the necessary modules in order, passing the appropriate arguments.
7.  Each module runs its specific task (e.g., running EDA, training a model).
