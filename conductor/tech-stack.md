# Technology Stack - Kaggle ML Arena

## Core Technologies
- **Programming Language:** Python 3.12+
- **Dependency & Environment Management:** `uv` (Fast Python package installer and resolver)

## Machine Learning & Data Science
- **Frameworks:**
  - **AutoGluon:** For automated machine learning and ensemble stacking.
  - **Scikit-Learn:** Core library for preprocessing modules and model interfaces.
  - **Optuna:** Hyperparameter optimization and automated pipeline search.
- **Data Handling:** Pandas, NumPy.

## Infrastructure & Configuration
- **Configuration Management:** 
  - **OmegaConf:** Hierarchical YAML configuration.
  - **Pydantic:** Data validation and settings management using Python type annotations.
- **CLI & UI:**
  - **Rich:** For beautiful console formatting and tables.
  - **Textual:** For TUI-based task management and monitoring.
- **Automation:**
  - **Playwright:** Headless browser automation for fetching Kaggle scores.
  - **Kaggle API:** Official tool for data download and submissions.

## Quality Assurance
- **Testing:** `pytest` for unit and e2e testing.
- **Static Analysis:** `ruff` or `flake8` (implied by typical `uv` setups).
