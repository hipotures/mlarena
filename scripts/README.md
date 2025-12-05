# Competition Tools

Centralne, uniwersalne narzędzia dla wszystkich konkursów Kaggle.

## Zawartość

### `submissions_tracker.py`
System śledzenia submissions z local CV, public i private scores.

**Użycie z linii komend:**

```bash
# Dodaj nową submission
python scripts/submissions_tracker.py add \
    --project playground-series-s5e11 \
    submission-20231116-model-v1.csv \
    autogluon-medium \
    --local-cv 0.85432 \
    --cv-std 0.00123 \
    --notes "Initial baseline"

# Zaktualizuj wyniki z leaderboard
python scripts/submissions_tracker.py update \
    --project playground-series-s5e11 \
    1 \
    --public 0.85123 \
    --private 0.84987

# Wyświetl listę submissions
python scripts/submissions_tracker.py list \
    --project playground-series-s5e11 \
    --sort-by public_score \
    --limit 10

# Eksportuj do CSV
python scripts/submissions_tracker.py export \
    --project playground-series-s5e11
```

**Użycie w kodzie:**

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from submissions_tracker import SubmissionsTracker

# Inicjalizacja
project_root = Path(__file__).parent.parent.parent
tracker = SubmissionsTracker(project_root)

# Dodaj submission
tracker.add_submission(
    filename="submission-20231116.csv",
    model_name="autogluon-medium",
    local_cv_score=0.85432,
    cv_std=0.00123,
    notes="Baseline model with default params"
)

# Zaktualizuj scores
tracker.update_scores(
    submission_id=1,
    public_score=0.85123,
    private_score=0.84987
)

# Wyświetl top 5 submissions
tracker.display_submissions(limit=5, sort_by="public_score")

# Pobierz best submissions
best = tracker.get_best_submissions(metric="public_score", top_n=3)
```

## Integracja z projektem

Narzędzia automatycznie integrują się z funkcją `create_submission()` w każdym projekcie:

```python
from code.utils.submission import create_submission

# Submission zostanie automatycznie dodana do trackera
create_submission(
    predictions=y_pred,
    test_ids=test_df['id'],
    model_name="lgbm-v2",
    local_cv_score=0.85432,
    cv_std=0.00123,
    notes="LGBM with tuned hyperparameters",
    config={"learning_rate": 0.01, "n_estimators": 1000}
)
```

## Format danych

Tracker przechowuje dane w `[projekt]/submissions/submissions.json`:

```json
[
  {
    "id": 1,
    "timestamp": "2023-11-16 23:30:00",
    "filename": "submission-20231116233000.csv",
    "model_name": "autogluon-medium",
    "local_cv_score": 0.85432,
    "cv_std": 0.00123,
    "public_score": 0.85123,
    "private_score": null,
    "notes": "Baseline model",
    "config": {
      "preset": "medium_quality",
      "time_limit": 3600
    }
  }
]
```

## Przydatne funkcje

### Porównanie local CV vs public score
```python
tracker.display_submissions(sort_by="local_cv_score")
# Zobacz, czy local CV koreluje z public score
```

### Export do analizy
```python
tracker.export_to_csv()
# Otwórz submissions_tracking.csv w Excel/Pandas do analizy
```

### Znajdź best submission
```python
best_local = tracker.get_best_submissions("local_cv_score", top_n=1)[0]
best_public = tracker.get_best_submissions("public_score", top_n=1)[0]

if best_local['id'] != best_public['id']:
    print("⚠️  Local CV nie koreluje z public score!")
```

## All Scripts Overview

### User-Facing Scripts (Main CLI Tools)

These are the primary scripts you'll interact with directly:

#### `experiment_manager.py`
Main orchestrator for the modular experiment pipeline (EDA → Model → Submit → Fetch).
```bash
# Initialize project
uv run python scripts/experiment_manager.py init-project --project <name>

# Run EDA
uv run python scripts/experiment_manager.py eda --project <name>

# Train model
uv run python scripts/experiment_manager.py model --project <name> \
    --experiment-id exp-... --template gpu-dev-5m

# List experiments
uv run python scripts/experiment_manager.py list --project <name>
```

#### `ml_runner.py`
Generic ML runner with template-based training. Used internally by `experiment_manager.py`.
Supports any model module that implements `train()` and `predict()` functions.

#### `autogluon_runner.py`
Direct AutoGluon runner with built-in templates (fast-cpu, dev-gpu, best-gpu, etc.).
Alternative to using experiment_manager for quick iterations.

#### `submission_workflow.py`
End-to-end submission pipeline: upload to Kaggle + score scraping via Playwright/CDP.
```bash
# Upload and fetch score
uv run python scripts/submission_workflow.py pull-score \
    --project <name> \
    --filename submission-20251117.csv \
    --experiment-id exp-...
```

#### `submissions_tracker.py`
Track submissions with local CV, public, and private scores (documented above).

#### `experiment_logger.py`
Git-based experiment tracking with code snapshots and reproducibility.
```bash
# List experiments
python scripts/experiment_logger.py --project <name> list

# Show experiment details
python scripts/experiment_logger.py --project <name> show exp-...

# Restore code snapshot
python scripts/experiment_logger.py --project <name> restore exp-...
```

#### `optuna_runner.py`
Hyperparameter tuning orchestrator using Optuna with TPE sampler.
Supports XGBoost, LightGBM, CatBoost with presets (quick, thorough, extreme).

#### `stacking_runner.py`
Ensemble and stacking framework for combining multiple model predictions.
Supports weighted blending, rank averaging, power averaging, and meta-learning.

#### `feature_runner.py`
Feature engineering orchestrator with two-stage pipeline (feat_stage + cv_stage).
Prevents data leakage by separating global and per-fold transformers.

#### `kaggle_scraper.py`
Scrapes Kaggle leaderboard/submissions via Playwright/CDP.
Requires Chrome with remote debugging port (--remote-debugging-port=9222).

### Infrastructure Scripts (Internal Utilities)

These scripts are used internally by the main tools:

#### `template_loader.py`
Loads and merges global + project templates from YAML files.
Validates template structure and handles project overrides.

#### `pipeline_loader.py`
Loads declarative pipeline definitions from YAML.
Default pipeline: EDA → preprocess → feat → model → predict → tune → stack → submit → fetch-score.

#### `feature_eda.py`
Feature analysis and visualization utilities.
Used by feature engineering pipeline for exploratory analysis.

#### `generate_oof_optuna.py`
Generates out-of-fold predictions for Optuna tuning studies.
Specialized helper for hyperparameter optimization workflows.

#### `ai_helper.py`
AI assistant utilities for code generation and task automation.
Experimental tool for Claude Code integration.

#### `template_configurator.py`
Interactive UI tool for configuring experiment templates.
Helps create custom model.yaml and preprocess.yaml configurations.

### Script Categories Summary

**Primary Entry Points** (use these directly):
- `experiment_manager.py` - Main orchestrator ⭐
- `autogluon_runner.py` - Direct AutoGluon runner
- `submissions_tracker.py` - Score tracking
- `experiment_logger.py` - Git-based reproducibility

**Advanced Workflows**:
- `optuna_runner.py` - Hyperparameter tuning
- `stacking_runner.py` - Model ensembling
- `feature_runner.py` - Feature engineering

**Automation**:
- `submission_workflow.py` - Kaggle upload + score scraping
- `kaggle_scraper.py` - Leaderboard scraping

**Internal Infrastructure** (usually called by other scripts):
- `ml_runner.py` - Generic model training
- `template_loader.py` - Template management
- `pipeline_loader.py` - Pipeline definitions
- `feature_eda.py` - Feature utilities
- `generate_oof_optuna.py` - OOF predictions
- `ai_helper.py` - AI assistance
- `template_configurator.py` - Template UI

For detailed usage of each script, run: `python scripts/<script_name>.py --help`
