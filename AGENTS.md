## Repository Overview

Kaggle competition framework with modular pipeline architecture. The system orchestrates experiments through a module registry pattern with state-driven execution.

**Core Entry Point**: `scripts/mla.py` -> `src/mlarena/cli/main.py`

## Architecture

### Four-Layer System

```
Scripts Layer (scripts/mla.py)
  -> Core Package (src/mlarena/)
     - core/registry.py: Decorator-based module registration
     - core/pipeline.py: Execution orchestration + dependency resolution
     - core/experiment.py: State persistence (state.json)
     - modules/: init, eda, preprocess, model, predict, submit, fetch-score
  -> Project Layer (projects/kaggle/[competition]/)
     - code/utils/config.py: Competition constants (TARGET_COLUMN, etc.)
     - code/models/: Model implementations (train/predict)
     - code/preprocessing/: Custom preprocessing modules (fit_transform/transform)
     - templates/: model/*.yaml, preprocess/*.yaml (override globals)
  -> Tracking Layer
     - experiments/{id}/state.json: Module status tracking
     - submissions/submissions.json: Leaderboard tracking
     - Git hash capture on every run
```

### Module Registry Pattern

**File**: `src/mlarena/core/registry.py`

All modules self-register via decorator:

```python
@ModuleRegistry.register
class MyModule(BaseModule):
    name = "my_module"
    dependencies = {"other_module"}

    def execute(self) -> ModuleResult:
        # Implementation
```

Registry discovers modules on startup (`src/mlarena/cli/main.py`) and dynamically creates CLI subcommands.

### State Management

**File**: `src/mlarena/core/experiment.py`

Three experiment types:

1. **Fixed** (`experiments/init/`, `experiments/eda/`): Setup modules, always overwrite
2. **Named** (`experiments/pre-{template}/`): Preprocessing, cached if input unchanged
3. **Timestamped** (`experiments/exp-YYYYMMDD-HHMMSS/`): Pipeline runs, one per model

State structure:

```python
{
    "experiment_id": "exp-20251208-002830",
    "modules": {
        "model": {
            "status": "completed",  # pending|running|completed|failed
            "payload": {"local_cv_score": 0.923},
            "invocation": {...},
            "error": null
        }
    },
    "git": {"hash": "...", "dirty": false}
}
```

### Pipeline Execution

**File**: `src/mlarena/core/pipeline.py`

1. Dependency resolution via topological sort
2. Mark stale runs as failed
3. Skip completed modules (unless `--force`)
4. File-lock state.json during execution
5. Capture module output (payload, artifacts, errors)

### Template Resolution

**Order**: Project templates -> Global templates

**Structure**: Each template in its own file, filename = template name (without `.yaml` extension)

**Directories**:
- Global: `src/mlarena/templates/model/*.yaml`, `src/mlarena/templates/preprocess/*.yaml`
- Project: `projects/kaggle/{comp}/templates/model/*.yaml`, `projects/kaggle/{comp}/templates/preprocess/*.yaml`

**Template File Format** (direct content, no `templates:` wrapper):

```yaml
# src/mlarena/templates/model/cpu-dev-5m.yaml
model: autogluon_baseline
config:
  preset: medium
  time_limit: 300
  use_gpu: false
```

**Meta-templates** (preprocessing chains):

```yaml
# src/mlarena/templates/preprocess/full-pipeline.yaml
chain: [imputer, encoder, feature_selector]
```

## Integration Points

### 1. Model Files

**Location**: `projects/kaggle/{comp}/code/models/{name}.py` OR `src/mlarena/defaults/models/{name}.py`

**Interface**:

```python
def train(train_df, val_df, config, artifacts=None):
    """
    artifacts may include:
      - orig_df: external/original dataset from preprocessing (optional)
      - sample_weight: dataframe/series with sample weights (optional)

    Returns:
      - predictor  OR  (predictor, training_summary_dict)
    """
```

**Resolution**: Project-local takes precedence. Error if both exist.

### 2. Preprocessing Files

**Location**: `projects/kaggle/{comp}/code/preprocessing/{name}.py` OR `src/mlarena/defaults/preprocessing/{name}.py`

**Interface**:

```python
def fit_transform(train_df, val_df, test_df, config, orig_df=None):
    """
    Returns either:
      (train_df, val_df, test_df, state_dict)                  # legacy
      (train_df, val_df, test_df, orig_df, state_dict)         # supports external/orig dataset
    """
```

**Chaining**: Multiple templates run sequentially; outputs flow through `train_processed.csv`, `test_processed.csv`,
and (if present) `orig_processed.csv`.

**Global preprocess submodules** (`src/mlarena/defaults/preprocessing/*.py`, may be overridden per-project):

- `external_dataset` (load/align external/orig; no merge)
- `sanity_check` (basic data validation)
- `imputer`, `missing_values_imputer` (missing value handling)
- `rare_category_handler` (rare category bucketing)
- `encoder`, `categorical_encoder` (categorical encoding)
- `feature_engineer` (feature creation)
- `feature_selector` (feature selection)
- `imbalance_handler` (class imbalance handling / weights)
- `drift_detector`, `adversarial_validation` (drift/AV utilities + weights)
- `datetime_handler` (datetime parsing/features)
- `outlier_handler` (outlier handling)
- `scaler` (scaling/normalization)
- `target_transformer` (target transforms)
- `identity`, `noop` (pass-through / smoke)

### External/Original Dataset Handling

- Use `external_dataset` to load/align an external dataset (produces `orig_processed.csv`).
- Do not concatenate train+orig during preprocessing; if you want to train on both, merge inside the model (e.g. a model implementation that consumes `orig_df`).

### 3. Project Config

**File**: `projects/kaggle/{comp}/code/utils/config.py`

**Critical constants**:

```python
PROJECT_ROOT = Path(__file__).parent.parent.parent
TARGET_COLUMN = "target"              # MUST match data
AUTOGLUON_PROBLEM_TYPE = "binary"     # binary|regression|multiclass
AUTOGLUON_EVAL_METRIC = "roc_auc"
ID_COLUMN = "id"                      # Auto-detected from sample_submission
SUBMISSION_PROBAS = False             # False=labels, True=probabilities
```

**Loading**: `src/mlarena/utils/project.py` imports via `importlib`.

### 4. Submission Creation

**Project wrapper**: `code/utils/submission.py` -> **Core logic**: `src/kaggle_tools/submission.py`

Automatically:
- Validates submission format
- Creates timestamped CSV
- Logs experiment (git hash + code snapshot)
- Tracks submission (links to experiment)

### 5. Submission Queue

**Script**: `scripts/submission_queue.py`
**Queue file**: `projects/kaggle/{project}/submissions/queue.json`
**Utilities**: `src/mlarena/utils/kaggle_api.py` (Kaggle API utilities)

**Submit module parameter**: `submit.queue_submit=true`

**Features**:
- Queue submissions for later batch processing
- Duplicate detection via Kaggle API (prevents re-submission)
- Error tracking with timestamps
- Thread-safe operations (FileLock)
- Auto-cleanup on successful fetch-score (with --continue-flow)

**Queue commands**:
```bash
# List queue
python scripts/submission_queue.py --project {project} list

# Submit by queue #, experiment-id, or CSV filename
python scripts/submission_queue.py --project {project} submit 1
python scripts/submission_queue.py --project {project} submit exp-20251226-103504
python scripts/submission_queue.py --project {project} submit submission.csv.gz

# Submit with auto fetch-score (waits 30s, fetches score, removes on success)
python scripts/submission_queue.py --project {project} submit 1 --continue-flow

# Remove from queue
python scripts/submission_queue.py --project {project} remove 1
```

**Queue entry structure**:
```json
{
  "id": 1,
  "experiment_id": "exp-20251226-103504",
  "submission_file": "experiments/exp-20251226-103504/artifacts/predict/submission.csv.gz",
  "kaggle_message": "0.71234 | feat: 42 | exp-20251226-103504 | model | preprocess | submission.csv.gz",
  "project": "project-name",
  "competition": "competition-slug",
  "added_timestamp": "20251226 143022",
  "submission_attempts": [{"timestamp": "...", "success": true/false, "error": "..."}],
  "last_error": null,
  "status": "pending|submitted|completed|failed"
}
```

## Auto-Flow Logic

**File**: `src/mlarena/cli/main.py`

**Prerequisites** (manual, one-time):
- `mla init --project <name>` - initialize project structure and download data
- `mla eda --project <name>` - run exploratory data analysis
- Auto-flow validates both init and eda are completed before starting
- Validation fails with clear error message if prerequisites missing

**Phase 1: Preprocessing**
- Run preprocessing chain (each named `pre-{template}`)
- Skip if completed + input unchanged (unless `--force`)

**Phase 2: Pipeline**
- Model creates NEW timestamped experiment_id
- Predict/submit/fetch-score reuse same experiment_id
- All run sequentially, fail fast
- Auto-commit: `"auto-flow({project}): {modules} | local {cv} | public {score}"`

## Project Structure

```
projects/kaggle/[competition]/
  code/
    utils/
      config.py           # Competition constants
      submission.py       # Wrapper for src/kaggle_tools
    models/
      autogluon_baseline.py
    preprocessing/
      custom_step.py
  templates/
    model/                # Override globals (individual files)
      custom-model.yaml
    preprocess/           # Override globals (individual files)
      custom-preprocess.yaml
  data/
    train.csv.gz, test.csv.gz, sample_submission.csv.gz  # Compressed CSV files
  experiments/
    init/state.json       # Fixed: setup
    eda/state.json        # Fixed: setup
    pre-{template}/       # Named: preprocessing (single step or chain)
      artifacts/preprocess/
        train_processed.csv.gz, test_processed.csv.gz
        orig_processed.csv.gz  # If external dataset used
        sample_weights.csv.gz  # If AV/imbalance handling used
    exp-YYYYMMDD-HHMMSS/  # Timestamped: pipeline
      artifacts/
        model/leaderboard.csv.gz
        predict/submission-TIMESTAMP.csv.gz
  submissions/
    submissions.json      # Leaderboard tracking
    queue.json            # Submission queue (batch processing)
```

## CSV Compression

All CSV files in the framework use gzip compression (`.csv.gz`) for efficient storage and I/O.

**Benefits:**
- **Space savings**: 60-90% reduction in file size
- **Faster I/O**: Less data to read from disk
- **Automatic handling**: Pandas `compression='infer'` detects format from file extension
- **Kaggle compatible**: Kaggle API accepts `.csv.gz` submissions

**Backward Compatibility:**
- Code automatically handles both `.csv` and `.csv.gz` files
- Fallback logic in `src/mlarena/utils/project.py` prioritizes `.csv.gz`, falls back to `.csv`
- All modules use `compression='infer'` for seamless read/write

**Migration:**
To compress existing CSV files in a project:
```bash
# Dry-run (preview only)
python scripts/migrate_csv_to_gz.py --project PROJECT_NAME --dry-run

# Compress data/ and submissions/
python scripts/migrate_csv_to_gz.py --project PROJECT_NAME

# Include experiments/ directory
python scripts/migrate_csv_to_gz.py --project PROJECT_NAME --include-experiments

# All projects at once
python scripts/migrate_csv_to_gz.py --all-projects --include-experiments
```

**Manual compression/decompression:**
```bash
# Compress
gzip data/train.csv  # Creates train.csv.gz, removes train.csv

# Decompress
gunzip data/train.csv.gz  # Creates train.csv, removes train.csv.gz
```

## Critical Pitfalls

1. **Uncommitted changes** -> Git hash missing, can't reproduce
2. **Wrong TARGET_COLUMN** -> AutoGluon fails
3. **Module already completed** -> won’t re-run (use `--force`)
4. **Model/preprocess ambiguity** -> exists both project-local and global
5. **Wrong directory** -> file not found (run from repo root)
6. **Chrome not running** -> `fetch-score` fails (remote-debug Chrome)
7. **External dataset merge in preprocess** -> don’t do it; keep orig separate and merge in model stage if needed

## Development Guidelines

### Code Style
- 4-space indentation
- `snake_case` for functions/modules
- `CamelCase` for classes
- Constants in `code/utils/config.py`
- Documentation should be written in English

### Smoke Test

**Quick model-only test:**
```bash
uv run python scripts/mla.py model --project <proj> --profile smoke skip_submit=true
```

**Full pipeline test (preprocess → model → predict → submit):**
```bash
# Prerequisites (one-time setup):
uv run python scripts/mla.py init -p <project>
uv run python scripts/mla.py eda -p <project>

# Auto-flow pipeline:
uv run python scripts/mla.py -p <project>
```
- Example: `uv run python scripts/mla.py -p Titanic`
- **Duration**: ~2-3 minutes for complete flow (after init/eda)
- Auto-runs: preprocess → model → predict → submit → fetch-score
- Use `--profile smoke` for faster testing (60s time limit instead of default)
- **Note**: init and eda must be run manually first

### Commit Format
- `feat:`, `fix:`, `experiment:`
- Example: `"feat: add AutoGluon baseline"`

### Never Commit
- Data files (`data/*.csv`, `*.zip`)
- Model outputs (`AutogluonModels/`, `*.pkl`)
- Experiment artifacts (e.g., model files, predictions, large intermediates) **excluding** `experiments/**/state.json`

### Always Commit
- Code changes
- Config updates
- Documentation
- Experiment state tracking (`experiments/**/state.json`)

## See Also

- `docs/MLA_WORKFLOW_GUIDE.md`
- `docs/configs.md`
- `README.md`
- Correct test for the Titanic project: `uv run python scripts/mla.py -p Titanic` (no need for `--profile smoke` since the dataset is small; use `--force` only if you want to test AI behavior)
