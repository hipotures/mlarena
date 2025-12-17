# MLArena Utility Scripts

Standalone helper scripts for MLArena workflow management. These scripts are not part of the main pipeline but provide useful utilities for project maintenance and advanced workflows.

## Available Utilities

### 🧹 `clean.py` - Artifact Cleanup

Removes AutoGluon artifacts and model files to save disk space while preserving experiment tracking data.

**Usage:**
```bash
# Remove AutoGluon models and cache files
uv run python scripts/utils/clean.py --project Titanic --artifacts

# Remove all experiment artifacts
uv run python scripts/utils/clean.py --project Titanic --experiments

# Dry run to see what would be deleted
uv run python scripts/utils/clean.py --project Titanic --artifacts --dry-run
```

**What it removes:**
- `AutogluonModels/` - Trained model files
- `.predictor/` - AutoGluon cache
- `experiments/*/artifacts/` - Generated artifacts (preserves state.json)

**What it preserves:**
- `experiments/*/state.json` - Experiment tracking
- `submissions/submissions.json` - Submission history
- `data/` - Competition data

---

### 🔄 `sync.py` - Project Synchronization

Synchronizes Kaggle projects between local and remote machines using rsync.

**Usage:**
```bash
# Sync project from local to remote
uv run python scripts/utils/sync.py --project Titanic --from local --to remote

# Sync from remote to local
uv run python scripts/utils/sync.py --project Titanic --from remote --to local

# Dry run
uv run python scripts/utils/sync.py --project Titanic --from local --to remote --dry-run
```

**Configuration:**
Edit the script to configure your remote hosts and paths.

---

### ⚖️ `av_weights_mix.py` - Adversarial Validation with External Data

Advanced preprocessing helper for competitions with external datasets. Performs adversarial validation to generate sample weights, optionally merging original datasets.

**Usage:**
```bash
# Basic adversarial validation
uv run python scripts/utils/av_weights_mix.py \
  --project playground-series-s5e12 \
  --orig-path data/diabetes_dataset.csv

# Union mode with source flag
uv run python scripts/utils/av_weights_mix.py \
  --project playground-series-s5e12 \
  --orig-path data/diabetes_dataset.csv \
  --mode union \
  --source-flag is_original

# Advanced configuration
uv run python scripts/utils/av_weights_mix.py \
  --project playground-series-s5e12 \
  --orig-path data/diabetes_dataset.csv \
  --mode align \
  --time-limit 600 \
  --presets best_quality \
  --included-model-types GBM XGB CAT \
  --output data/av_weights_custom.csv
```

**Modes:**
- `align` (default): Keep only competition columns, fill missing with NA, weights for original train only
- `union`: Take union of all columns, optionally add source flag column

**Output:**
- Saves sample weights to specified path (default: `data/train_av_weights_mix.csv`)
- Use these weights in model training for better generalization

---

## Integration with Main Pipeline

These utilities are **standalone** and do not integrate directly with the `mla.py` pipeline. They are designed for:

- **Post-experiment cleanup** (`clean.py`)
- **Multi-machine workflows** (`sync.py`)
- **Advanced preprocessing scenarios** (`av_weights_mix.py`)

For standard MLArena workflows, use `scripts/mla.py` instead.

## Core Pipeline Scripts

The main MLArena entry points are located in `scripts/`:

- **`mla.py`** - Main CLI for running experiments (init → eda → preprocess → model → submit)
- **`experiment_logger.py`** - Experiment tracking (used internally)
- **`submissions_tracker.py`** - Submission history management (CLI + library)
- **`template_loader.py`** - YAML template loader (used internally)
- **`ai_helper.py`** - AI-powered config generation (used during init)

---

## See Also

- [Main README](../../README.md) - MLArena overview
- [MLA Workflow Guide](../../docs/MLA_WORKFLOW_GUIDE.md) - Pipeline documentation
- [CLAUDE.md](../../CLAUDE.md) - Development guidelines
