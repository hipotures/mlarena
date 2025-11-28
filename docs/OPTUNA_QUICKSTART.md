# Optuna System - Quick Start

5-minute guide to get started with hyperparameter tuning in Kaggle competitions.

## Installation

```bash
# System is pre-installed with kaggle-fork1
uv sync

# Verify installation
uv run python -c "from kaggle_tools import optuna, preprocessing, stacking; print('✓ OK')"
```

## Basic Workflow

### Step 1: Initialize Project

```bash
# Create project structure + download data
uv run python scripts/experiment_manager.py init-project \
    --project my-competition
```

### Step 2: Tune Hyperparameters

```bash
# Quick smoke test (20 trials, 30min)
uv run python scripts/optuna_runner.py \
    --project my-competition \
    --model xgboost \
    --preset quick

# Production run (100 trials, 2h)
uv run python scripts/optuna_runner.py \
    --project my-competition \
    --model xgboost \
    --preset thorough
```

### Step 3: Train with Best Params

```bash
# Uses cached best_params.json from tuning
uv run python scripts/experiment_manager.py model \
    --project my-competition \
    --model xgboost_optuna \
    --auto-submit
```

## Common Commands

**Feature Engineering:**
```bash
# Transform features
uv run python scripts/feature_runner.py \
    --project my-competition \
    --feature-set baseline
```

**Model Blending:**
```bash
# Ensemble multiple models
uv run python scripts/stacking_runner.py \
    --project my-competition \
    --models model1.csv model2.csv model3.csv \
    --blend-method weighted \
    --blend-weights 0.5 0.3 0.2
```

**Dashboard:**
```bash
# Launch Optuna dashboard (http://localhost:8080)
uv run python scripts/optuna_runner.py \
    --project my-competition \
    --model xgboost \
    --preset thorough \
    --dashboard
```

## Configuration

Edit `projects/kaggle/my-competition/configs/project.yaml`:

```yaml
optuna:
  n_trials: 100      # Number of optimization trials
  timeout: 7200      # Max time (seconds)
  cv_folds: 5        # Cross-validation folds

  param_space:
    xgboost:
      learning_rate: [0.001, 0.3, log]
      max_depth: [3, 10, int]
```

## Presets

| Preset | Trials | Time | Use Case |
|--------|-------:|------|----------|
| `quick` | 20 | 30 min | Smoke test |
| `thorough` | 100 | 2 hours | Production |
| `extreme` | 500 | 24 hours | Final push |

## Example: Complete Pipeline

```bash
#!/bin/bash
PROJECT="playground-series-s5e11"

# 1. Feature engineering
uv run python scripts/experiment_manager.py feat \
    --project $PROJECT \
    --feature-set baseline

# 2. Tune 3 models (parallel)
for MODEL in xgboost lightgbm catboost; do
    uv run python scripts/optuna_runner.py \
        --project $PROJECT \
        --model $MODEL \
        --preset thorough \
        --use-transformed &
done
wait

# 3. Ensemble best results
uv run python scripts/stacking_runner.py \
    --project $PROJECT \
    --models xgb.csv lgb.csv cat.csv \
    --blend-method rank
```

## Next Steps

- **Full Guide:** [OPTUNA_GUIDE.md](./OPTUNA_GUIDE.md) - Complete documentation
- **Examples:** See `tests/test_optuna_e2e.py` for code examples
- **Templates:** `config/templates/kaggle_competition/code/models/`

## Troubleshooting

**Study not resuming?**
```bash
# Check storage path
ls projects/kaggle/my-competition/experiments/optuna.db

# Use --resume flag
uv run python scripts/optuna_runner.py --resume ...
```

**Out of memory?**
```bash
# Reduce CV folds
--cv-folds 3

# Run trials sequentially
--n-jobs 1
```

**Need help?**
```bash
# Command help
uv run python scripts/optuna_runner.py --help

# Full docs
cat docs/OPTUNA_GUIDE.md
```

## Pro Tips

1. **Always commit before experiments** - Enables reproducibility via git hash
2. **Use dashboard for long runs** - Monitor progress at http://localhost:8080
3. **Start with quick preset** - Validate setup before thorough tuning
4. **Blend diverse models** - Best results from complementary models (correlation < 0.95)

## Architecture

```
ExperimentManager
  ├── feat   → FeaturePipeline (data leakage protection)
  ├── tune   → Optuna StudyManager (hyperparameter search)
  ├── model  → Train with best params
  └── stack  → Ensemble predictions
```

**Key Features:**
- SQLite persistence (resume interrupted runs)
- Parquet storage (50% smaller than CSV)
- Git-based reproducibility
- Optuna dashboard visualization
- Data leakage protection (two-stage pipeline)

---

**For complete documentation, see [OPTUNA_GUIDE.md](./OPTUNA_GUIDE.md)**
