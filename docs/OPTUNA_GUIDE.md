# Optuna Hyperparameter Tuning System - Complete Guide

Comprehensive guide to using the Optuna system for hyperparameter tuning, feature engineering, and model ensembling in Kaggle competitions.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Feature Engineering](#feature-engineering)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Model Ensembling](#model-ensembling)
- [Complete Workflow Examples](#complete-workflow-examples)
- [Configuration Reference](#configuration-reference)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

The Optuna system provides three core capabilities:

1. **Feature Engineering** (`feat` module) - Transform features with data leakage protection
2. **Hyperparameter Tuning** (`tune` module) - Optimize XGBoost/LightGBM/CatBoost with Optuna
3. **Model Ensembling** (`stack` module) - Blend predictions from multiple models

### Architecture

```
ExperimentManager (orchestration)
    ├── feat    → feature_runner.py → FeaturePipeline
    ├── tune    → optuna_runner.py → StudyManager + CVObjective
    └── stack   → stacking_runner.py → Blenders + MetaLearner
```

### Key Features

- **Data Leakage Protection** - Two-stage pipeline (feat_stage + cv_stage)
- **Persistent Storage** - SQLite for Optuna studies, Parquet for transformed data
- **Resume Support** - Continue interrupted tuning sessions
- **Dashboard** - Optuna web UI for visualization (http://localhost:8080)
- **Experiment Tracking** - Git-based reproducibility system

## Quick Start

### 1. Setup New Competition

```bash
# Initialize project structure + download data
uv run python scripts/experiment_manager.py init-project \
    --project playground-series-s5e11

# Installs project.yaml with Optuna defaults
```

### 2. Feature Engineering

```bash
# Run feature engineering
uv run python scripts/experiment_manager.py feat \
    --project playground-series-s5e11 \
    --feature-set baseline

# Output: experiments/{experiment_id}/features/
#   - train_transformed.parquet
#   - test_transformed.parquet
#   - pipeline/feat_stage.pkl
```

### 3. Hyperparameter Tuning

```bash
# Tune XGBoost with "thorough" preset (100 trials, 2h)
uv run python scripts/experiment_manager.py tune \
    --project playground-series-s5e11 \
    --experiment-id exp-20251127-123456 \
    --model xgboost \
    --preset thorough \
    --use-transformed

# Output: experiments/{experiment_id}/optuna/
#   - best_params_xgboost.json
#   - trials_xgboost.csv
#   - optuna.db (SQLite study)
```

### 4. Model Ensembling

```bash
# Blend multiple model predictions
uv run python scripts/stacking_runner.py \
    --project playground-series-s5e11 \
    --models submission1.csv submission2.csv submission3.csv \
    --strategy blend \
    --blend-method weighted \
    --blend-weights 0.5 0.3 0.2

# Output: submissions/ensemble-blend-{timestamp}.csv
```

## Feature Engineering

### Two-Stage Pipeline

The system uses a two-stage pipeline to prevent data leakage:

**feat_stage** (Global, Safe):
- Fitted on FULL train set
- Applied to both train and test
- Examples: LogTransformer, StandardScaler, Polynomial Features

**cv_stage** (Per-Fold, Risky):
- Fitted SEPARATELY per CV fold
- CRITICAL: Prevents validation leakage
- Examples: TargetEncoding, FrequencyEncoding

### Configuration

Create `configs/preprocessing/baseline.yaml`:

```yaml
preprocessing:
  storage_format: parquet
  compression: snappy

  feat_stage:
    enabled: true
    transformers:
      - type: LogTransformer
        columns: [feature1, feature2]
      - type: StandardScalerTransformer
        columns: [feature3, feature4]

  cv_stage:
    enabled: true
    transformers:
      - type: TargetEncodingTransformer
        columns: [category_col]
        target: target
        smoothing: 10.0
        min_samples_leaf: 5
```

### Available Transformers

**feat_stage (Safe):**
- `LogTransformer` - Log(1+x) transformation
- `StandardScalerTransformer` - Zero mean, unit variance
- `PolynomialFeaturesTransformer` - Interaction features
- `BinningTransformer` - Quantile-based binning

**cv_stage (Risky):**
- `TargetEncodingTransformer` - Mean target encoding
- `FrequencyEncodingTransformer` - Category frequency
- `TargetAggregatesTransformer` - Target statistics per group
- `WeightOfEvidenceTransformer` - WoE for binary classification

### Usage Examples

**Standalone Feature Engineering:**

```bash
# Preview transformations
uv run python scripts/feature_runner.py \
    --project playground-series-s5e11 \
    --feature-set baseline \
    --preview

# Apply and save
uv run python scripts/feature_runner.py \
    --project playground-series-s5e11 \
    --feature-set baseline
```

**Integration with Experiment Manager:**

```bash
uv run python scripts/experiment_manager.py feat \
    --project playground-series-s5e11 \
    --experiment-id exp-20251127-123456 \
    --feature-set advanced
```

### Data Leakage Protection

**CRITICAL TEST:**

```python
from kaggle_tools.preprocessing.cv_stage import TargetEncodingTransformer

# Simulate CV fold
train_fold = train_df.iloc[:400]
val_fold = train_df.iloc[400:]  # Contains category B

# Fit ONLY on train_fold (category B is unseen)
encoder = TargetEncodingTransformer(columns=['cat'], target='target')
encoder.fit(train_fold)

# Transform validation
val_transformed = encoder.transform(val_fold)

# ✓ CORRECT: B gets global mean from train_fold
# ✗ LEAKAGE: B would get its own target mean from val_fold
```

## Hyperparameter Tuning

### Presets

Three built-in presets (from `configs/presets/`):

| Preset     | Trials | Timeout | Use Case |
|------------|-------:|---------|----------|
| `quick`    | 20     | 30 min  | Smoke test |
| `thorough` | 100    | 2 hours | Production |
| `extreme`  | 500    | 24 hours | Final push |

### Optuna Configuration

Default configuration (`configs/project.yaml`):

```yaml
optuna:
  storage: "sqlite:///experiments/optuna.db"
  study_name: "xgboost_study"
  direction: maximize

  n_trials: 50
  timeout: null  # No timeout
  n_jobs: 1      # Sequential trials

  cv_folds: 5
  early_stopping_rounds: 50

  sampler: TPESampler  # Tree-structured Parzen Estimator
  pruner: MedianPruner  # Kill unpromising trials early

  param_space:
    xgboost:
      learning_rate: [0.001, 0.3, log]
      max_depth: [3, 10, int]
      subsample: [0.5, 1.0, float]
      # ... more parameters
```

### Parameter Space Format

Format: `[min, max, type]` where type is:
- `"int"` - Integer range
- `"float"` - Continuous range
- `"log"` - Log-uniform distribution
- `["choice1", "choice2"]` - Categorical

Examples:

```yaml
learning_rate: [0.001, 0.3, log]  # Log scale (good for learning rates)
max_depth: [3, 10, int]           # Integer
subsample: [0.5, 1.0, float]      # Continuous
booster: ["gbtree", "dart"]       # Categorical
```

### Usage Examples

**Basic Tuning:**

```bash
uv run python scripts/optuna_runner.py \
    --project playground-series-s5e11 \
    --model xgboost \
    --preset thorough
```

**With Dashboard:**

```bash
# Launch dashboard (http://localhost:8080)
uv run python scripts/optuna_runner.py \
    --project playground-series-s5e11 \
    --model lightgbm \
    --preset thorough \
    --dashboard
```

**Custom Parameters:**

```bash
uv run python scripts/optuna_runner.py \
    --project playground-series-s5e11 \
    --model catboost \
    --n-trials 200 \
    --timeout 7200 \
    --cv-folds 10
```

**Resume Interrupted Study:**

```bash
uv run python scripts/optuna_runner.py \
    --project playground-series-s5e11 \
    --model xgboost \
    --resume
```

### Integration with Model Templates

Model templates (e.g., `code/models/xgboost_optuna.py`) automatically:

1. Check for cached `best_params.json`
2. Run Optuna if not found
3. Train final model with best params
4. Save results to experiment directory

Example usage:

```bash
uv run python scripts/experiment_manager.py model \
    --project playground-series-s5e11 \
    --experiment-id exp-20251127-123456 \
    --model xgboost_optuna
```

### Model-Specific Features

**XGBoost:**
- Tree method: `hist` (fast)
- Early stopping: Yes
- GPU support: Yes

**LightGBM:**
- Native categorical: No (encode first)
- Early stopping: Yes
- GPU support: Yes

**CatBoost:**
- Native categorical: YES (auto-detect)
- Early stopping: Yes
- GPU support: Yes (task_type="GPU")

## Model Ensembling

### Blending Strategies

Three blending methods:

**1. Weighted Blending:**
```bash
uv run python scripts/stacking_runner.py \
    --project playground-series-s5e11 \
    --models model1.csv model2.csv model3.csv \
    --strategy blend \
    --blend-method weighted \
    --blend-weights 0.5 0.3 0.2
```

**2. Rank Averaging (Robust):**
```bash
uv run python scripts/stacking_runner.py \
    --project playground-series-s5e11 \
    --models model1.csv model2.csv model3.csv \
    --strategy blend \
    --blend-method rank
```

**3. Power Averaging:**
```bash
uv run python scripts/stacking_runner.py \
    --project playground-series-s5e11 \
    --models model1.csv model2.csv \
    --strategy blend \
    --blend-method power \
    --blend-power 2.0  # Quadratic mean
```

### Meta-Learning (Stacking)

```bash
uv run python scripts/stacking_runner.py \
    --project playground-series-s5e11 \
    --models model1.csv model2.csv model3.csv \
    --strategy meta \
    --meta-model xgboost
```

**Note:** Meta-learning requires out-of-fold predictions (future enhancement).

### Ensemble Diversity

The runner reports diversity metrics:

```
Diversity Metrics:
  Average pairwise correlation: 0.87
  Min correlation: 0.82
  Max correlation: 0.93
  ⚠ High correlation - models may be too similar
```

**Recommendations:**
- `avg_corr > 0.95` - Models too similar (diminishing returns)
- `0.7 < avg_corr < 0.95` - Good diversity
- `avg_corr < 0.7` - Excellent diversity (complementary models)

## Complete Workflow Examples

### Example 1: Quick Iteration (Dev)

```bash
PROJECT="playground-series-s5e11"
EXP_ID="exp-$(date +%Y%m%d%H%M%S)"

# 1. Feature engineering
uv run python scripts/experiment_manager.py feat \
    --project $PROJECT \
    --experiment-id $EXP_ID \
    --feature-set baseline

# 2. Quick tuning (20 trials, 30min)
uv run python scripts/experiment_manager.py tune \
    --project $PROJECT \
    --experiment-id $EXP_ID \
    --model xgboost \
    --preset quick \
    --use-transformed

# 3. Train final model
uv run python scripts/experiment_manager.py model \
    --project $PROJECT \
    --experiment-id $EXP_ID \
    --model xgboost_optuna

# 4. Submit
uv run python scripts/experiment_manager.py submit \
    --project $PROJECT \
    --experiment-id $EXP_ID \
    --auto-submit
```

### Example 2: Production Pipeline

```bash
PROJECT="playground-series-s5e11"

# 1. EDA (initial)
uv run python scripts/experiment_manager.py eda \
    --project $PROJECT \
    --notes "baseline analysis"
# Output: exp-{timestamp}

# 2. Feature engineering + tuning (parallel models)
for MODEL in xgboost lightgbm catboost; do
    EXP_ID="exp-$(date +%Y%m%d%H%M%S)-${MODEL}"

    uv run python scripts/experiment_manager.py feat \
        --project $PROJECT \
        --experiment-id $EXP_ID \
        --feature-set advanced

    uv run python scripts/experiment_manager.py tune \
        --project $PROJECT \
        --experiment-id $EXP_ID \
        --model $MODEL \
        --preset thorough \
        --use-transformed \
        --dashboard &  # Background with dashboard
done

wait  # Wait for all tuning to complete

# 3. Ensemble best models
uv run python scripts/stacking_runner.py \
    --project $PROJECT \
    --models submission-xgb.csv submission-lgb.csv submission-cat.csv \
    --strategy blend \
    --blend-method weighted \
    --blend-weights 0.4 0.3 0.3
```

### Example 3: Advanced Ensembling

```bash
# Create diverse models with different feature sets
MODELS=(
    "xgb_baseline.csv"
    "xgb_advanced.csv"
    "lgb_baseline.csv"
    "lgb_advanced.csv"
    "cat_baseline.csv"
)

# Level 1: Blend within model type
uv run python scripts/stacking_runner.py \
    --project $PROJECT \
    --models xgb_baseline.csv xgb_advanced.csv \
    --blend-method weighted \
    --output-name xgb_blend.csv

uv run python scripts/stacking_runner.py \
    --project $PROJECT \
    --models lgb_baseline.csv lgb_advanced.csv \
    --blend-method weighted \
    --output-name lgb_blend.csv

# Level 2: Blend across model types
uv run python scripts/stacking_runner.py \
    --project $PROJECT \
    --models xgb_blend.csv lgb_blend.csv cat_baseline.csv \
    --blend-method rank \
    --output-name final_ensemble.csv
```

## Configuration Reference

### Project Configuration

`configs/project.yaml` - Default configuration for all experiments:

```yaml
preprocessing:
  storage_format: parquet  # parquet or csv
  compression: snappy      # snappy, gzip, or None
  save_transformed_data: true

  feat_stage:
    enabled: false
    transformers: []

  cv_stage:
    enabled: false
    transformers: []

optuna:
  storage: "sqlite:///experiments/optuna.db"
  study_name: "default_study"
  direction: maximize
  n_trials: 50
  timeout: null
  n_jobs: 1
  cv_folds: 5
  early_stopping_rounds: 50
  sampler: TPESampler
  pruner: MedianPruner
  param_space: {}  # See presets for examples

stacking:
  blend_method: weighted
  blend_power: 2.0
  meta_model: logistic
  calibration:
    enabled: false
    method: isotonic

system:
  random_seed: 42
  verbose: true
```

### Preset Configuration

`configs/presets/thorough.yaml`:

```yaml
optuna:
  n_trials: 100
  timeout: 7200  # 2 hours
  cv_folds: 5
  early_stopping_rounds: 50

  param_space:
    xgboost:
      learning_rate: [0.001, 0.3, log]
      max_depth: [3, 10, int]
      min_child_weight: [1, 10, int]
      subsample: [0.5, 1.0, float]
      colsample_bytree: [0.5, 1.0, float]
      gamma: [0.0, 5.0, float]
      reg_alpha: [0.0, 10.0, log]
      reg_lambda: [0.0, 10.0, log]

    lightgbm:
      learning_rate: [0.001, 0.3, log]
      num_leaves: [15, 255, int]
      max_depth: [3, 12, int]
      min_child_samples: [5, 100, int]
      subsample: [0.5, 1.0, float]
      colsample_bytree: [0.5, 1.0, float]
      reg_alpha: [0.0, 10.0, log]
      reg_lambda: [0.0, 10.0, log]

    catboost:
      learning_rate: [0.001, 0.3, log]
      depth: [4, 10, int]
      l2_leaf_reg: [0.1, 10.0, log]
      bagging_temperature: [0.0, 1.0, float]
      random_strength: [0.0, 10.0, float]
```

## Best Practices

### 1. Data Leakage Prevention

**✓ DO:**
- Use `feat_stage` for target-independent transformations
- Use `cv_stage` for target-dependent transformations
- Verify unseen categories get global mean (not their own target)
- Run unit tests: `pytest tests/test_data_leakage.py`

**✗ DON'T:**
- Fit target encoding on full train set
- Mix feat_stage and cv_stage transformers
- Skip cv_stage for any target-dependent feature

### 2. Hyperparameter Tuning

**✓ DO:**
- Start with `quick` preset for smoke testing
- Use `thorough` for production
- Monitor dashboard during long runs
- Save study to SQLite for resumption
- Use early stopping to speed up bad trials

**✗ DON'T:**
- Use `extreme` without 24h+ compute budget
- Run without early stopping (wastes time)
- Ignore pruning (kills bad trials early)
- Forget to set random_seed for reproducibility

### 3. Model Ensembling

**✓ DO:**
- Blend models with correlation < 0.95
- Use rank averaging for robust blending
- Weight by CV scores or optimize weights
- Stack diverse models (different algorithms, features)

**✗ DON'T:**
- Blend nearly-identical models (diminishing returns)
- Use power averaging with p > 5 (numerical instability)
- Over-complicate stacking (often simple blend wins)

### 4. Experiment Organization

**✓ DO:**
- Create experiment per feature set or model variant
- Use descriptive `--notes` for tracking
- Commit before running experiments (for git hash)
- Use `--force` carefully (overwrites modules)

**✗ DON'T:**
- Reuse experiment_id for different experiments
- Delete experiments (breaks reproducibility)
- Skip git commits (can't reproduce)

## Troubleshooting

### Issue: Optuna study not resuming

**Problem:**
```
[W 2025-11-27 12:00:00,000] A new study created...
```
Expected to resume existing study.

**Solution:**
```bash
# Check storage path matches
ls experiments/optuna.db

# Ensure study_name matches
# In config: study_name: "xgboost_study"

# Or use --resume flag
uv run python scripts/optuna_runner.py --resume ...
```

### Issue: "Data leakage detected" in tests

**Problem:**
```python
AssertionError: LEAKAGE DETECTED: B encoding is 0.5000, expected 0.4000
```

**Solution:**
Check transformer is fitted per-fold:

```python
# ✗ WRONG - fitting on full data
encoder.fit(train_df)

# ✓ CORRECT - fitting per-fold
for train_fold, val_fold in cv_folds:
    encoder_fold = encoder.clone()  # Fresh instance
    encoder_fold.fit(train_fold)    # Fit on fold only
    val_transformed = encoder_fold.transform(val_fold)
```

### Issue: Dashboard not loading

**Problem:**
```
Connection refused at http://localhost:8080
```

**Solution:**
```bash
# Ensure optuna-dashboard is installed
uv add optuna-dashboard

# Check port is free
lsof -i :8080

# Launch manually
optuna-dashboard sqlite:///experiments/optuna.db

# Or use --dashboard flag
uv run python scripts/optuna_runner.py --dashboard ...
```

### Issue: Out of memory during tuning

**Problem:**
```
MemoryError: Unable to allocate array
```

**Solution:**
```bash
# Reduce CV folds
--cv-folds 3  # Instead of 5

# Reduce parallel jobs
--n-jobs 1  # Sequential trials

# Use smaller dataset for tuning (sample)
# In code:
train_sample = train_df.sample(frac=0.5, random_state=42)
```

### Issue: CatBoost categorical features not working

**Problem:**
```
ValueError: categorical_features indices are out of range
```

**Solution:**
CatBoost auto-detects categorical features if:
1. Column dtype is `object` or `category`
2. Numeric columns with low cardinality (< 5% unique)

```python
# Explicit categorical
cat_features = [0, 1, 2]  # Indices
model = CatBoostClassifier(cat_features=cat_features)

# Or convert dtypes
df['cat_col'] = df['cat_col'].astype('category')
```

### Issue: ModuleNotFoundError after init-project

**Problem:**
```
ModuleNotFoundError: No module named 'utils.config'
```

**Solution:**
Ensure you're running from project root and code is in `sys.path`:

```bash
# Check current directory
pwd
# Should be: /mnt/ml/kaggle-fork1

# Run from root
uv run python scripts/experiment_manager.py ...
```

## Additional Resources

- **Optuna Documentation:** https://optuna.readthedocs.io/
- **PRD (Product Requirements):** `/home/xai/.claude/plans/OPTUNA_SYSTEM_PRD.md`
- **Unit Tests:** `tests/test_data_leakage.py`, `tests/test_optuna_e2e.py`
- **Templates:** `config/templates/kaggle_competition/`

## Contributing

When adding new transformers or models:

1. Follow naming conventions (`*Transformer`, `*Blender`, `*Learner`)
2. Add unit tests to `tests/`
3. Update this guide with examples
4. Add to `__init__.py` exports
5. Document parameter space in preset YAML

## License

This system is part of the kaggle-projects repository and follows its license.
