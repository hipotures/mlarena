# Configuration Files

This directory contains configuration files for the competition project.

## templates.yaml

Defines experiment templates that combine model implementations with their hyperparameter configurations.

### File Location

```
projects/kaggle/playground-series-s5e11/configs/templates.yaml
```

### Structure

```yaml
templates:
  <template-name>:
    model: <model_module_name>
    config:
      hyperparameters:
        presets: <autogluon_preset>
        time_limit: <seconds>
        use_gpu: <true|false>
        excluded_models:
          - <MODEL_TYPE>
          - <MODEL_TYPE>
        # ... other AutoGluon fit() parameters
      model:
        # ... model-specific parameters
      preprocessing:
        # ... preprocessing-specific parameters
```

### Template Fields

#### Top Level

- **`<template-name>`** (string): Unique identifier for the template
  - Used with `--template` flag in `experiment_manager.py` and `ml_runner.py`
  - Convention: `{compute}-{variant}` (e.g., `best-cpu-fe11`, `dev-gpu`, `exp01-tier1`)

- **`model`** (string): Python module name from `code/models/`
  - Must exist as `code/models/<model>.py`
  - Examples: `autogluon_baseline`, `autogluon_features_11`, `exp01_tier1_features`

#### Config Section

##### hyperparameters

AutoGluon `TabularPredictor.fit()` parameters:

- **`presets`** (string): AutoGluon quality preset
  - Values: `medium_quality`, `best_quality`, `high_quality`, `extreme_quality`
  - Higher quality = more models, longer training

- **`time_limit`** (integer): Training time limit in seconds
  - Examples: `300` (5min), `3600` (1h), `7200` (2h), `86400` (24h)

- **`use_gpu`** (boolean): Enable GPU acceleration
  - `true` or `false`
  - Requires CUDA-compatible GPU

- **`excluded_models`** (list, optional): Model types to exclude from training
  - Common values:
    - `NN_TORCH` - Neural network (PyTorch)
    - `NN_MXNET` - Neural network (MXNet)
    - `FASTAI` - FastAI tabular
    - `XGB` - XGBoost
    - `CAT` - CatBoost
    - `LR` - Linear models
    - `KNN` - K-Nearest Neighbors
    - `RF` - Random Forest
    - `XT` - Extra Trees
  - Use when specific models cause errors or are too slow

- **`num_bag_folds`** (integer, optional): Number of bagging folds
  - Default: 8 (or based on preset)
  - Higher = better ensemble, longer training

- **`num_stack_levels`** (integer, optional): Number of stacking levels
  - Default: 1 (or based on preset)
  - Higher = deeper ensemble, longer training

##### model

Model-specific configuration passed to the model implementation:

- **`sample_fraction`** (float): Fraction of training data to use
  - Range: 0.0 to 1.0
  - Used for quick experiments or memory constraints

- **`cv_folds`** (integer): Cross-validation folds
  - Used by custom model implementations

- **`n_estimators`** (integer): Number of estimators
  - Used by ensemble models (TabICL, custom implementations)

- **`batch_size`** (integer): Batch size for gradient-based models

- **`device`** (string): Device for PyTorch/TabICL models
  - Values: `cuda`, `cpu`

- **`norm_methods`** (list): Normalization methods
  - Values: `none`, `power`, `quantile`, `standard`

- **`output_labels`** (boolean): Output class labels vs probabilities

- **`label_threshold`** (float): Classification threshold
  - Range: 0.0 to 1.0

- **`num_trials`** (integer): Hyperparameter search trials (Optuna)

- **`searcher`** (string): Hyperparameter search strategy
  - Values: `auto`, `random`, `bayesian`

- **`leaderboard_rows`** (integer): Number of rows to show in leaderboard

##### preprocessing

Preprocessing configuration for feature engineering pipelines:

- **`feature_set`** (string): Feature engineering variant
  - Examples: `tier1_critical`, `tier2_encoding`, `tier1_with_transfer`
  - Defined in model implementation

- **`include_tier1`** (boolean): Include baseline features
  - Used in multi-tier feature engineering

- **`use_original_dataset`** (boolean): Use original competition data
  - For transfer learning scenarios

### Example Templates

#### 1. Basic CPU Template

```yaml
best-cpu:
  model: autogluon_baseline
  config:
    hyperparameters:
      presets: best_quality
      time_limit: 3600
      use_gpu: false
```

**Usage**: 1-hour CPU training with best quality preset, all models enabled.

#### 2. Feature Engineering Template

```yaml
best-cpu-fe11:
  model: autogluon_features_11
  config:
    hyperparameters:
      presets: best_quality
      time_limit: 7200
      use_gpu: false
      excluded_models:
        - NN_TORCH
```

**Usage**: 2-hour CPU training with feature variant 11, excluding neural networks.

#### 3. GPU Development Template

```yaml
dev-gpu:
  model: autogluon_baseline
  config:
    hyperparameters:
      presets: medium_quality
      time_limit: 300
      use_gpu: true
```

**Usage**: 5-minute GPU smoke test with medium quality.

#### 4. Custom Model Template

```yaml
exp03-lgbm-optuna:
  model: exp03_lgbm_optuna
  config:
    hyperparameters:
      n_trials: 50
      n_folds: 5
      early_stopping_rounds: 50
      verbose_eval: 100
      use_gpu: false
    preprocessing:
      feature_set: tier1_critical
```

**Usage**: LightGBM with Optuna hyperparameter optimization on tier1 features.

#### 5. TabICL Template

```yaml
tabicl-full:
  model: tabicl_skrub
  config:
    model:
      sample_fraction: 1.0
      cv_folds: 1
      n_estimators: 32
      batch_size: 8
      device: cuda
      norm_methods:
        - none
        - power
        - quantile
      output_labels: true
      label_threshold: 0.5
    hyperparameters:
      use_gpu: true
```

**Usage**: Full TabICL training with multiple normalization methods on GPU.

### Common Use Cases

#### Quick Iteration

For fast experiments during development:

```yaml
fast-cpu:
  model: autogluon_baseline
  config:
    hyperparameters:
      presets: medium_quality
      time_limit: 60
      use_gpu: false
      excluded_models:
        - NN_TORCH  # XGBoost-only for speed
```

#### Excluding Problematic Models

When specific models crash or are too slow:

```yaml
best-cpu-stable:
  model: autogluon_baseline
  config:
    hyperparameters:
      presets: best_quality
      time_limit: 3600
      use_gpu: false
      excluded_models:
        - NN_TORCH    # Crashes on this dataset
        - FASTAI      # Out of memory
        - NN_MXNET    # Deprecated
```

#### Feature Engineering Series

Systematic feature exploration:

```yaml
# Baseline features
best-cpu-fe00:
  model: autogluon_features_00
  config:
    hyperparameters:
      presets: best_quality
      time_limit: 3600
      use_gpu: false

# Variant 1: Add interaction terms
best-cpu-fe01:
  model: autogluon_features_01
  config:
    hyperparameters:
      presets: best_quality
      time_limit: 3600
      use_gpu: false

# Variant 2: Add polynomial features
best-cpu-fe02:
  model: autogluon_features_02
  config:
    hyperparameters:
      presets: best_quality
      time_limit: 3600
      use_gpu: false
```

### Running Templates

#### Via Experiment Manager (Recommended)

```bash
uv run python scripts/experiment_manager.py model \
    --project playground-series-s5e11 \
    --template best-cpu-fe11 \
    --auto-submit \
    --wait-seconds 45
```

#### Via ML Runner (Direct)

```bash
uv run python scripts/ml_runner.py \
    --project playground-series-s5e11 \
    --template exp01-tier1 \
    --auto-submit
```

#### Via AutoGluon Runner (Legacy)

```bash
uv run python scripts/autogluon_runner.py \
    --project playground-series-s5e11 \
    --template dev-gpu
```

**Note**: `autogluon_runner.py` has limited template support compared to `ml_runner.py`.

### Template Naming Conventions

- **`fast-*`**: Quick smoke tests (< 5 minutes)
- **`dev-*`**: Development iteration (5-10 minutes)
- **`best-*`**: Production quality (1-2 hours)
- **`extreme-*`**: Maximum quality (24+ hours)
- **`time8-*`**: 8-hour training jobs
- **`*-cpu`**: CPU-only training
- **`*-gpu`**: GPU-accelerated training
- **`*-fe##`**: Feature engineering variant number
- **`exp##-*`**: Numbered experiment series

### Validation

Templates are validated at runtime:

1. **Model exists**: `code/models/<model>.py` must exist
2. **Valid hyperparameters**: AutoGluon must accept all hyperparameters
3. **GPU availability**: If `use_gpu: true`, CUDA must be available

### Troubleshooting

#### Template not found

```
Error: Template 'my-template' not found
```

- Check template name spelling
- Verify `templates.yaml` syntax (valid YAML)
- Ensure template is not commented out

#### Model import error

```
ModuleNotFoundError: No module named 'models.my_model'
```

- Create `code/models/my_model.py`
- Implement required functions (see existing models)

#### Invalid hyperparameters

```
TypeError: fit() got an unexpected keyword argument 'invalid_param'
```

- Check AutoGluon documentation for valid parameters
- Remove unknown parameters from `hyperparameters` section

#### GPU not available

```
RuntimeError: CUDA out of memory
```

- Set `use_gpu: false` for CPU training
- Reduce `batch_size` if using custom models
- Exclude memory-intensive models (`NN_TORCH`, `FASTAI`)

### Best Practices

1. **Start with fast templates** for validation before long runs
2. **Exclude problematic models** rather than debugging for hours
3. **Use consistent naming** for template series (fe01, fe02, ...)
4. **Document model changes** in git commit messages
5. **Track template performance** in submissions tracker
6. **Version control** all template changes

### Related Files

- `code/models/` - Model implementations referenced by templates
- `scripts/experiment_manager.py` - Main pipeline orchestrator
- `scripts/ml_runner.py` - Direct model runner
- `experiments/*/state.json` - Experiment execution logs
- `submissions/submissions.json` - Performance tracking
