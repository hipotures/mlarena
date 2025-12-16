# Configuration Files

**Related Documentation:**
- [MLA_WORKFLOW_GUIDE.md](MLA_WORKFLOW_GUIDE.md) - Main workflow guide with practical examples
- [ARCHITECTURE.md](ARCHITECTURE.md) - MLArena architecture overview

---

This directory contains configuration files for the competition project.

## templates/model.yaml

Defines experiment templates that combine model implementations with their hyperparameter configurations. Preprocess templates live alongside in `templates/preprocess.yaml`.

### File Location

```
projects/kaggle/<project>/templates/model.yaml
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
  - Used with `--model-template` flag in `mla.py`
  - Convention: `{compute}-{variant}-{time}[-{special}]` (e.g., `cpu-best-1h-fe11`, `gpu-dev-5m`, `cpu-fast-1m-tier1`)

- **`model`** (string): Python module name from `code/models/`
  - Must exist as `code/models/<model>.py`
  - Examples: `autogluon_baseline`, `autogluon_features_11`, `exp01_tier1_features`

#### Config Section

##### hyperparameters

AutoGluon `TabularPredictor.fit()` parameters:

- **`presets`** (string): AutoGluon quality preset
  - Values: `medium`, `best`, `high`, `extreme`
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

##### Sample Weight Configuration

Controls how AutoGluon handles sample weights for training and evaluation:

- **`sample_weight_strategy`** (string, optional): Strategy for sample weighting
  - `null` (default): Use weights from preprocessing artifacts (legacy behavior)
  - `"auto_weight"`: AutoGluon automatic balancing (experimental)
  - `"balance_weight"`: Equal class weights for classification
  - `"<column_name>"`: Use specific column from train_df as weights
  - **Note**: When using preprocessing modules like `adversarial_validation` or `imbalance_handler` that return weights via artifacts, leave this as `null`

- **`weight_evaluation`** (boolean, optional): Use sample weights for evaluation metrics
  - `null` (default): Auto-detect
    - `true` when using explicit weights (artifacts, custom column)
    - `false` when using `auto_weight` or `balance_weight`
  - `true`: Apply weights to validation/test metrics (weighted CV scores)
  - `false`: Ignore weights for evaluation (only use for training)
  - **AutoGluon Warning**: Setting `weight_evaluation: true` with `sample_weight_strategy: "auto_weight"` or `"balance_weight"` is not recommended by AutoGluon docs. Use appropriate `eval_metric` instead.

**Examples**:

```yaml
# Example 1: Use weights from preprocessing (default)
cpu-best-1h-av:
  model: autogluon_baseline
  config:
    preset: best
    time_limit: 3600
    # sample_weight_strategy: null (implicit)
    # weight_evaluation: null (auto-detect -> true if weights present)
  preprocess_template: av_weights  # Returns weights via artifacts
```

```yaml
# Example 2: Auto-balancing without weighted evaluation (recommended)
cpu-best-1h-balanced:
  model: autogluon_baseline
  config:
    preset: best
    time_limit: 3600
    sample_weight_strategy: "auto_weight"
    weight_evaluation: false  # Recommended by AutoGluon
```

```yaml
# Example 3: Equal class weights
cpu-best-1h-equal:
  model: autogluon_baseline
  config:
    preset: best
    time_limit: 3600
    sample_weight_strategy: "balance_weight"
    # weight_evaluation: null (auto -> false)
```

```yaml
# Example 4: Disable weighted evaluation with preprocessing weights
cpu-best-1h-av-unweighted-eval:
  model: autogluon_baseline
  config:
    preset: best
    time_limit: 3600
    weight_evaluation: false  # Use weights for training, not evaluation
  preprocess_template: av_weights
```

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
cpu-best-1h:
  model: autogluon_baseline
  config:
    hyperparameters:
      presets: best
      time_limit: 3600
      use_gpu: false
```

**Usage**: 1-hour CPU training with best quality preset, all models enabled.

#### 2. Feature Engineering Template

```yaml
cpu-best-2h-fe11:
  model: autogluon_features_11
  config:
    hyperparameters:
      presets: best
      time_limit: 7200
      use_gpu: false
      excluded_models:
        - NN_TORCH
```

**Usage**: 2-hour CPU training with feature variant 11, excluding neural networks.

#### 3. GPU Development Template

```yaml
gpu-dev-5m:
  model: autogluon_baseline
  config:
    hyperparameters:
      presets: medium
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

#### 6. Hyperparameter Optimization (HPO) Template

```yaml
test_hpo_medium:
  model: autogluon_baseline
  hpo_preset: hpo_boost_medium  # 50 trials, conservative ranges
  config:
    preset: best
    time_limit: 3600
    use_gpu: false
    included_model_types: [GBM, XGB, CAT]

    # Optional: Override preset defaults
    # num_trials: 150      # Override medium default (50)
    # searcher: bayesian   # Override auto
```

**Usage**: AutoGluon native HPO with 50 trials on boost models (GBM, XGB, CAT).

**Available HPO presets**:
- `hpo_boost_medium` - 50 trials, conservative ranges (1-2h)
- `hpo_boost_high` - 100 trials, broader ranges (4-6h)
- `hpo_boost_best` - 200 trials, exhaustive ranges (8-12h)

**How HPO works**:
1. Template specifies `hpo_preset` which references a preset file
2. Preset defines `num_trials`, `scheduler`, `searcher` and search spaces
3. Search spaces use YAML notation: `[min, max, log]` → `space.Real(min, max, log=True)`
4. Template can override any preset defaults
5. Only boost models (GBM, XGB, CAT) use custom search spaces
6. Other models use AutoGluon defaults

**HPO preset files**:
- Global: `src/mlarena/templates/model/hpo/*.yaml`
- Project: `projects/{comp}/templates/model/hpo/*.yaml` (overrides global)

**Example preset** (`hpo_boost_medium.yaml`):
```yaml
hpo:
  num_trials: 50
  scheduler: local
  searcher: auto

search_space:
  GBM:
    learning_rate: [0.01, 0.3, log]
    num_leaves: [20, 150]
    lambda_l1: [1e-5, 10.0, log]
  XGB:
    learning_rate: [0.01, 0.3, log]
    max_depth: [3, 8]
  CAT:
    learning_rate: [0.01, 0.3, log]
    depth: [4, 8]
```

### Common Use Cases

#### Quick Iteration

For fast experiments during development:

```yaml
cpu-fast-1m:
  model: autogluon_baseline
  config:
    hyperparameters:
      presets: medium
      time_limit: 60
      use_gpu: false
      excluded_models:
        - NN_TORCH  # XGBoost-only for speed
```

#### Excluding Problematic Models

When specific models crash or are too slow:

```yaml
cpu-best-1h-stable:
  model: autogluon_baseline
  config:
    hyperparameters:
      presets: best
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
cpu-best-1h-fe00:
  model: autogluon_features_00
  config:
    hyperparameters:
      presets: best
      time_limit: 3600
      use_gpu: false

# Variant 1: Add interaction terms
cpu-best-1h-fe01:
  model: autogluon_features_01
  config:
    hyperparameters:
      presets: best
      time_limit: 3600
      use_gpu: false

# Variant 2: Add polynomial features
cpu-best-1h-fe02:
  model: autogluon_features_02
  config:
    hyperparameters:
      presets: best
      time_limit: 3600
      use_gpu: false
```

### Running Templates

#### Via MLArena (Recommended)

```bash
uv run python scripts/mla.py model \
    --project <project> \
    --model-template cpu-best-2h-fe11 \
    --auto-submit \
    --wait-seconds 45
```

#### Direct Module Invocation

```bash
# Run specific module with template
uv run python scripts/mla.py model \
    --project <project> \
    --model-template cpu-dev-5m-tier1

# With preprocessing template
uv run python scripts/mla.py model \
    --project <project> \
    --model-template gpu-dev-5m \
    --preprocess-template baseline
```

**Note**: MLArena (`mla.py`) is the unified entry point. Legacy runners (`ml_runner.py`, `autogluon_runner.py`) have been deprecated.

### Template Naming Conventions

Format: `{compute}-{variant}-{time}[-{special}]`

**Compute prefix:**
- `cpu-*`: CPU-only training
- `gpu-*`: GPU-accelerated training

**Variant:**
- `fast`: Quick smoke tests (< 5 minutes)
- `dev`: Development iteration (5-10 minutes)
- `best`: Production quality (1-8 hours)
- `extreme`: Maximum quality (24+ hours)

**Time:**
- `1m`, `5m`: Minutes (60s, 300s)
- `1h`, `2h`, `8h`: Hours (3600s, 7200s, 28800s)
- `24h`: 24 hours (86400s)

**Special (optional):**
- `fe##`: Feature engineering variant number (e.g., `fe11`)
- `av`, `av-gbm`, `av-xgb`: AutoGluon Variant weights with specific models
- `tier1`, `stable`: Custom experiment identifiers

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
- Verify `templates/model.yaml` syntax (valid YAML)
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
