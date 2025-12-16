# tune Module - Hyperparameter Optimization

The `tune` module provides Optuna-based hyperparameter search for AutoGluon models. It's designed for experimental HPO workflows and quick parameter exploration on small data samples.

## Overview

- **Module name**: `tune`
- **Dependencies**: `model` (requires trained baseline model)
- **Part of auto-flow**: No (manual use only)
- **Source**: `src/mlarena/modules/tune.py`
- **Status**: Legacy module (use AutoGluon native HPO via model templates instead)

## Purpose

The `tune` module was designed to:

- Explore hyperparameter search spaces using Optuna
- Find optimal parameters before running full training
- Test parameter sensitivity on small data samples

**Important**: For production use, prefer **AutoGluon native HPO** via model templates with `hpo_preset` and `hyperparameter_tune_kwargs`. See [HPO Guide](MLA_WORKFLOW_GUIDE.md#hyperparameter-optimization-hpo) for details.

## Prerequisites

### Installation

Optuna must be installed separately:

```bash
uv pip install optuna
```

### Required Module

The `tune` module depends on the `model` module being completed first (though it doesn't use the trained model directly - it trains its own trials).

## Usage

### Basic Usage

```bash
# Run HPO with default settings (10 trials, 60s per trial)
uv run python scripts/mla.py tune --project <competition-slug>
```

### Advanced Usage

```bash
# Custom trial count and time limit
uv run python scripts/mla.py tune \
  --project titanic \
  --n-trials 20 \
  --time-limit 120 \
  --tune-template custom_search_space
```

## CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--n-trials` | int | 10 | Number of Optuna trials to run |
| `--time-limit` | int | 60 | Time limit per trial in seconds |
| `--tune-template` | str | tune | Template name defining search space |

## Template Configuration

Tune templates define the hyperparameter search space for Optuna.

### Template Structure

```yaml
# projects/kaggle/<slug>/templates/tune/my_search.yaml
search_space:
  learning_rate:
    type: float
    low: 0.001
    high: 0.1
    log: true

  num_boost_round:
    type: int
    low: 50
    high: 500
    log: false

  max_depth:
    type: int
    low: 3
    high: 10

  preset:
    type: categorical
    choices: [medium, high, best_quality]
```

### Parameter Types

#### 1. Float Parameters

```yaml
parameter_name:
  type: float
  low: 0.001        # minimum value
  high: 1.0         # maximum value
  log: true         # use log scale (optional, default: false)
```

#### 2. Integer Parameters

```yaml
parameter_name:
  type: int
  low: 10           # minimum value
  high: 1000        # maximum value
  log: false        # use log scale (optional, default: false)
```

#### 3. Categorical Parameters

```yaml
parameter_name:
  type: categorical
  choices: [option1, option2, option3]
```

## How It Works

1. **Sampling**: Module samples a small subset of training data (max 300 rows per trial)
2. **Trial training**: Each trial trains an AutoGluon model with suggested parameters
3. **Evaluation**: Model performance is evaluated using the leaderboard
4. **Optimization**: Optuna maximizes the evaluation metric
5. **Output**: Best parameters and score are saved to `tune_result.json`

### Trial Process

For each Optuna trial:

```python
# 1. Sample parameters from search space
params = {
    "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
    "max_depth": trial.suggest_int("max_depth", 3, 10),
}

# 2. Sample small training subset (max 300 rows)
sample_df = train_df.sample(300, random_state=trial.number)

# 3. Train AutoGluon with sampled parameters
predictor = TabularPredictor(...)
predictor.fit(sample_df, hyperparameters=params, time_limit=60)

# 4. Return best model score
return leaderboard.iloc[0]["score"]
```

## Outputs

The module creates the following artifacts in `experiments/<exp_id>/artifacts/tune/`:

1. **tune_result.json**: Best parameters and score
   ```json
   {
     "best_params": {
       "learning_rate": 0.0234,
       "max_depth": 7,
       "num_boost_round": 234
     },
     "best_value": 0.892,
     "template": "custom_search_space"
   }
   ```

2. **trial_N/**: Individual trial AutoGluon outputs (one directory per trial)

## Examples

### Example 1: Quick HPO

```bash
# Fast exploration with 5 trials, 30s each
uv run python scripts/mla.py tune \
  --project titanic \
  --n-trials 5 \
  --time-limit 30
```

### Example 2: Learning Rate Search

```yaml
# templates/tune/lr_search.yaml
search_space:
  learning_rate:
    type: float
    low: 0.0001
    high: 0.5
    log: true
```

```bash
uv run python scripts/mla.py tune \
  --project titanic \
  --tune-template lr_search \
  --n-trials 15
```

### Example 3: Tree-Based Model HPO

```yaml
# templates/tune/tree_hpo.yaml
search_space:
  max_depth:
    type: int
    low: 3
    high: 12

  num_leaves:
    type: int
    low: 20
    high: 200
    log: true

  learning_rate:
    type: float
    low: 0.01
    high: 0.3
    log: true

  min_child_samples:
    type: int
    low: 5
    high: 100
```

## Limitations

1. **Small sample size**: Only 300 rows per trial (fast but may not generalize)
2. **No cross-validation**: Single train evaluation (higher variance)
3. **Legacy status**: AutoGluon native HPO is more robust
4. **Time-consuming**: Even with small samples, many trials take hours
5. **No early stopping**: All trials run to completion

## Recommended Alternative: AutoGluon Native HPO

Instead of using `tune`, use AutoGluon's built-in HPO via model templates:

```yaml
# templates/model/hpo_model.yaml
model: autogluon_baseline
config:
  preset: medium
  time_limit: 3600
  hpo_preset: ray
  num_trials: 50
  scheduler: ASHA
  searcher: bayes
```

See [HPO Guide](MLA_WORKFLOW_GUIDE.md#hyperparameter-optimization-hpo) for complete details.

### Advantages of Native HPO

- Uses full training data (not just 300 rows)
- Integrated with AutoGluon's ensemble system
- Supports advanced schedulers (ASHA, Hyperband)
- Better resource management
- Production-ready

## Error Handling

- **Optuna missing**: Clear error with installation instructions
- **No training data**: Fails with "train missing" error
- **Target column mismatch**: Validates target exists in config
- **Trial failures**: Optuna continues with remaining trials
- **All trials fail**: Returns empty result with failure status

## Template Locations

- **Global templates**: `src/mlarena/templates/tune/*.yaml`
- **Project templates**: `projects/kaggle/<slug>/templates/tune/*.yaml`

Project templates override global templates when names collide.

## Performance Tips

1. **Start with fewer trials**: 5-10 trials often sufficient for initial exploration
2. **Use log scale**: Set `log: true` for learning rates and tree counts
3. **Narrow search space**: Based on previous experiments
4. **Monitor trial times**: Adjust `--time-limit` based on data size
5. **Check trial outputs**: Review individual trial directories for issues

## Integration Notes

- Results are saved to state.json but **not automatically applied** to subsequent runs
- To use best parameters, manually update your model template
- Consider using tune results to inform native HPO search spaces

## See Also

- [HPO Guide](MLA_WORKFLOW_GUIDE.md#hyperparameter-optimization-hpo) - AutoGluon native HPO (recommended)
- [Model Templates](model_templates.md) - How to configure HPO in model templates
- [Optuna Documentation](https://optuna.readthedocs.io/) - Advanced Optuna features
