# stack Module - Ensemble Predictions

The `stack` module creates ensemble submissions by averaging predictions from multiple models. It's a simple yet effective way to combine model predictions and often improves leaderboard scores.

## Overview

- **Module name**: `stack`
- **Dependencies**: `predict` (requires at least one prediction file)
- **Part of auto-flow**: No (manual use only)
- **Source**: `src/mlarena/modules/stack.py`

## Purpose

The `stack` module enables you to:

- Combine predictions from multiple models via simple averaging
- Create ensemble submissions without training meta-models
- Quickly test ensemble performance
- Leverage model diversity for improved scores

## Usage

### Basic Usage

```bash
# Stack multiple prediction files
uv run python scripts/mla.py stack \
  --project <competition-slug> \
  stack.prediction_files=[submission1.csv,submission2.csv,submission3.csv]
```

### With Explicit Columns

```bash
# Specify ID and target columns explicitly
uv run python scripts/mla.py stack \
  --project titanic \
  stack.prediction_files=[pred1.csv,pred2.csv,pred3.csv] \
  stack.id_column=PassengerId \
  stack.target_column=Survived
```

### Stacking Previous Experiment

```bash
# Stack using prediction from completed predict module
uv run python scripts/mla.py stack \
  --project titanic \
  --exp-id exp-20251216-123045
```

## CLI Overrides

| Override | Type | Default | Description |
|----------|------|---------|-------------|
| `stack.prediction_files` | list | None | Paths to prediction CSV files to ensemble |
| `stack.id_column` | str | None | ID column name (defaults to first column) |
| `stack.target_column` | str | None | Target column name (defaults to last column) |

## How It Works

### Ensemble Process

1. **Load predictions**: Read all specified CSV files
2. **Validate structure**: Ensure all files have consistent ID columns
3. **Extract predictions**: Get target column values from each file
4. **Average**: Compute arithmetic mean across all predictions
5. **Create submission**: Combine IDs with averaged predictions

### Averaging Formula

```python
# Simple arithmetic mean
averaged_prediction = (pred1 + pred2 + pred3 + ... + predN) / N
```

### Column Detection

If column names are not specified:
- **ID column**: First column in the first prediction file
- **Target column**: Last column in the first prediction file

## Outputs

The module creates the following artifacts in `experiments/<exp_id>/artifacts/stack/`:

1. **stacked_submission.csv**: Averaged predictions ready for submission
2. **Payload metadata**: List of input files used in the ensemble

### Output Format

```csv
PassengerId,Survived
1,0.234
2,0.876
3,0.123
...
```

## Examples

### Example 1: Stack Two Models

```bash
# Create ensemble from two different model experiments
uv run python scripts/mla.py stack \
  --project titanic \
  --prediction-files \
    projects/kaggle/titanic/experiments/exp-20251216-100000/artifacts/predict/submission.csv \
    projects/kaggle/titanic/experiments/exp-20251216-110000/artifacts/predict/submission.csv
```

### Example 2: Stack Multiple Preprocessing Variants

```bash
# Combine predictions from same model with different preprocessing
uv run python scripts/mla.py stack \
  --project titanic \
  --prediction-files \
    experiments/exp-baseline/artifacts/predict/submission.csv \
    experiments/exp-with-fe/artifacts/predict/submission.csv \
    experiments/exp-with-scaling/artifacts/predict/submission.csv
```

### Example 3: Stack Diverse Model Types

```bash
# Ensemble different model architectures
uv run python scripts/mla.py stack \
  --project titanic \
  --prediction-files \
    submissions/xgb_submission.csv \
    submissions/lgbm_submission.csv \
    submissions/rf_submission.csv \
    submissions/nn_submission.csv
```

## Typical Workflow

### 1. Train Multiple Models

```bash
# Train baseline model
uv run python scripts/mla.py model --project titanic --model-template baseline

# Train with different preprocessing
uv run python scripts/mla.py preprocess \
    --project playground-series-s5e1 \
    --preprocess-template feature_interactions

# Train different model type
uv run python scripts/mla.py model --project titanic --model-template cpu-xgb-8h
```

### 2. Generate Predictions

Each model run creates a submission CSV in its experiment directory.

### 3. Create Ensemble

```bash
# Stack all three predictions
uv run python scripts/mla.py stack \
  --project titanic \
  --prediction-files \
    experiments/exp-20251216-100000/artifacts/predict/submission.csv \
    experiments/exp-20251216-110000/artifacts/predict/submission.csv \
    experiments/exp-20251216-120000/artifacts/predict/submission.csv
```

### 4. Submit Ensemble

```bash
# Submit the stacked predictions
uv run python scripts/mla.py submit \
  --project titanic \
  --exp-id <stack-exp-id>
```

## When Stacking Helps

Ensembling works best when models are **diverse**:

### Good Diversity Sources

1. **Different algorithms**: XGBoost + LightGBM + Random Forest
2. **Different preprocessing**: Raw features vs. engineered features
3. **Different hyperparameters**: Deep trees vs. shallow trees
4. **Different training subsets**: Various CV folds or bootstraps
5. **Different feature sets**: Different feature selection methods

### Limited Benefit Cases

- Averaging very similar models (minimal diversity)
- Models with highly correlated errors
- One model significantly better than others (better to use single best model)

## Advanced Techniques

### Weighted Averaging (Manual)

The `stack` module uses simple averaging. For weighted averaging:

1. Note individual model scores
2. Calculate weights (e.g., proportional to CV score)
3. Manually create weighted ensemble in a custom script

### Multi-Level Stacking

```bash
# Level 1: Stack base models
uv run python scripts/mla.py stack --project proj --prediction-files base1.csv base2.csv base3.csv

# Level 2: Stack level-1 ensemble with additional models
uv run python scripts/mla.py stack --project proj --prediction-files stacked1.csv base4.csv base5.csv
```

## Best Practices

1. **Start with 2-3 models**: Validate improvement before adding more
2. **Use diverse models**: Maximize prediction diversity
3. **Check individual scores**: Ensure all models have reasonable performance
4. **Validate on CV**: Test ensemble on local validation before submitting
5. **Track experiments**: Document which models are in each ensemble
6. **Don't overfit**: Too many similar models may overfit to public leaderboard

## Limitations

1. **Simple averaging only**: No weighted or learned ensembles
2. **Regression/probability only**: Not designed for multi-class label voting
3. **No validation**: Module doesn't verify prediction quality
4. **Memory constraints**: All files loaded into memory simultaneously

## Alternative Approaches

### AutoGluon Ensembling

AutoGluon automatically creates ensembles during training. To leverage this:

```yaml
# model template with ensembling
config:
  preset: best_quality  # enables aggressive ensembling
  num_bag_folds: 5      # use bagging
  num_stack_levels: 1   # enable stacking
```

See [Model Templates](model_templates.md) for details.

### Weighted Ensemble Script

Create a custom Python script for sophisticated ensembling:

```python
import pandas as pd

# Load predictions
pred1 = pd.read_csv("sub1.csv")
pred2 = pd.read_csv("sub2.csv")
pred3 = pd.read_csv("sub3.csv")

# Weighted average (based on CV scores)
weights = [0.5, 0.3, 0.2]  # sum to 1.0
ensemble = (
    weights[0] * pred1["target"] +
    weights[1] * pred2["target"] +
    weights[2] * pred3["target"]
)

# Create submission
submission = pd.DataFrame({
    "id": pred1["id"],
    "target": ensemble
})
submission.to_csv("weighted_ensemble.csv", index=False)
```

## Error Handling

- **No prediction files**: Fails with "no predictions" error
- **Files not found**: Skips missing files, fails if all missing
- **Mismatched IDs**: No automatic alignment (files must have matching row order)
- **Different column names**: Uses first file's structure, may fail if inconsistent
- **Missing columns**: Falls back to first/last column defaults

## Validation Tips

1. **Check file consistency**: Verify all predictions have same ID order
2. **Compare file lengths**: All should have same number of rows
3. **Sanity check values**: Ensure predictions are in valid range (e.g., 0-1 for probabilities)
4. **Review individual models**: Remove underperforming models before ensembling

## Template Locations

The `stack` module doesn't use templates (operates directly on prediction files).

## Integration with Pipeline

Stack is designed for manual experimentation after multiple model runs:

```bash
# 1. Train multiple experiments
for template in baseline cpu-xgb-8h cpu-rf-8h; do
  uv run python scripts/mla.py model --project proj --model-template $template
done

# 2. Collect prediction files
predictions=$(find experiments/*/artifacts/predict/submission.csv)

# 3. Create ensemble
uv run python scripts/mla.py stack --project proj --prediction-files $predictions
```

## See Also

- [Model Module](modules/model.rst) - Train individual models
- [Predict Module](modules/predict.rst) - Generate prediction files
- [Model Templates](model_templates.md) - AutoGluon ensembling configuration
- [AutoGluon Ensembling](https://auto.gluon.ai/stable/tutorials/tabular/advanced/tabular-model-ensembling.html) - Native ensemble methods
