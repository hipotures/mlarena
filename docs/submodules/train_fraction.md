# Train Fraction Sub-Module

## Overview

The **train_fraction** sub-module provides a way to subsample the training data and create internal validation (tuning) and evaluation (holdout) splits. This is useful for speeding up development on large datasets or creating consistent local evaluation setups.

**Module Name**: `train_fraction`  
**Location**: `src/mlarena/defaults/preprocessing/train_fraction.py`

## Capabilities
- **Subsampling**: Reduce the training set size to a fraction of the original.
- **Tuning Split**: Create a validation set (`tuning_processed.csv.gz`) for hyperparameter optimization.
- **Evaluation Split**: Create a holdout set (`eval_processed.csv.gz`) for offline metric calculation.
- **Deterministic**: Uses a single shuffle with a fixed `random_state`.
- **4-way Split**: Logic: `[Train | Tuning | Eval | Discard]`.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_fraction` | float | `1.0` | Fraction of original data to use for training (0, 1] |
| `valid_fraction` | float | `0.0` | Fraction of original data for validation/tuning [0, 1) |
| `eval_fraction` | float | `0.0` | Fraction of original data for offline evaluation [0, 1) |
| `random_state` | int | `42` | Seed for reproducibility |

**Note**: The sum of `train_fraction + valid_fraction + eval_fraction` must be ≤ 1.0.

## Examples

### Fast Development (10% Data)
```yaml
train_fast:
  module: train_fraction
  cache: true
  config:
    train_fraction: 0.1
```

### Full Local Evaluation Setup
```yaml
train_with_holdout:
  module: train_fraction
  cache: true
  config:
    train_fraction: 0.7
    valid_fraction: 0.1
    eval_fraction: 0.1
    # 10% is discarded
```

## Artifacts
- `eval_processed.csv.gz`: The holdout evaluation set (if `eval_fraction > 0`). Saved in the experiment's artifact directory.
- `tuning_processed.csv.gz`: The validation set (if `valid_fraction > 0`), automatically handled by the pipeline.

## State Dictionary (`fit_transform` return)
```python
{
    "train_fraction": 0.7,
    "valid_fraction": 0.1,
    "eval_fraction": 0.1,
    "input_rows": 100000,
    "train_rows": 70000,
    "tuning_rows": 10000,
    "eval_rows": 10000,
    "discarded_rows": 10000,
    "eval_path": "experiments/pre-.../eval_processed.csv.gz"
}
```

## Notes & Tips
- **Positioning**: This module should typically be **first** in your preprocessing chain if you want to speed up all subsequent steps.
- **Evaluation Data**: The `eval_processed.csv.gz` is a "true holdout" - it is saved immediately and is NOT transformed by any subsequent preprocessing steps in the chain.
- **Validation Data**: The `tuning_out` is returned in the `val_df` slot, meaning it WILL be transformed by any following sub-modules in the chain (e.g., if you have `[train_fraction, scaler]`, both train and tuning sets will be scaled).
- **Test Data**: The submission `test_df` is passed through completely unchanged.
