# pseudo_labeling

## Overview
Trains a quick model on labeled data and adds confident predictions from the test
set as pseudo-labeled rows. This is a semi-supervised augmentation step.

## Parameters
- `include_cols` (list[str] | null): Explicit numeric columns to use.
- `exclude_cols` (list[str]): Columns to exclude.
- `use_original_features_only` (bool): Restrict to original features.
- `model_type` (str): `logreg` or `rf`.
- `logreg_max_iter` (int): Logistic regression iterations.
- `logreg_c` (float): Logistic regression C.
- `rf_n_estimators` (int): Random forest trees.
- `rf_max_depth` (int | null): Random forest max depth.
- `confidence_threshold` (float): Minimum confidence for pseudo labels.
- `max_pseudo_fraction` (float | null): Max fraction of test rows to add.
- `use_soft_labels` (bool): Use probabilities for binary targets.
- `weight_by_confidence` (bool): Add/scale `sample_weight` by confidence.
- `scale_features` (bool): Standardize for logistic regression.
- `missing_strategy` (str): `mean`, `median`, `zero`.
- `fit_on_val` (bool): Include validation rows in model fit if targets exist.
- `allow_regression` (bool): Enable regression pseudo-labeling (random subset).
- `regression_keep_fraction` (float): Fraction of test rows to add for regression.
- `random_state` (int): RNG seed.

## Example
```yaml
module: pseudo_labeling
config:
  model_type: logreg
  confidence_threshold: 0.9
  max_pseudo_fraction: 0.2
```

## Artifacts
- `pseudo_model.pkl`: Fitted model.
- `scaler.pkl`: Optional scaler for logistic regression.
- `summary.json`: Transformation summary.

## Notes
- Adds pseudo-labeled rows to training data only.
- Confidence-based selection is applied for classification.
- Use with care when test distribution differs from train.
