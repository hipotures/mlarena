# mixup_augmentation

## Overview
Creates mixed samples by combining pairs of rows: `x_new = lam*x_i + (1-lam)*x_j`.
Targets are mixed for regression or kept hard for classification (unless soft labels
are enabled).

## Parameters
- `include_cols` (list[str] | null): Explicit numeric columns to use.
- `exclude_cols` (list[str]): Columns to exclude.
- `use_original_features_only` (bool): Restrict to original features.
- `augment_ratio` (float): Fraction of train rows to add.
- `alpha` (float): Beta distribution parameter for lambda.
- `lambda_clip` (float | null): Optional clip for lambda (0-0.5).
- `allow_soft_labels` (bool): Allow soft labels (binary only).
- `hard_label_threshold` (float): Threshold for hard labels.
- `random_state` (int): RNG seed.

## Example
```yaml
module: mixup_augmentation
config:
  augment_ratio: 0.3
  alpha: 0.2
  allow_soft_labels: false
```

## Artifacts
- `summary.json`: Transformation summary.

## Notes
- Works best for numeric/encoded features.
- Soft labels are not supported for multiclass in this module.
- Use after encoding/scaling for stable behavior.
