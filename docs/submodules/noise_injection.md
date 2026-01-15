# noise_injection

## Overview
Augments the training data by injecting noise into numeric features. The module
creates additional rows and leaves validation/test data unchanged.

## Parameters
- `include_cols` (list[str] | null): Explicit numeric columns to use.
- `exclude_cols` (list[str]): Columns to exclude.
- `use_original_features_only` (bool): Restrict to original features.
- `noise_type` (str): `gaussian` or `swap`.
- `augment_ratio` (float): Fraction of train rows to add (e.g., 0.3 adds 30%).
- `gaussian_sigma` (float): Gaussian noise sigma.
- `gaussian_scale_by_std` (bool): Scale noise by column std.
- `swap_prob` (float): Swap probability per cell (swap noise).
- `random_state` (int): RNG seed.

## Example
```yaml
module: noise_injection
config:
  noise_type: gaussian
  augment_ratio: 0.3
  gaussian_sigma: 0.01
```

## Artifacts
- `summary.json`: Transformation summary.

## Notes
- Applies only to training data; targets are copied from original rows.
- Use after feature engineering so the noise applies to final features.
- If you already use strong regularization, this may not help.
