# Scaler Sub-Module

## Overview

The **scaler** sub-module provides standardization and distribution transformations for numerical features. It's designed for models sensitive to feature scale/distribution (e.g., linear models, neural networks, distance-based algorithms). Supports multiple scaling methods, log transformations, and winsorization.

## Purpose

- Standardize numerical features to consistent scales
- Transform distributions to be closer to normal (Gaussian)
- Apply log transformations to handle skewed data
- Clip outliers using quantile-based winsorization
- Prepare features for scale-sensitive models

## Libraries

- `sklearn.preprocessing`: StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
- `numpy`: Log transformations and mathematical operations
- `pandas`: DataFrame manipulations

## Parameters

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scaling_method` | str | `"none"` | Scaling method to apply |
| `numeric_include` | List[str] \| None | `None` | Specific columns to scale (None = all numeric) |
| `numeric_exclude` | List[str] | `[]` | Columns to exclude from scaling |

### Scaling Methods

- **`none`**: No scaling applied (pass-through)
- **`standard`**: Z-score normalization (mean=0, std=1)
- **`minmax`**: Scale to [0, 1] range
- **`robust`**: Scale using median and IQR (robust to outliers)
- **`quantile_normal`**: Transform to normal distribution using quantiles
- **`quantile_uniform`**: Transform to uniform distribution using quantiles

### Transformation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_transform` | List[str] | `[]` | Columns for log1p transformation (applied before scaling) |
| `clip_lower_quantile` | float \| None | `None` | Lower quantile for winsorization (0.0-1.0) |
| `clip_upper_quantile` | float \| None | `None` | Upper quantile for winsorization (0.0-1.0) |
| `n_quantiles` | int | `1000` | Number of quantiles for QuantileTransformer |
| `random_state` | int | `42` | Random state for QuantileTransformer |

## Examples

### Example 1: Standard Scaling (Z-score)

```yaml
scaler_standard:
  module: scaler
  cache: true
  config:
    scaling_method: "standard"
    numeric_exclude: ["id", "year"]  # Don't scale these
```

**Effect**: All numeric columns (except id, year) scaled to mean=0, std=1.

### Example 2: MinMax Scaling to [0, 1]

```yaml
scaler_minmax:
  module: scaler
  cache: true
  config:
    scaling_method: "minmax"
```

**Effect**: All numeric features scaled to range [0, 1].

### Example 3: Robust Scaling (Outlier-Resistant)

```yaml
scaler_robust:
  module: scaler
  cache: true
  config:
    scaling_method: "robust"
    clip_lower_quantile: 0.01  # Clip bottom 1%
    clip_upper_quantile: 0.99  # Clip top 1%
```

**Effect**: Scale using median and IQR, with outlier clipping.

### Example 4: Log Transformation + Standard Scaling

```yaml
scaler_log_standard:
  module: scaler
  cache: true
  config:
    scaling_method: "standard"
    log_transform: ["price", "volume", "count"]  # Right-skewed columns
```

**Effect**:
1. Creates new columns: `price_log`, `volume_log`, `count_log`
2. Applies standard scaling to ALL numeric columns (including log-transformed)

### Example 5: Quantile Transform to Normal Distribution

```yaml
scaler_quantile_normal:
  module: scaler
  cache: true
  config:
    scaling_method: "quantile_normal"
    n_quantiles: 1000
    random_state: 42
```

**Effect**: Transform features to approximate normal distribution using quantile mapping.

### Example 6: Winsorization Only (No Scaling)

```yaml
scaler_clip_only:
  module: scaler
  cache: true
  config:
    scaling_method: "none"
    clip_lower_quantile: 0.05  # Bottom 5%
    clip_upper_quantile: 0.95  # Top 5%
```

**Effect**: Clip extreme values without scaling.

### Example 7: Specific Columns Only

```yaml
scaler_selected:
  module: scaler
  cache: true
  config:
    scaling_method: "standard"
    numeric_include: ["age", "income", "balance"]  # Only scale these
```

**Effect**: Scale only specified columns, leave others unchanged.

## Artifacts

The scaler module creates the following artifacts in `submodules/scaler/`:

### Files Created

1. **`scaler.pkl`** (if scaling_method != "none")
   - Fitted sklearn scaler object
   - Used for inference-time transformations

2. **`summary.json`**
   - Standard preprocessing report
   - Additional `scaling` section with:
     - Scaling method used
     - List of scaled columns
     - List of log-transformed columns
     - List of clipped columns
     - Clip bounds per column
     - Column statistics (before/after)

### Summary JSON Structure

```json
{
  "version": "1.0",
  "train": { /* shape changes */ },
  "test": { /* shape changes */ },
  "config": { /* sanitized config */ },
  "scaling": {
    "method": "standard",
    "scaled_columns": ["age", "income", "balance"],
    "log_transformed_columns": ["price_log", "volume_log"],
    "clipped_columns": ["age", "income"],
    "clip_bounds": {
      "age": {"lower": 18.0, "upper": 80.0},
      "income": {"lower": 10000.0, "upper": 500000.0}
    },
    "column_stats": {
      "age": {
        "before": {"mean": 45.2, "std": 15.8, "min": 0, "max": 120},
        "after": {"mean": 0.0, "std": 1.0, "min": -1.72, "max": 2.21}
      }
    }
  }
}
```

## State Dictionary

The `state_dict` returned by `fit_transform` contains:

```python
{
    "version": "1.0",
    "config": {...},  # User config (without _ prefixes)
    "scaling_method": "standard",
    "scaled_columns": ["col1", "col2", ...],
    "log_transformed_columns": ["col1_log", ...],
    "clipped_columns": ["col1", "col2", ...],
    "clip_bounds": {
        "col1": {"lower": 10.0, "upper": 100.0},
        ...
    },
    "scaler_path": "submodules/scaler/scaler.pkl",  # If scaling applied
    "column_stats": {...}  # Statistics before/after
}
```

## Transformation Order

The scaler applies transformations in this order:

1. **Log Transformation** (if specified)
   - Creates new columns with `_log` suffix
   - Uses `log1p` to handle zeros
   - Automatically shifts negative values

2. **Winsorization/Clipping** (if specified)
   - Calculates quantiles from **train data only**
   - Clips all datasets (train/val/test) to same bounds
   - Prevents extreme outliers from affecting scaler fit

3. **Scaling** (if method != "none")
   - Fits scaler on **train data only**
   - Transforms all datasets using fitted scaler
   - Saves scaler for inference

## Notes

### When to Use Scaling

**Models that benefit from scaling:**
- Linear models (LogisticRegression, LinearRegression, Ridge, Lasso)
- Support Vector Machines (SVM, SVR)
- Neural Networks
- K-Nearest Neighbors (KNN)
- Principal Component Analysis (PCA)
- Clustering algorithms (K-Means, DBSCAN)

**Models that DON'T need scaling:**
- Tree-based models (RandomForest, XGBoost, LightGBM, CatBoost)
- Naive Bayes
- Decision Trees

### Scaling Method Selection

| Method | When to Use | Pros | Cons |
|--------|-------------|------|------|
| `standard` | General purpose, assume normal distribution | Standard practice, interpretable | Sensitive to outliers |
| `minmax` | Bounded features, neural networks | Preserves zero, bounded output | Very sensitive to outliers |
| `robust` | Data with outliers | Robust to outliers | Less standardized |
| `quantile_normal` | Non-normal distributions | Creates normal distribution | Loses extreme values info |
| `quantile_uniform` | Uniform distribution needed | Creates uniform distribution | Loses extreme values info |

### Log Transformation Tips

**When to use log transformation:**
- Right-skewed distributions (long tail on right)
- Features with wide range (e.g., income, price, counts)
- Exponential relationships

**Warning:**
- Cannot handle negative values (scaler auto-shifts if needed)
- Cannot handle zeros in original log (scaler uses log1p)
- Creates new columns (increases feature count)

### Winsorization vs Clipping

**Winsorization** (this module): Clips values to quantile bounds
- Example: `clip_lower_quantile=0.01` → bottom 1% set to 1st percentile value
- **Preserves all rows**, just bounds extreme values

**Outlier Removal** (use `outlier_handler` module instead): Removes rows
- Drops rows with extreme values
- **Reduces dataset size**

### Quantile Transformers

**QuantileTransformer** is powerful but has trade-offs:

**Pros:**
- Handles arbitrary distributions
- Robust to outliers
- Creates specific target distribution (normal/uniform)

**Cons:**
- Loses magnitude information
- May overfit to train distribution
- Expensive with large `n_quantiles`

**Recommendation**: Start with `standard` or `robust`, try `quantile_normal` if distribution is very non-normal.

## Tuning Tips

### 1. Check Feature Distributions First

```python
import matplotlib.pyplot as plt

# Plot distributions before scaling
train_df[numeric_cols].hist(bins=50, figsize=(15, 10))
plt.show()
```

Identify:
- Skewed distributions → consider `log_transform`
- Outliers → consider `clip_*_quantile`
- Different scales → definitely need scaling

### 2. Start Simple

```yaml
# First try: standard scaling only
scaler_simple:
  config:
    scaling_method: "standard"
```

Then add complexity if needed:
```yaml
# Add log transform for skewed columns
scaler_improved:
  config:
    scaling_method: "standard"
    log_transform: ["price", "volume"]
    clip_lower_quantile: 0.01
    clip_upper_quantile: 0.99
```

### 3. Validate Transformations

Check `summary.json` after running:
```bash
cat experiments/pre-scaler/artifacts/preprocess/submodules/scaler/summary.json
```

Look at `column_stats` to verify:
- Scaled columns have mean ≈ 0, std ≈ 1 (for standard)
- No extreme outliers remain after clipping

### 4. Tree-Based Models

If using **only** tree-based models (XGBoost, LightGBM, RandomForest):
```yaml
scaler_skip:
  config:
    scaling_method: "none"  # Skip scaling entirely
```

Trees are scale-invariant, so scaling adds no value.

### 5. Mixed Model Ensemble

If using **both** linear and tree-based models:
```yaml
# Scale for linear models, trees ignore scaling anyway
scaler_ensemble:
  config:
    scaling_method: "robust"  # Robust to outliers for better linear models
    clip_lower_quantile: 0.01
    clip_upper_quantile: 0.99
```

## Common Patterns

### Pattern 1: Preprocessing for Linear Model

```yaml
chain_linear:
  chain: [sanity_check, imputer, scaler_standard, encoder]

scaler_standard:
  module: scaler
  config:
    scaling_method: "standard"
```

### Pattern 2: Preprocessing for Neural Network

```yaml
chain_nn:
  chain: [sanity_check, imputer, scaler_minmax, encoder]

scaler_minmax:
  module: scaler
  config:
    scaling_method: "minmax"  # Bounded [0,1] for activation functions
```

### Pattern 3: Handling Skewed Financial Data

```yaml
scaler_financial:
  module: scaler
  config:
    scaling_method: "robust"
    log_transform: ["income", "loan_amount", "credit_limit"]
    clip_lower_quantile: 0.01
    clip_upper_quantile: 0.99
```

### Pattern 4: AutoGluon with Scaling (Mixed Models)

```yaml
chain_autogluon_mixed:
  chain: [sanity_check, imputer, scaler_robust, autogluon_booster]

scaler_robust:
  module: scaler
  config:
    scaling_method: "robust"  # Helps linear/nn models in AutoGluon ensemble
```

## Edge Cases

### Empty Numeric Columns

If no numeric columns found after exclusions:
- Module returns unchanged DataFrames
- `state_dict["scaled_columns"] = []`
- Message: "No numeric columns to process"

### Columns Only in Train or Only in Test

- Module only scales columns present in **both** train and test
- Columns unique to train or test are left unchanged

### Negative Values with Log Transform

- Module automatically shifts values: `log1p(x + shift)` where `shift = abs(min(x)) + 1`
- Ensures all values are non-negative before log

### NaN Values

- Scaling methods handle NaNs (produce NaN in output)
- **Recommendation**: Run `imputer` module **before** `scaler`

## Troubleshooting

### "No numeric columns to process"

**Cause**: All numeric columns excluded or none exist

**Fix**: Check `numeric_exclude` and `_dataset.ignored_columns`

### "ValueError: Input contains NaN"

**Cause**: Some scalers don't handle NaN

**Fix**: Add `imputer` module before `scaler` in chain:
```yaml
my_pipeline:
  chain: [sanity_check, imputer, scaler]  # imputer first!
```

### Scaled values still have outliers

**Cause**: Scaling doesn't remove outliers, just rescales them

**Fix**: Use `clip_*_quantile` to bound values before scaling:
```yaml
scaler:
  config:
    clip_lower_quantile: 0.01
    clip_upper_quantile: 0.99
    scaling_method: "standard"
```

### Log-transformed columns not scaled

**Cause**: `log_transform` creates new columns, but they're automatically added to `numeric_cols`

**Effect**: Log-transformed columns **are** scaled if `scaling_method != "none"`

### Different train/test scales after scaling

**Cause**: Test data has different distribution than train

**Fix**: This is expected - scaler fits on train only. If test has extreme values, use:
```yaml
scaler:
  config:
    clip_lower_quantile: 0.01  # Clip test to same bounds
    clip_upper_quantile: 0.99
```

## See Also

- **imputer** - Fill missing values before scaling
- **outlier_handler** - Remove outlier rows (different from clipping)
- **feature_engineer** - Create interactions (may need scaling after)
- **autogluon_booster** - AutoGluon-specific preprocessing
