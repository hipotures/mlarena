# Rare Category Handler - Sub-module 3

## Overview

The Rare Category Handler reduces cardinality and handles rare categorical values by:
- Grouping rare categories into a special `"__RARE__"` label
- Limiting to top-K most frequent categories
- Detecting and flagging potential ID columns (high uniqueness)

This preprocessing step reduces overfitting and feature space size before encoding.

## When to Use

- **High cardinality categorical features** (hundreds or thousands of unique values)
- **Rare categories** that appear very infrequently in training data
- **Before encoding** to reduce the number of encoded features
- **To detect ID columns** that should be excluded from modeling

## Parameters

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_freq` | int | `10` | Minimum absolute frequency for a category to be kept |
| `min_freq_ratio` | float | `0.01` | Minimum relative frequency (0-1) for a category to be kept |
| `top_k` | int or null | `null` | If set, keep only top K most frequent categories |
| `rare_label` | str | `"__RARE__"` | Label used for grouped rare categories |

### ID Detection Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `detect_id_like_columns` | bool | `true` | Enable automatic detection of ID-like columns |
| `id_unique_fraction_threshold` | float | `0.95` | Uniqueness threshold for ID detection (0-1) |

### Column Selection Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `protected_categorical_columns` | list[str] | `[]` | Columns to exclude from processing |

## How It Works

### 1. Category Frequency Analysis

For each categorical column:
- Count frequency of each category value in training data
- Calculate both absolute count and relative frequency

### 2. Category Selection

Categories are kept if they meet **both** conditions:
- Absolute frequency ≥ `min_freq`
- Relative frequency ≥ `min_freq_ratio`

**OR** if `top_k` is set:
- Keep only the K most frequent categories

### 3. Rare Category Grouping

All categories not selected in step 2 are mapped to `rare_label`.

### 4. ID Column Detection

If `detect_id_like_columns = true`:
- Calculate uniqueness: `unique_count / total_rows`
- Flag column as ID if uniqueness ≥ `id_unique_fraction_threshold`
- ID columns are **skipped** (not processed)

### 5. Unseen Categories

Categories in test/validation that weren't seen in training are automatically mapped to `rare_label`.

## Examples

### Example 1: Basic Usage (Default Settings)

```yaml
rare_category_handler:
  module: rare_category_handler
  cache: true
  config:
    min_freq: 10
    min_freq_ratio: 0.01
    rare_label: "__RARE__"
```

**Effect**: Categories appearing <10 times OR <1% of data are grouped to `__RARE__`.

### Example 2: Top-K Limiting

```yaml
rare_category_handler:
  module: rare_category_handler
  cache: true
  config:
    top_k: 50
    rare_label: "__OTHER__"
```

**Effect**: Keep only 50 most frequent categories per column, group rest to `__OTHER__`.

### Example 3: Strict Filtering

```yaml
rare_category_handler:
  module: rare_category_handler
  cache: true
  config:
    min_freq: 100
    min_freq_ratio: 0.05
    rare_label: "__RARE__"
```

**Effect**: Only keep categories with ≥100 occurrences AND ≥5% frequency.

### Example 4: Disable ID Detection

```yaml
rare_category_handler:
  module: rare_category_handler
  cache: true
  config:
    detect_id_like_columns: false
```

**Effect**: Process all categorical columns, don't skip ID-like columns.

### Example 5: Protect Specific Columns

```yaml
rare_category_handler:
  module: rare_category_handler
  cache: true
  config:
    min_freq: 10
    min_freq_ratio: 0.01
    protected_categorical_columns: ["country", "language"]
```

**Effect**: Don't process `country` and `language` columns, keep all their values.

## Artifacts Generated

### 1. `category_mappings.json`

Complete mapping of old → new categories for each processed column:

```json
{
  "category_mappings": {
    "city": {
      "New York": "New York",
      "Los Angeles": "Los Angeles",
      "Chicago": "Chicago",
      "Small Town A": "__RARE__",
      "Small Town B": "__RARE__"
    }
  },
  "column_stats": {
    "city": {
      "unique_before": 1000,
      "unique_after": 51,
      "n_rare_categories": 997,
      "n_kept_categories": 50,
      "reduction_ratio": 0.051
    }
  },
  "detected_id_columns": ["customer_id", "transaction_id"]
}
```

### 2. `summary.json`

Standard transformation summary (shape changes, columns modified).

## Use Cases

### Use Case 1: High Cardinality Reduction

**Problem**: Product category with 5000 unique values, most appearing only once.

**Solution**:
```yaml
config:
  min_freq: 50
  min_freq_ratio: 0.001
```

**Result**: Reduce to ~100 meaningful categories, group rare products.

### Use Case 2: Preventing Overfitting on Rare Values

**Problem**: Country column with 200 countries, 150 appear <5 times.

**Solution**:
```yaml
config:
  min_freq: 10
  top_k: 30
```

**Result**: Keep top 30 countries, group rest as "__RARE__".

### Use Case 3: ID Column Detection

**Problem**: Column `user_hash` has 99.8% unique values.

**Solution**:
```yaml
config:
  detect_id_like_columns: true
  id_unique_fraction_threshold: 0.95
```

**Result**: `user_hash` detected as ID, skipped from processing, logged in report.

## Integration with Other Sub-modules

### Before Encoding

```yaml
preprocessing_pipeline:
  chain: [sanity_check, imputer, rare_category_handler, encoder]
```

**Why**: Reduce cardinality before one-hot or target encoding.

### After Imputation

```yaml
preprocessing_pipeline:
  chain: [sanity_check, imputer, rare_category_handler, scaler]
```

**Why**: Imputation may fill missing values with mode, creating more categories to handle.

## Parameter Tuning Guidelines

### `min_freq` vs `min_freq_ratio`

- **High variance datasets**: Use `min_freq_ratio` (relative)
- **Large datasets (>100K rows)**: Use `min_freq` (absolute)
- **Small datasets (<10K rows)**: Use `min_freq_ratio`

### `top_k` Selection

- **Too low** (e.g., 10): Loss of information, underfitting
- **Too high** (e.g., 1000): No reduction, overfitting risk
- **Recommendation**: Start with 50-100 for most datasets

### `id_unique_fraction_threshold`

- **0.95**: Conservative (default)
- **0.90**: More aggressive detection
- **0.99**: Only flag truly unique columns

## Edge Cases

### Empty Categories After Filtering

If all categories in a column are rare:
- Column will have only `__RARE__` values
- Consider removing such columns in subsequent steps

### Test Set Has New Categories

New categories in test/validation are automatically mapped to `rare_label`.

### Column Becomes Constant

After grouping, column may have only 1 unique value (e.g., all `__RARE__`).
- Use `sanity_check` sub-module after this step to remove constant columns

## Performance Considerations

- **Memory**: O(n_rows * n_categorical_cols) for value counting
- **Speed**: Fast (pandas value_counts), <1 second for millions of rows
- **Scalability**: Works well on datasets with billions of categories

## Recommendations

1. **Always run before encoding** to reduce encoded feature space
2. **Combine with `min_freq_ratio`** for relative thresholding
3. **Use `top_k`** for extreme cardinality (>1000 unique values)
4. **Check `detected_id_columns`** in artifacts to verify ID detection
5. **Tune thresholds** based on validation performance

## Common Pitfalls

❌ **Setting `min_freq` too high** → Loss of information
✅ **Start conservative** (min_freq=10), increase if needed

❌ **Disabling ID detection** → Processing ID columns as features
✅ **Keep enabled** unless you're sure there are no IDs

❌ **Using only `min_freq` on small datasets** → Unbalanced thresholds
✅ **Use `min_freq_ratio`** for small/medium datasets

## State Dictionary

The `state_dict` returned by `fit_transform` contains:

```python
{
    "version": "1.0",
    "config": {...},  # Cleaned config (no _ prefixes)
    "category_mappings": {
        "column_name": {
            "original_value": "new_value",
            ...
        }
    },
    "detected_id_columns": ["col1", "col2"],
    "column_stats": {
        "column_name": {
            "unique_before": 1000,
            "unique_after": 51,
            "n_rare_categories": 949,
            "n_kept_categories": 50,
            "reduction_ratio": 0.051
        }
    },
    "n_categorical_processed": 5,
    "n_id_detected": 2
}
```
