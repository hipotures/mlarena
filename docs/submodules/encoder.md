# Encoder Sub-Module

## Overview

Universal categorical encoding with multiple strategies, compatible with AutoGluon's native categorical support.

**Module Name**: `encoder`
**Category**: Feature Transformation
**Libraries**: `sklearn.preprocessing`, `category_encoders`, `pandas`

## Purpose

Provides flexible categorical encoding with support for:
- **None** - Leave categories as-is (for AutoGluon native support)
- **One-Hot Encoding** - Binary features for each category
- **Ordinal Encoding** - Integer labels for categories
- **Target Mean Encoding** - Replace categories with target mean (smoothed)
- **CatBoost Encoding** - Ordered target statistics
- **Feature Hashing** - Fixed-size hash space for high-cardinality features

## Parameters

### Required Parameters

None - all parameters have defaults.

### Optional Parameters

#### `encoding_method` (str, default: `"one_hot"`)

Encoding strategy to use.

**Choices**: `none` | `one_hot` | `ordinal` | `target_mean` | `catboost` | `hashing`

- `none` - Keep categorical columns as-is (recommended for AutoGluon)
- `one_hot` - One-hot encoding (sklearn)
- `ordinal` - Ordinal encoding (sklearn)
- `target_mean` - Target mean encoding with smoothing
- `catboost` - CatBoost-style ordered target encoding (requires `category_encoders`)
- `hashing` - Feature hashing for high-cardinality features

#### `include_cols` (list, default: `null`)

List of specific columns to encode. If `null`, encodes all categorical columns (excluding system columns).

**Example**:
```yaml
include_cols: ["Sex", "Embarked", "Cabin"]
```

#### `exclude_cols` (list, default: `null`)

List of columns to exclude from encoding (in addition to system columns like ID, target).

**Example**:
```yaml
exclude_cols: ["Name", "Ticket"]  # Don't encode these high-cardinality columns
```

#### `max_cardinality` (int, default: `50`)

**CRITICAL PARAMETER**: Automatically excludes categorical columns with more than this many unique values from `one_hot` and `ordinal` encoding methods.

**Purpose**: Prevents explosion of one-hot features and overfitting on high-cardinality columns like Name, Ticket, Cabin.

**Behavior**:
- If column has >50 unique values (default), it's auto-excluded
- Warning is shown with list of excluded columns
- Does NOT apply if column is in `include_cols` (explicit override)
- Does NOT apply to `target_mean`, `catboost`, `hashing` methods (they handle high cardinality)

**Example**:
```yaml
encoder:
  module: encoder
  cache: true
  config:
    encoding_method: "one_hot"
    max_cardinality: 100  # Increase threshold to encode more columns
```

**Recommendation**:
- Keep default (50) for one-hot encoding
- Use `rare_category_handler` sub-module BEFORE encoder to reduce cardinality
- For very high cardinality (>100), use `target_mean` or `hashing` instead

#### `drop_first` (bool, default: `false`)

For `one_hot` encoding: drop the first category to avoid multicollinearity.

**Recommended**: `false` for tree-based models, `true` for linear models.

#### `handle_unknown` (str, default: `"ignore"`)

How to handle categories in test set that weren't seen in training.

**Choices**: `ignore` | `error` | `use_encoded_value`

- `ignore` - Assign all zeros (one-hot) or use unknown_value (ordinal)
- `error` - Raise error if unknown category encountered
- `use_encoded_value` - Use `unknown_value` parameter (ordinal only)

#### `unknown_value` (int, default: `-1`)

Value to assign to unknown categories when `handle_unknown="use_encoded_value"` (ordinal encoding only).

#### `hash_dim` (int, default: `8`)

Hash space dimension for `hashing` method. Higher values reduce collisions but increase feature count.

**Recommended**: 8-16 for moderate cardinality, 32-64 for high cardinality.

#### `target_encoding_smoothing` (float, default: `1.0`)

Smoothing parameter for `target_mean` and `catboost` encodings. Higher values = more regularization (closer to global mean).

**Formula**: `(count * mean + smoothing * global_mean) / (count + smoothing)`

**Recommended**: 1.0-10.0

#### `target_encoding_min_samples` (int, default: `1`)

Minimum samples required for a category to get its own encoding (target_mean only). Categories below threshold use global mean.

**Recommended**: 1-10 depending on dataset size.

#### `keep_original` (bool, default: `false`)

If `true`, keeps original categorical columns alongside encoded versions.

**Use case**: When you want both encoded features AND native categorical support (AutoGluon).

## Encoding Strategies

### 1. None (`encoding_method: "none"`)

**Best for**: AutoGluon with native categorical support

Pass-through - no encoding performed.

**Example**:
```yaml
encoder:
  module: encoder
  cache: true
  config:
    encoding_method: "none"
```

**Output**: Original DataFrame unchanged.

---

### 2. One-Hot Encoding (`encoding_method: "one_hot"`)

**Best for**: Linear models, neural networks, low-cardinality features

Creates binary (0/1) column for each category.

**Example**:
```yaml
encoder:
  module: encoder
  cache: true
  config:
    encoding_method: "one_hot"
    drop_first: false
    handle_unknown: "ignore"
```

**Output**:
- Original column: `Sex = ["male", "female"]`
- Encoded: `Sex_male`, `Sex_female` (or just `Sex_female` if `drop_first=true`)

**Artifact**: `onehot_encoder.pkl`

---

### 3. Ordinal Encoding (`encoding_method: "ordinal"`)

**Best for**: Tree-based models, ordinal relationships

Assigns integer labels to categories.

**Example**:
```yaml
encoder:
  module: encoder
  cache: true
  config:
    encoding_method: "ordinal"
    handle_unknown: "use_encoded_value"
    unknown_value: -1
```

**Output**:
- Original column: `Embarked = ["S", "C", "Q"]`
- Encoded: `Embarked = [0, 1, 2]`

**Artifact**: `ordinal_encoder.pkl`

---

### 4. Target Mean Encoding (`encoding_method: "target_mean"`)

**Best for**: High-cardinality categorical features, tree-based models

Replaces categories with smoothed mean of target variable.

**Requires**: Target column must be present in training data.

**Example**:
```yaml
encoder:
  module: encoder
  cache: true
  config:
    encoding_method: "target_mean"
    target_encoding_smoothing: 5.0
    target_encoding_min_samples: 10
```

**Output**:
- Original column: `Cabin = ["C85", "C123", "E46", ...]`
- Encoded: `Cabin_te = [0.75, 0.32, 0.68, ...]` (smoothed target means)

**Artifacts**: `target_encodings.json` (category → target mean mapping)

**Warning**: Can leak information if not used carefully. Consider using with cross-validation.

---

### 5. CatBoost Encoding (`encoding_method: "catboost"`)

**Best for**: High-cardinality features, tree-based models, reducing overfitting

Ordered target statistics (similar to target encoding but with ordering to reduce leakage).

**Requires**:
- Target column must be present
- `category_encoders` library (falls back to `target_mean` if not installed)

**Example**:
```yaml
encoder:
  module: encoder
  cache: true
  config:
    encoding_method: "catboost"
```

**Output**:
- Original column: `Name = ["John Smith", "Jane Doe", ...]`
- Encoded: `Name_cb = [0.42, 0.71, ...]` (ordered target statistics)

**Artifact**: `catboost_encoder.pkl`

---

### 6. Feature Hashing (`encoding_method: "hashing"`)

**Best for**: Very high-cardinality features, memory constraints

Projects categories into fixed-size hash space.

**Example**:
```yaml
encoder:
  module: encoder
  cache: true
  config:
    encoding_method: "hashing"
    hash_dim: 16
```

**Output**:
- Original column: `Ticket = ["A/5 21171", "PC 17599", ...]`
- Encoded: `Ticket_hash_0`, `Ticket_hash_1`, ..., `Ticket_hash_15` (16 features)

**Note**: No artifact saved (hashing is deterministic and stateless).

---

## Common Use Cases

### Use Case 1: AutoGluon with Native Categories

```yaml
# No encoding - let AutoGluon handle categories
encoder:
  module: encoder
  cache: true
  config:
    encoding_method: "none"
```

### Use Case 2: Mixed Encoding (Native + Encoded)

```yaml
# Encode high-cardinality features, keep low-cardinality as categorical
encoder:
  module: encoder
  cache: true
  config:
    encoding_method: "target_mean"
    include_cols: ["Name", "Ticket", "Cabin"]  # High-cardinality only
    keep_original: true  # Keep original for AutoGluon
```

### Use Case 3: One-Hot for Linear Model

```yaml
encoder:
  module: encoder
  cache: true
  config:
    encoding_method: "one_hot"
    max_cardinality: 50  # Auto-exclude high-cardinality (default)
    drop_first: true  # Avoid multicollinearity
```

### Use Case 4: Ordinal for Tree Models

```yaml
encoder:
  module: encoder
  cache: true
  config:
    encoding_method: "ordinal"
    handle_unknown: "use_encoded_value"
    unknown_value: -1
```

---

## Artifacts Generated

### `summary.json`

Standard preprocessing report with shape changes and new columns.

### `onehot_encoder.pkl` (one_hot method)

Fitted `sklearn.preprocessing.OneHotEncoder` object.

### `ordinal_encoder.pkl` (ordinal method)

Fitted `sklearn.preprocessing.OrdinalEncoder` object.

### `target_encodings.json` (target_mean method)

Mapping of category → smoothed target mean for each column.

```json
{
  "encodings": {
    "Cabin": {
      "C85": 0.75,
      "C123": 0.32,
      ...
    }
  },
  "global_mean": 0.38
}
```

### `catboost_encoder.pkl` (catboost method)

Fitted `category_encoders.CatBoostEncoder` object.

---

## State Dictionary

```python
{
  "version": "1.0",
  "config": {...},  # User config (without _ prefixes)
  "encoding_method": "one_hot",
  "encoded_columns": ["Sex", "Embarked", "Pclass"],
  "encoded_columns_info": {
    "n_features_out": 8,
    "feature_names": ["Sex_male", "Sex_female", ...],
    "drop_first": false
  },
  "keep_original": false
}
```

---

## Performance Considerations

### One-Hot Encoding

- **Memory**: Can explode with high-cardinality features (1000 categories = 1000 columns)
- **Speed**: Fast encoding, slower training with many features
- **Recommendation**: Use only for low-cardinality features (<50 categories)

### Target Mean Encoding

- **Risk**: Overfitting if not smoothed properly
- **Speed**: Fast, memory-efficient
- **Recommendation**: Increase `smoothing` for small datasets or rare categories

### Feature Hashing

- **Trade-off**: Fixed size but potential hash collisions
- **Speed**: Very fast, memory-efficient
- **Recommendation**: Use `hash_dim` ~10% of unique categories for good trade-off

---

## Edge Cases

### High-Cardinality Protection (Built-in)

**Automatic protection**: Encoder auto-excludes columns with >`max_cardinality` unique values (default: 50).

**Example on Titanic**:
```
Warning: Excluded 3 high-cardinality columns from one_hot encoding (>50 unique values):
['Name', 'Ticket', 'Cabin']. Consider using 'target_mean', 'catboost', or 'hashing'
for these columns, or increase 'max_cardinality' parameter.
```

**Solutions for high-cardinality columns**:
1. **Best**: Use `rare_category_handler` sub-module BEFORE encoder (groups rare categories, extracts features)
2. **Alternative**: Use `encoding_method: "target_mean"` or `"hashing"` (no cardinality limits)
3. **Override**: Set `max_cardinality: 1000` or use `include_cols: ["Name"]` to force encoding

### Missing Categories in Test

With `handle_unknown="ignore"`:
- One-hot: All zeros (no category matches)
- Ordinal: Uses `unknown_value` (-1 by default)
- Target mean: Uses `global_mean`

### Target Encoding Without Target

If using `target_mean` or `catboost` methods, target column must be present in config. If not, error will be raised.

---

## Tuning Tips

### For Tree-Based Models (XGBoost, LightGBM, CatBoost)

1. **Try `encoding_method: "none"` first** - Tree models handle categorical naturally
2. If encoding needed:
   - Low cardinality (<50): `ordinal` or `one_hot`
   - High cardinality (>50): `target_mean` or `catboost`

### For Linear Models (Logistic Regression, SVM)

1. **Use `one_hot` with `drop_first: true`**
2. **Exclude high-cardinality columns** or use `hashing`

### For Neural Networks

1. **Use `one_hot` for low-cardinality**
2. **Use `hashing` for high-cardinality**
3. Consider entity embeddings (not supported in this module)

---

## Chain Compatibility

Works well after:
- `rare_category_handler` - Reduces cardinality before encoding
- `imputer` - Fills missing values before encoding

Works well before:
- `scaler` - Scales the encoded numerical features
- `feature_selector` - Selects best encoded features

**Example Chain**:
```yaml
full_pipeline:
  chain: [rare_category_handler, encoder, scaler, feature_selector]
```

---

## See Also

- **rare_category_handler** - Reduce cardinality before encoding
- **autogluon_booster** - Optimize features for AutoGluon
- **feature_selector** - Select best features after encoding
