# Sanity Check - Data Quality and Type Enforcement

## Overview

The `sanity_check` sub-module performs basic data cleaning and quality checks to identify and fix obvious problems before running the rest of the preprocessing pipeline.

**Purpose**: Enforce data types, remove problematic columns (constant values, high missing rates), detect and fix infinite values, and optionally remove duplicate rows.

**Libraries**: `pandas`, `numpy`

**Location**: `src/mlarena/defaults/preprocessing/sanity_check.py`

---

## Parameters

### Required Parameters

None - all parameters are optional with sensible defaults.

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_unique_fraction` | float | `0.01` | Minimum fraction of unique values required to keep a column. Columns with fewer unique values are considered "nearly constant" and dropped. Range: `0.0` to `1.0`. |
| `max_missing_fraction` | float | `0.95` | Maximum fraction of missing values allowed. Columns with more missing values are dropped. Range: `0.0` to `1.0`. |
| `max_missing_fraction_row` | float \| null | `null` | Maximum fraction of missing values allowed per row. Rows above this threshold are dropped. Range: `0.0` to `1.0`. |
| `drop_duplicates` | bool | `true` | Whether to remove duplicate rows. Keeps first occurrence. |
| `column_types_override` | dict | `{}` | Dictionary mapping column names to target dtypes (e.g., `{"Age": "float64", "Pclass": "int32"}`). Forces type conversion. |
| `ignore_columns` | list | `[]` | List of column names that should never be dropped, in addition to automatic protection of ID and target columns. |

---

## Protected Columns

The following columns are **automatically protected** from being dropped:
- ID column (from `_dataset.id_column`)
- Target column (from `_dataset.target`)
- Ignored columns (from `_dataset.ignored_columns`)
- Columns listed in `ignore_columns` parameter

---

## Behavior

### 1. Infinite Value Detection

- Scans all numeric columns for `inf` and `-inf` values
- Replaces infinite values with `NaN`
- Logs count of infinite values per column

### 2. Constant Column Detection

- Calculates `unique_fraction = nunique / len(df)` for each column
- If `unique_fraction < min_unique_fraction`, marks column for removal
- Protected columns are never removed

**Example**:
- Dataset with 1000 rows, column has 5 unique values → `unique_fraction = 0.005`
- If `min_unique_fraction = 0.01`, column is dropped (too few unique values)

### 3. High Missing Column Detection

- Calculates `missing_fraction = null_count / len(df)` for each column
- If `missing_fraction > max_missing_fraction`, marks column for removal
- Protected columns are never removed

### 4. Duplicate Row Removal

- If `drop_duplicates = true`, removes duplicate rows
- Keeps first occurrence
- Applies to train, test, and validation sets independently
- Resets indices after removal

### 5. Type Enforcement

- Attempts to convert columns specified in `column_types_override`
- If conversion fails, logs warning and continues
- Does not fail the entire pipeline

### 6. Row Missingness Filter

- If `max_missing_fraction_row` is set, rows with missing fraction above the threshold are dropped.
- Applies independently to train, test, val, and orig datasets.

---

## Artifacts Generated

### 1. `sanity_report.json`

Detailed report of detected issues and actions taken:

```json
{
  "issues_found": {
    "constant_columns": [
      {
        "column": "Sex",
        "unique_count": 2,
        "unique_fraction": 0.002244
      }
    ],
    "high_missing_columns": [
      {
        "column": "Cabin",
        "missing_count": 687,
        "missing_fraction": 0.771
      }
    ],
    "infinite_values": {
      "train_Price": 5,
      "test_Price": 3
    },
    "duplicate_rows_train": 12,
    "duplicate_rows_test": 0,
    "type_mismatches": []
  },
  "columns_dropped": ["Sex", "Cabin"],
  "columns_dropped_count": 2,
  "duplicates_removed": {
    "train": 12,
    "test": 0
  },
  "types_changed": {
    "Age": {"from": "object", "to": "float64"}
  },
  "protected_columns": ["PassengerId", "Survived"],
  "final_columns": ["PassengerId", "Survived", "Pclass", "Name", "Age", ...],
  "final_column_count": 10
}
```

### 2. `summary.json`

Standard preprocessing report with before/after shapes and column changes.

---

## State Dict

The `state_dict` returned by `fit_transform()` contains:

```python
{
    "version": "1.0",
    "config": {...},  # User config (without internal _ params)
    "issues_found": {...},  # Same as sanity_report.json
    "columns_dropped": [...],  # List of dropped column names
    "columns_dropped_count": 2,
    "duplicates_removed_train": 12,
    "duplicates_removed_test": 0,
    "types_changed": {...}  # Column type conversions
}
```

---

## Examples

### Example 1: Default Configuration

```yaml
sanity_check:
  module: sanity_check
  cache: true
  config: {}  # Use all defaults
```

**Behavior**:
- Drops columns with < 1% unique values
- Drops columns with > 95% missing values
- Removes duplicate rows
- No type enforcement

### Example 2: Small Dataset (Titanic)

For small datasets (< 1000 rows), lower the `min_unique_fraction`:

```yaml
sanity_check_titanic:
  module: sanity_check
  cache: true
  config:
    min_unique_fraction: 0.0001  # 0.01% instead of 1%
    max_missing_fraction: 0.95
    drop_duplicates: true
```

**Why**: In a dataset with 891 rows (Titanic), a column with 3 unique values has `unique_fraction = 0.0034` (0.34%), which would be dropped with default `0.01` threshold. This is often undesirable for categorical features like `Sex` or `Pclass`.

### Example 3: Aggressive Cleaning

```yaml
sanity_check_aggressive:
  module: sanity_check
  cache: true
  config:
    min_unique_fraction: 0.05  # Require 5% unique values
    max_missing_fraction: 0.5  # Drop columns with >50% missing
    drop_duplicates: true
    ignore_columns: ["ImportantFeature"]  # Don't drop this even if nearly constant
```

### Example 4: Type Enforcement

```yaml
sanity_check_typed:
  module: sanity_check
  cache: true
  config:
    min_unique_fraction: 0.01
    max_missing_fraction: 0.95
    drop_duplicates: true
    column_types_override:
      Age: float64
      Pclass: int32
      Fare: float64
      Sex: category
```

### Example 5: No Dropping, Only Fixing

```yaml
sanity_check_conservative:
  module: sanity_check
  cache: true
  config:
    min_unique_fraction: 0.0  # Never drop constant columns
    max_missing_fraction: 1.0  # Never drop high-missing columns
    drop_duplicates: false  # Keep duplicates
    column_types_override:
      Age: float64  # Only enforce types
```

---

## Tuning Recommendations

### For Small Datasets (< 1000 rows)

- Set `min_unique_fraction: 0.0001` or lower
- Categorical features naturally have low unique fractions

### For Large Datasets (> 100k rows)

- Default `min_unique_fraction: 0.01` is usually fine
- Consider raising `max_missing_fraction` if data is naturally sparse

### For High-Cardinality Data (e.g., text, IDs)

- Add high-cardinality columns to `ignore_columns` to prevent accidental removal
- Or use `rare_category_handler` sub-module instead

### For Dirty Data

- Lower `max_missing_fraction` to `0.7` or `0.8` to aggressively clean
- Set `drop_duplicates: true` (default)

---

## Edge Cases

### 1. All Columns Dropped

If `min_unique_fraction` is too high or `max_missing_fraction` is too low, all non-protected columns might be dropped.

**Solution**: Adjust thresholds or add important columns to `ignore_columns`.

### 2. Type Conversion Failures

If `column_types_override` specifies invalid conversions (e.g., converting text to numeric), the conversion is skipped and logged in `type_mismatches`.

**Solution**: Check `sanity_report.json` for `type_mismatches` and fix column names or types.

### 3. Different Columns in Train vs. Test

After sanity check, train and test may have different columns if one dataset has more constant/high-missing columns.

**Solution**: This is expected behavior. Downstream sub-modules should handle this using `align_columns()` utility.

### 4. Protected Column is Nearly Constant

Protected columns (ID, target, ignored) are never dropped, even if they fail `min_unique_fraction` check.

**Solution**: This is intentional. If you want to drop a protected column, remove it from the protection list first.

---

## Chain Compatibility

### Recommended Position

**First** in any preprocessing chain.

```yaml
my_pipeline:
  chain: [sanity_check, imputer, scaler, encoder]
```

### Why First?

- Removes problematic columns before other sub-modules process them
- Prevents downstream errors from constant/high-missing columns
- Cleans infinite values early

### Can Be Skipped?

Yes, but not recommended. Without sanity checks:
- Imputation may fail on all-NA columns
- Scalers may fail on constant columns
- Models may fail on infinite values

---

## Common Issues

### Issue: Important categorical column dropped

**Symptom**: Column like `Sex` or `Pclass` is missing after sanity check.

**Cause**: `min_unique_fraction` threshold too high for dataset size.

**Solution**:
```yaml
config:
  min_unique_fraction: 0.0001  # Lower threshold
  # Or protect the column:
  ignore_columns: ["Sex", "Pclass"]
```

### Issue: Too many columns dropped

**Symptom**: Only ID and target remain.

**Cause**: Thresholds too aggressive for this dataset.

**Solution**:
```yaml
config:
  min_unique_fraction: 0.001  # Lower from 0.01
  max_missing_fraction: 0.99  # Raise from 0.95
```

### Issue: Type conversion warnings

**Symptom**: Logs show `"Could not convert column 'X' to dtype"`.

**Cause**: Column contains non-numeric data or nulls that prevent conversion.

**Solution**:
- Check column data: `df['X'].value_counts()`
- Fix data issues first or remove column from `column_types_override`

---

## Performance

- **Speed**: Very fast (< 1 second on datasets up to 1M rows)
- **Memory**: Low (only copies DataFrames for reporting)
- **I/O**: Minimal (saves 2 JSON reports)

---

## Related Sub-Modules

- **imputer**: Run after `sanity_check` to handle remaining missing values
- **rare_category_handler**: Better for handling low-frequency categories than dropping them
- **outlier_handler**: Handles extreme values (not infinite values)

---

## Version History

- **v1.0** (2025-12-09): Initial implementation
  - Basic sanity checks
  - Infinite value detection
  - Duplicate removal
  - Type enforcement
