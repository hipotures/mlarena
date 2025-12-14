# Dataset Merger Sub-Module

> **⚠️ DEPRECATION NOTICE**
>
> **This module is deprecated in favor of [`external_dataset`](./external_dataset.md).**
>
> - **`dataset_merger`**: Merges external data directly into training set during preprocessing (inflexible)
> - **`external_dataset`**: Provides external dataset as separate file, allowing models to decide if/how to merge (flexible)
>
> **Migration**: Use `external_dataset` for new projects. Existing projects using `dataset_merger` will continue to work, but we recommend migrating to `external_dataset` for better flexibility.
>
> **Key Benefit**: With `external_dataset`, models can optionally merge train+orig, while preprocessing modules like adversarial validation can ignore orig entirely.

## Overview

The **dataset_merger** sub-module merges external/original datasets with Kaggle competition training data. It handles column alignment, name mapping, and optional source tracking to enable training on combined datasets while maintaining feature consistency.

**Module Name**: `dataset_merger` (⚠️ deprecated)
**Location**: `config/code/preprocessing/dataset_merger.py`
**Replacement**: [`external_dataset`](./external_dataset.md)

## Capabilities

- **Dataset merging**: Combine external/original dataset with Kaggle train data
- **Column alignment modes**: Intersection (`align`) or union (`union`) of columns
- **Schema mapping**: Map column names between datasets when schemas differ
- **Source tracking**: Optional flag column to distinguish Kaggle vs original rows
- **Validation**: Ensures target column exists, validates mapping, checks file paths
- **Detailed reporting**: Column alignment summary, merge statistics, missing value analysis

## Parameters

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `orig_path` | str | **required** | Path to original dataset CSV file (relative to project root) |
| `mode` | str | `"align"` | Column alignment mode: `align` (intersection) or `union` (all columns) |
| `source_flag` | str \| null | `null` | Column name for source tracking. If set, adds binary flag: 0=Kaggle, 1=original |
| `column_mapping` | dict | `{}` | Map original column names to Kaggle names. Example: `{"orig_col": "kaggle_col"}` |

### Behavior Details

- **`orig_path`**: Required parameter. Path is resolved relative to project root (e.g., `data/original_dataset.csv`)
- **`mode`**:
  - `"align"`: Keep only columns present in BOTH datasets (intersection). Missing values filled with NA
  - `"union"`: Keep ALL columns from both datasets. Kaggle-only and original-only columns preserved
- **`source_flag`**: When enabled, adds binary column to train AND test (for feature consistency). Useful for tracking synthetic vs real data
- **`column_mapping`**: Applied to original dataset BEFORE merging. All mapping keys must exist in original dataset

## Column Alignment Modes

### `align` Mode (Intersection)
Keeps only columns that exist in both Kaggle and original datasets. This is the safest default when you want consistent features.

**Example scenario**:
- Kaggle: `[id, age, income, target]` (26 columns)
- Original: `[age, income, education, zip_code]` (31 columns)
- Result: `[age, income]` + target (3 columns total)

**Use when**:
- Original dataset is superset with extra features you don't need
- You want to avoid NA values from union mode
- Feature sets are mostly overlapping

### `union` Mode (All Columns)
Keeps ALL columns from both datasets. Missing columns filled with NA.

**Example scenario**:
- Kaggle: `[id, age, income, target]` (26 columns)
- Original: `[age, income, education, zip_code]` (31 columns)
- Result: `[id, age, income, education, zip_code]` + target (32 columns)
- Kaggle rows: `education` and `zip_code` = NA
- Original rows: `id` = NA

**Use when**:
- You want to preserve all available features
- Subsequent imputation module will handle missing values
- Original has valuable features not in Kaggle

## Source Flag

The `source_flag` parameter adds a binary column to track data origin:
- `0` = Kaggle competition data
- `1` = Original/external dataset

**Important**: Flag is added to BOTH train and test datasets to maintain feature consistency. Test always gets flag=0 (Kaggle).

**Use cases**:
- Mixing synthetic and real data
- Adversarial validation on competition vs external data
- Debugging distribution shifts
- Models that might learn source patterns

## Column Mapping

When Kaggle and original datasets use different column names for same features, use `column_mapping` to align them.

**Validation**:
- All keys in `column_mapping` must exist in original dataset (error if not found)
- Mapping applied to original dataset BEFORE merging
- After mapping, standard column alignment (align/union) proceeds

**Example**:
```yaml
column_mapping:
  "original_age": "age"           # Rename original_age → age
  "orig_target": "diagnosed_diabetes"  # Rename orig_target → target
```

## Output

### Returned DataFrames
- **train_merged**: Kaggle train + original dataset (rows concatenated)
- **val_df**: Unchanged (pass-through)
- **test_df**: With `source_flag` column added if enabled (for feature consistency)

### State Dictionary
```python
{
    "column_alignment": {
        "mode": "union",
        "matched_columns": ["age", "income", ...],
        "kaggle_only_columns": ["id"],
        "original_only_columns": ["education", "zip_code", ...],
        "num_matched": 25,
        "num_kaggle_only": 1,
        "num_original_only": 6,
        "final_columns_train": ["age", "income", ..., "target"],
        "final_columns_test": ["age", "income", ...]  # excludes target
    },
    "merge_statistics": {
        "kaggle_rows": 700000,
        "original_rows": 100000,
        "merged_rows": 800000,
        "kaggle_columns": 26,
        "original_columns": 31,
        "final_columns": 32
    },
    "source_flag": "is_kaggle",  # or null
    "column_mapping": {"orig_col": "kaggle_col"}  # or {}
}
```

## Examples

### Basic Merge (Align Mode, No Mapping)
Simplest case: merge external dataset with column intersection.

```yaml
merge_basic:
  module: dataset_merger
  cache: true
  config:
    orig_path: data/original_dataset.csv
    mode: align
    source_flag: null
    column_mapping: {}
```

**Result**: Only common columns kept, merged train returned.

### Union Mode with Source Tracking
Keep all columns, track which rows are Kaggle vs original.

```yaml
merge_union_tracked:
  module: dataset_merger
  cache: true
  config:
    orig_path: data/diabetes_dataset.csv
    mode: union
    source_flag: is_kaggle
    column_mapping: {}
```

**Result**:
- All columns preserved (with NA fills)
- `is_kaggle` column: 0=Kaggle, 1=original
- Test also gets `is_kaggle=0` for feature consistency

### Column Name Mapping
Original and Kaggle use different column names.

```yaml
merge_with_mapping:
  module: dataset_merger
  cache: true
  config:
    orig_path: data/medical_records.csv
    mode: align
    source_flag: null
    column_mapping:
      patient_age: age
      blood_sugar_level: glucose
      body_mass_idx: bmi
      hypertension_flag: high_blood_pressure
```

**Result**: Original columns renamed before merge, then intersection kept.

### Full-Featured: Union + Source + Mapping
Combine all features for maximum flexibility.

```yaml
merge_full_featured:
  module: dataset_merger
  cache: true
  config:
    orig_path: data/external_diabetes.csv
    mode: union
    source_flag: data_source
    column_mapping:
      patient_id: id
      diagnosed: diagnosed_diabetes
      hba1c_level: hba1c
```

**Result**:
- Columns mapped first
- All columns kept (union)
- Source tracked in `data_source` column
- Test gets `data_source=0`

### Preprocessing Chain: Merge → Impute → Scale
Typical pipeline when original has missing values.

```yaml
# Template file: templates/preprocess/merge_impute_scale.yaml
chain:
  - dataset_merger
  - imputer
  - scaler

# Individual configs
dataset_merger:
  module: dataset_merger
  cache: true
  config:
    orig_path: data/original_dataset.csv
    mode: union
    source_flag: null
    column_mapping: {}

imputer:
  module: imputer
  cache: true
  config:
    numeric_strategy: median
    categorical_strategy: most_frequent

scaler:
  module: scaler
  cache: true
  config:
    numeric_strategy: standard
```

## Common Patterns

### Pattern 1: Synthetic Data Augmentation
Merge Kaggle train with synthetically generated data.

```yaml
merge_synthetic:
  module: dataset_merger
  cache: true
  config:
    orig_path: data/synthetic_train.csv
    mode: union
    source_flag: is_synthetic  # Track for debugging
    column_mapping: {}
```

### Pattern 2: Historical Data Integration
Add historical data to competition dataset.

```yaml
merge_historical:
  module: dataset_merger
  cache: true
  config:
    orig_path: data/historical_diabetes_2020.csv
    mode: align  # Only common features
    source_flag: null  # Don't need tracking
    column_mapping:
      year_2020_age: age
      glucose_2020: glucose
```

### Pattern 3: Public Dataset Enrichment
Merge with public medical dataset.

```yaml
merge_public_data:
  module: dataset_merger
  cache: true
  config:
    orig_path: data/nhanes_diabetes.csv
    mode: union  # Keep all NHANES features
    source_flag: data_source
    column_mapping:
      RIDAGEYR: age
      LBXGLU: glucose
      BMXBMI: bmi
```

## Validation & Error Handling

### Required Validations
- `orig_path` must be provided (error if missing)
- Original dataset file must exist (error if not found)
- Target column must exist in Kaggle train (error if missing)
- All `column_mapping` keys must exist in original dataset (error if not found)
- Test dataset must NOT have target column (error if present)

### Warnings
- If no columns match between datasets (after mapping), warning issued
- If original dataset has target column, it's automatically included in merge

## Technical Notes

### Project Root Resolution
Path resolution handles preprocessing chains (5 levels deep):
```
artifacts/preprocess → artifacts → step → chain → experiments → project_root
```

This ensures `orig_path: data/file.csv` resolves correctly to `{project_root}/data/file.csv`.

### Target Column Handling
- Target column automatically detected from Kaggle train
- Included in merged train dataset
- **Excluded** from test dataset (standard preprocessing behavior)
- If original has target with same name, both targets are concatenated

### Memory Considerations
- Large original datasets loaded into memory completely
- Consider dataset size when using `mode: union` (more columns = more memory)
- Caching enabled by default to avoid re-merging on subsequent runs

### Cache Behavior
When `cache: true`:
- Module skips re-execution if inputs unchanged
- Original dataset file changes detected via file hash
- Config changes force re-execution

## Troubleshooting

### FileNotFoundError: Original dataset not found
**Cause**: `orig_path` doesn't exist or is incorrect
**Solution**: Verify path is relative to project root: `{project}/data/file.csv`

### ValueError: Column mapping references non-existent columns
**Cause**: `column_mapping` keys don't exist in original dataset
**Solution**: Check original column names, update mapping

### KeyError: Target column not in Kaggle train
**Cause**: Target column name mismatch or missing
**Solution**: Verify `code/utils/config.py` has correct `TARGET_COLUMN`

### Empty DataFrame after merge
**Cause**: No matching columns between datasets in `align` mode
**Solution**: Use `mode: union` or fix `column_mapping`

### Test dataset has target column error
**Cause**: Original dataset or test setup issue
**Solution**: Check that test.csv doesn't have target (standard Kaggle format)

## Migration Guide

### From `dataset_merger` to `external_dataset`

**Old Template** (dataset_merger):
```yaml
module: dataset_merger
config:
  orig_path: data/diabetes.csv
  mode: align
```
**Result**: Train = Kaggle + External (merged)

**New Template** (external_dataset):
```yaml
module: external_dataset
config:
  orig_path: data/diabetes.csv
  mode: align
```
**Result**: Train = Kaggle, Orig = External (separate files)

**Model Code Update**:
```python
# OLD: Model receives merged data automatically
def train(train_df, val_df, config, artifacts=None):
    predictor.fit(train_df)  # Already contains orig

# NEW: Model decides if/how to merge
def train(train_df, val_df, config, artifacts=None):
    if artifacts and 'orig_df' in artifacts:
        orig_df = artifacts['orig_df']
        train_df = pd.concat([train_df, orig_df])  # Manual merge
    predictor.fit(train_df)
```

## See Also

- **[external_dataset.md](./external_dataset.md)** - ⭐ Recommended replacement (flexible merge strategy)
- [imputer.md](imputer.md) - Handle missing values from union mode
- [scaler.md](scaler.md) - Scale merged features
- [feature_selector.md](feature_selector.md) - Select relevant features after merge
- [sub-preproc.md](sub-preproc.md) - Preprocessing chain documentation
