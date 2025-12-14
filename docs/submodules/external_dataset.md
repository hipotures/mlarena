# External Dataset Loader

**Module**: `external_dataset`
**Type**: Preprocessing sub-module
**Status**: ✅ Active (replaces deprecated `dataset_merger`)

## Overview

The `external_dataset` module loads and aligns external/original datasets with Kaggle competition data, providing them as **separate datasets** rather than merging them during preprocessing.

This allows models to decide whether and how to merge the external data with training data, providing maximum flexibility.

### Key Difference from `dataset_merger`

| Feature | `dataset_merger` (deprecated) | `external_dataset` (new) |
|---------|------------------------------|--------------------------|
| **Merging strategy** | Concatenates train+orig during preprocessing | Provides orig as separate file |
| **Return value** | 4-tuple (train_merged, val, test, state) | 5-tuple (train, val, test, **orig**, state) |
| **Flexibility** | Model receives merged data (no choice) | Model decides if/how to merge |
| **Use case** | Always merge external data | Optional merge, AV can ignore orig |

## Configuration

### Template Format

```yaml
# config/templates/preprocess/external_dataset.yaml
module: external_dataset
cache: true
config:
  orig_path: data/original_dataset.csv     # Required: path to external CSV
  mode: align                               # Optional: 'align' (default) or 'union'
  source_flag: null                         # Optional: column name for source tracking
  column_mapping: {}                        # Optional: rename original columns
```

### Parameters

#### `orig_path` (required)
Path to external dataset CSV file.

- **Relative paths**: resolved from project root (e.g., `data/diabetes_original.csv`)
- **Absolute paths**: used as-is

#### `mode` (optional, default: `align`)
Column alignment strategy when merging datasets.

- **`align`** (intersection): Keep only common columns between Kaggle and external datasets
  - Safe default - ensures feature compatibility
  - Drops columns unique to either dataset

- **`union`** (all columns): Keep all columns from both datasets
  - Fills missing columns with `NA`
  - Use when you want to preserve all features

#### `source_flag` (optional, default: `null`)
Column name to track data source (0=Kaggle, 1=External).

- **`null`**: No source tracking
- **String**: Adds column to train/test/orig with source indicator
  - Useful for models to differentiate data sources
  - Example: `"is_kaggle"`, `"data_source"`

#### `column_mapping` (optional, default: `{}`)
Dictionary mapping original column names to Kaggle column names.

```yaml
column_mapping:
  original_name: kaggle_name
  Age: age
  Blood_Pressure: bp
```

## Output

### Files Created

```
experiments/pre-{template}/
└── artifacts/
    └── preprocess/
        ├── train_processed.csv      # Kaggle train (unchanged)
        ├── test_processed.csv       # Kaggle test (with source_flag if enabled)
        └── orig_processed.csv       # External dataset (aligned & mapped)
```

### State Dictionary

```python
{
    "version": "1.0",
    "module": "external_dataset",
    "orig_path": "/absolute/path/to/orig.csv",
    "orig_rows": 1000,
    "alignment": {
        "mode": "align",
        "matched_columns": 25,
        "kaggle_only_columns": ["id"],
        "external_only_columns": ["extra_col"],
    },
    "source_flag": "is_kaggle",  # or null
    "column_mapping_applied": True,
    "match_rate": 0.96,
}
```

## Usage Examples

### Example 1: Basic Usage (Column Alignment)

```yaml
# config/templates/preprocess/load_diabetes.yaml
module: external_dataset
config:
  orig_path: data/diabetes_original.csv
  mode: align  # Keep only common columns
```

**Result**:
- Kaggle train: 70,000 rows × 15 columns
- External orig: 768 rows × 15 columns (aligned)
- Test: as-is

### Example 2: With Column Mapping

```yaml
# External dataset has different column names
module: external_dataset
config:
  orig_path: data/public_diabetes.csv
  mode: align
  column_mapping:
    Glucose_Level: Glucose
    BP: BloodPressure
    Skin_Thickness: SkinThickness
```

**Result**:
- Original columns renamed to match Kaggle format
- Then aligned (common columns only)

### Example 3: With Source Tracking

```yaml
module: external_dataset
config:
  orig_path: data/diabetes_original.csv
  mode: union  # Keep all columns
  source_flag: data_source
```

**Result**:
- Train: `data_source = 0` (Kaggle)
- Orig: `data_source = 1` (External)
- Test: `data_source = 0` (Kaggle)
- Models can use `data_source` as a feature or for sample weighting

### Example 4: In Preprocessing Chain

```yaml
# config/templates/preprocess/full_pipeline.yaml
chain:
  - external_dataset       # Load orig dataset
  - scaler                 # Scale train, test, AND orig
  - feature_selector       # Select features on all 3 datasets
```

**Flow**:
1. `external_dataset`: Provides orig as separate file
2. `scaler`: Fits on train, transforms train/test/orig
3. `feature_selector`: Applies same feature selection to all datasets

## Model Integration

Models receive `orig_df` via the `artifacts` parameter.

### Example Model: Merge Train + Orig

```python
# config/code/models/autogluon_baseline.py
def train(train_df, val_df, config, artifacts=None):
    # Check for orig_df in artifacts
    if artifacts and 'orig_df' in artifacts:
        orig_df = artifacts['orig_df']

        # Merge train + orig
        train_df = pd.concat([train_df, orig_df], ignore_index=True)
        print(f"Merged {len(orig_df)} external rows into training")

    # Train model on merged data
    predictor = TabularPredictor(label=target).fit(train_df)
    return predictor, {"used_orig": orig_df is not None}
```

### Example Model: Ignore Orig (Adversarial Validation)

```python
# AV models train on Kaggle data only
def train(train_df, val_df, config, artifacts=None):
    # AV trains binary classifier: train vs test
    # Orig is passed through but not used
    av_classifier = train_av_model(train_df, test_df)

    # artifacts['orig_df'] exists but is ignored
    return av_classifier, {"used_orig": False}
```

## Comparison: align vs union Mode

### Mode: `align` (Intersection)

**Example**:
- Kaggle columns: `[id, age, glucose, bp, bmi, target]`
- External columns: `[age, glucose, bp, bmi, cholesterol]`
- **Result**: `[age, glucose, bp, bmi]` (5 → 4 columns)

**Dropped**:
- Kaggle-only: `id`, `target` (target preserved automatically)
- External-only: `cholesterol`

### Mode: `union` (All Columns)

**Example**:
- Kaggle columns: `[id, age, glucose, bp, bmi, target]`
- External columns: `[age, glucose, bp, bmi, cholesterol]`
- **Result**: `[id, age, glucose, bp, bmi, cholesterol, target]` (6 columns)

**Filled with NA**:
- External dataset: `id=NA`, `target=NA`
- Kaggle train: `cholesterol=NA`

## Troubleshooting

### Low Match Rate Warning

```
⚠ Low column match rate (35.2%). Consider using 'column_mapping' to align column names.
```

**Solution**: Use `column_mapping` to rename external columns.

```yaml
column_mapping:
  Original_Name: Kaggle_Name
  # Map all mismatched columns
```

### Missing Columns Error (Mode: align)

```
ValueError: No common columns found between Kaggle and external datasets.
```

**Solution**: Either:
1. Add `column_mapping` to align names
2. Change to `mode: union` to keep all columns

### File Not Found

```
FileNotFoundError: External dataset not found: data/diabetes.csv
```

**Solution**: Verify path is correct relative to project root.

```bash
ls -la projects/kaggle/{competition}/data/
```

## Migration from `dataset_merger`

**Old** (`dataset_merger`):
```yaml
module: dataset_merger
config:
  orig_path: data/diabetes.csv
  mode: align
```

**Result**: Train = Kaggle + External (merged during preprocessing)

---

**New** (`external_dataset`):
```yaml
module: external_dataset
config:
  orig_path: data/diabetes.csv
  mode: align
```

**Result**: Train = Kaggle, Orig = External (separate files)

**Model Update Required**:
```python
# OLD: Model receives merged data automatically
def train(train_df, val_df, config, artifacts=None):
    # train_df already contains orig data
    predictor.fit(train_df)

# NEW: Model decides if/how to merge
def train(train_df, val_df, config, artifacts=None):
    if artifacts and 'orig_df' in artifacts:
        orig_df = artifacts['orig_df']
        train_df = pd.concat([train_df, orig_df])  # Manual merge
    predictor.fit(train_df)
```

## Best Practices

1. **Use `align` mode by default** - Safer, ensures feature compatibility
2. **Add `source_flag` for debugging** - Helps identify data source in model
3. **Use `column_mapping` liberally** - Align names upfront to avoid issues
4. **Chain with other preprocessing** - External dataset flows through entire pipeline
5. **Models opt-in to merging** - Check `artifacts['orig_df']` before merging

## See Also

- [dataset_merger (deprecated)](./dataset_merger.md) - Old merge-during-preprocessing approach
- [Preprocessing Chain System](../MLA_WORKFLOW_GUIDE.md#preprocessing-chains)
- [Model Integration](../configs.md#model-templates)
