# MLArena Preprocessing Sub-Modules - Developer Guide

## Overview

This directory contains utilities for building universal preprocessing sub-modules.
Each sub-module is a standalone preprocessing template that can be chained with others.

## Architecture

- **Template-level chaining**: Each sub-module is a separate template
- **One code per capability**: Sub-module behavior controlled by YAML parameters
- **Independent and composable**: Sub-modules work in any order
- **MLArena pattern**: Follows `fit_transform(train, val, test, config)` interface

## Creating a New Sub-Module

### Step 1: Copy Template

```bash
cp src/mlarena/preprocessing/TEMPLATE.py src/mlarena/defaults/preprocessing/my_submodule.py
```

### Step 2: Implement fit_transform

Follow the template structure:
1. Extract and validate config
2. Create artifact directory
3. Perform transformation
4. Save artifacts (fitted objects, reports)
   - **`train_processed.csv.gz`**: Transformed training data.
   - **`test_processed.csv.gz`**: Transformed test data.
   - **`tuning_processed.csv.gz`**: (Optional) Validation data for HPO (typically from `train_fraction`).
   - **`eval_processed.csv.gz`**: (Optional) Offline evaluation data, not used during training but shown in leaderboard (typically from `train_fraction`).
5. Return (train_df, val_df, test_df, orig_df, state_dict)

### Step 3: Add to Templates

Create a new file `src/mlarena/templates/preprocess/my_submodule.yaml` with the following content:

```yaml
my_submodule:
  module: my_submodule
  cache: true
  config:
    param1: value1
    param2: value2
```

### Step 4: Test

```bash
uv run python scripts/mla.py preprocess \
    --project my_project \
    --preprocess-template my_submodule
```

### Step 5: Create Documentation

Create `docs/submodules/{submodule_name}.md` with detailed parameter documentation:

```bash
# Create documentation file
touch docs/submodules/my_submodule.md
```

The documentation should include:
- **Overview**: What the sub-module does
- **Parameters**: All config parameters with types, defaults, and descriptions
- **Examples**: Common use cases with YAML configs
- **Artifacts**: What files/reports are generated
- **Notes**: Edge cases, recommendations, tuning tips

## Sub-Module Checklist

- [ ] Handles train, val (optional), and test DataFrames
- [ ] Validates all required config parameters
- [ ] Creates sub-module artifact directory
- [ ] Saves fitted objects (scalers, encoders) if needed
- [ ] Saves report JSON with transformation summary
- [ ] Returns complete state_dict for reproducibility
- [ ] Handles missing columns gracefully
- [ ] Works with both numeric and categorical data
- [ ] Doesn't leak information from test to train
- [ ] Logs shape changes and new columns created
- [ ] **Documentation created** in `docs/submodules/{name}.md`

## Available Sub-Modules

Detailed documentation for each sub-module:

- **[sanity_check.md](sanity_check.md)**: Basic cleaning and data type enforcement.
- **[imputer.md](imputer.md)**: Missing value imputation (numeric and categorical).
- **[rare_category_handler.md](rare_category_handler.md)**: Group rare categories to reduce cardinality.
- **[encoder.md](encoder.md)**: Categorical encoding (one-hot, target, catboost, hashing).
- **[categorical_encoder.md](categorical_encoder.md)**: Advanced pandas-native categorical conversion using EDA metadata.
- **[scaler.md](scaler.md)**: Numerical scaling and distribution transformations.
- **[drift_detector.md](drift_detector.md)**: Detection and removal of drifting features between train/test.
- **[feature_interactions.md](feature_interactions.md)**: Create simple arithmetic interaction features (add, sub, mul, div) between numeric pairs.
- **[feature_polynomial.md](feature_polynomial.md)**: Create polynomial and interaction features using sklearn.
- **[feature_group_agg.md](feature_group_agg.md)**: Create group-based aggregations (groupby + agg + merge).
- **[feature_selector.md](feature_selector.md)**: Automated feature selection using multiple methods.
- **[imbalance_handler.md](imbalance_handler.md)**: Handling class imbalance via weighting or resampling.
- **[outlier_handler.md](outlier_handler.md)**: Detection and handling of numeric outliers.
- **[datetime_handler.md](datetime_handler.md)**: Parsing and expanding datetime columns.
- **[target_transformer.md](target_transformer.md)**: Transformations for regression target columns.
- **[adversarial_validation.md](adversarial_validation.md)**: Distribution shift handling via AV weighting.
- **[external_dataset.md](external_dataset.md)**: Loading and aligning external/original datasets.
- **[train_fraction.md](train_fraction.md)**: Training data subsampling and validation/evaluation splitting.
- **[utility_modules.md](utility_modules.md)**: Minimal modules like `noop` and `identity` for testing.

## Utilities Available

### validation.py

- `validate_config(config, required, optional)` - Check required/optional params, set defaults
- `validate_column_exists(df, columns, context)` - Verify columns present in DataFrame
- `infer_column_types(df)` - Auto-detect numeric/categorical/datetime/boolean columns
- `validate_choice(value, choices, param_name)` - Validate value is in allowed list
- `validate_numeric_range(value, min, max, param_name)` - Validate numeric parameter

### artifacts.py

- `save_fitted_object(obj, artifact_dir, name)` - Pickle and save sklearn transformers
- `load_fitted_object(artifact_dir, name)` - Load pickled objects
- `save_report(data, artifact_dir, name)` - Save JSON reports
- `load_report(artifact_dir, name)` - Load JSON reports
- `get_submodule_artifact_dir(artifact_dir, submodule_name)` - Get sub-module artifact path

### dataframe_utils.py

- `get_numeric_columns(df, exclude)` - List numeric columns
- `get_categorical_columns(df, exclude)` - List categorical columns
- `get_datetime_columns(df, exclude)` - List datetime columns
- `get_boolean_columns(df, exclude)` - List boolean columns
- `safe_drop_columns(df, columns)` - Drop existing columns only
- `align_columns(train_df, test_df, fill_value)` - Sync train/test columns
- `get_constant_columns(df, threshold)` - Find columns with constant values
- `get_high_missing_columns(df, threshold)` - Find columns with high missing rates
- `copy_dataframe(df)` - Create deep copy

### report.py

- `log_transformation_summary(before_df, after_df, submodule_name)` - Shape/column changes
- `create_preprocessing_report(train_before, train_after, test_before, test_after, config)` - Standard report format
- `create_column_stats_report(df, columns)` - Statistics per column
- `create_missing_values_report(df)` - Missing values analysis
- `create_data_quality_report(df)` - Comprehensive quality metrics

## Best Practices

1. **Always validate config first** - Fail fast with clear errors
2. **Save artifacts to sub-module directory** - Keep organized
3. **Log all transformations** - Shape changes, new columns
4. **Handle edge cases** - Empty DataFrames, all-NA columns
5. **Use type hints** - Makes code self-documenting
6. **Write docstrings** - Explain config parameters
7. **Test independently** - Each sub-module should work alone
8. **Test in chains** - Verify compatibility with others

## Common Patterns

### Pattern 1: Column Selection

```python
from mlarena.preprocessing.utils import dataframe_utils, validation

# Get dataset config
dataset_config = config.get("_dataset", {})
id_column = dataset_config.get("id_column", "id")
target_column = dataset_config.get("target")
ignored_columns = dataset_config.get("ignored_columns", [])

# Exclude system columns
exclude_cols = [id_column, target_column] + ignored_columns

# Get column types
numeric_cols = dataframe_utils.get_numeric_columns(train_df, exclude=exclude_cols)
categorical_cols = dataframe_utils.get_categorical_columns(train_df, exclude=exclude_cols)
```

### Pattern 2: Config Validation

```python
from mlarena.preprocessing.utils import validation

# Define required and optional parameters
required_params = ["method", "threshold"]
optional_params = {
    "n_features": 100,
    "random_state": 42,
    "verbose": False,
}

# Validate and set defaults
validation.validate_config(config, required_params, optional_params)

# Validate choice parameter
validation.validate_choice(
    config["method"],
    ["variance", "mi", "correlation"],
    "method"
)

# Validate numeric range
validation.validate_numeric_range(
    config["threshold"],
    min_value=0.0,
    max_value=1.0,
    param_name="threshold"
)
```

### Pattern 3: Fitted Object Management

```python
from sklearn.preprocessing import StandardScaler
from mlarena.preprocessing.utils import artifacts

# Create sub-module artifact directory
submodule_dir = artifacts.get_submodule_artifact_dir(artifact_dir, "scaler")

# Fit
scaler = StandardScaler()
scaler.fit(train_df[numeric_cols])

# Save
scaler_path = artifacts.save_fitted_object(scaler, submodule_dir, "scaler.pkl")

# Transform
train_df[numeric_cols] = scaler.transform(train_df[numeric_cols])
test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])
if val_df is not None:
    val_df[numeric_cols] = scaler.transform(val_df[numeric_cols])

# Save relative path in state
state_dict["scaler_path"] = str(scaler_path.relative_to(artifact_dir))
```

### Pattern 4: Report Generation

```python
from mlarena.preprocessing.utils import report, artifacts, dataframe_utils

# Save copies for reporting
train_df_original = dataframe_utils.copy_dataframe(train_df)
test_df_original = dataframe_utils.copy_dataframe(test_df)

# ... perform transformations ...

# Generate report
summary = report.create_preprocessing_report(
    train_before=train_df_original,
    train_after=train_df,
    test_before=test_df_original,
    test_after=test_df,
    config=config,
)

# Save report
artifacts.save_report(summary, submodule_dir, "summary.json")
```

### Pattern 5: Handling Missing Columns Gracefully

```python
from mlarena.preprocessing.utils import dataframe_utils

# Only transform columns that exist in both train and test
common_numeric = set(dataframe_utils.get_numeric_columns(train_df)) & \
                 set(dataframe_utils.get_numeric_columns(test_df))

# Or align columns before transformation
train_df, test_df = dataframe_utils.align_columns(train_df, test_df, fill_value=0)
```

## Chaining Sub-Modules

Sub-modules are chained at the template level using meta-templates:

```yaml
# src/mlarena/templates/preprocess.yaml
templates:
  # Individual sub-modules
  sanity_check:
    module: sanity_check
    cache: true
    config:
      drop_duplicates: true

  imputer:
    module: imputer
    cache: true
    config:
      numeric_strategy: "mean"

  # Meta-template (chain)
  my_pipeline:
    chain: [sanity_check, imputer, scaler, encoder]
```

Run with:
```bash
uv run python scripts/mla.py preprocess \
    --project my_project \
    --preprocess-template my_pipeline
```

## Troubleshooting

### "Module not found"

Check that:
- File is in `src/mlarena/defaults/preprocessing/` or `{project}/code/preprocessing/`
- Filename matches template's `module:` field
- Function `fit_transform` is defined

### "Config validation failed"

Check that template has all required parameters with correct types.

### "Artifact directory not found"

Ensure you're using `config.get("_artifact_dir")` correctly.

### "Column not found in DataFrame"

Use `validation.validate_column_exists()` to check columns before accessing them.

### "Import error: cannot import preprocessing.utils"

Make sure you're importing from the correct path:
```python
from mlarena.preprocessing.utils import validation, artifacts, dataframe_utils, report
```

## Sub-Module Template Locations

- **Global sub-modules**: `src/mlarena/defaults/preprocessing/`
- **Project-local sub-modules**: `projects/kaggle/{competition}/code/preprocessing/`
- **Template file**: `src/mlarena/preprocessing/TEMPLATE.py`

## Artifact Structure

### Chain-Based Directory Structure

Each preprocessing chain (single template, meta-template, or CLI chain) gets its own experiment directory:

```
experiments/
  # Single template
  pre-{template}/
    0-{template}/
      artifacts/preprocess/
        ├── train_processed.csv
        ├── test_processed.csv
        └── submodules/
            └── {submodule_name}/
                ├── summary.json
                ├── {fitted_object}.pkl
                └── {custom_artifacts}
      state.json

  # Meta-template: chain: [sanity_check, imputer]
  pre-my_pipeline/
    0-sanity_check/
      artifacts/preprocess/
        ├── train_processed.csv
        ├── test_processed.csv
        └── submodules/...
      state.json
    1-imputer/
      artifacts/preprocess/
        ├── train_processed.csv  ← Input from 0-sanity_check
        ├── test_processed.csv
        └── submodules/...
      state.json

  # CLI chain: --preprocess-template noop,imputer,scaler
  pre-chain-a1b2c3d4/  ← Hash of template list
    0-noop/
      artifacts/...
    1-imputer/
      artifacts/...
    2-scaler/
      artifacts/...

## Chain State Format

**For detailed payload format documentation, see:** [State Payload Formats](../state_payload_formats.md)

When preprocessing chains execute, each step creates its own state entry:

```json
{
  "experiment_id": "pre-my_pipeline/abc123def/1-imputer",
  "modules": {
    "preprocess": {  // Note: Still uses "preprocess" as module name
      "status": "completed",
      "payload": {...}
    }
  }
}
```

For the final chain output, query the last step's state.json.

**Key Points:**
- **Chain isolation**: Each unique chain gets separate directory
- **Sub-module indexing**: `{idx}-{template}` prevents duplicates (e.g., chain with same template twice)
- **Data flow**: Sub-module N loads output from sub-module N-1 within same chain
- **Cache validation**: Chain hash ensures different template orders don't share cache

## Next Steps

1. Implement your sub-module using the TEMPLATE.py
2. Add it to `src/mlarena/templates/preprocess.yaml`
3. Test it standalone
4. Test it in chains with other sub-modules
5. Verify artifacts are saved correctly
6. Check state.json for reproducibility
