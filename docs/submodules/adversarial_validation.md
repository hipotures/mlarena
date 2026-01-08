# Adversarial Validation Preprocessing Module

> **⚠️ IMPORTANT: Sample Weight Limitations**
>
> **Custom models (with `model:` in template) do NOT receive sample weights automatically!** Only the fallback inline AutoGluon gets weights automatically. For custom models, you MUST load weights manually from preprocessing state. See [Integration with Models](#integration-with-models) for the required pattern.

## Overview

The Adversarial Validation (AV) preprocessing module addresses **distribution shift** between training and test sets by:

1. Training a binary classifier to distinguish train samples from test samples
2. Using the classifier's predictions to generate sample weights
3. Assigning higher weights to training samples that resemble the test distribution

This helps models focus on learning patterns that generalize to the test set, improving leaderboard scores when train/test distributions differ.

### When to Use

- **Strong distribution shift**: Train and test sets come from different time periods, sources, or populations
- **Leaderboard discrepancy**: Local CV doesn't match public leaderboard scores
- **Known data collection differences**: Test set has different preprocessing, sampling, or quality

### When NOT to Use

- **Identical distributions**: Train and test are randomly split from same source
- **Small datasets**: AV requires enough samples to train reliable classifier
- **Post-resampling**: Never use AV after SMOTE or other resampling (breaks row alignment)

## Capabilities

- **Reuses MLArena Model Infrastructure**: Trains AV classifier using existing AutoGluon models
- **Configurable Transformations**: Four weight transformation methods with different characteristics
- **Clean Integration**: Works seamlessly in preprocessing chains
- **State Persistence**: Saves weights path in state.json for downstream models
- **Comprehensive Outputs**: Weights CSV, raw probabilities, AV statistics, model artifacts

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `presets` | str | `"medium_quality_faster_train"` | AutoGluon preset for AV classifier |
| `time_limit` | int | `600` | Training time limit in seconds |
| `included_model_types` | list[str] \| null | `null` | Limit model types (e.g., `["GBM", "CAT", "XGB"]`) |
| `drop_columns` | list[str] | `[]` | Additional columns to drop beyond ID/target |
| `drop_prefixes` | list[str] | `[]` | Drop columns whose names start with these prefixes (e.g., `["mean_"]`) |
| `weight_transform` | str | `"odds_ratio_normalized"` | Weight transformation method (see below) |
| `weights_output_name` | str | `"sample_weights.csv"` | Output CSV filename |
| `weight_column_name` | str | `"__sample_weight__"` | Column header in weights CSV |

## Weight Transformation Methods

### 1. `raw`

**Formula**: Returns raw AV probabilities

**Range**: [0, 1]

**Use When**: You want to apply custom transformations downstream

**Example**:
```python
# P(sample is from test) = 0.234
weight = 0.234
```

### 2. `odds_ratio`

**Formula**: `p / (1 - p)`

**Range**: [0, ∞)

**Use When**: Standard odds ratio weighting without caps

**Example**:
```python
# P(is_test) = 0.75
weight = 0.75 / (1 - 0.75) = 3.0
```

### 3. `odds_ratio_capped`

**Formula**: `clip(p / (1 - p), upper=2.0)`

**Range**: [0, 2.0]

**Use When**: Preventing extreme weights from dominating training

**Example**:
```python
# P(is_test) = 0.90
weight = min(0.90 / 0.10, 2.0) = 2.0  # Capped
```

### 4. `odds_ratio_normalized` (RECOMMENDED)

**Formula**: `clip(p / (1 - p), upper=2.0) / mean`

**Range**: Centered around 1.0

**Use When**: Standard practice for most use cases

**Characteristics**:
- Balances weights (mean = 1.0)
- Prevents extreme outliers
- Works well with AutoGluon's default behavior

**Example**:
```python
# Raw odds: [3.0, 0.5, 2.0, 1.0]
# Mean: 1.625
# Normalized: [1.85, 0.31, 1.23, 0.62]
```

## Examples

### Example 0: Using Same Model With/Without AV Weights

**Scenario**: You have a custom model that optionally supports AV weights. You want to test both configurations.

**Without AV weights** (baseline):
```bash
# Use preprocessing WITHOUT AV module
uv run python scripts/mla.py --project playground-series-s5e12 \
    --preprocess-template basic_preprocessing \
    --model-template my_model
```

Output:
```
ℹ No AV weights in preprocessing state (training without weights)
Training model...
Local CV: 0.845
```

**With AV weights**:
```bash
# Use preprocessing WITH AV module
uv run python scripts/mla.py --project playground-series-s5e12 \
    --preprocess-template av_with_external \
    --model-template my_model
```

Output:
```
✓ Loaded AV weights: mean=1.000, std=0.559
Training model...
Local CV: 0.851  # Potentially improved with weights
```

**Key Insight**: Same model code works in BOTH scenarios. The model automatically detects and uses weights when available, trains normally when not.

### Example 1: Basic Standalone Usage

```yaml
# src/mlarena/templates/preprocess/av_basic.yaml
module: adversarial_validation
cache: true
config:
  presets: medium_quality_faster_train
  time_limit: 600
  weight_transform: odds_ratio_normalized
```

**Command**:
```bash
uv run python scripts/mla.py --project playground-series-s5e12 \
    --preprocess-template av_basic \
    --model-template baseline
```

### Example 2: With External Dataset Chain (No Preprocess Merge)

```yaml
# projects/kaggle/playground-series-s5e12/templates/preprocess/av_with_external.yaml
chain:
  - test_external_diabetes
  - adversarial_validation_step  # MUST be LAST

adversarial_validation_step:
  module: adversarial_validation
  cache: true
  config:
    presets: best_quality
    time_limit: 3600
    weight_transform: odds_ratio_normalized
    drop_prefixes: ["mean_"]  # e.g., exclude target-encoded features
```

**WARNING**: Do NOT add resampling modules after AV - weights will mismatch!

### Example 3: Fast Development Iteration

```yaml
# src/mlarena/templates/preprocess/av_smoke.yaml
module: adversarial_validation
cache: true
config:
  presets: medium_quality_faster_train
  time_limit: 60  # 1 minute for smoke test
  included_model_types: ["GBM"]  # Only GBM for speed
  weight_transform: odds_ratio_capped
```

### Example 4: Production Quality AV

```yaml
# src/mlarena/templates/preprocess/av_production.yaml
module: adversarial_validation
cache: true
config:
  presets: best_quality
  time_limit: 7200  # 2 hours
  included_model_types: ["GBM", "CAT", "XGB", "NN_TORCH"]
  weight_transform: odds_ratio_normalized
```

### Example 5: Different Weight Transformations

```yaml
# src/mlarena/templates/preprocess/av_raw_weights.yaml
module: adversarial_validation
cache: true
config:
  presets: medium_quality_faster_train
  time_limit: 600
  weight_transform: raw  # Get raw probabilities
  weights_output_name: av_raw_weights.csv
```

### Example 6: Custom Column Dropping

```yaml
# src/mlarena/templates/preprocess/av_custom_drops.yaml
module: adversarial_validation
cache: true
config:
  presets: medium_quality_faster_train
  time_limit: 600
  drop_columns:
    - temporal_feature_1  # Drop time-based features
    - temporal_feature_2
  weight_transform: odds_ratio_normalized
```

## Integration with Models

---

### ⚠️ CRITICAL LIMITATION: Custom Models Do NOT Get Weights Automatically

**MLArena's architecture does NOT pass sample weights to custom models!**

The model module (`src/mlarena/modules/model.py`) has two execution paths:

1. **Fallback Inline AutoGluon (when `model:` not specified in template)**
   - ✅ **AUTOMATICALLY gets weights** from preprocessing state
   - Weights are loaded and added to `train_df` as `__sample_weight__` column
   - No code changes needed

2. **Custom Models (when `model: path/to/model.py` in template)**
   - ❌ **DOES NOT get weights automatically**
   - Model's `train()` function receives only: `(train_df, val_df, config, artifacts)`
   - **YOU MUST load weights manually** (see pattern below)

This is a fundamental limitation of the current MLArena architecture where custom models use dynamic loading that doesn't include sample_weight in the interface.

---

### Loading Weights in Custom Models

```python
# projects/kaggle/{comp}/code/models/my_custom_model.py

import json
from pathlib import Path
import pandas as pd
from autogluon.tabular import TabularPredictor


def train(train_df, val_df, config, artifacts):
    """
    Custom model with optional AV weights support.

    Handles two cases:
    1. Preprocessing WITH AV module → loads weights from state.json
    2. Preprocessing WITHOUT AV module → trains without weights
    """

    # 1. Try to load weights from preprocessing state (OPTIONAL)
    sample_weight = None
    preprocess_state_path = config.system.experiment_dir.parent / "state.json"

    # Only attempt to load weights if preprocessing state exists
    if preprocess_state_path.exists():
        try:
            with open(preprocess_state_path) as f:
                state = json.load(f)

            # Navigate to custom_module_state (where AV module stores weights_path)
            custom_state = state.get("modules", {}).get("preprocess", {}).get("payload", {}).get("custom_module_state", {})
            weights_path_str = custom_state.get("weights_path")

            # If weights_path exists in state, try to load it
            if weights_path_str:
                weights_path = Path(weights_path_str)

                # Handle relative paths
                if not weights_path.is_absolute():
                    weights_path = config.system.project_root / weights_path

                # Load weights if file exists
                if weights_path.exists():
                    weights_df = pd.read_csv(weights_path)
                    sample_weight = weights_df.iloc[:, 0]  # First column (MLArena format)
                    print(f"✓ Loaded AV weights: mean={sample_weight.mean():.3f}, std={sample_weight.std():.3f}")
                else:
                    print(f"⚠ Weights path in state but file not found: {weights_path}")
            else:
                print("ℹ No AV weights in preprocessing state (training without weights)")

        except Exception as e:
            print(f"⚠ Failed to load weights: {e}")
            print("  Continuing without sample weights...")
            sample_weight = None
    else:
        print("ℹ No preprocessing state found (training without weights)")

    # 2. Prepare training data (with or without weights)
    train_data = train_df.copy()

    if sample_weight is not None:
        # Validate weight count matches training data
        if len(sample_weight) != len(train_data):
            print(f"✗ ERROR: Weight count ({len(sample_weight)}) != train count ({len(train_data)})")
            print("  Training WITHOUT weights to avoid errors")
            sample_weight = None
        else:
            train_data["__sample_weight__"] = sample_weight

    # 3. Create TabularPredictor with optional sample_weight
    predictor = TabularPredictor(
        label=config.dataset.target,
        path=str(artifacts["model_dir"]),
        sample_weight="__sample_weight__" if sample_weight is not None else None,
    )

    # 4. Train model
    predictor.fit(
        train_data,
        presets=config.config.get("preset", "medium_quality"),
        time_limit=config.config.get("time_limit", 600),
    )

    # 5. Extract results
    leaderboard = predictor.leaderboard(silent=True)
    best_score = None
    if not leaderboard.empty and "score_val" in leaderboard.columns:
        best_score = float(leaderboard["score_val"].max())

    return {
        "local_cv_score": best_score,
        "model_path": str(artifacts["model_dir"]),
        "used_sample_weights": sample_weight is not None,  # Track if weights were used
    }
```

**Key Points:**

1. **AV module is OPTIONAL** - code must handle both cases:
   - ✅ Preprocessing WITH AV → `custom_module_state.weights_path` exists
   - ✅ Preprocessing WITHOUT AV → `custom_module_state` is empty or missing

2. **Defensive loading**:
   - Check if `preprocess_state_path` exists
   - Check if `weights_path` exists in state
   - Check if weights file exists on disk
   - Validate weight count matches training data
   - Use try-except to handle errors gracefully

3. **Clear feedback**:
   - Print messages showing whether weights were loaded
   - Return `used_sample_weights` flag in training summary

4. **Fallback to no weights**:
   - If ANY step fails, set `sample_weight = None`
   - Model trains successfully without weights (no crash)

### ✅ Automatic Weights (Fallback Inline AutoGluon Only)

**ONLY works when NO `model:` is specified in template!**

When using fallback inline AutoGluon baseline:
```yaml
# src/mlarena/templates/model/baseline.yaml
# NO model: key specified - uses fallback
config:
  preset: medium_quality
  time_limit: 600
```

The system automatically:
1. Loads `sample_weights.csv` from preprocessing state
2. Adds weights as `__sample_weight__` column to `train_df`
3. Passes `sample_weight="__sample_weight__"` to `TabularPredictor`

**However:** This is NOT recommended for production because:
- Fallback AutoGluon has limited configurability
- Cannot use custom model logic or ensembles
- For production, use custom models with manual weight loading

## Output

### File Structure

```
experiments/pre-adversarial_validation/
  0-adversarial_validation/
    artifacts/
      preprocess/
        sample_weights.csv              # Single-column weights CSV
        av_predictions.csv              # Raw AV probabilities (debugging)
        av_model/                       # AutoGluon AV classifier
          models/
            LightGBM/
            CatBoost/
            WeightedEnsemble/
    state.json                          # Module state with weights_path
```

### sample_weights.csv Format

**CRITICAL**: Must be single column (system uses `.iloc[:, 0]`)

```csv
__sample_weight__
0.523
1.234
0.891
0.456
...
```

**Requirements**:
- ONE column only (header name doesn't matter)
- Row count MUST match train_processed.csv
- Row order MUST match train_processed.csv
- NO ID column (index-based matching)

### av_predictions.csv Format

```csv
av_prob
0.234
0.567
0.423
0.189
...
```

### state.json Format

```json
{
  "modules": {
    "preprocess": {
      "status": "completed",
      "payload": {
        "custom_module_state": {
          "weights_path": "/path/to/sample_weights.csv",
          "weight_transform": "odds_ratio_normalized",
          "av_auc": 0.623,
          "av_rows": 1000000,
          "presets": "medium_quality_faster_train",
          "time_limit": 600,
          "weight_stats": {
            "mean": 1.0,
            "std": 0.45,
            "min": 0.02,
            "max": 2.0
          }
        }
      }
    }
  }
}
```

## Technical Notes

### Model Architecture

The AV classifier:
- Uses AutoGluon TabularPredictor with binary classification
- Trains on concatenated train+test with `__is_test__` label (0=train, 1=test)
- Returns AUC score as indicator of distribution shift strength
- **High AUC (>0.7)**: Strong shift, weights will be meaningful
- **Low AUC (<0.55)**: Weak shift, consider skipping AV

### Memory Considerations

Training AV classifier doubles memory usage:
- Concatenates train+test for training
- Stores full AutoGluon model in artifacts
- For very large datasets (>2M rows), consider:
  - Sampling for AV training only
  - Using `presets="good_quality_faster_train"`
  - Limiting `included_model_types`

### Preprocessing Chain Constraints

**CRITICAL**: AV must be LAST preprocessing step

**Why**: Weights are matched by row index, not ID
- Any resampling/shuffling after AV breaks alignment
- Row count must match between weights and final train_processed.csv

**Valid chain**:
```yaml
chain:
  - dataset_merger
  - feature_interactions
  - feature_polynomial
  - adversarial_validation  # LAST
```

**Invalid chain**:
```yaml
chain:
  - adversarial_validation
  - smote_resampling  # BREAKS row alignment!
```

## Troubleshooting

### Issue: "Weights don't seem to have any effect on my model"

**Cause**: You're using a custom model and weights are NOT being passed to it

**Diagnosis**:
1. Check your model template - does it have `model: path/to/model.py`?
2. If YES, your model is NOT getting weights automatically

**Solutions**:
1. **Option A (Recommended)**: Add weight loading code to your custom model (see [Loading Weights in Custom Models](#loading-weights-in-custom-models))
2. **Option B (Quick test)**: Remove `model:` from template to use fallback AutoGluon (gets weights automatically)

**Verification**:
```python
# Add this to your model's train() function to diagnose weight loading
def train(train_df, val_df, config, artifacts):
    # Check if weights column exists in training data
    if "__sample_weight__" in train_df.columns:
        weights = train_df["__sample_weight__"]
        print(f"✓ Sample weights detected in train_df:")
        print(f"  Count: {len(weights)}")
        print(f"  Mean: {weights.mean():.3f}")
        print(f"  Std: {weights.std():.3f}")
        print(f"  Range: [{weights.min():.3f}, {weights.max():.3f}]")
        has_weights = True
    else:
        print("ℹ No sample weights in train_df (training without AV weights)")
        has_weights = False

    # Your model training code here...
    predictor = TabularPredictor(
        label=config.dataset.target,
        sample_weight="__sample_weight__" if has_weights else None,
    )
    # ...
```

### Error: "Train/test column mismatch"

**Cause**: Train and test datasets have different features

**Solutions**:
1. Check if preprocessing created train-only features (e.g., target encoding with leakage)
2. Add problematic columns to `drop_columns`
3. Fix upstream preprocessing to ensure consistent features

### Error: "Weights row count doesn't match train row count"

**Cause**: Preprocessing chain modified train row count after AV

**Solutions**:
1. Move AV to END of preprocessing chain
2. Remove any resampling/filtering modules after AV
3. Check if validation split is altering train_df

### Warning: "AV AUC is very low (< 0.55)"

**Meaning**: Train and test distributions are very similar

**Solutions**:
1. This is actually good news - distributions match!
2. Consider skipping AV entirely (won't help much)
3. If you still want weights, use `weight_transform: raw` to see actual probabilities

### Warning: "AV AUC is very high (> 0.85)"

**Meaning**: Very strong distribution shift

**Implications**:
1. Most weights will be capped at 2.0
2. Consider using `weight_transform: odds_ratio_capped` with higher cap
3. Investigate root cause of shift (data leakage? time period?)

### Error: "AV classifier model not found"

**Cause**: `av_classifier.py` missing from `src/mlarena/defaults/models/`

**Solutions**:
1. Verify file exists at correct path
2. Check file permissions
3. Ensure not in gitignore

### Performance: AV training is very slow

**Solutions**:
1. Reduce `time_limit` (e.g., 300s instead of 600s)
2. Use faster preset: `presets: good_quality_faster_train`
3. Limit model types: `included_model_types: ["GBM", "CAT"]`
4. For smoke tests: `included_model_types: ["GBM"]` with `time_limit: 60`

### Memory: OOM during AV training

**Solutions**:
1. Reduce AutoGluon memory usage via presets
2. Sample train+test for AV training only (advanced)
3. Use fewer model types
4. Run on machine with more RAM

## Comparison with Project-Local AV

### Old Approach (Project-Local)

```python
# projects/kaggle/{comp}/code/utils/adversarial_validation.py
def compute_adversarial_weights(train_df, test_df, ...):
    # Custom implementation
    # Duplicated across projects
    # No caching or state tracking
```

### New Approach (Global Module)

```yaml
# src/mlarena/templates/preprocess/adversarial_validation.yaml
module: adversarial_validation
cache: true
config:
  presets: medium_quality_faster_train
  time_limit: 600
```

**Benefits**:
- **Reusable**: Works across all projects
- **Cached**: Preprocessing results saved and reused
- **Tracked**: State in state.json with git hash
- **Configurable**: Easy YAML-based configuration
- **Documented**: Centralized documentation

**⚠️ Important Note**:
Both old and new approaches have the **same limitation**: custom models must load weights manually. The system architecture doesn't pass sample_weight to custom model `train()` functions. Only fallback inline AutoGluon gets weights automatically.

### Backwards Compatibility

Old project-local `compute_adversarial_weights()` functions remain for backwards compatibility but are marked as deprecated. New code should use the global preprocessing module.

## See Also

- [dataset_merger.md](dataset_merger.md) - For merging external datasets before AV
- [MLA_WORKFLOW_GUIDE.md](../MLA_WORKFLOW_GUIDE.md) - Complete workflow guide
- [configs.md](../configs.md) - Template configuration reference
