# State Payload Format Variations

This document explains the different payload structures used in `state.json` files across different module types and preprocessing configurations.

---

## Overview

The `state.json` file tracks execution status and outputs for each module. The `payload` field within each module entry varies depending on:

1. **Module type** (preprocess vs model vs submit)
2. **Preprocessing mode** (single-step vs chain)
3. **Custom module state** (weights, eval data, etc.)

---

## Basic State Structure

All `state.json` files share this base structure:

```json
{
  "experiment_id": "exp-20251217-152730",
  "modules": {
    "module_name": {
      "status": "completed",
      "payload": { ... },
      "invocation": { ... },
      "error": null
    }
  },
  "git": {
    "hash": "a475f26",
    "dirty": false
  }
}
```

---

## Preprocessing Payload Variations

### 1. Single-Step Preprocessing

**Experiment ID format:** `pre-{template_name}`

**Example:** `experiments/pre-baseline/state.json`

```json
{
  "experiment_id": "pre-baseline/0-baseline",
  "modules": {
    "preprocess": {
      "status": "completed",
      "payload": {
        "train_processed": "artifacts/preprocess/train_processed.csv.gz",
        "test_processed": "artifacts/preprocess/test_processed.csv.gz",
        "orig_processed": null,
        "tuning_processed": null,
        "shapes": {
          "train": [891, 12],
          "test": [418, 11]
        },
        "custom_module_state": {}
      },
      "invocation": {
        "preprocess_template": "baseline",
        "force": false
      }
    }
  }
}
```

**Key fields:**
- `train_processed` - Path to processed training data (relative to experiment dir)
- `test_processed` - Path to processed test data
- `orig_processed` - Path to processed external/original dataset (optional)
- `tuning_processed` - Path to tuning/validation data (optional)
- `shapes` - DataFrame dimensions before/after transformation
- `custom_module_state` - Module-specific data (see below)

---

### 2. Chain Preprocessing (Meta-Template)

**Experiment ID format:** `pre-{chain_name}/{hash}/{step}-{template}`

**Example:** `experiments/pre-full-pipeline/abc123def/2-scaler/state.json`

```json
{
  "experiment_id": "pre-full-pipeline/abc123def/2-scaler",
  "modules": {
    "preprocess": {
      "status": "completed",
      "payload": {
        "train_processed": "artifacts/preprocess/train_processed.csv.gz",
        "test_processed": "artifacts/preprocess/test_processed.csv.gz",
        "orig_processed": "artifacts/preprocess/orig_processed.csv.gz",
        "shapes": {
          "train": [891, 45],
          "test": [418, 44],
          "orig": [1000, 45]
        },
        "input_source": "1-encoder",
        "chain_step": 2,
        "custom_module_state": {
          "scaler_path": "artifacts/preprocess/submodules/scaler/scaler.pkl"
        }
      },
      "invocation": {
        "preprocess_template": "scaler",
        "chain_exp_id": "pre-full-pipeline/abc123def",
        "input_source": "1-encoder",
        "is_last_in_chain": true
      }
    }
  }
}
```

**Additional fields:**
- `input_source` - Previous step in chain that provided input data
- `chain_step` - Index of this step in the chain (0-based)
- `chain_exp_id` - Full chain experiment ID with hash

**Note:** Module name is still `"preprocess"` even in chains, NOT the step name.

---

### 3. Custom Module State

The `custom_module_state` field stores module-specific artifacts:

#### Adversarial Validation Weights

```json
{
  "custom_module_state": {
    "weights_path": "projects/kaggle/titanic/experiments/pre-av/0-adversarial_validation/artifacts/preprocess/sample_weights.csv.gz",
    "av_score": 0.532,
    "discriminator_auc": 0.468
  }
}
```

**Used by:** `adversarial_validation.py`

**Purpose:** Sample weights for reweighting train data to match test distribution.

---

#### External Dataset Alignment

```json
{
  "custom_module_state": {
    "orig_path": "artifacts/preprocess/orig_processed.csv.gz",
    "orig_shape": [1000, 45],
    "alignment_method": "inner_join",
    "id_column": "PassengerId"
  }
}
```

**Used by:** `external_dataset.py`

**Purpose:** Track external dataset processing and alignment.

---

#### Train Fraction / Validation Split

```json
{
  "custom_module_state": {
    "eval_path": "artifacts/preprocess/eval_processed.csv.gz",
    "train_fraction": 0.8,
    "eval_fraction": 0.2,
    "random_state": 42,
    "original_train_size": 891,
    "sampled_train_size": 712,
    "eval_size": 179
  }
}
```

**Used by:** `train_fraction.py`

**Purpose:** Document train/val split for model validation.

---

#### Imbalance Handling

```json
{
  "custom_module_state": {
    "weights_path": "artifacts/preprocess/submodules/imbalance_handler/sample_weights.csv.gz",
    "method": "class_weight",
    "class_distribution": {
      "0": 549,
      "1": 342
    },
    "weights": {
      "0": 0.811,
      "1": 1.302
    }
  }
}
```

**Used by:** `imbalance_handler.py`

**Purpose:** Class weights for handling imbalanced datasets.

---

## Model Payload

**Example:** `experiments/exp-20251217-152730/state.json`

```json
{
  "experiment_id": "exp-20251217-152730",
  "modules": {
    "model": {
      "status": "completed",
      "payload": {
        "local_cv_score": 0.8234,
        "local_cv": 0.8234,
        "model_path": "artifacts/model/model",
        "leaderboard_path": "artifacts/model/leaderboard.csv",
        "best_model": "WeightedEnsemble_L2",
        "training_time": 245.3,
        "preprocess_source": "pre-baseline/0-baseline"
      },
      "invocation": {
        "model_template": "cpu-dev-5m",
        "preprocess_template": "baseline",
        "preprocess_exp_dir": "/path/to/experiments/pre-baseline/0-baseline"
      }
    }
  }
}
```

**Key fields:**
- `local_cv_score` or `local_cv` - Cross-validation score (both names supported for backward compatibility)
- `model_path` - Path to saved model directory
- `leaderboard_path` - Path to AutoGluon leaderboard CSV
- `best_model` - Name of best model from leaderboard
- `preprocess_source` - Which preprocessing experiment was used

---

## Predict Payload

```json
{
  "modules": {
    "predict": {
      "status": "completed",
      "payload": {
        "submission_file": "artifacts/predict/submission-20251217152745.csv",
        "prediction_count": 418,
        "model_source": "artifacts/model/model"
      }
    }
  }
}
```

---

## Submit Payload

```json
{
  "modules": {
    "submit": {
      "status": "completed",
      "payload": {
        "submitted": true,
        "submission_file": "projects/kaggle/titanic/submissions/submission-20251217152745.csv",
        "kaggle_message": "Successfully submitted to Titanic",
        "queued": false
      }
    }
  }
}
```

**If queued:**
```json
{
  "payload": {
    "submitted": false,
    "queued": true,
    "queue_file": "projects/kaggle/titanic/submissions/queue.json",
    "queue_number": 3
  }
}
```

---

## Fetch-Score Payload

```json
{
  "modules": {
    "fetch-score": {
      "status": "completed",
      "payload": {
        "score": 0.7987,
        "score_type": "public",
        "fetch_method": "cdp",
        "timestamp": "2025-12-17 15:28:10"
      }
    }
  }
}
```

---

## Loading Payloads in Code

### Loading from Model Module

The model module handles multiple payload structures:

```python
# From src/mlarena/modules/model.py (simplified)

# Try to get preprocess payload
modules = state.get("modules", {})

# Option 1: Single-step preprocessing
preprocess_module = modules.get("preprocess")

# Option 2: Chain preprocessing (if module name differs)
if not preprocess_module:
    preprocess_module = next(iter(modules.values()), {})

preprocess_payload = preprocess_module.get("payload", {})

# Extract paths
train_path = preprocess_payload.get("train_processed")
test_path = preprocess_payload.get("test_processed")
orig_path = preprocess_payload.get("orig_processed")

# Extract custom state
custom_state = preprocess_payload.get("custom_module_state", {})
weights_path = custom_state.get("weights_path")
eval_path = custom_state.get("eval_path")
```

---

## Payload Evolution

### Backward Compatibility

Older experiments may use different field names:

```json
// Old format (still supported):
{
  "payload": {
    "local_cv": 0.8234  // Legacy name
  }
}

// New format:
{
  "payload": {
    "local_cv_score": 0.8234  // Preferred name
  }
}
```

**Code handles both:**
```python
local_cv = payload.get("local_cv_score") or payload.get("local_cv")
```

---

## Common Access Patterns

### Get Final Preprocessing Output (Chains)

```python
from pathlib import Path
import json

# Find last step in chain
chain_dir = Path("experiments/pre-full-pipeline/abc123def")
steps = sorted([d for d in chain_dir.iterdir() if d.is_dir()],
               key=lambda p: int(p.name.split("-")[0]))
last_step = steps[-1]

# Load state
state_file = last_step / "state.json"
with open(state_file) as f:
    state = json.load(f)

# Get payload
payload = state["modules"]["preprocess"]["payload"]
train_path = payload["train_processed"]
```

---

### Get Sample Weights (if exist)

```python
# Check custom_module_state for weights
custom_state = payload.get("custom_module_state", {})
weights_path = custom_state.get("weights_path")

if weights_path:
    import pandas as pd
    weights = pd.read_csv(weights_path)
```

---

### Get Model CV Score

```python
# Load experiment state
with open("experiments/exp-20251217-152730/state.json") as f:
    state = json.load(f)

model_payload = state["modules"]["model"]["payload"]
cv_score = model_payload.get("local_cv_score") or model_payload.get("local_cv")
```

---

## Summary

**Payload structure depends on:**
1. Module type (preprocess, model, predict, submit, fetch-score)
2. Preprocessing configuration (single vs chain)
3. Module-specific features (weights, eval data, etc.)

**Common patterns:**
- All payloads have module-specific required fields
- `custom_module_state` stores module-specific artifacts
- Preprocessing chains use `input_source` to track data flow
- Model loading handles multiple preprocessing formats automatically

**Best practices:**
- Always check for `None` when accessing optional fields (`orig_processed`, `tuning_processed`)
- Handle both `local_cv_score` and `local_cv` for backward compatibility
- Use `custom_module_state` for module-specific data that doesn't fit standard fields
- Document new custom_module_state fields when creating new preprocessing modules

---

## See Also

- **State Management**: [architecture.md](architecture.md#experiment-state-snapshot)
- **Preprocessing Contract**: [submodules/README.md](submodules/README.md#chain-state-format)
- **Model Integration**: [README.md](../README.md#experiment-tracking-and-reproducibility)
