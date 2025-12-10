# Target Transformer Sub-Module

## Overview

The **target_transformer** sub-module applies configurable transformations to the target column for regression problems. It supports log, Box-Cox, and Yeo–Johnson, with optional clipping and automatic shifting for non-positive values. The fitted transformer (for PowerTransformer-based methods) is saved for inverse-transformation during prediction.

**Module Name**: `target_transformer`  
**Location**: `config/code/preprocessing/target_transformer.py`

## Purpose
- Stabilize variance and normalize skewed targets
- Prevent extreme values via quantile clipping before transform
- Provide reproducible transformer artifacts for inference-time inverse transforms

## Parameters

### Core
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_transform` | str | `"none"` | Transformation: `none`, `log1p`, `boxcox`, `yeo_johnson` |
| `standardize` | bool | `true` | Standardize output of PowerTransformer (Box-Cox/Yeo-Johnson) |

### Clipping (pre-transform)
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clip_lower_quantile` | float \| null | `null` | Lower quantile bound (0-1); clips target before transform |
| `clip_upper_quantile` | float \| null | `null` | Upper quantile bound (0-1); clips target before transform |

### Shifting (log/Box-Cox safety)
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shift_before_log` | bool | `true` | Auto-shift if min(target) ≤ 0 for `log1p` / `boxcox` |
| `shift_value` | float \| null | `null` | Manual shift override (added before log/Box-Cox); set to force a specific offset |

## Examples

### 1) Log1p with Auto-Shift
```yaml
target_log1p:
  module: target_transformer
  cache: true
  config:
    target_transform: "log1p"
    shift_before_log: true   # auto-fix non-positive targets
```

### 2) Box-Cox with Manual Shift and Clipping
```yaml
target_boxcox:
  module: target_transformer
  cache: true
  config:
    target_transform: "boxcox"
    shift_value: 1.0
    clip_lower_quantile: 0.01
    clip_upper_quantile: 0.99
    standardize: true
```

### 3) Yeo–Johnson (Handles Non-Positive)
```yaml
target_yeojohnson:
  module: target_transformer
  cache: true
  config:
    target_transform: "yeo_johnson"
    standardize: true
```

### 4) Pass-Through (No Transform)
```yaml
target_none:
  module: target_transformer
  cache: true
  config:
    target_transform: "none"
```

## Artifacts
- `target_transform_report.json`: method, clip bounds, shift used, standardize flag, transformer path (if applicable), sanitized config.
- `summary.json`: standard preprocessing summary (shapes/columns).
- `power_transformer.pkl` (for `boxcox` / `yeo_johnson`): fitted `sklearn.preprocessing.PowerTransformer`.

## State Dictionary (returned)
```python
{
    "version": "1.0",
    "method": "boxcox",
    "target_column": "target",
    "clip_bounds": {"lower": 0.01, "upper": 0.99},
    "shift_used": 1.0,
    "standardize": true,
    "transformer_path": "submodules/target_transformer/power_transformer.pkl",
    "config": {...}  # user config without internal keys
}
```

## Notes & Tips
- Use only for regression targets; ensure `_dataset.target` is set.
- `log1p`/`boxcox` require strictly positive inputs; `shift_before_log` or `shift_value` will offset data first.
- Clipping uses train quantiles and applies the same bounds to validation data.
- Yeo–Johnson handles non-positive values without shifting; still benefits from clipping.
- Store/track `transformer_path` for inverse-transform of predictions at inference time.***
