# Tune Module

## Overview

The `tune` module performs hyperparameter optimization using Optuna. It searches for optimal parameters defined in a search space and runs AutoGluon on a small training subset for speed.

**Module Name**: `tune`
**Location**: `src/mlarena/modules/tune.py`

## Usage

**Command**:
```bash
uv run python scripts/mla.py tune --project <project> [options]
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tune_template` | str | `"tune"` | Name of the template defining the search space. |
| `n_trials` | int | `10` | Number of Optuna trials to run. |
| `time_limit` | int | `60` | Time limit (seconds) per AutoGluon trial. |

## Template Configuration

Define search spaces in `src/mlarena/templates/tune/{name}.yaml`:

```yaml
search_space:
  learning_rate:
    type: float
    low: 0.001
    high: 0.1
    log: true
  num_leaves:
    type: int
    low: 10
    high: 100
```

## Status
**Experimental / TODO**. This module is currently under active development. APIs may change.
