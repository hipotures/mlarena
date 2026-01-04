# Stack Module

## Overview

The `stack` module provides functionality for ensembling predictions from multiple experiments. It calculates the arithmetic mean of prediction columns from provided submission files.

**Module Name**: `stack`
**Location**: `src/mlarena/modules/stack.py`

## Usage

**Command**:
```bash
uv run python scripts/mla.py stack --project <project> [options]
```

## Parameters

| Parameter |
| Type | Default | Description |
|-----------|------|---------|-------------|
| `prediction_files` | List[str] | `[]` | List of paths to submission CSV files to ensemble. |
| `id_column` | str | `None` | Name of the ID column (auto-detected if None). |
| `target_column` | str | `None` | Name of the target column (auto-detected if None). |

## Examples

### Manual Stacking
```bash
uv run python scripts/mla.py stack -p Titanic \
  prediction_files="['experiments/exp-1/sub.csv', 'experiments/exp-2/sub.csv']"
```

### Auto-Stacking (Experimental)
If run in a pipeline after `predict`, it can automatically pick up the last prediction file (though usually intended for multi-file input).

## Status
**Experimental**. Please verify results manually.

