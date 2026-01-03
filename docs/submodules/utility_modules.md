# Utility Preprocessing Sub-Modules

This document covers minimal and utility sub-modules used for testing, infrastructure verification, or simple pass-through operations.

## No-op (`noop`)

The **noop** (No-operation) sub-module is a smoke test utility. it performs no data transformations but generates a standard preprocessing report and logs its execution.

**Module Name**: `noop`  
**Location**: `src/mlarena/defaults/preprocessing/noop.py`

### Capabilities
- Verification of preprocessing pipeline infrastructure.
- Generates `noop_report.json` and `summary.json`.

### Example
```yaml
noop_test:
  module: noop
  cache: false
```

---

## Identity (`identity`)

The **identity** sub-module is the most minimal pass-through. It returns the data exactly as received without any reporting overhead or artifact creation.

**Module Name**: `identity`  
**Location**: `src/mlarena/defaults/preprocessing/identity.py`

### Capabilities
- Zero-overhead pass-through.
- Useful as a placeholder in templates.

### Example
```yaml
identity_step:
  module: identity
```
