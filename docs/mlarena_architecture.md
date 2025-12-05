# MLArena Architecture Redesign Plan

## Overview

Redesign the ML experiment pipeline into a modular, testable architecture with a single entry point (`mla.py`). This replaces `experiment_manager.py`, `ml_runner.py`, `autogluon_runner.py`, and related scripts.

**Execution contract for AI agent:** Każdą fazę realizuj osobno. Po zakończeniu fazy `X` dołącz krótki raport (zakres, kluczowe decyzje, testy) do pliku `docs/mlarena_architecture-phaseX.md`. Ten plik tworzymy per fazę; ma służyć jako dziennik wdrożenia.

**Key decisions:**
- Package location: `src/mlarena/` (new package alongside `kaggle_tools`)
- Migration: No backwards compatibility - remove old scripts immediately
- Documentation: `docs/mlarena_architecture.md`

---

## Table of Contents

- [Overview](#overview)
- [Phase 1: Core Package Structure](#phase-1-core-package-structure)
- [Phase 2: Core Abstractions](#phase-2-core-abstractions)
- [Phase 3: Module Implementations](#phase-3-module-implementations)
- [Phase 4: CLI Entry Point](#phase-4-cli-entry-point)
- [Phase 5: Testing Strategy](#phase-5-testing-strategy)
- [Phase 6: Remove Old Scripts](#phase-6-remove-old-scripts)
- [Phase 7: Migration Strategy (playground-series-s5e12)](#phase-7-migration-strategy-playground-series-s5e12)
- [Phase 8: Configuration Mapping](#phase-8-configuration-mapping)
- [Phase 9: Implementation Details](#phase-9-implementation-details)
- [Phase 10: Backward Compatibility & Rollout](#phase-10-backward-compatibility--rollout)
- [Phase 11: Concurrency & Recovery](#phase-11-concurrency--recovery)
- [Phase 12: Enhanced Caching Strategy](#phase-12-enhanced-caching-strategy)
- [Phase 13: Test Plan](#phase-13-test-plan)
- [Phase 14: Enhanced Migration Strategy](#phase-14-enhanced-migration-strategy)
- [Success Criteria](#success-criteria)

---

## Phase Report Template (for docs/mlarena_architecture-phaseX.md)

- **Zakres**: co zostało zrobione w tej fazie.
- **Decyzje / odchylenia**: kluczowe wybory, różnice względem planu.
- **Testy**: komendy, wyniki (pass/fail), pokrycie jeśli dotyczy.
- **Ryzyka / następne kroki**: otwarte kwestie, blokery, plan na kolejną fazę.
- **Artefakty / PR**: linki do commitów/PR i główne ścieżki plików.

## Phase 1: Core Package Structure

### 1.1 Create Directory Structure

```
src/mlarena/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── module.py         # BaseModule ABC, ModuleResult, ModuleContext
│   ├── experiment.py     # ExperimentState (state.json management)
│   ├── registry.py       # ModuleRegistry (discovery + registration)
│   ├── pipeline.py       # PipelineExecutor (dependency resolution)
│   └── config.py         # TemplateLoader, load_pipeline_def
├── modules/
│   ├── __init__.py       # Auto-imports all modules for registration
│   ├── eda.py
│   ├── preprocess.py
│   ├── feat.py
│   ├── model.py
│   ├── predict.py
│   ├── tune.py
│   ├── stack.py
│   ├── submit.py
│   └── fetch_score.py
├── cli/
│   ├── __init__.py
│   ├── main.py           # CLI entry point logic
│   └── formatters.py     # Rich table/panel formatters
└── utils/
    ├── __init__.py
    ├── git.py            # Git operations (get_git_info, etc.)
    ├── kaggle_api.py     # Kaggle CLI wrapper
    └── time.py           # UTC timestamps
```

### 1.2 Update pyproject.toml

Add `src/mlarena` to packages:

```toml
[tool.setuptools.packages.find]
where = ["src"]
include = ["kaggle_tools*", "mlarena*"]
```

---

## Phase 2: Core Abstractions

### 2.1 BaseModule ABC (`core/module.py`)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from pathlib import Path

@dataclass
class ModuleResult:
    success: bool
    payload: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[Path] = field(default_factory=list)
    error: Optional[str] = None

@dataclass
class ModuleContext:
    project_name: str
    project_root: Path
    experiment_id: str
    experiment_dir: Path
    artifact_dir: Path
    cli_args: Dict[str, Any]
    state: "ExperimentState"
    config_module: Any

class BaseModule(ABC):
    name: str = ""
    description: str = ""
    dependencies: Set[str] = set()

    def __init__(self, context: ModuleContext): ...

    @classmethod
    def register_cli_args(cls, parser) -> None: ...

    def set_invocation_params(self, params: Dict[str, Any]) -> None: ...

    @abstractmethod
    def execute(self) -> ModuleResult: ...

    def can_run(self) -> tuple[bool, str]: ...
```

### 2.2 ExperimentState (`core/experiment.py`)

Manages `state.json` with:
- `experiment_id`, `project`, `created_at`, `git`, `pipeline`, `modules`, `run`
- Methods: `start_module()`, `complete_module()`, `fail_module()`, `save()`
- **CRITICAL**: Each module's `invocation` params stored in state.json

```python
@dataclass
class ModuleEntry:
    name: str
    status: str  # pending | running | completed | failed
    started_at: Optional[str]
    finished_at: Optional[str]
    pid: Optional[int]
    invocation: Dict[str, Any]  # CLI params for reproducibility
    payload: Dict[str, Any]
    error: Optional[str]

@dataclass
class ExperimentState:
    experiment_id: str
    project: str
    modules: Dict[str, ModuleEntry]
    # ...

    @classmethod
    def load_or_create(cls, project_root, project_name, experiment_id=None): ...
```

### 2.3 ModuleRegistry (`core/registry.py`)

```python
class ModuleRegistry:
    _modules: Dict[str, Type[BaseModule]] = {}

    @classmethod
    def register(cls, module_class): ...  # Decorator

    @classmethod
    def discover(cls) -> None:
        """Import mlarena.modules to trigger registration"""
        from mlarena import modules
```

### 2.4 PipelineExecutor (`core/pipeline.py`)

Handles automatic dependency resolution:

```python
class PipelineExecutor:
    def _resolve_execution_order(self, target_module: str) -> List[str]:
        """Topological sort: returns modules to run (deps first, skip completed)"""

    def run_module(self, module_name: str, force=False, skip_deps=False) -> Dict[str, ModuleResult]:
        """Run module and its dependencies"""
```

**Dependency resolution algorithm:**
1. BFS from target module through `dependencies` set
2. Skip modules with status="completed"
3. Reverse order (dependencies first)
4. Execute sequentially, stop on failure

---

## Phase 3: Module Implementations

### 3.1 Module Dependencies Graph

```
eda         -> (none)
preprocess  -> (none)
feat        -> (none)
model       -> preprocess
predict     -> model
tune        -> preprocess
stack       -> predict (multiple)
submit      -> predict
fetch-score -> submit
```

### 3.2 Port Each Module

| Module | Source Lines | Key Logic |
|--------|--------------|-----------|
| eda | experiment_manager.py:549-699 | ydata-profiling, problem_type_guess |
| preprocess | ml_runner.py:499-581 | fit_transform, caching |
| model | ml_runner.py:759-795 | train(), config merge, code snapshot |
| predict | ml_runner.py:797-837 | predict(), submission creation |
| submit | submission_workflow.py:200-350 | Kaggle CLI upload |
| fetch-score | submission_workflow.py:400-500 | submissions list scraping |
| tune | experiment_manager.py:1028-1081 | Optuna placeholder |
| stack | experiment_manager.py:1083-1122 | Stacking placeholder |
| feat | experiment_manager.py:989-1026 | Feature engineering placeholder |

### 3.3 Module CLI Args Pattern

Each module registers its own args:

```python
@register_module
class ModelModule(BaseModule):
    name = "model"
    dependencies = {"preprocess"}

    @classmethod
    def register_cli_args(cls, parser):
        parser.add_argument("--model-template", required=True)
        parser.add_argument("--preprocess-template", default="identity")
        parser.add_argument("--time-limit", type=int)
        parser.add_argument("--preset")
        parser.add_argument("--use-gpu", type=int, choices=[0, 1])
```

---

## Phase 4: CLI Entry Point

### 4.1 Create `scripts/mla.py`

Thin wrapper:

```python
#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from mlarena.cli.main import main
if __name__ == "__main__":
    main()
```

### 4.2 CLI Structure (`cli/main.py`)

```bash
# Module commands (dynamic from registry)
mla --project COMP model --model-template dev-gpu
mla --project COMP eda --eda-notes "exploration"
mla --project COMP submit --auto-submit

# Global options
--project, -p        # Required: competition name
--experiment-id, -e  # Optional: resume existing
--pipeline           # Pipeline definition (default: "default")
--force, -f          # Rerun completed modules
--skip-deps          # Don't auto-run dependencies

# Special commands
mla init --project COMP                     # Initialize new project (creates folders, copies templates, downloads Kaggle data)
mla --project COMP list                     # List experiments
mla --project COMP modules                  # List available modules
```

### 4.3 Invocation Recording

Every CLI call stores params in state.json:

```json
{
  "modules": {
    "model": {
      "invocation": {
        "command": "model",
        "model_template": "dev-gpu",
        "time_limit": 3600,
        "use_gpu": 1
      }
    }
  },
  "run": {
    "invocation": {
      "argv": ["model", "--project", "s5e11", "--model-template", "dev-gpu"],
      "cli_args": {"project": "s5e11", "model_template": "dev-gpu"}
    }
  }
}
```

---

## Phase 5: Testing Strategy

### 5.1 Unit Tests Structure

```
tests/
├── unit/
│   ├── test_module.py        # BaseModule, ModuleResult
│   ├── test_experiment.py    # ExperimentState
│   ├── test_registry.py      # ModuleRegistry
│   ├── test_pipeline.py      # PipelineExecutor
│   └── modules/
│       ├── test_eda.py
│       ├── test_model.py
│       └── ...
└── integration/
    └── test_full_pipeline.py
```

### 5.2 Test Fixtures

```python
@pytest.fixture
def mock_context(tmp_path):
    """Isolated ModuleContext for testing"""
    # Create minimal project structure
    # Mock config_module
    # Return ready-to-use context
```

### 5.3 Module Testability

Each module must be testable in isolation:
- Accept `ModuleContext` with mock dependencies
- No global state
- Return `ModuleResult` (inspectable)

---

## Phase 6: Remove Old Scripts

Delete after mlarena is functional:
- `scripts/experiment_manager.py` (~2000 lines)
- `scripts/ml_runner.py` (~900 lines)
- `scripts/autogluon_runner.py` (~400 lines)
- Keep `scripts/submission_workflow.py` (used by submit module internally)
- Keep `scripts/submissions_tracker.py` (used by submit module)
- Keep `scripts/experiment_logger.py` (used for legacy log reading)

---

## Implementation Order

### Week 1: Foundation
1. [ ] Create `src/mlarena/` directory structure
2. [ ] Implement `core/module.py` (BaseModule, ModuleResult, ModuleContext)
3. [ ] Implement `core/experiment.py` (ExperimentState, ModuleEntry)
4. [ ] Implement `core/registry.py` (ModuleRegistry)
5. [ ] Implement `core/pipeline.py` (PipelineExecutor)
6. [ ] Write unit tests for core classes
7. [ ] Update pyproject.toml

### Week 2: Modules
1. [ ] Port `modules/eda.py`
2. [ ] Port `modules/preprocess.py`
3. [ ] Port `modules/model.py` (most complex)
4. [ ] Port `modules/predict.py`
5. [ ] Port `modules/submit.py`
6. [ ] Port `modules/fetch_score.py`
7. [ ] Stub `modules/tune.py`, `modules/stack.py`, `modules/feat.py`

### Week 3: CLI & Integration
1. [ ] Implement `cli/main.py`
2. [ ] Implement `cli/formatters.py`
3. [ ] Create `scripts/mla.py` entry point
4. [ ] Write integration tests
5. [ ] Test with real competition (s5e11 or s5e12)

### Week 4: Cleanup & Documentation
1. [ ] Remove old scripts
2. [ ] Write `docs/mlarena_architecture.md`
3. [ ] Update `CLAUDE.md`
4. [ ] Update `scripts/README.md`

---

## Critical Files to Reference

| Purpose | File | Lines |
|---------|------|-------|
| Module implementations | scripts/experiment_manager.py | 549-1191 |
| MLRunner training logic | scripts/ml_runner.py | 223-838 |
| Template loading | scripts/ml_runner.py | 42-117 |
| ExperimentManager class | scripts/experiment_manager.py | 227-550 |
| Submission workflow | scripts/submission_workflow.py | 180-500 |
| State.json structure | scripts/experiment_manager.py | 307-423 |
| Pipeline definition | config/pipelines/default.yaml | all |

---

## State.json Enhanced Schema

```json
{
  "experiment_id": "exp-YYYYMMDD-HHMMSS",
  "project": "competition-name",
  "created_at": "ISO8601",
  "git": {
    "hash": "...",
    "branch": "...",
    "has_uncommitted_changes": bool
  },
  "pipeline": {
    "name": "default",
    "modules": ["eda", "preprocess", "model", "predict", "submit", "fetch-score"]
  },
  "modules": {
    "module_name": {
      "status": "pending|running|completed|failed",
      "started_at": "ISO8601",
      "finished_at": "ISO8601",
      "updated_at": "ISO8601",
      "pid": int|null,
      "invocation": {
        "command": "model",
        "model_template": "dev-gpu",
        "time_limit": 3600
      },
      "error": null|"message",
      "...module_specific_payload..."
    }
  },
  "run": {
    "created_at": "ISO8601",
    "updated_at": "ISO8601",
    "invocation": {
      "argv": ["..."],
      "cli_args": {...}
    }
  }
}
```

---

## Resume Experiment Flow

```bash
# Start new experiment (auto-generates exp-id)
mla -p s5e11 eda --eda-notes "initial"
# Output: Experiment: exp-20251205-143000

# Continue same experiment
mla -p s5e11 -e exp-20251205-143000 model --model-template dev-gpu

# Run module in middle of flow (auto-runs dependencies)
mla -p s5e11 submit
# Resolves: eda -> preprocess -> model -> predict -> submit
# Skips already completed modules

# Force rerun
mla -p s5e11 -e exp-20251205-143000 model --model-template best-gpu --force
```

---

## Phase 7: Migration Strategy (playground-series-s5e12)

### 7.1 State.json Compatibility

Obecny format state.json jest **kompatybilny** z nowym systemem. Jedyna zmiana to sciezki absolutne:

```json
// OBECNE (stare sciezki z fork1)
"project_root": "/mnt/ml/kaggle-fork1/projects/kaggle/playground-series-s5e12"

// NOWE (relatywne lub aktualne)
"project_root": "/mnt/ml/kaggle/projects/kaggle/playground-series-s5e12"
```

**Decyzja**: Nowy system bedzie uzywal **sciezek relatywnych** w state.json, rekonstruujac absolutne w runtime.

### 7.2 Skrypt Migracji (jednorazowy)

```python
# scripts/migrate_state_json.py
"""Migrate state.json files to new path format."""
import json
from pathlib import Path

def migrate_experiment(state_path: Path, old_base: str, new_base: str):
    with open(state_path) as f:
        data = json.load(f)

    # Replace absolute paths in config
    config = data.get("modules", {}).get("model", {}).get("config", {})
    if config:
        for section in ["system", "dataset"]:
            for key, val in config.get(section, {}).items():
                if isinstance(val, str) and old_base in val:
                    config[section][key] = val.replace(old_base, new_base)

    # Add invocation if missing (dla starszych eksperymentow)
    for mod_name, mod_data in data.get("modules", {}).items():
        if "invocation" not in mod_data:
            mod_data["invocation"] = {
                "migrated": True,
                "original_cli_args": "unknown"
            }

    with open(state_path, "w") as f:
        json.dump(data, f, indent=2)

# Usage:
# python scripts/migrate_state_json.py --project playground-series-s5e12 \
#     --old-base /mnt/ml/kaggle-fork1 --new-base /mnt/ml/kaggle
```

### 7.3 Co migrowac dla s5e12

| Element | Akcja | Naklad |
|---------|-------|--------|
| experiments/*/state.json | Skrypt migracji (sciezki) | Niski - 5 min |
| submissions/submissions.json | Bez zmian (format OK) | Brak |
| experiments/*.json (legacy) | Ignorowac (stary format) | Brak |
| templates/*.yaml | Bez zmian | Brak |

**Calkowity naklad dla s5e12**: ~15 minut (skrypt + weryfikacja)

---

## Phase 8: Configuration Mapping

### 8.1 TemplateLoader - Reuzycie istniejacych plikow

Nowy `TemplateLoader` bedzie **identyczny** w zachowaniu do obecnego `template_loader.py`:

```python
# src/mlarena/core/config.py

GLOBAL_TEMPLATE_DIR = REPO_ROOT / "config" / "templates"
GLOBAL_PIPELINE_DIR = REPO_ROOT / "config" / "pipelines"

def load_templates(kind: str, project_root: Path) -> Dict[str, Dict]:
    """
    Load templates with local override.

    Sources (in order of priority):
    1. config/templates/{kind}.yaml (global)
    2. projects/kaggle/{proj}/templates/{kind}.yaml (local override)
    """
    global_path = GLOBAL_TEMPLATE_DIR / f"{kind}.yaml"
    local_path = project_root / "templates" / f"{kind}.yaml"

    merged = _read_yaml(global_path).get("templates", {})
    if local_path.exists():
        local = _read_yaml(local_path).get("templates", {})
        merged.update(local)  # Local overrides global

    return merged

def load_pipeline_def(name: str, project_root: Path) -> Dict:
    """Load pipeline definition from YAML."""
    global_path = GLOBAL_PIPELINE_DIR / f"{name}.yaml"
    local_path = project_root / "pipelines" / f"{name}.yaml"

    path = local_path if local_path.exists() else global_path
    return _read_yaml(path).get("pipeline", {})
```

### 8.2 Mapowanie Template Names

| Obecna nazwa | Nowa nazwa | Uwagi |
|--------------|------------|-------|
| `fast-cpu` | `cpu-fast-1m` | Juz istnieje w YAML |
| `dev-cpu` | `cpu-dev-5m` | Aliasy w CLI |
| `dev-gpu` | `gpu-dev-5m` | Aliasy w CLI |
| `best-cpu` | `cpu-best-1h` | Aliasy w CLI |
| `best-gpu` | `gpu-best-1h` | Aliasy w CLI |

**Implementacja aliasow w CLI**:

```python
# cli/main.py
TEMPLATE_ALIASES = {
    "fast-cpu": "cpu-fast-1m",
    "dev-cpu": "cpu-dev-5m",
    "dev-gpu": "gpu-dev-5m",
    "best-cpu": "cpu-best-1h",
    "best-gpu": "gpu-best-1h",
    "extreme-gpu": "gpu-extreme-24h",
}

def resolve_template(name: str) -> str:
    return TEMPLATE_ALIASES.get(name, name)
```

### 8.3 Pliki konfiguracyjne - bez zmian

| Plik | Lokalizacja | Zmiana |
|------|-------------|--------|
| model.yaml | config/templates/model.yaml | Bez zmian |
| preprocess.yaml | config/templates/preprocess.yaml | Bez zmian |
| default.yaml | config/pipelines/default.yaml | Bez zmian |
| project templates | projects/kaggle/*/templates/*.yaml | Bez zmian |

---

## Phase 9: Implementation Details

### 9.1 Logging & Telemetry

```python
# src/mlarena/utils/logging.py
import logging
from rich.logging import RichHandler
from rich.console import Console

console = Console()

def setup_logging(level: str = "INFO", log_file: Path = None):
    """Configure logging with Rich handler + optional file."""
    handlers = [RichHandler(console=console, show_time=True, show_path=False)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=handlers,
    )
    return logging.getLogger("mlarena")

# W kazdym module:
logger = logging.getLogger("mlarena.modules.model")
logger.info("Starting model training...")
logger.debug(f"Config: {config}")
```

**Telemetry w state.json**:

```json
{
  "modules": {
    "model": {
      "timing": {
        "started_at": "2025-12-05T14:30:00Z",
        "finished_at": "2025-12-05T15:30:00Z",
        "duration_seconds": 3600
      },
      "resources": {
        "peak_memory_mb": 8192,
        "gpu_used": true,
        "gpu_memory_mb": 4096
      }
    }
  }
}
```

### 9.2 Error Handling & Retry

```python
# src/mlarena/core/module.py
from functools import wraps
from typing import Callable
import time

class ModuleError(Exception):
    """Base exception for module errors."""
    pass

class RetryableError(ModuleError):
    """Error that can be retried."""
    pass

class FatalError(ModuleError):
    """Error that should stop pipeline."""
    pass

def with_retry(max_attempts: int = 3, delay: float = 1.0):
    """Decorator for retryable operations."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except RetryableError as e:
                    last_error = e
                    logger.warning(f"Attempt {attempt+1}/{max_attempts} failed: {e}")
                    time.sleep(delay * (attempt + 1))  # Exponential backoff
            raise FatalError(f"All {max_attempts} attempts failed") from last_error
        return wrapper
    return decorator

# Usage in module:
class SubmitModule(BaseModule):
    @with_retry(max_attempts=3, delay=5.0)
    def _upload_to_kaggle(self, submission_path: Path):
        result = subprocess.run(["kaggle", "competitions", "submit", ...])
        if result.returncode != 0:
            raise RetryableError(f"Kaggle upload failed: {result.stderr}")
```

### 9.3 Artifact Caching

```python
# src/mlarena/core/cache.py
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional
import pickle

class ArtifactCache:
    """Cache preprocessed data and model artifacts."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def compute_key(self, config: Dict[str, Any]) -> str:
        """Compute cache key from config hash."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def get(self, key: str, artifact_type: str) -> Optional[Any]:
        """Retrieve cached artifact."""
        cache_path = self.cache_dir / key / f"{artifact_type}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        return None

    def put(self, key: str, artifact_type: str, data: Any) -> Path:
        """Store artifact in cache."""
        cache_path = self.cache_dir / key / f"{artifact_type}.pkl"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)
        return cache_path

# Usage in PreprocessModule:
def execute(self) -> ModuleResult:
    cache = ArtifactCache(self.context.experiment_dir / "preprocess_cache")
    cache_key = cache.compute_key(self.preprocess_config)

    cached = cache.get(cache_key, "features")
    if cached and not self._invocation_params.get("recompute"):
        return ModuleResult(success=True, payload={"cached": True, ...})

    # Compute features...
    cache.put(cache_key, "features", features_df)
```

### 9.4 GPU/CPU Management

```python
# src/mlarena/utils/resources.py
import os
from typing import Optional

def detect_gpu() -> dict:
    """Detect available GPU resources."""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "available": True,
                "count": torch.cuda.device_count(),
                "names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                "memory_gb": [torch.cuda.get_device_properties(i).total_memory / 1e9
                             for i in range(torch.cuda.device_count())],
            }
    except ImportError:
        pass
    return {"available": False}

def configure_resources(use_gpu: bool, num_gpus: int = 1, num_cpus: Optional[int] = None):
    """Configure environment for training."""
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if num_cpus:
        os.environ["OMP_NUM_THREADS"] = str(num_cpus)
        os.environ["MKL_NUM_THREADS"] = str(num_cpus)

    return {
        "use_gpu": use_gpu,
        "num_gpus": num_gpus if use_gpu else 0,
        "num_cpus": num_cpus or os.cpu_count(),
    }
```

### 9.5 Integration with submission_workflow.py

`SubmitModule` bedzie **delegowac** do istniejacego `SubmissionRunner`:

```python
# src/mlarena/modules/submit.py
from mlarena.core.module import BaseModule, ModuleResult
from mlarena.core.registry import register_module

@register_module
class SubmitModule(BaseModule):
    name = "submit"
    description = "Submit to Kaggle and fetch score"
    dependencies = {"predict"}

    @classmethod
    def register_cli_args(cls, parser):
        parser.add_argument("--auto-submit", action="store_true")
        parser.add_argument("--skip-score-fetch", action="store_true")
        parser.add_argument("--wait-seconds", type=int, default=30)
        parser.add_argument("--cdp-url", default="http://localhost:9222")
        parser.add_argument("--skip-git", action="store_true")

    def execute(self) -> ModuleResult:
        # Import existing workflow (reuse, don't rewrite)
        import sys
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from submission_workflow import SubmissionRunner, SubmissionArtifact

        # Get submission file from predict module
        predict_data = self.context.state.get_module("predict")
        submission_file = predict_data.payload.get("submission_file")

        artifact = SubmissionArtifact(
            path=self.context.project_root / submission_file,
            filename=Path(submission_file).name,
            project_root=self.context.project_root,
            competition=self.context.project_name,
            local_cv_score=predict_data.payload.get("local_cv"),
        )

        runner = SubmissionRunner(
            artifact=artifact,
            wait_seconds=self._invocation_params.get("wait_seconds", 30),
            cdp_url=self._invocation_params.get("cdp_url", "http://localhost:9222"),
            auto_submit=self._invocation_params.get("auto_submit", False),
            skip_browser=self._invocation_params.get("skip_score_fetch", False),
            skip_git=self._invocation_params.get("skip_git", False),
            experiment_id=self.context.experiment_id,
        )

        result = runner.execute()

        return ModuleResult(
            success=True,
            payload={
                "public_score": result.get("public_score") if result else None,
                "submission_file": submission_file,
                "kaggle_message": runner.kaggle_message,
            }
        )
```

### 9.6 CDP / Offline Fallback (submit + fetch-score)

- **CDP niedostępny lub użytkownik nie jest zalogowany do Kaggle**: `fetch-score` nie blokuje pipeline; zwraca `success=True` z `payload.fetch_failed=True` i `error="cdp_unavailable|not_authenticated"`. Ostrzeżenie trafia do loga, a szczegóły do `state.json` (klucz `fetch_failed_reason`).
- **Zachowanie submit**: upload do Kaggle musi się powieść. Gdy CDP brak, `submit` kończy się sukcesem, ustawia `public_score=None`, a `fetch-score` może być uruchomiony (i prawdopodobnie zakończy się soft-failure) lub pominięty.
- **CLI**: `--skip-score-fetch` nadal pomija scrapera; brak CDP jest traktowany jak `skip` (bez błędu), ale zapisuje flagę `fetch_failed=True`.
- **Przykład logów (brak CDP)**:
  ```
  [warning] fetch-score: CDP unavailable (http://localhost:9222). Skipping score scrape.
  [info]    fetch-score: Marked fetch_failed=True, public_score=None
  ```
- **Przykład wpisu w state.json** (fragment):
  ```json
  "modules": {
    "fetch-score": {
      "status": "completed",
      "payload": {
        "public_score": null,
        "fetch_failed": true,
        "fetch_failed_reason": "cdp_unavailable"
      },
      "error": "cdp_unavailable"
    }
  }
  ```

### 9.7 Git Snapshot Policy

```python
# src/mlarena/utils/git.py
from pathlib import Path
import subprocess
import shutil

class GitSnapshot:
    """Manage code snapshots for reproducibility."""

    @staticmethod
    def get_info(repo_root: Path) -> dict:
        """Get current git state."""
        def run_git(cmd):
            try:
                return subprocess.check_output(
                    cmd, cwd=repo_root, stderr=subprocess.DEVNULL
                ).decode().strip()
            except:
                return None

        status = run_git(["git", "status", "--porcelain"])
        return {
            "hash": run_git(["git", "rev-parse", "HEAD"]),
            "hash_short": run_git(["git", "rev-parse", "--short", "HEAD"]),
            "branch": run_git(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "has_uncommitted_changes": bool(status),
            "uncommitted_files": status.split("\n") if status else [],
        }

    @staticmethod
    def warn_uncommitted(git_info: dict, console) -> None:
        """Warn user about uncommitted changes."""
        if git_info.get("has_uncommitted_changes"):
            console.print("[yellow]WARNING: Uncommitted changes detected![/yellow]")
            console.print(f"   Files: {', '.join(git_info['uncommitted_files'][:5])}")
            console.print("   Consider committing before running experiments.")

    @staticmethod
    def create_snapshot(source_dir: Path, dest_dir: Path) -> Path:
        """Copy code directory for reproducibility."""
        if dest_dir.exists():
            shutil.rmtree(dest_dir)

        shutil.copytree(
            source_dir,
            dest_dir,
            ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", ".git", "*.egg-info",
                "AutogluonModels", "*.pkl", "*.parquet"
            ),
        )
        return dest_dir
```

**Policy w ModelModule**:

```python
def execute(self) -> ModuleResult:
    # 1. Check git state
    git_info = GitSnapshot.get_info(REPO_ROOT)
    GitSnapshot.warn_uncommitted(git_info, console)

    # 2. Train model...

    # 3. Create snapshot AFTER successful training
    snapshot_dir = GitSnapshot.create_snapshot(
        self.context.project_root / "code",
        self.context.experiment_dir / "code_snapshot"
    )

    return ModuleResult(
        success=True,
        payload={
            "git": git_info,
            "code_snapshot": str(snapshot_dir.relative_to(self.context.project_root)),
            ...
        }
    )
```

---

---

## Phase 10: Backward Compatibility & Rollout

### 10.1 CLI Aliases (1 sprint transition)

Stare skrypty jako aliasy przez 1 sprint (2 tygodnie):

```bash
# scripts/experiment_manager.py (deprecated wrapper)
#!/usr/bin/env python
"""DEPRECATED: Use mla.py instead. This wrapper will be removed in 2 weeks."""
import sys
import warnings
from pathlib import Path

warnings.warn(
    "\n" + "="*70 + "\n"
    "DEPRECATION WARNING:\n"
    "  experiment_manager.py is deprecated and will be removed in 2 weeks.\n"
    "  Use mla.py instead:\n"
    f"    OLD: python scripts/experiment_manager.py {' '.join(sys.argv[1:])}\n"
    f"    NEW: python scripts/mla.py {' '.join(sys.argv[1:])}\n"
    "="*70,
    DeprecationWarning,
    stacklevel=2
)

# Map old commands to new
COMMAND_MAP = {
    "eda": "eda",
    "model": "model",
    "list": "list",
    "init": "init",
}

# Forward to mla.py with mapped command
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from mlarena.cli.main import main

# Inject mapped command if needed
if len(sys.argv) > 1 and sys.argv[1] in COMMAND_MAP:
    sys.argv[1] = COMMAND_MAP[sys.argv[1]]

main()
```

### 10.2 Lock on Old Entry Point

Po 1 sprincie (2025-12-19):

```bash
# scripts/experiment_manager.py (locked)
#!/usr/bin/env python
"""REMOVED: Use mla.py instead."""
import sys
print("ERROR: experiment_manager.py has been removed.")
print("Use: python scripts/mla.py")
print("See: docs/mlarena_architecture.md for migration guide")
sys.exit(1)
```

### 10.3 Rollout Timeline

| Week | Action | Status |
|------|--------|--------|
| W1 (Dec 5-11) | Deploy mla.py + wrappers | Both systems work |
| W2 (Dec 12-18) | Team tests mla.py | Deprecation warnings |
| W3 (Dec 19+) | Remove old scripts | Only mla.py |

---

## Phase 11: Concurrency & Recovery

### 11.1 Experiment Lockfile

```python
# src/mlarena/core/lock.py
import fcntl
from pathlib import Path
from contextlib import contextmanager

class ExperimentLock:
    """File-based lock for concurrent experiment access."""

    def __init__(self, experiment_dir: Path):
        self.lock_file = experiment_dir / ".experiment.lock"
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        self.fd = None

    @contextmanager
    def acquire(self, timeout: int = 5):
        """Acquire exclusive lock on experiment."""
        self.fd = open(self.lock_file, "w")
        try:
            fcntl.flock(self.fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield
        except BlockingIOError:
            raise RuntimeError(
                f"Experiment is locked by another process. "
                f"If you're sure no other process is running, remove: {self.lock_file}"
            )
        finally:
            if self.fd:
                fcntl.flock(self.fd.fileno(), fcntl.LOCK_UN)
                self.fd.close()

# Usage in PipelineExecutor:
def run_module(self, module_name: str, ...):
    with ExperimentLock(self.state.experiment_dir).acquire():
        # Execute module safely
        ...
```

### 11.2 SIGTERM Handling

```python
# src/mlarena/core/signals.py
import signal
import sys
from typing import Callable

class GracefulShutdown:
    """Handle SIGTERM/SIGINT gracefully."""

    def __init__(self, state: ExperimentState, module_name: str):
        self.state = state
        self.module_name = module_name
        self.shutdown_requested = False

    def __enter__(self):
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        return self

    def __exit__(self, *args):
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    def _handle_signal(self, signum, frame):
        console.print(f"[yellow]Received signal {signum}, shutting down gracefully...[/yellow]")
        self.shutdown_requested = True
        self.state.fail_module(
            self.module_name,
            f"Interrupted by signal {signum}",
            {"interrupted": True, "signal": signum}
        )
        sys.exit(1)

# Usage in BaseModule:
def execute(self) -> ModuleResult:
    with GracefulShutdown(self.context.state, self.name):
        # Run module logic
        ...
```

### 11.3 Module Status Recovery

```python
# src/mlarena/core/experiment.py (extension)

def recover_stale_modules(self) -> List[str]:
    """Mark stale 'running' modules as 'failed'."""
    recovered = []
    for name, entry in self.modules.items():
        if entry.status == "running":
            pid = entry.pid
            if pid and not self._is_pid_active(pid):
                self.fail_module(
                    name,
                    f"Process {pid} no longer active (system crash or kill)",
                    {"stale_recovery": True}
                )
                recovered.append(name)
    return recovered

@staticmethod
def _is_pid_active(pid: int) -> bool:
    """Check if process is still running."""
    try:
        os.kill(pid, 0)  # Signal 0 checks existence
        return True
    except OSError:
        return False

# Called on ExperimentState.load_or_create():
state = cls.load_or_create(...)
recovered = state.recover_stale_modules()
if recovered:
    console.print(f"[yellow]Recovered stale modules: {', '.join(recovered)}[/yellow]")
```

### 11.4 Retry Policy for fetch-score

```python
# src/mlarena/modules/fetch_score.py
@register_module
class FetchScoreModule(BaseModule):
    name = "fetch-score"
    dependencies = {"submit"}

    @classmethod
    def register_cli_args(cls, parser):
        parser.add_argument("--max-retries", type=int, default=3)
        parser.add_argument("--retry-delay", type=int, default=30)

    @with_retry(max_attempts=3, delay=30.0)
    def _scrape_score(self):
        """Scrape score with exponential backoff."""
        # Playwright scraping logic
        # Raises RetryableError on timeout/network issues
        ...

    def execute(self) -> ModuleResult:
        max_retries = self._invocation_params.get("max_retries", 3)

        for attempt in range(max_retries):
            try:
                score = self._scrape_score()
                return ModuleResult(success=True, payload={"public_score": score})
            except RetryableError as e:
                if attempt == max_retries - 1:
                    # Last attempt failed - mark as partial success
                    console.print(f"[yellow]Score fetch failed after {max_retries} attempts[/yellow]")
                    return ModuleResult(
                        success=True,  # Don't block pipeline
                        payload={"public_score": None, "fetch_failed": True},
                        error=f"Score unavailable: {e}"
                    )
                time.sleep(self._invocation_params.get("retry_delay", 30))
```

---

## Phase 12: Enhanced Caching Strategy

### 12.1 Composite Cache Key

```python
# src/mlarena/core/cache.py (enhanced)

def compute_cache_key(
    config: Dict[str, Any],
    git_hash: str,
    data_checksum: Optional[str] = None
) -> str:
    """
    Compute cache key from config + git + data.

    Key invalidated when:
    - Config changes (hyperparameters, template)
    - Code changes (git hash)
    - Data changes (checksum)
    """
    components = {
        "config": config,
        "git_hash": git_hash[:8],  # Short hash
        "data_checksum": data_checksum or "none",
    }

    key_str = json.dumps(components, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]

def compute_data_checksum(data_path: Path, sample_size: int = 1000) -> str:
    """
    Fast checksum based on:
    - File size
    - First/last N rows
    - Column names + dtypes
    """
    import pandas as pd

    file_stat = data_path.stat()
    df_head = pd.read_csv(data_path, nrows=sample_size)
    df_tail = pd.read_csv(data_path, skiprows=lambda i: i > 0 and i < file_stat.st_size - sample_size)

    signature = {
        "size": file_stat.st_size,
        "mtime": file_stat.st_mtime,
        "columns": list(df_head.columns),
        "dtypes": {col: str(dtype) for col, dtype in df_head.dtypes.items()},
        "head_hash": hashlib.md5(df_head.to_json().encode()).hexdigest()[:8],
        "tail_hash": hashlib.md5(df_tail.to_json().encode()).hexdigest()[:8],
    }

    return hashlib.md5(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
```

### 12.2 Cache Metadata

```python
# Cached artifacts stored with metadata:
{
    "cache_key": "a1b2c3d4e5f6g7h8",
    "created_at": "2025-12-05T14:30:00Z",
    "git_hash": "abc123de",
    "config": {...},
    "data_checksums": {
        "train": "checksum123",
        "test": "checksum456"
    },
    "artifacts": {
        "train_fe.parquet": {"size_mb": 12.5, "rows": 100000, "cols": 50},
        "test_fe.parquet": {"size_mb": 6.2, "rows": 50000, "cols": 49},
        "state.pkl": {"size_kb": 2.3}
    }
}
```

### 12.3 Cache Invalidation

```python
def get_cached_features(self, cache_key: str) -> Optional[CachedFeatures]:
    """Retrieve cached features with validation."""
    meta_path = self.cache_dir / cache_key / "meta.json"

    if not meta_path.exists():
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    # Validate git hash matches
    current_git = GitSnapshot.get_info(REPO_ROOT)["hash"][:8]
    if meta["git_hash"] != current_git:
        console.print(f"[yellow]Cache invalidated: git hash changed[/yellow]")
        return None

    # Validate data checksums match
    current_train_checksum = compute_data_checksum(self.train_path)
    if meta["data_checksums"]["train"] != current_train_checksum:
        console.print(f"[yellow]Cache invalidated: train data changed[/yellow]")
        return None

    # Load cached artifacts
    return self._load_cached_artifacts(cache_key)
```

---

## Phase 13: Test Plan

### 13.1 Minimal Test Coverage Requirements

| Component | Target Coverage | Critical Paths |
|-----------|----------------|----------------|
| core/module.py | 90% | BaseModule.can_run(), execute() flow |
| core/experiment.py | 95% | start/complete/fail module, state.json persistence |
| core/pipeline.py | 90% | Dependency resolution, execution order |
| core/registry.py | 100% | Register, discover, get |
| modules/model.py | 80% | Config merge, training flow, error handling |
| modules/submit.py | 70% | Kaggle upload (mocked), score fetch retry |

### 13.2 Mocking Strategy

```python
# tests/conftest.py
import pytest
from unittest.mock import MagicMock, patch

@pytest.fixture
def mock_kaggle_api():
    """Mock Kaggle CLI calls."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Successfully submitted to competition",
            stderr=""
        )
        yield mock_run

@pytest.fixture
def mock_playwright():
    """Mock Playwright score scraping."""
    with patch("playwright.async_api.async_playwright") as mock_pw:
        mock_page = MagicMock()
        mock_page.evaluate.return_value = "0.92345"  # Mock score

        mock_context = MagicMock()
        mock_context.pages = [mock_page]

        mock_browser = MagicMock()
        mock_browser.contexts = [mock_context]

        mock_playwright = MagicMock()
        mock_playwright.chromium.connect_over_cdp.return_value = mock_browser

        mock_pw.return_value.__aenter__.return_value = mock_playwright
        yield mock_pw

@pytest.fixture
def mock_autogluon():
    """Mock AutoGluon TabularPredictor."""
    with patch("autogluon.tabular.TabularPredictor") as mock_predictor:
        mock_instance = MagicMock()
        mock_instance.fit.return_value = None
        mock_instance.predict.return_value = [0.1, 0.9, 0.5]
        mock_instance.leaderboard.return_value = pd.DataFrame({
            "model": ["LightGBM", "CatBoost"],
            "score_val": [0.92, 0.91]
        })

        mock_predictor.return_value = mock_instance
        yield mock_predictor
```

### 13.3 Critical Test Cases

```python
# tests/unit/test_pipeline.py
def test_dependency_resolution_simple():
    """Test: model -> preprocess dependency resolved."""
    executor = PipelineExecutor(...)

    # Mark preprocess as pending, model as pending
    order = executor._resolve_execution_order("model")

    assert order == ["preprocess", "model"]

def test_dependency_resolution_skip_completed():
    """Test: Skip completed modules in chain."""
    state.complete_module("preprocess", {})

    order = executor._resolve_execution_order("model")

    assert order == ["model"]  # preprocess skipped

def test_module_lock_concurrent_access():
    """Test: Second process blocked by lock."""
    with ExperimentLock(exp_dir).acquire():
        with pytest.raises(RuntimeError, match="locked by another process"):
            with ExperimentLock(exp_dir).acquire():
                pass

def test_stale_module_recovery():
    """Test: Stale 'running' module marked as failed."""
    state.start_module("model", {})
    state.modules["model"].pid = 99999  # Non-existent PID

    recovered = state.recover_stale_modules()

    assert "model" in recovered
    assert state.get_module_status("model") == "failed"
```

### 13.4 Integration Tests

```python
# tests/integration/test_full_pipeline.py
def test_end_to_end_pipeline(tmp_path, mock_kaggle_api, mock_autogluon):
    """Test: Complete EDA -> Model -> Predict -> Submit flow."""
    # Setup minimal project
    project_root = setup_test_project(tmp_path, "test-comp")

    # Run pipeline
    result = run_mla_command([
        "--project", "test-comp",
        "model",
        "--model-template", "cpu-fast-1m",
        "--skip-submit"
    ])

    # Verify state
    state = ExperimentState.load_existing(project_root, result.experiment_id)
    assert state.get_module_status("model") == "completed"
    assert state.get_module_status("predict") == "completed"

    # Verify artifacts
    submission_file = state.modules["predict"].payload["submission_file"]
    assert (project_root / submission_file).exists()
```

### 13.5 CI Execution

- **Per-PR (fast)**: `uv run pytest tests/unit` z mockami Kaggle/Playwright; cel < 2 min.
- **Nightly (pełne)**: `uv run pytest tests/integration` (mockowane zewnętrzne usługi) + unit; sprawdza spójność pipeline end-to-end.
- **Pokrycie**: uruchamiaj `--cov=src/mlarena --cov-report=xml` i egzekwuj progi z tabeli 13.1 w CI; build failuje, gdy pokrycie spadnie poniżej celu.

---

## Phase 14: Enhanced Migration Strategy

### 14.1 Automatic Backup

```python
# scripts/migrate_state_json.py (enhanced)
import shutil
from datetime import datetime

def migrate_experiment(
    state_path: Path,
    old_base: str,
    new_base: str,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Migrate state.json with automatic backup.

    Returns:
        Migration report with changes made.
    """
    # 1. Create backup
    backup_path = state_path.parent / f"state.json.bak-{datetime.now():%Y%m%d-%H%M%S}"
    shutil.copy2(state_path, backup_path)
    console.print(f"[green]Backup created: {backup_path.name}[/green]")

    # 2. Load and analyze
    with open(state_path) as f:
        data = json.load(f)

    report = {
        "experiment_id": data.get("experiment_id"),
        "paths_replaced": 0,
        "modules_updated": 0,
        "changes": []
    }

    # 3. Detect paths per-file (not global replace)
    config = data.get("modules", {}).get("model", {}).get("config", {})
    if config:
        for section in ["system", "dataset"]:
            section_data = config.get(section, {})
            for key, val in section_data.items():
                if isinstance(val, str) and old_base in val:
                    old_val = val
                    new_val = val.replace(old_base, new_base)

                    if dry_run:
                        report["changes"].append(f"  {section}.{key}: {old_val} -> {new_val}")
                    else:
                        section_data[key] = new_val

                    report["paths_replaced"] += 1

    # 4. Add invocation if missing
    for mod_name, mod_data in data.get("modules", {}).items():
        if "invocation" not in mod_data:
            if not dry_run:
                mod_data["invocation"] = {
                    "migrated": True,
                    "original_cli_args": "unknown",
                    "migration_timestamp": datetime.now().isoformat()
                }
            report["modules_updated"] += 1

    # 5. Save (unless dry-run)
    if not dry_run:
        with open(state_path, "w") as f:
            json.dump(data, f, indent=2)
        console.print(f"[green]Migrated: {state_path}[/green]")
    else:
        console.print(f"[yellow]DRY RUN: {state_path}[/yellow]")

    return report
```

### 14.2 CLI Interface

```bash
# Dry-run mode (preview changes)
python scripts/migrate_state_json.py \
    --project playground-series-s5e12 \
    --old-base /mnt/ml/kaggle-fork1 \
    --new-base /mnt/ml/kaggle \
    --dry-run

# Output:
# Found 45 experiments to migrate
#
# exp-20251201-013304:
#   - Paths replaced: 8
#   - Modules updated: 1
#   Changes:
#     system.project_root: /mnt/ml/kaggle-fork1/... -> /mnt/ml/kaggle/...
#     dataset.train_path: /mnt/ml/kaggle-fork1/... -> /mnt/ml/kaggle/...
#   Backup: state.json.bak-20251205-143000
#
# Total: 8 paths replaced, 1 module updated
# Run without --dry-run to apply

# Actual migration
python scripts/migrate_state_json.py \
    --project playground-series-s5e12 \
    --old-base /mnt/ml/kaggle-fork1 \
    --new-base /mnt/ml/kaggle
```

### 14.3 Per-File Path Detection

```python
def detect_base_paths(state_data: Dict) -> List[str]:
    """Detect all base paths in state.json for smart replacement."""
    paths = set()

    def extract_paths(obj, prefix=""):
        if isinstance(obj, dict):
            for key, val in obj.items():
                extract_paths(val, f"{prefix}.{key}")
        elif isinstance(obj, str):
            # Check if looks like absolute path
            if obj.startswith("/") and "kaggle" in obj:
                # Extract base path (up to /projects/kaggle/competition-name/)
                match = re.match(r"(.*?/projects/kaggle/[^/]+)/", obj)
                if match:
                    paths.add(match.group(1))

    extract_paths(state_data)
    return sorted(paths)

# Usage:
detected_bases = detect_base_paths(data)
console.print(f"[cyan]Detected base paths:[/cyan]")
for base in detected_bases:
    console.print(f"  - {base}")

# Ask user to confirm replacement
if not auto_confirm:
    response = input(f"Replace {detected_bases[0]} with {new_base}? [y/N] ")
    if response.lower() != "y":
        return
```

### 14.4 Scope Guard

- Skrypt migracyjny operuje wyłącznie na `projects/kaggle/<project>/experiments/*/state.json`; inne katalogi są ignorowane, aby uniknąć przypadkowych podmian ścieżek.

### 14.5 Rollback Procedure

```bash
# If migration fails or causes issues:

# 1. List backups
ls experiments/exp-*/state.json.bak-*

# 2. Restore single experiment
cp experiments/exp-20251201-013304/state.json.bak-20251205-143000 \
   experiments/exp-20251201-013304/state.json

# 3. Bulk restore script
python scripts/restore_backups.py --project playground-series-s5e12 --timestamp 20251205-143000
```

---

## Success Criteria

1. **Single entry point**: Only `mla.py` needed for all operations
2. **Modular**: Each module in separate file, testable in isolation
3. **Dependencies**: Auto-run preceding modules when needed
4. **Reproducibility**: All CLI params stored in state.json
5. **Resumable**: `--experiment-id` continues existing experiment
6. **Clean**: No fallbacks to old code, old scripts removed after 1 sprint
7. **Compatible configs**: Existing YAML templates work without changes
8. **Migrated s5e12**: Existing experiments accessible via new system
9. **Concurrent-safe**: Lockfile prevents parallel access to same experiment
10. **Recovery**: Stale modules auto-detected and marked failed
11. **Tested**: >80% coverage on critical paths, mocked external APIs
12. **Backed up**: All migrations create automatic .bak files
