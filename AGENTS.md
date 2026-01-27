# AI Agent Guide (@AGENTS.md)

**Role**: You are an AI agent working on the `mlarena` framework. Your goal is to navigate, understand, and modify this specific codebase efficiently.

**Optimization Rule**: Do NOT scan every file. Use the map below to jump directly to the relevant components.

## 🗺️ Codebase Map (Where to look)

| Component | Path | Description |
| :--- | :--- | :--- |
| **CLI & Entry** | `scripts/mla.py` → `src/mlarena/cli/main.py` | Command parsing, auto-flow logic, `main()` function. |
| **Pipeline Logic** | `src/mlarena/core/pipeline.py` | Dependency resolution, execution graph, skipping completed. |
| **State/Registry** | `src/mlarena/core/experiment.py`, `src/mlarena/core/registry.py` | State persistence (`state.json`) and module registration (`@ModuleRegistry`). |
| **Modules** | `src/mlarena/modules/` | Individual modules (`model.py`, `preprocess.py`, `submit.py`, etc.). |
| **Templates** | `src/mlarena/templates/` | Global YAML templates. Look here for default configs. |
| **Defaults** | `src/mlarena/defaults/` | Default implementations for `models/` (train) and `preprocessing/` (fit_transform). |
| **Projects** | `projects/kaggle/<slug>/` | User code location. `code/models/`, `code/utils/config.py`. |
| **Queues** | `src/mlarena/utils/queue.py` | Task queue implementation (`mla queue`). |

## 📁 Detailed Directory Tree

```
mlarena/                                    # Repository root
│
├── src/mlarena/                           # Framework source code
│   ├── cli/
│   │   └── main.py                       # CLI entry point, command parsing
│   ├── core/
│   │   ├── conf.py                       # OmegaConf + Pydantic config system
│   │   ├── pipeline.py                   # Execution engine, dependency resolution
│   │   ├── experiment.py                 # State management (state.json)
│   │   ├── registry.py                   # Module discovery (@ModuleRegistry)
│   │   └── module.py                     # BaseModule class
│   ├── modules/                          # Pipeline modules
│   │   ├── init.py                       # Project initialization
│   │   ├── eda.py                        # Exploratory data analysis
│   │   ├── preprocess.py                 # Preprocessing pipeline
│   │   ├── model.py                      # Model training
│   │   ├── predict.py                    # Prediction generation
│   │   ├── submit.py                     # Kaggle submission
│   │   ├── fetch_score.py                # Score scraping
│   │   ├── mcts/                         # ⚡ MCTS search implementation
│   │   │   ├── mcts_module.py           # MCTS pipeline module
│   │   │   └── tree_search.py           # Tree search algorithm
│   │   └── rant/                         # ⚡ RANT search implementation
│   │       ├── rant_module.py           # RANT pipeline module
│   │       └── random_tree.py           # Random tree algorithm
│   ├── defaults/                         # Default implementations
│   │   ├── models/                       # 🏋️ Model trainers
│   │   │   ├── autogluon_tabular.py
│   │   │   ├── catboost_model.py
│   │   │   ├── lightgbm_model.py
│   │   │   └── xgboost_model.py
│   │   └── preprocessing/                # 🔧 Preprocessing steps
│   │       ├── identity.py               # Pass-through (template)
│   │       ├── imputer.py                # Missing value imputation
│   │       ├── encoder.py                # Categorical encoding
│   │       ├── feature_selector.py       # Feature selection
│   │       ├── dae_embeddings.py         # Denoising autoencoder
│   │       └── [30+ other modules]
│   ├── search_spaces/                    # 🎲 MCTS/RANT search space definitions
│   │   └── preprocess/                   # Preprocessing search spaces
│   │       ├── imputer.yaml              # Imputation hyperparameters
│   │       ├── encoder.yaml              # Encoder hyperparameters
│   │       ├── feature_selector.yaml     # Feature selection ranges
│   │       └── [25+ other search spaces]
│   ├── templates/                        # 📋 Global YAML templates
│   │   ├── model/                        # Model templates
│   │   │   ├── baseline.yaml
│   │   │   ├── cpu-dev-5m.yaml
│   │   │   └── hpo/                      # HPO-specific templates
│   │   ├── preprocess/                   # Preprocessing templates
│   │   │   ├── baseline.yaml
│   │   │   ├── identity.yaml
│   │   │   └── [chains and custom configs]
│   │   ├── profiles/                     # Config profiles
│   │   │   ├── smoke.yaml                # Fast testing
│   │   │   └── dev.yaml                  # Development
│   │   └── project/                      # Project scaffolding
│   └── utils/                            # Shared utilities
│       ├── queue.py                      # Task queue
│       ├── artifacts.py                  # Artifact management
│       └── report.py                     # Report generation
│
├── conf/                                  # 🌐 Global configuration
│   ├── generator_config.yaml             # Template generator settings
│   └── preprocess/                       # Preprocessing configs
│       └── mla_super_chain.yaml          # MCTS default settings
│
├── scripts/                               # 🔨 Entry points and utilities
│   ├── mla.py                            # ⭐ Main CLI entry point
│   ├── mcts_oracle.py                    # 🧙 MCTS oracle (best variant selector)
│   ├── submissions_tracker.py            # Submission tracking
│   ├── experiment_logger.py              # Experiment logging
│   ├── task_queue.py                     # Task queue runner
│   ├── task_queue_textual.py            # TUI for task queue
│   └── utils/
│       ├── clean.py                      # Artifact cleanup
│       └── sync.py                       # Project synchronization
│
└── projects/kaggle/<slug>/               # 🏆 Competition projects
    ├── data/                             # Raw competition data
    │   ├── train.csv
    │   ├── test.csv
    │   └── sample_submission.csv
    ├── code/                             # Competition-specific code
    │   ├── models/                       # Custom model implementations
    │   │   └── my_model.py
    │   ├── preprocessing/                # Custom preprocessing modules
    │   │   └── my_preprocess.py
    │   └── utils/
    │       └── config.py                 # ⚙️ Project constants (TARGET_COLUMN, etc.)
    ├── conf/                             # 📝 Project-level configs
    │   └── preprocess/                   # Project preprocessing configs
    │       └── mla_super_chain.yaml      # MCTS settings override
    ├── templates/                        # Project template overrides
    │   ├── model/                        # Model templates (override globals)
    │   │   └── custom-model.yaml
    │   └── preprocess/                   # Preprocessing templates
    │       └── custom-preprocess.yaml
    ├── experiments/                      # 🧪 Experiment results
    │   ├── logs/                         # 📊 MCTS/RANT logs (START HERE!)
    │   │   ├── mcts.log                  # ← MCTS execution log
    │   │   └── rant.log                  # ← RANT execution log
    │   ├── db/                           # 💾 Optuna study databases
    │   │   └── mcts_study.db             # SQLite database for MCTS
    │   ├── eda/                          # EDA experiment
    │   │   ├── state.json                # Module state + metadata
    │   │   └── artifacts/                # Generated reports
    │   └── exp-YYYYMMDD-HHMMSS/         # Timestamped experiments
    │       ├── state.json                # ← Experiment state (read this!)
    │       ├── state.lock                # File lock for concurrent writes
    │       └── artifacts/                # Model outputs, predictions, etc.
    ├── submissions/                      # Kaggle submissions
    │   └── submissions.json              # Submission tracking
    └── queue/                            # Task queue storage
        └── tasks.json                    # Queued experiments

# NFS Mount (for distributed computing)
/mnt/mlarena/                             # 🖥️ Remote computing server mount
└── projects/kaggle/<slug>/               # Same structure as local
    └── experiments/                      # ← Check here if local is empty!
        └── exp-YYYYMMDD-HHMMSS/
```

### 🔍 Quick Location Guide

| What you need | Where to find it |
|---------------|------------------|
| **MCTS logs** | `projects/kaggle/<slug>/experiments/logs/mcts.log` |
| **RANT logs** | `projects/kaggle/<slug>/experiments/logs/rant.log` |
| **Optuna DB** | `projects/kaggle/<slug>/experiments/db/mcts_study.db` |
| **Search spaces** | `src/mlarena/search_spaces/preprocess/*.yaml` |
| **Experiment results** | `projects/kaggle/<slug>/experiments/exp-*/state.json` |
| **Remote results** | `/mnt/mlarena/projects/kaggle/<slug>/experiments/` |
| **Global templates** | `src/mlarena/templates/{model,preprocess}/` |
| **Project templates** | `projects/kaggle/<slug>/templates/` |
| **MCTS config** | `conf/preprocess/mla_super_chain.yaml` (global)<br>`projects/kaggle/<slug>/conf/preprocess/mla_super_chain.yaml` (project) |
| **Preprocessing code** | `src/mlarena/defaults/preprocessing/*.py` |
| **Model code** | `src/mlarena/defaults/models/*.py` |

## 🏗️ Architecture Shortcuts

-   **Adding a Module**: Create file in `src/mlarena/modules/`, inherit `BaseModule`, decorate with `@ModuleRegistry.register`.
-   **Adding a Preprocess Step**: Copy `src/mlarena/defaults/preprocessing/identity.py` to `src/mlarena/defaults/preprocessing/`.
-   **Config System**: Uses `OmegaConf` + `Pydantic`. Root config logic in `src/mlarena/core/conf.py`. Project config (`code/utils/config.py`) is imported dynamically.
-   **Magic Flags**: Use dot notation for overrides, e.g., `project=playground-series-s6e1`, `mcts.enabled=true`, `mcts.budget=2000`, `force=true`.
-   **Artifacts**: Always use `self.context.artifact_dir`. Never hardcode paths.
-   **State**: `self.context.state` contains the `state.json` data.

## 🖥️ Compute & Storage Architecture (NFS Workflow)

Środowisko pracuje w modelu rozproszonym, co wpływa na lokalizację plików:
- **Local Dev Server**: Środowisko pracy agenta (`~/ml/kaggle`). Tu modyfikujemy kod, szablony i skrypty.
- **Computing Server**: Zdalna maszyna wykonująca obliczenia. Eksportuje ona swój folder roboczy `/home/xai/ml/mlarena` przez NFS.
- **NFS Mount**: Zasób ze zdalnego serwera jest zamontowany lokalnie w `/mnt/mlarena`.
- **Rsync Sync**: Lokalne zmiany z `~/ml/kaggle` są synchronizowane na serwer obliczeniowy przez NFS (struktura 1:1).
- **Path Mapping**: Eksperymenty i ich stan (`state.json`) znajdują się fizycznie na NFS. Agent musi mapować lokalne ścieżki projektów na `/mnt/mlarena/...`, aby analizować wyniki.

## 🤖 Abstract Task Examples

### Task 1: "Create 32 experiments based on EDA"
**Goal**: Generate multiple model runs with different preprocessing and model parameters.

**Strategy**:
1.  **Read EDA**: Check `projects/kaggle/<proj>/experiments/eda/state.json` or reports in `artifacts/`.
2.  **Create Templates**: Generate YAML files in `projects/kaggle/<proj>/templates/model/` and `preprocess/`.
    *   *Naming*: `run1-lgbm-imputed.yaml`, `run2-xgb-scaled.yaml`.
3.  **Queue Tasks**: Use the Task Queue to schedule them.
    *   `uv run python scripts/mla.py queue add -p <proj> --model-template run1-lgbm-imputed`
    *   `uv run python scripts/mla.py queue add -p <proj> --model-template run2-xgb-scaled`

### Task 2: "Fix bug in submission validation"
**Goal**: Submission fails because column order is wrong.

**Strategy**:
1.  **Locate Logic**: Go to `src/mlarena/modules/submit.py` -> `_validate_submission()`.
2.  **Check Config**: Look at `src/mlarena/utils/project.py` (how `sample_submission` is loaded).
3.  **Fix**: Modify `_validate_submission` to allow reordered columns if names match.

### Task 3: "Add new preprocessing method 'Winsorization'"
**Goal**: Implement a new outlier handling technique.

**Strategy**:
1.  **Implementation**: Create `src/mlarena/defaults/preprocessing/winsorizer.py`.
2.  **Interface**: Implement `fit_transform(train, val, test, config)`.
3.  **Template**: Add `winsorizer.yaml` to `src/mlarena/templates/preprocess/`.
4.  **Docs**: Create `docs/submodules/winsorizer.md`.

## ⚠️ Critical Rules for Agents

1.  **Do NOT edit `experiments/**/state.json` manually**. Let the framework handle state.
2.  **Do NOT import project code globally**. Use `importlib` or local imports inside functions to avoid breaking CLI discovery.
3.  **ALWAYS check `projects/kaggle/<proj>/code/utils/config.py`** before assuming column names (e.g., `TARGET_COLUMN`).
4.  **Respect `mla_retention`**. If disk space is low, suggest using this flag for AutoGluon models.
5.  **Always check `/mnt/mlarena` for experiment results** if the local `experiments/` folder is empty. Use 1:1 path mapping between the local repository and the NFS mount point.
6.  **Console output should use `rich`** for improved visualization of data/logs.
7.  **Default output paths**: If a script/task does not specify where to write logs or outputs, save to `/tmp` by default. Never write to the project root.

## 📚 Documentation Index
-   **Workflow**: `docs/MLA_WORKFLOW_GUIDE.md`
-   **Submodules**: `docs/submodules/README.md`
-   **Config**: `docs/configs.md`

## 🧠 Recently Added Memories

- **Submissions Tracker**: The `scripts/submissions_tracker.py` script now scans `experiments/*/state.json` to populate the submissions list, merging with `submissions/submissions.json`. It supports both legacy and new experiment schemas. IDs are preserved for existing entries, and new entries get sequential IDs. Sorting by public_score is fixed.
- **Config Fixes**: Fixed missing `requires_preproc: [{group: imputer}]` for `polynomial` and `mixed` feature engineering variants in `random_preprocess_config_titanic.yaml` and `random_preprocess_config.yaml` to prevent PolynomialFeatures from failing on NaNs.
- **Experiment Generation**: Fixed bug in `generate_random_preprocess_experiments.py` where `_order_by_requires` did not skip imputer requirements when EDA indicated no missing values, causing a ValueError.
- **TUI Usability**: User resolved a Textual TUI mouse issue (hover works, click ignores) by enabling 'mouse reporting' in their terminal configuration.
- **CC Usage Script**: The `scripts/cc_usage_web.py` script now handles "Page not found" on the usage page as a valid "free_plan" state, returning null usage stats instead of an error.
- **Manual Experimentation**: Created documentation for manual experimentation in 'docs/manual_experimentation/', including 'HOWTO.md' and a 'GEMINI.md' summary. This documents the workflow for cloning experiments, file naming (suffix _01, _02...), full file isolation, and queueing with 'scripts/task_queue.py'.
- **CLI Help**: The `mla` CLI uses a hybrid help system: `main.py` handles global help, while `experiments` and `submissions` modules have internal `argparse` parsers. patched `main.py` to pass `--help` through to these specific modules to expose their custom flags (like `--sort-by`). Other modules (model, preprocess) rely on global help.
- **In-Memory Pipeline**: Implemented 'Functional In-Memory Pipeline' in 'src/mlarena/modules/preprocess.py'. It allows defining a preprocessing template with a 'steps' list, which is executed as a single 'sklearn.pipeline.Pipeline' without intermediate file I/O, while preserving all module states.
