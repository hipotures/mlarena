# AI Agent Guide (@AGENTS.md)

**Role**: You are an AI agent working on the `mlarena` framework. Your goal is to navigate, understand, and modify this specific codebase efficiently.

**Optimization Rule**: Do NOT scan every file. Use the map below to jump directly to the relevant components.

## 🗺️ Codebase Map (Where to look)

| Component | Path | Description |
| :--- | :--- | :--- |
| **CLI & Entry** | `scripts/mla.py` → `src/mlarena/cli/main.py` | Command parsing, auto-flow logic, `main()` function. |
| **Pipeline Logic** | `src/mlarena/core/pipeline.py` | Dependency resolution, execution graph, skipping completed. |
| **State/Registry** | `src/mlarena/core/experiment.py`, `registry.py` | State persistence (`state.json`) and module registration (`@ModuleRegistry`). |
| **Modules** | `src/mlarena/modules/` | Individual modules (`model.py`, `preprocess.py`, `submit.py`, etc.). |
| **Templates** | `src/mlarena/templates/` | Global YAML templates. Look here for default configs. |
| **Defaults** | `src/mlarena/defaults/` | Default implementations for `models/` (train) and `preprocessing/` (fit_transform). |
| **Projects** | `projects/kaggle/<slug>/` | User code location. `code/models/`, `code/utils/config.py`. |
| **Queues** | `src/mlarena/utils/queue.py` | Task queue implementation (`mla queue`). |

## 🏗️ Architecture Shortcuts

-   **Adding a Module**: Create file in `src/mlarena/modules/`, inherit `BaseModule`, decorate with `@ModuleRegistry.register`.
-   **Adding a Preprocess Step**: Copy `src/mlarena/defaults/preprocessing/TEMPLATE.py` to `src/mlarena/defaults/preprocessing/`.
-   **Config System**: Uses `OmegaConf`. Project config (`code/utils/config.py`) is imported dynamically.
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
    *   `python scripts/mla.py queue add -p <proj> --model-template run1-lgbm-imputed`
    *   `python scripts/mla.py queue add -p <proj> --model-template run2-xgb-scaled`

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

## 📚 Documentation Index
-   **Workflow**: `docs/MLA_WORKFLOW_GUIDE.md`
-   **Submodules**: `docs/submodules/README.md`
-   **Config**: `docs/configs.md`