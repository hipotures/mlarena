# Development TODO

> **Note:** This file tracks development tasks. Consider moving items to GitHub Issues for better project management.

## Performance Optimization

### Completed
- [x] Remove toplevel pandas imports from modules (moved to execute() functions) - **reduced from 4.5s to 2.5s**
- [x] Remove toplevel pandas import from utils/init/core.py
- [x] Skip ModuleRegistry.clear() on every run (use cached imports)

### Planned Optimizations (target: <1s for completed modules)
- [ ] **State caching** - Don't reload state.json when module status is already `completed` (estimated savings: ~0.5s)
  - Cache state in memory after first load
  - Only reload when file mtime changes
  - Skip JSON parsing for repeat calls

- [ ] **Lazy config loading** - Don't load project config.py for completed modules (estimated savings: ~0.3s)
  - Move `load_project_config()` inside module execute()
  - Only load when module actually runs
  - Completed modules don't need config

- [ ] **Skip pipeline loading** - Don't parse pipeline YAML for already-completed modules (estimated savings: ~0.2s)
  - Check module status before loading pipeline
  - Pipeline only needed when module executes

- [ ] **Parallel module discovery** - Import modules concurrently using ThreadPoolExecutor (estimated savings: ~0.2s)
  - Current: sequential import of 10 modules
  - Use threads to parallelize imports

- [ ] **Bytecode compilation** - Pre-compile modules to .pyc in production (estimated savings: ~0.1s)
  - Run `python -m compileall src/` during deployment
  - Ensure .pyc files are committed or generated at install time

**Expected total**: 4.5s → 2.5s (current) → <1s (with all optimizations)

## Experiment Manager Enhancements

- [ ] Add explicit `aborted` status to ExperimentManager modules (separate from `failed`) to mark user-interrupted runs without implying an error. Update state handling, list views, and restart logic accordingly.

## CLI meta-commands (non-flow)

- [ ] Dodać tryb pipeline bez podawania modułu: `mla --project X` uruchamia domyślny pipeline, `--from preprocess` startuje od preprocess, `--from model` od model itd.
- [ ] Dodać subkomendę `admin` (nie jako moduł) z akcjami:
  - `admin list --project X` (lista eksperymentów + statusy modułów z state.json)
  - `admin clean --project X [--artifacts|--models|--cache]` z potwierdzeniem
  - `admin gc --project X --older-than 7d` (sprzątanie starych eksperymentów)
  - `admin status --project X` (podsumowanie ostatnich runów)
- [ ] Utrzymać kompatybilność: najpierw obsłużyć meta-komendy, potem moduły, na końcu tryb pipeline.
- [ ] Zarezerwować przestrzeń nazw modułów (brak modułów `admin`, `pipeline`, `list`).
- [ ] Opcjonalny alias `mla list-experiments --project X` jako cienka nakładka na `experiment_logger.py`.
- [ ] W `admin clean` dodać ochronę przed przypadkowym usuwaniem: pokazanie targetów i prompt.
