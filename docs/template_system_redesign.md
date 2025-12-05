# Template System Redesign (preprocess + model, no fallback)

Goal: decouple preprocessing from modeling, with first-class templates for each, a simple Rich-based CLI configurator that only emits CLI flags (no hidden state), and experiment state fully captured in `state.json`.

## Core concepts
- Two template sources:
  - `templates/preprocess.yaml` – defines FE pipelines (modules in `code/preprocessing/*.py`), their params, and optional caching behavior.
  - `templates/model.yaml` – defines model configs (modules in `code/models/*.py`) similar to today’s `templates.yaml`.
- No `templates.yaml` fallback once migrated; runner errors if required template not found.
- Experiment state stores both selections (`preprocess_template`, `model_template`) plus resolved configs and CLI overrides.

## Preprocess interface
- Each preprocess template points to a module exporting:
  - `fit_transform(train_df, val_df, test_df, config)` -> `(train_fe, val_fe, test_fe, state_dict)`
  - `transform(df, state_dict, config)` -> `df_fe`
- Runner saves:
  - `features/train_fe.parquet`, `features/val_fe.parquet`, `features/test_fe.parquet`
  - `features/state.pkl` (or similar) with `state_dict`
- Runner flags:
  - `--preprocess-template <name>`
  - `--use-preprocessed` (skip recompute; load parquet + state)
- Stage=predict loads `state.pkl` and transforms test without refit.

## Model interface
- Unchanged: model module gets already-processed `train_df/val_df/test_df`.
- Runner flags:
  - `--model-template <name>`
  - Existing overrides (`--time-limit`, `--preset`, `--use-gpu`, etc.) still apply.
  - NEW: `--model-name <module>` lets you override the module from the template (train/all only; predict always uses the saved config). Use this to reuse the same template params with a different model file (e.g., switch to `autogluon_shiftaware` without cloning templates).

## Runner flow (ml_runner)
1. Load preprocess template (if provided); fit/transform; cache outputs; optionally skip when `--use-preprocessed`.
2. Load model template; train/predict on processed frames.
3. Record in `state.json`:
   - `preprocess_template`, resolved preprocess config, cache paths
   - `model_template`, resolved model config
   - CLI overrides and flags (e.g., `skip-submit`)

## Rich configurator (UX)
- Pure flag builder: user picks preprocess + model template from lists (read from YAML), toggles checkboxes (`skip-submit`, `use-preprocessed`, etc.), and the tool prints/runs the equivalent CLI command.
- No hidden state beyond normal runner; everything reproducible via printed command and `state.json`.

## Files & layout
- New template files:
  - `templates/preprocess.yaml`
  - `templates/model.yaml`
- Preprocess modules live in `code/preprocessing/` (mirrors model modules in `code/models/`).
- Artifacts per experiment:
  - `experiments/<exp-id>/features/` (parquet + state)
  - `experiments/<exp-id>/artifacts/` (model)
  - `experiments/<exp-id>/state.json` (records both templates + overrides)

Global vs project resolution
----------------------------
- Templates: merged global (`config/templates/*.yaml`) + project (`projects/kaggle/<proj>/templates/*.yaml`); local entries override with a warning. If a name exists in both, local wins, but we print the override info.
- Models: loaded from project `code/models/` or global `config/code/models/`. If the same filename exists in both, the runner fails (no shadowing). Use a new filename for project-only variants; change globals only for cross-project behavior.
- Preprocessing: loaded from project `code/preprocessing/` or global `config/code/preprocessing/` (e.g., `identity.py`). Same-name clashes also fail to avoid ambiguity. Keep shared utilities global; add project-specific pipelines under the project path with unique names.

## Migration plan (no fallback)
1. Implement dual-template loading in runner; remove use of `templates.yaml`.
2. Add CLI flags for preprocess/model templates and `--use-preprocessed`.
3. Add caching of processed data + state.
4. Add Rich configurator that only builds/runs the CLI.
5. Migrate existing templates into `templates/model.yaml`; create baseline preprocess templates in `templates/preprocess.yaml`.
6. Delete legacy template path and code after parity tests.

## TODO: declarative pipeline + moduły (do wdrożenia)

Co trzeba dopisać, żeby nowe moduły dało się podłączać bez grzebania w kodzie:

1) **Deklaratywny pipeline (YAML)**  
   - Plik: `config/pipelines/<name>.yaml`.  
   - Zawartość: kolejność modułów, wymagane artefakty wej./wyj., flagi sterujące (np. skip-submit), mapowanie nazwa→handler.  
   - `experiment_manager`/CLI: ładuje wskazany pipeline, waliduje dostępność modułów i odpala sekwencję; usuń sztywne `MODULES` z kodu.

2) **Kontrakt modułu (inputs/outputs)**  
   - Każdy moduł opisany w YAML (patrz pkt 3) ma zadeklarowane: wymagane wejścia (ścieżki/klucze), produkowane artefakty (ścieżki pod `experiments/<id>/...`), oraz eventy/stany zapisywane do `state.json`.  
   - Preprocess: przenieść cache do `experiments/<id>/features/` (train/val/test + `state.pkl`), w `predict` używać `transform(state.pkl)` zamiast `fit_transform`/cache zewnętrznego; zapis ścieżek w state.json zgodnie z kontraktem.  
   - Model/predict: jasno zdefiniować gdzie lądują modele (`experiments/<id>/artifacts/`) i predykcje (`experiments/<id>/predictions/`), zamiast rozrzuconych lokalizacji.

3) **Rejestr modułów (YAML)**  
   - Plik: `config/modules.yaml` (lub per-projekt `templates/modules.yaml`).  
   - Mapa `nazwa_modułu` → `{handler: code path + entrypoints (run(), validate()), inputs, outputs}`.  
   - `experiment_manager` przy starcie pipeline sprawdza, czy wszystkie moduły mają wpis w rejestrze, importuje handler i wywołuje go według kontraktu.  
   - Dodanie nowego modułu = dodać plik z funkcjami + wpisać go do rejestru; brak zmian w kodzie orkiestratora.
