Template resolution: global vs. project
======================================

Goal
----
- Allow global templates in `config/templates/*.yaml` to be reused by all projects.
- Allow per-project overrides/augments via `projects/kaggle/<proj>/templates/*.yaml`.
- When names collide, project-level template wins, but emit a warning so users know a global template is shadowed.

Desired behavior
----------------
1. Load order:
   - Global: `config/templates/model.yaml`, `config/templates/preprocess.yaml` (and any other global template files).
   - Local: `projects/kaggle/<proj>/templates/model.yaml`, `.../preprocess.yaml` (if present).
2. Merge rule:
   - Start with globals.
   - For each template name present in local, overwrite the global entry with the local one.
   - If the same name exists in both, log a warning: `Template '<name>' overridden by project/<proj> template`.
3. Missing files:
   - If local file is absent, use only global.
   - If neither exists, raise a clear error.
4. Validation:
   - After merge, validate schema (must contain `templates:` mapping, each entry has `model`/`config` or `module`/`config` depending on file).
   - Fail fast with file path in the error.
5. Caching/paths:
- Do not mix caches; keep preprocessing/model artifacts under the project’s `preprocess_cache/` and `experiments/`.
- Template resolution should not rewrite paths; only the merge layer changes.
- Global templates are read-only from project perspective; projects may only override via local files, not mutate globals.

Warning policy
--------------
- On duplicate names, emit a single warning per name per run, mentioning both source files.
- If a project overrides and keeps the same name intentionally, the warning is informational (not fatal).
- Set `KAGGLE_TEMPLATE_NO_WARN=1` to suppress warnings in batch runs if needed.

Model code resolution (global vs. project)
------------------------------------------
- Global model code lives in `config/code/models/` (e.g., `autogluon_baseline.py`).
- Project-specific models live in `projects/kaggle/<proj>/code/models/`.
- The loader checks for a local file first; if the same filename exists both locally and globally, it fails with an error (no shadowing). Use a new filename for project-only variants.
- Do not edit global model files when iterating on a single project; add/modify a project-local file instead. Touch global code only when you need a cross-project change.

Preprocessing code resolution (global vs. project)
--------------------------------------------------
- Global preprocessing utilities live in `config/code/preprocessing/` (e.g., `identity.py`).
- Project-specific preprocessing lives in `projects/kaggle/<proj>/code/preprocessing/`.
- If a module name exists both globally and locally, execution fails to avoid ambiguity; choose a distinct name for project-only pipelines.
- Keep `identity.py` and other shared utilities global; do not duplicate or edit them locally unless the change is meant to be shared across projects.

Implementation sketch (future work)
-----------------------------------
- Extract template loading into a helper `load_templates(kind, project)`:
  - `kind` in {model, preprocess}.
  - Read global file if it exists; read local file if it exists.
  - Merge dicts with local taking precedence; accumulate warnings.
  - Return merged templates + warnings.
- Update runners (`ml_runner`, etc.) to call the helper instead of reading a single file.
- Add a unit test with a fake project:
  - Global defines `foo`, `bar`; local defines `foo` (different) and `baz`.
  - Assert final set: `foo` from local, `bar` from global, `baz` from local, plus warning for `foo`.
