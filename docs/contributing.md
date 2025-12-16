# Contributing Guide

## Workflow

- Work from the repository root (`scripts/mla.py` is the entry point).
- Keep experiments and data out of commits (`data/*.csv`, `AutogluonModels/`, `experiments/` artifacts).
- Prefer small, focused changes; follow commit prefixes such as `feat:`, `fix:`, or `docs:`.

## Coding standards

- Python: PEP 8 formatting and Google-style docstrings (`Args`, `Returns`, `Raises`, `Examples` where useful).
- Modules self-register via `ModuleRegistry.register`; preserve module names and dependencies.
- Avoid logic changes in preprocessing defaults unless intentionally fixing a bug; these are shared across projects.

## Testing

- Run unit tests when present: `uv run python -m pytest`.
- For a fast integration check, run a smoke pipeline: `uv run python scripts/mla.py --project <slug> --smoke --skip-submit`.
- Ensure new docs or schema changes do not break existing templates or project overrides.

## Documentation

- Update Markdown docs in `docs/` and keep the `docs/index.rst` toctree in sync.
- Ensure all new modules are documented in both `.md` files and `.rst` reference docs.
- Mark superseded docs with a `.DEPR` suffix rather than deleting them outright.

## Submitting changes

- Keep PR descriptions concise and note any manual steps (e.g., Kaggle credential setup).
- If git commits are created automatically by auto-flow, verify messages before pushing.
- When adding new templates or preprocessing steps, document parameters in `docs/modules/preprocessing.rst` or `docs/model_templates.md`.
