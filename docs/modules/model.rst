model Module
============

Train a model using a YAML template, loading preprocessed data when provided, and save artifacts in a new experiment directory.

- **Depends on:** none (preprocessing is pulled by name, not dependency)
- **Key overrides:** ``model_template=<name>``, ``preprocess_template=<name>``, ``model.time_limit=<int>``, ``model.preset=<str>``, ``common.use_gpu=<bool>``, ``model.mla_retention=<bool>``
- **Profiles:** ``--profile smoke``, ``--profile dev``
- **Outputs:** model artifacts under ``experiments/<exp>/artifacts/``, leaderboard CSV, optional submission file, local CV metric in state payload

Model Cleanup (mla_retention)
----------------------------

AutoGluon models can be automatically cleaned up after training to save ~99% disk space while keeping prediction functional.

.. code-block:: bash

   uv run python scripts/mla.py model -p <project> model.mla_retention=true

**Behavior**:
- Calls ``predictor.delete_models(models_to_keep='best')`` and ``predictor.save_space()``.
- Keeps only the best model (usually the ensemble) - **prediction still works**.
- Deletes intermediate models (LightGBM, CatBoost, etc.) that aren't part of the final ensemble.
- Significantly reduces experiment size (e.g., from 30MB+ to <1MB for small datasets).

Source: ``src/mlarena/modules/model.py``  
See also: ``docs/model_templates.md`` for template configuration.
