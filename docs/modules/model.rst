model Module
============

Train a model using a YAML template, loading preprocessed data when provided, and save artifacts in a new experiment directory.

- **Depends on:** none (preprocessing is pulled by name, not dependency)
- **Key overrides:** ``model_template=<name>``, ``preprocess_template=<name>``, ``model.time_limit=<int>``, ``model.preset=<str>``, ``common.use_gpu=<bool>``
- **Profiles:** ``--profile smoke``, ``--profile dev``
- **Outputs:** model artifacts under ``experiments/<exp>/artifacts/``, leaderboard CSV, optional submission file, local CV metric in state payload

Source: ``src/mlarena/modules/model.py``  
See also: ``docs/model_templates.md`` for template configuration.
