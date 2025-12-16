model Module
============

Train a model using a YAML template, loading preprocessed data when provided, and save artifacts in a new experiment directory.

- **Depends on:** none (preprocessing is pulled by name, not dependency)
- **Key flags:** ``--model-template``, ``--preprocess-template``, ``--time-limit``, ``--preset``, ``--use-gpu``, ``--dev``, ``--smoke``
- **Outputs:** model artifacts under ``experiments/<exp>/artifacts/``, leaderboard CSV, optional submission file, local CV metric in state payload

Source: ``src/mlarena/modules/model.py``  
See also: ``docs/model_templates.md`` for template configuration.
