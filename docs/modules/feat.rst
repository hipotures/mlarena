feat Module
===========

Apply lightweight feature transformations (log1p, ratios, column drops) defined in a feature template.

- **Depends on:** none
- **Key flags:** ``--feat-template`` (default: identity)
- **Outputs:** ``train_features.csv`` and ``test_features.csv`` under ``experiments/<exp>/artifacts/feat/`` plus metadata JSON

Source: ``src/mlarena/modules/feat.py``
