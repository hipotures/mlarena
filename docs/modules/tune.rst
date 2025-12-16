tune Module
===========

Run Optuna-based hyperparameter search on a sampled training subset using AutoGluon.

- **Depends on:** model
- **Key flags:** ``--tune-template``, ``--n-trials``, ``--time-limit``
- **Outputs:** ``tune_result.json`` with best parameters and score under ``experiments/<exp>/artifacts/tune/``

Source: ``src/mlarena/modules/tune.py``
