preprocess-tune Module
======================

Optuna-driven preprocessing tuning (FAST evaluation only). Builds preprocessing pipelines from the super-chain + search spaces and evaluates them with a FAST model template.

- **Depends on:** EDA summary (``experiments/eda/artifacts/eda/eda_summary.json`` must exist)
- **Key overrides:** ``preprocess_tune.model_template=<name>``, ``preprocess_tune.n_trials=<int>``, ``preprocess_tune.optuna_workers=<int>``, ``preprocess_tune.study_name=<name>``
- **Outputs:** trial artifacts under ``projects/kaggle/<slug>/experiments/optuna_<study>/trial_XXXX/`` and best-chain templates in ``projects/kaggle/<slug>/templates/preprocess/``

Basic usage
-----------

.. code-block:: bash

   uv run python scripts/mla.py pre tune --project Titanic \
     preprocess_tune.model_template=mcts \
     preprocess_tune.n_trials=20

Key parameters
--------------

- ``preprocess_tune.super_chain``: path to super-chain YAML (default: ``conf/preprocess/super_chain_optuna.yaml``)
- ``preprocess_tune.study_name``: Optuna study name
- ``preprocess_tune.n_trials``: number of trials
- ``preprocess_tune.optuna_workers``: parallel Optuna workers (`n_jobs`)
- ``preprocess_tune.max_trial_sec``: hard timeout per trial
- ``preprocess_tune.allow_heavy_steps`` / ``preprocess_tune.allow_heavy_variants``: heavy gating
- ``preprocess_tune.max_features_out``: hard cap for feature count
- ``preprocess_tune.storage_url``: Optuna storage (SQLite by default)
- ``preprocess_tune.optuna_storage_timeout``: SQLite connection timeout (seconds)
- ``preprocess_tune.model_template``: FAST model template (if omitted, uses ``evaluation.model`` from the super-chain)
- ``preprocess_tune.seed``: trial seed (default: ``common.seed``)
- ``preprocess_tune.quiet_preprocess_panel``: suppress PREPROCESS panels during tuning
- ``preprocess_tune.quiet_model_panel``: suppress MODEL panels during tuning
- ``preprocess_tune.model_verbosity``: AutoGluon verbosity override (default from ``model.verbosity``)
- ``preprocess_tune.model_cleanup`` / ``preprocess_tune.ag_cleanup``: remove ``model_fast`` and ``*_processed.csv.gz`` after each trial
- ``preprocess_tune.cleanup_processed``: remove only ``*_processed.csv.gz`` after each trial

Quiet panels
------------

To suppress PREPROCESS / PREPROCESS COMPLETED panels during tuning:

.. code-block:: bash

   uv run python scripts/mla.py pre tune --project Titanic \
     preprocess_tune.quiet_preprocess_panel=true \
     preprocess_tune.quiet_model_panel=true

You can also set ``preprocess.quiet_preprocess_panel=true`` (global for preprocess panels).

Source: ``src/mlarena/modules/preprocess_tune.py``
