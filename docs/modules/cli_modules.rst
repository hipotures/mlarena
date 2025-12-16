CLI Modules
===========

Registered MLArena modules exposed as CLI subcommands. Use ``uv run python scripts/mla.py modules`` to list what is available in your environment.

.. list-table::
   :header-rows: 1
   :widths: 18 18 18 46

   * - Module
     - Depends on
     - Key flags
     - Purpose / outputs
   * - init
     - -
     - ``--competition`` (override slug), ``--skip-download``, ``--target-column``, ``--problem-type``, ``--metric``, ``--id-column``, ``--ignore-columns``, ``--submit-probas``/``--submit-labels``
     - Scaffold a Kaggle project, download data, and persist config/state under ``experiments/init/``.
   * - eda
     - -
     - ``--eda-notes``
     - Run ydata-profiling when available; write HTML/JSON profiles and summary payload to ``experiments/eda/``.
   * - preprocess
     - -
     - ``--preprocess-template`` (required), ``--cache``
     - Execute a single preprocessing step or chain element; outputs processed train/test (and optional orig) CSVs plus shape metadata.
   * - model
     - -
     - ``--model-template`` (defaults to baseline), ``--preprocess-template``, ``--time-limit``, ``--preset``, ``--use-gpu``, ``--dev``, ``--smoke``
     - Train a model from YAML template; saves artifacts in a new ``exp-YYYYMMDD-HHMMSS`` experiment.
   * - predict
     - model
     - ``--predict-suffix``
     - Load model artifact and generate submission CSV; reuses preprocessing context when available.
   * - submit
     - predict
     - ``--skip-submit``, ``--message``, ``--auto-submit``
     - Validate submission vs. sample, optionally upload to Kaggle, and record payload.
   * - fetch-score
     - submit
     - ``--score-placeholder``
     - Fetch latest public leaderboard score via Kaggle CLI; stores payload in experiment artifacts.
   * - feat
     - -
     - ``--feat-template`` (default: identity)
     - Apply lightweight feature transformations (log1p/ratios/drops) and write feature CSVs.
   * - tune
     - model
     - ``--tune-template`` (search space), ``--n-trials``, ``--time-limit``
     - Run Optuna search on a sampled training subset using AutoGluon; saves best params/score.
   * - stack
     - predict
     - ``--prediction-files``, ``--id-column``, ``--target-column``
     - Average multiple submission files into a single stacked submission.
