Model Implementations
=====================

Default model modules reside in ``src/mlarena/defaults/models/``. Project-specific models can be added under ``projects/kaggle/<slug>/code/models/`` and will override global names when they match.

Interface
---------

Each model module should expose:

``train(train_df, val_df, config, artifacts=None)`` returning a predictor (or predictor, summary)

and optionally ``predict(predictor, test_df, config, artifacts=None)`` when custom prediction logic is needed.

Available defaults
------------------

.. list-table::
   :header-rows: 1
   :widths: 22 20 58

   * - Name
     - Source
     - Description
   * - autogluon_baseline
     - ``src/mlarena/defaults/models/autogluon_baseline.py``
     - AutoGluon Tabular baseline with template-driven presets, time limits, GPU flags, optional sample weights, and optional external/orig merge (``merge_orig`` in model template config).
   * - av_classifier
     - ``src/mlarena/defaults/models/av_classifier.py``
     - Lightweight AutoGluon classifier used by adversarial validation preprocessing to score covariate shift between train and test.

Tips for custom models
----------------------

- Place your module in ``projects/kaggle/<slug>/code/models/<name>.py`` and reference it from a model template via ``model: <name>``.
- Consume preprocessing artifacts through the ``artifacts`` dict (for example, sample weights or external/orig datasets) when relevant. For the default AutoGluon model you can disable orig merging by setting ``merge_orig: false`` in the model template.
- Keep dependencies local to the model file to avoid global import overhead; heavy imports should live inside ``train``/``predict`` functions.
