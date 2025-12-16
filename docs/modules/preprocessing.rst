Preprocessing Modules Reference
===============================

Default preprocessing modules live in ``src/mlarena/defaults/preprocessing/`` and can be overridden per project via ``projects/kaggle/<slug>/code/preprocessing/<name>.py``. Each module implements ``fit_transform`` and returns processed train/test data plus a state dictionary saved in ``state.json``.

Sanity Check
------------

Path: ``src/mlarena/defaults/preprocessing/sanity_check.py``

.. list-table::
   :header-rows: 1
   :widths: 22 14 14 50

   * - Parameter
     - Type
     - Default
     - Description
   * - column_types_override
     - dict
     - {}
     - Explicit dtype overrides per column.
   * - min_unique_fraction
     - float
     - 0.01
     - Drop columns with fewer distinct values than this fraction.
   * - max_missing_fraction
     - float
     - 0.95
     - Drop columns with missing values above this fraction.
   * - drop_duplicates
     - bool
     - True
     - Remove duplicate rows.
   * - ignore_columns
     - list
     - []
     - Extra columns to never drop (in addition to ID/target/ignored list).

External Dataset
----------------

Path: ``src/mlarena/defaults/preprocessing/external_dataset.py``

.. list-table::
   :header-rows: 1
   :widths: 22 14 14 50

   * - Parameter
     - Type
     - Default
     - Description
   * - orig_path
     - str
     - required
     - CSV path to the external/original dataset.
   * - mode
     - str
     - align
     - ``align`` keeps intersecting columns; ``union`` keeps all.
   * - source_flag
     - str | null
     - null
     - Column name to tag source rows (0=Kaggle, 1=external).
   * - column_mapping
     - dict
     - {}
     - Rename external columns to match Kaggle column names.

Imputer
-------

Path: ``src/mlarena/defaults/preprocessing/imputer.py``

.. list-table::
   :header-rows: 1
   :widths: 22 14 14 50

   * - Parameter
     - Type
     - Default
     - Description
   * - numeric_strategy
     - str
     - mean
     - Numeric imputation strategy (mean, median, most_frequent, constant, knn, iterative).
   * - categorical_strategy
     - str
     - most_frequent
     - Categorical imputation strategy (most_frequent or constant).
   * - column_strategies
     - dict
     - {}
     - Per-column overrides of the global strategies.
   * - fill_value
     - int | float | str
     - 0
     - Value for ``constant`` strategy.
   * - knn_n_neighbors
     - int
     - 5
     - Neighbors used by ``knn`` imputer.
   * - iterative_estimator
     - str
     - bayesian_ridge
     - Estimator for the experimental iterative imputer.
   * - iterative_max_iter
     - int
     - 10
     - Iterations for the iterative imputer.
   * - treat_outliers_as_na
     - bool
     - False
     - Convert detected outliers to NA before imputing.
   * - outlier_method
     - str
     - iqr
     - Method used when ``treat_outliers_as_na`` is enabled.
   * - outlier_threshold
     - float | null
     - null
     - Threshold for outlier detection (auto-selected when null).

Missing Values Imputer
----------------------

Path: ``src/mlarena/defaults/preprocessing/missing_values_imputer.py``

.. list-table::
   :header-rows: 1
   :widths: 22 14 14 50

   * - Parameter
     - Type
     - Default
     - Description
   * - numeric_strategy
     - str
     - median
     - Numeric imputation strategy (mean, median, most_frequent, constant).
   * - categorical_strategy
     - str
     - most_frequent
     - Categorical imputation strategy (most_frequent or constant).
   * - fill_value
     - int | float | str
     - 0
     - Value for ``constant`` strategy.

Rare Category Handler
---------------------

Path: ``src/mlarena/defaults/preprocessing/rare_category_handler.py``

.. list-table::
   :header-rows: 1
   :widths: 22 14 14 50

   * - Parameter
     - Type
     - Default
     - Description
   * - min_freq
     - int
     - 10
     - Minimum absolute count to keep a category.
   * - min_freq_ratio
     - float
     - 0.01
     - Minimum relative frequency to keep a category.
   * - top_k
     - int | null
     - null
     - Keep only the top-K categories (others mapped to ``rare_label``).
   * - rare_label
     - str
     - __RARE__
     - Label used for rare categories.
   * - detect_id_like_columns
     - bool
     - True
     - Skip columns that look like IDs.
   * - id_unique_fraction_threshold
     - float
     - 0.95
     - Threshold for considering a column ID-like.
   * - protected_categorical_columns
     - list
     - []
     - Columns that should never be bucketed.

Categorical Encoder
-------------------

Path: ``src/mlarena/defaults/preprocessing/categorical_encoder.py``

.. list-table::
   :header-rows: 1
   :widths: 22 14 14 50

   * - Parameter
     - Type
     - Default
     - Description
   * - max_cardinality
     - int
     - 50
     - Maximum distinct values when reading EDA metadata.
   * - exclude_text_type
     - bool
     - False
     - Skip columns marked as Text by EDA.
   * - include_numeric_categories
     - bool
     - True
     - Treat low-cardinality numeric columns as categorical.
   * - enable_auto_detect
     - bool
     - True
     - Auto-detect numeric categorical columns when EDA data is missing.
   * - auto_detect_threshold
     - int
     - 25
     - Max unique values for auto-detected numeric categoricals.

Encoder
-------

Path: ``src/mlarena/defaults/preprocessing/encoder.py``

.. list-table::
   :header-rows: 1
   :widths: 22 14 14 50

   * - Parameter
     - Type
     - Default
     - Description
   * - encoding_method
     - str
     - one_hot
     - Encoding strategy (none, one_hot, ordinal, target_mean, target_mean_oof, catboost, hashing).
   * - include_cols
     - list | null
     - null
     - Explicit columns to encode.
   * - exclude_cols
     - list | null
     - null
     - Columns to skip during encoding.
   * - max_cardinality
     - int
     - 50
     - Ignore higher-cardinality categoricals for one-hot encoding.
   * - drop_first
     - bool
     - False
     - Drop the first category (one-hot only).
   * - handle_unknown
     - str
     - ignore
     - Strategy for unseen categories.
   * - unknown_value
     - int
     - -1
     - Value for unknown categories when applicable.
   * - hash_dim
     - int
     - 8
     - Output dimension for hashing encoder.
   * - target_encoding_smoothing
     - float
     - 1.0
     - Smoothing factor for target encoding.
   * - target_encoding_min_samples
     - int
     - 1
     - Minimum samples for target encoding.
   * - oof_folds
     - int
     - 5
     - Folds for out-of-fold target mean encoding.
   * - oof_shuffle
     - bool
     - True
     - Shuffle folds for OOF target encoding.
   * - oof_random_state
     - int
     - 42
     - Random seed for OOF encoding.
   * - oof_feature_prefix
     - str
     - mean_
     - Prefix for target mean OOF features.
   * - keep_original
     - bool
     - False
     - Preserve original categorical columns alongside encodings.

Datetime Handler
----------------

Path: ``src/mlarena/defaults/preprocessing/datetime_handler.py``

.. list-table::
   :header-rows: 1
   :widths: 22 14 14 50

   * - Parameter
     - Type
     - Default
     - Description
   * - datetime_cols
     - list
     - []
     - Columns to parse as datetimes.
   * - datetime_formats
     - dict
     - {}
     - Optional per-column strptime formats.
   * - expand_datetime_cols
     - list | null
     - null
     - Explicit columns to expand (fallbacks to detected).
   * - time_features_set
     - str
     - basic
     - Feature bundle (basic, extended, none, custom).
   * - custom_features
     - list
     - []
     - Custom datetime parts to add when ``time_features_set=custom``.
   * - cyclical_features
     - list
     - []
     - Datetime parts to encode cyclically (hour, dayofweek, month, weekofyear).
   * - time_diff_pairs
     - list
     - []
     - List of column pairs for time deltas.
   * - time_diff_default_unit
     - str
     - days
     - Unit for time differences (seconds, minutes, hours, days).
   * - drop_original_datetime
     - bool
     - False
     - Drop raw datetime columns after expansion.

Feature Engineer
----------------

Path: ``src/mlarena/defaults/preprocessing/feature_engineer.py``

.. list-table::
   :header-rows: 1
   :widths: 22 14 14 50

   * - Parameter
     - Type
     - Default
     - Description
   * - interaction_types
     - list
     - []
     - Arithmetic interactions to build (add, sub, mul, div).
   * - numeric_pairs
     - list
     - []
     - Explicit column pairs for interactions.
   * - auto_pair_numeric
     - bool
     - False
     - Automatically create numeric pairs.
   * - max_auto_pairs
     - int
     - 30
     - Cap on automatically generated pairs.
   * - poly_degree
     - int | null
     - null
     - Degree for polynomial features (>=2).
   * - poly_columns
     - list | null
     - null
     - Columns to include in polynomial expansion.
   * - poly_include_bias
     - bool
     - False
     - Include bias term in polynomial features.
   * - poly_interaction_only
     - bool
     - False
     - Use interaction-only polynomial terms.
   * - group_keys
     - list
     - []
     - Columns to group by for aggregations.
   * - group_value_cols
     - list
     - []
     - Value columns for group aggregations.
   * - aggs
     - list
     - []
     - Aggregations to compute (mean, std, min, max, count, nunique, etc.).
   * - max_generated_features
     - int
     - 200
     - Safety cap on new features created.

Feature Selector
----------------

Path: ``src/mlarena/defaults/preprocessing/feature_selector.py``

.. list-table::
   :header-rows: 1
   :widths: 22 14 14 50

   * - Parameter
     - Type
     - Default
     - Description
   * - selection_method
     - str
     - variance
     - Selection approach (variance, mi, correlation, model_importance, l1, rfe, none).
   * - k_features
     - int | null
     - null
     - Target feature count (when applicable).
   * - keep_fraction
     - float | null
     - 0.8
     - Fraction of features to keep.
   * - min_variance
     - float
     - 0.01
     - Variance threshold for the variance selector.
   * - min_importance
     - float
     - 0.001
     - Minimum importance for model-based selection.
   * - importance_model_type
     - str | null
     - lgbm
     - Model used to compute importances (lgbm, xgb, rf).
   * - n_estimators
     - int
     - 100
     - Estimators for model-based selection.
   * - max_depth
     - int
     - 5
     - Depth for model-based selection.
   * - random_state
     - int
     - 42
     - Seed for reproducibility.
   * - max_drop_fraction
     - float
     - 0.5
     - Maximum fraction of features that may be dropped.

Drift Detector
--------------

Path: ``src/mlarena/defaults/preprocessing/drift_detector.py``

.. list-table::
   :header-rows: 1
   :widths: 22 14 14 50

   * - Parameter
     - Type
     - Default
     - Description
   * - drift_metric
     - str
     - psi
     - Drift metric (psi, ks, chi2, model_auc).
   * - max_psi
     - float
     - 0.25
     - PSI threshold for drift.
   * - max_ks
     - float
     - 0.1
     - KS statistic threshold.
   * - max_pvalue
     - float
     - 0.01
     - P-value threshold for KS/chi2.
   * - min_auc
     - float
     - 0.6
     - Minimum AUC for model-based drift detection.
   * - action
     - str
     - flag_only
     - Action when drift is detected (none, drop, flag_only).
   * - max_drop_fraction
     - float
     - 0.2
     - Cap on fraction of columns that can be dropped.
   * - exclude_cols
     - list
     - []
     - Columns to skip when checking drift.
   * - random_state
     - int
     - 42
     - Seed for model-based detection.

Imbalance Handler
-----------------

Path: ``src/mlarena/defaults/preprocessing/imbalance_handler.py``

.. list-table::
   :header-rows: 1
   :widths: 22 14 14 50

   * - Parameter
     - Type
     - Default
     - Description
   * - imbalance_method
     - str
     - none
     - Strategy (none, class_weight, random_over, random_under, smote, smotenc, adasyn).
   * - sampling_strategy
     - str
     - auto
     - Sampling strategy (currently only ``auto`` is supported).
   * - use_sample_weights
     - bool
     - True
     - Whether to return weights instead of resampling where applicable.
   * - categorical_features
     - list
     - []
     - Categorical feature indices for SMOTENC.
   * - random_state
     - int
     - 42
     - Seed for stochastic samplers.

Outlier Handler
---------------

Path: ``src/mlarena/defaults/preprocessing/outlier_handler.py``

.. list-table::
   :header-rows: 1
   :widths: 22 14 14 50

   * - Parameter
     - Type
     - Default
     - Description
   * - outlier_method
     - str
     - iqr
     - Detection method (none, quantile, iqr, zscore, isolation_forest).
   * - lower_quantile
     - float
     - 0.01
     - Lower quantile cutoff (quantile method).
   * - upper_quantile
     - float
     - 0.99
     - Upper quantile cutoff (quantile method).
   * - iqr_factor
     - float
     - 1.5
     - Multiplier for IQR bounds.
   * - zscore_threshold
     - float
     - 3.0
     - Z-score threshold.
   * - isoforest_contamination
     - float
     - 0.05
     - Contamination rate for IsolationForest.
   * - action
     - str
     - clip
     - What to do with outliers (clip, set_na, flag_only).
   * - include_cols
     - list | null
     - null
     - Explicit columns to inspect.
   * - exclude_cols
     - list
     - []
     - Columns to skip.
   * - random_state
     - int
     - 42
     - Seed for IsolationForest.

Scaler
------

Path: ``src/mlarena/defaults/preprocessing/scaler.py``

.. list-table::
   :header-rows: 1
   :widths: 22 14 14 50

   * - Parameter
     - Type
     - Default
     - Description
   * - scaling_method
     - str
     - none
     - Scaling method (none, standard, minmax, robust, quantile_normal, quantile_uniform).
   * - numeric_include
     - list | null
     - null
     - Explicit numeric columns to scale.
   * - numeric_exclude
     - list
     - []
     - Numeric columns to leave untouched.
   * - log_transform
     - list
     - []
     - Columns to log-transform before scaling.
   * - clip_lower_quantile
     - float | null
     - null
     - Optional lower quantile clip.
   * - clip_upper_quantile
     - float | null
     - null
     - Optional upper quantile clip.
   * - n_quantiles
     - int
     - 1000
     - Quantiles for quantile transformer.
   * - random_state
     - int
     - 42
     - Seed for quantile transformer.

Target Transformer
------------------

Path: ``src/mlarena/defaults/preprocessing/target_transformer.py``

.. list-table::
   :header-rows: 1
   :widths: 22 14 14 50

   * - Parameter
     - Type
     - Default
     - Description
   * - target_transform
     - str
     - none
     - Target transformation (none, log1p, boxcox, yeo_johnson).
   * - clip_lower_quantile
     - float | null
     - null
     - Lower quantile clip before transform.
   * - clip_upper_quantile
     - float | null
     - null
     - Upper quantile clip before transform.
   * - shift_before_log
     - bool
     - True
     - Auto-shift targets to be positive for log/Box-Cox.
   * - shift_value
     - float | null
     - null
     - Manual shift override.
   * - standardize
     - bool
     - True
     - Standardize PowerTransformer output.

Adversarial Validation
----------------------

Path: ``src/mlarena/defaults/preprocessing/adversarial_validation.py``

.. list-table::
   :header-rows: 1
   :widths: 22 14 14 50

   * - Parameter
     - Type
     - Default
     - Description
   * - presets
     - str
     - medium_quality_faster_train
     - AutoGluon preset for the AV classifier.
   * - time_limit
     - int
     - 600
     - Training time budget for the AV model (seconds).
   * - included_model_types
     - list | null
     - null
     - Optional subset of AutoGluon model types.
   * - drop_columns
     - list
     - []
     - Columns to drop before training the AV classifier.
   * - drop_prefixes
     - list
     - []
     - Drop columns starting with any of these prefixes.
   * - weight_transform
     - str
     - odds_ratio_normalized
     - How to convert AV probabilities into weights (raw, odds_ratio, odds_ratio_capped, odds_ratio_normalized).
   * - weights_output_name
     - str
     - sample_weights.csv
     - Filename for the saved weights CSV.
   * - weight_column_name
     - str
     - __sample_weight__
     - Column name inside the weights CSV.

Identity
--------

Path: ``src/mlarena/defaults/preprocessing/identity.py``

No configuration; returns all inputs unchanged while preserving state metadata.

No-op
-----

Path: ``src/mlarena/defaults/preprocessing/noop.py``

No configuration; smoke-test module that writes a minimal report without transforming the data.
