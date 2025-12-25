# Analysis: S5E12 | Exploring FE + Optuna Ensemble

**Author**: Oleksii Zhukov
**URL**: https://www.kaggle.com/code/zhukovoleksiy/s5e12-exploring-fe-optuna-ensemble
**Votes**: 51
**Rank**: 9
**Analysis Date**: 2025-12-25

## Executive Summary

This notebook implements a sophisticated ensemble approach combining 25+ diverse models (GBDT variants, neural networks, random forests) with Optuna-optimized weighted blending. Key innovations include: (1) aggressive outlier removal, (2) extensive original dataset feature engineering (mean/count target encoding per column), (3) medical domain features (pulse pressure, BMI-waist interactions), (4) dual neural network architectures with focal loss, and (5) cross-validated Optuna weight optimization for ensemble blending. The approach achieves ensemble lift through extreme model diversity and sophisticated weighting.

## Reproducibility Assessment

**Overall Score**: MEDIUM

**Reason**: The notebook contains complete, runnable code with standard libraries (LightGBM, XGBoost, CatBoost, TensorFlow, Optuna, sklearn). However, several factors reduce reproducibility: (1) heavy reliance on GPU acceleration (CUDA for XGBoost/CatBoost, TensorFlow for NNs), (2) very long training times with 25+ models and Optuna trials, (3) aggressive outlier removal that drops variable numbers of rows per column (non-deterministic order), (4) some hyperparameters appear manually tuned without documented search process. The code is well-structured but requires significant computational resources and careful adaptation.

## Key Techniques

### 1. Feature Engineering

**Innovation**: Three-tier feature engineering approach: (1) aggressive quantile-based outlier removal, (2) original dataset statistical aggregations (mean/count per unique value), (3) medical domain features based on clinical risk factors.

**Code snippet**:
```python
# Outlier removal - removes top 50 extreme values per numeric column
def remove_top50_outliers_quantile(df, low_q=0.01, high_q=0.99):
    df_clean = df.copy()
    cols = df_clean.select_dtypes(include=["int", "float"]).columns

    for col in cols:
        q_low = df_clean[col].quantile(low_q)
        q_high = df_clean[col].quantile(high_q)

        outliers_below = df_clean[df_clean[col] < q_low].index
        outliers_above = df_clean[df_clean[col] > q_high].index
        outliers = list(outliers_below) + list(outliers_above)
        outliers_to_remove = outliers[:50]
        df_clean.drop(outliers_to_remove, inplace=True)

    return df_clean

# Original dataset aggregations
BASE = [col for col in train.columns if col not in ['id', target]]
ORIG = []

for col in BASE:
    # MEAN target encoding from original dataset
    mean_map = orig.groupby(col)[target].mean()
    new_mean_col_name = f"orig_mean_{col}"
    train = train.merge(mean_map, on=col, how='left')
    test = test.merge(mean_map, on=col, how='left')
    ORIG.append(new_mean_col_name)

    # COUNT of occurrences in original dataset
    new_count_col_name = f"orig_count_{col}"
    count_map = orig.groupby(col).size().reset_index(name=new_count_col_name)
    train = train.merge(count_map, on=col, how='left')
    test = test.merge(count_map, on=col, how='left')
    ORIG.append(new_count_col_name)

# Medical domain features
def create_medical_features(df):
    df = df.copy()

    # Pulse Pressure: arterial stiffness indicator
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']

    # Mean Arterial Pressure
    df['map_pressure'] = df['diastolic_bp'] + (df['pulse_pressure'] / 3)

    # BMI-Waist interaction (visceral fat proxy)
    df['bmi_waist_interaction'] = df['bmi'] * df['waist_to_hip_ratio']

    # High BP flag
    df['high_bp_flag'] = ((df['systolic_bp'] >= 130) |
                          (df['diastolic_bp'] >= 85)).astype(int)

    # Obesity flag
    df['obesity_flag'] = (df['bmi'] >= 30).astype(int)

    # Age groups
    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 35, 50, 65, 100],
        labels=[0, 1, 2, 3]
    ).astype(int)

    return df
```

**Reproducibility**: MEDIUM - The outlier removal is well-defined but drops rows (affects downstream). Original dataset aggregations are straightforward. Medical features use standard clinical thresholds.

**Impact**: High potential - original dataset feature engineering is proven in this competition. Medical features leverage domain knowledge. However, outlier removal is aggressive and may hurt generalization.

### 2. Target Encoding (OOF)

**Innovation**: Implements proper out-of-fold target encoding with Bayesian smoothing to prevent leakage while encoding categorical variables.

**Code snippet**:
```python
class TargetEncoderOOF:
    def __init__(self, n_splits=5, smooth=10, random_state=42):
        self.n_splits = n_splits
        self.smooth = smooth
        self.random_state = random_state
        self.map_dict = {}

    def fit_transform(self, X, y, cat_cols):
        X_encoded = X.copy()

        if y.nunique() > 20:
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        else:
            kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for col in cat_cols:
            global_mean = y.mean()
            agg = X.groupby(col)[y.name].agg(['count', 'mean'])
            counts = agg['count']
            means = agg['mean']

            # Smoothing: (mean * count + global * smooth) / (count + smooth)
            smooth_mean = (means * counts + global_mean * self.smooth) / (counts + self.smooth)
            self.map_dict[col] = smooth_mean

            X_encoded[f"TE_{col}"] = np.nan

            for train_idx, val_idx in kf.split(X, y):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr = y.iloc[train_idx]

                fold_agg = X_tr.groupby(col)[y.name].agg(['count', 'mean'])
                fold_counts = fold_agg['count']
                fold_means = fold_agg['mean']

                fold_smooth = (fold_means * fold_counts + global_mean * self.smooth) / (fold_counts + self.smooth)
                X_encoded.loc[val_idx, f"TE_{col}"] = X_val[col].map(fold_smooth)

            X_encoded[f"TE_{col}"] = X_encoded[f"TE_{col}"].fillna(global_mean)

        return X_encoded
```

**Reproducibility**: HIGH - Standard technique with clear implementation. Smoothing parameter (10) is reasonable default.

**Impact**: Medium-High - Properly implemented target encoding without leakage is valuable, especially combined with label/one-hot encoding.

### 3. Ensemble Strategy

**Innovation**: Optuna-based weight optimization across 25+ diverse models including GBDT variants, neural networks, random forests. Models selected for maximum diversity (different architectures, hyperparameters, boosting strategies).

**Code snippet**:
```python
class OptunaWeights:
    def __init__(self, random_state, n_trials=100):
        self.study = None
        self.weights = None
        self.random_state = random_state
        self.n_trials = n_trials

    def _objective(self, trial, y_true, y_preds):
        weights = [trial.suggest_float(f"weight{n}", 1e-15, 1)
                   for n in range(len(y_preds))]
        weighted_pred = np.average(np.array(y_preds).T, axis=1, weights=weights)
        score = roc_auc_score(y_true, weighted_pred)
        return score

    def fit(self, y_true, y_preds):
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        pruner = optuna.pruners.MedianPruner()
        self.study = optuna.create_study(
            sampler=sampler,
            pruner=pruner,
            study_name="OptunaWeights",
            direction="maximize",
        )
        objective_partial = partial(self._objective, y_true=y_true, y_preds=y_preds)
        self.study.optimize(objective_partial, n_trials=self.n_trials)
        self.weights = [self.study.best_params[f"weight{n}"]
                        for n in range(len(y_preds))]

# Model zoo includes:
# - Naked baselines (xgb_base, lgbm_base, cat_2, cat_4)
# - Manually tuned variants (xgb, lgbm, cat with different params)
# - Diverse architectures (rf_base, et_base, hist_grad)
# - Random/aggressive models (xgb_aggressive, lgbm_random)
# - Neural networks (keras_mlp2, keras_mlp3 with focal loss)
# - DART boosting (xgb_dart)
# - Extra trees mode (lgbm_xt)
```

**Reproducibility**: MEDIUM - Optuna optimization is reproducible with fixed seed, but requires significant compute (100 trials × 5 folds × 25 models). GPU requirement for CatBoost/XGBoost/TensorFlow reduces portability.

**Impact**: High - Ensemble diversity is excellent. Optuna optimization is superior to simple averaging. However, complexity and training time are substantial.

### 4. Neural Network Models

**Innovation**: Two custom Keras architectures - one with Wide & Deep structure and residual connections, another using focal loss for class imbalance.

**Code snippet**:
```python
def binary_focal_loss(alpha=0.8, gamma=2.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        p_t = tf.where(tf.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1.0), alpha, 1.0 - alpha)
        loss_val = -alpha_t * tf.pow(1.0 - p_t, gamma) * tf.math.log(p_t)
        return tf.reduce_mean(loss_val)
    return loss

class KerasTabularMLP3:
    def _build_model(self):
        inputs = keras.Input(shape=(self.input_dim,), dtype="float32")
        x = layers.BatchNormalization()(inputs)

        h1 = layers.Dense(256, activation="relu", kernel_initializer="he_normal")(x)
        h1 = layers.BatchNormalization()(h1)
        h1 = layers.Dropout(0.2)(h1)

        h2 = layers.Dense(128, activation="relu", kernel_initializer="he_normal")(h1)
        h2 = layers.BatchNormalization()(h2)
        h2 = layers.Dropout(0.2)(h2)

        h2 = layers.Concatenate()([h1, h2])  # Skip connection

        h3 = layers.Dense(64, activation="relu", kernel_initializer="he_normal")(h2)
        h3 = layers.BatchNormalization()(h3)
        h3 = layers.Dropout(0.1)(h3)

        outputs = layers.Dense(1, activation="sigmoid")(h3)

        model = keras.Model(inputs=inputs, outputs=outputs)

        fl = binary_focal_loss(alpha=self.alpha, gamma=self.gamma)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.lr),
            loss=fl,
            metrics=[keras.metrics.AUC(name="auc")],
        )
        return model
```

**Reproducibility**: MEDIUM - Requires TensorFlow/Keras with GPU. Early stopping makes training somewhat non-deterministic. Architectures are well-defined.

**Impact**: Medium - Neural networks can add diversity to GBDT-heavy ensembles, but tabular data typically favors tree models. Focal loss is appropriate for imbalanced data.

### 5. Model Diversity Strategy

**Innovation**: Explicit focus on model diversity through varied architectures, hyperparameters, and training strategies (DART, Extra Trees mode, aggressive learning rates, high randomness).

**Key model categories**:
```python
# Baseline models (standard configs)
xgb_base, lgbm_base, cat_2, cat_4

# Manually tuned (from other competitions)
xgb, lgbm, cat, lgbm_1, xgb_1, cat_1

# Random forests for stability
rf_base, et_base

# Special boosting modes
lgbm_xt (extra_trees=True)
xgb_dart (dropout regularization)

# Aggressive/random variants
xgb_aggressive (lr=0.55, depth=3)
lgbm_random (subsample=0.20)

# Shallow models
cat_shallow (depth=3)

# Sklearn alternative
hist_grad, hist_balanced

# Neural networks
keras_mlp2, keras_mlp3
```

**Reproducibility**: MEDIUM - All models use standard libraries, but mixture of CPU/GPU models and varied frameworks increases complexity.

**Impact**: High - Model diversity is the foundation of strong ensembles. This approach maximizes architectural and hyperparameter diversity.

## Implementation Recommendations

### Priority 1 (Implement first):
**Original Dataset Aggregations** - The mean/count features from the original dataset are straightforward to implement and have proven value in this competition. This is compatible with current MLA framework using `orig_df` in preprocessing.

```python
# In preprocessing module
def fit_transform(train_df, val_df, test_df, config, orig_df=None):
    if orig_df is None:
        return train_df, val_df, test_df, orig_df, {}

    target = config.get('target_column', 'diagnosed_diabetes')
    base_cols = [col for col in train_df.columns if col != target]

    for col in base_cols:
        # Mean target encoding from orig
        mean_map = orig_df.groupby(col)[target].mean()
        mean_col = f"orig_mean_{col}"
        train_df[mean_col] = train_df[col].map(mean_map).fillna(orig_df[target].mean())
        val_df[mean_col] = val_df[col].map(mean_map).fillna(orig_df[target].mean())
        test_df[mean_col] = test_df[col].map(mean_map).fillna(orig_df[target].mean())

        # Count from orig
        count_map = orig_df.groupby(col).size()
        count_col = f"orig_count_{col}"
        train_df[count_col] = train_df[col].map(count_map).fillna(0)
        val_df[count_col] = val_df[col].map(count_map).fillna(0)
        test_df[count_col] = test_df[col].map(count_map).fillna(0)

    return train_df, val_df, test_df, orig_df, {}
```

### Priority 2:
**Medical Domain Features** - Implement pulse pressure, MAP, BMI-waist interaction, and clinical threshold flags. These are dataset-specific but grounded in medical knowledge.

### Priority 3:
**OOF Target Encoding** - The TargetEncoderOOF class is well-implemented and could be adapted as a standalone preprocessing module. More sophisticated than simple target encoding.

### Priority 4 (Advanced):
**Optuna Ensemble Optimization** - For competitions where ensemble is critical, implement Optuna-based weight optimization. Requires significant compute but can extract additional performance from model diversity.

## MLA Integration Notes

**Preprocessing Module**: `preprocess-diabetes-orig-stats-gap.py`
- Extract mean/count aggregations from original dataset
- Add medical domain features (pulse_pressure, map_pressure, bmi_waist_interaction)
- Implement clinical threshold flags (high_bp_flag, obesity_flag, age_group)
- Ensure fillna strategies match notebook

**Preprocessing Module**: `preprocess-diabetes-target-encoder-oof.py`
- Port TargetEncoderOOF class
- Apply to categorical columns after other preprocessing
- Configure n_splits=5, smooth=10 as defaults

**Model Template**: `ensemble-optuna-weighted.yaml`
- Define base model list (start with 5-10 diverse models)
- Configure Optuna trials (start with 50, increase if beneficial)
- Set up cross-validation strategy
- Note: Requires custom model implementation, not easily templated

**Caveats**:
- GPU requirement for optimal performance (CatBoost/XGBoost CUDA, TensorFlow)
- Long training time with many models (consider subset for experimentation)
- Outlier removal strategy may need validation on other datasets
- Memory usage optimization (reduce_mem_usage) useful for large ensembles

## Code Snippets for Reference

```python
# Memory optimization for large datasets
def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and col_type.name != 'category':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
    return df

# Unified CV runner with framework-specific handling
def run_cv_optuna_blend(X_train, y_train, X_test, base_models, splitter,
                         random_state_list, n_splits, optuna_n_trials=100):
    for name, model in models.items():
        if 'lgb' in name:
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        elif 'xgb' in name:
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)
        elif 'cat' in name:
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                     use_best_model=True, early_stopping_rounds=200, verbose=0)
        elif any(x in name for x in ['rf', 'et', 'hist']):
            model.fit(X_tr, y_tr)  # No eval_set
        else:  # Custom keras/NN
            model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)

        val_pred = model.predict_proba(X_val)[:, 1]
        test_pred = model.predict_proba(X_test)[:, 1]

    # Optuna optimization per fold
    optweights = OptunaWeights(random_state=seed, n_trials=optuna_n_trials)
    blended_val = optweights.fit_predict(y_val.values, oof_preds_fold)
```

## Caveats and Limitations

### Dataset-Specific Assumptions:
- Medical feature definitions assume standard clinical ranges for blood pressure, BMI, age groups
- Original dataset aggregations assume same feature names and distributions between train/orig
- Outlier removal (top 50 per column at 1%/99% quantiles) is aggressive and may not generalize

### Computational Requirements:
- **GPU strongly recommended**: CatBoost GPU mode, XGBoost CUDA, TensorFlow GPU
- **Training time**: 25 models × 5 folds × Optuna trials (100) = very long (estimate 4-8 hours on GPU)
- **Memory**: Large ensemble requires significant RAM, especially with many features after encoding

### Portability Issues:
- TensorFlow version compatibility (protobuf < 4.21.0 requirement suggests older TF version)
- GPU-specific code paths (device="cuda", task_type="GPU") fail gracefully on CPU but much slower
- Some hyperparameters appear manually tuned without documented search (cat_tuned, xgb_aggressive)

### What Might Not Transfer:
- Aggressive outlier removal may hurt on cleaner datasets or different distributions
- Medical domain features are specific to diabetes health indicators
- Optuna weight optimization benefits diminish with fewer/less diverse models
- Neural network architectures designed for this specific feature count (after encoding ~70+ features)
- Some model configurations appear competition-specific (tuned on this leaderboard)

### Reproducibility Concerns:
- Outlier removal order-dependent (list concatenation before [:50] slice)
- Neural network training has inherent randomness despite seed setting
- Optuna TPE sampler has some non-determinism in parallel execution
- Early stopping in GBDT models creates minor variance across runs

### Integration Challenges:
- Ensemble approach requires custom model orchestration, not easily templated in MLA
- Mixed CPU/GPU model zoo requires careful resource management
- OOF target encoding needs integration with existing CV strategy
- Memory optimization may conflict with feature engineering expectations
