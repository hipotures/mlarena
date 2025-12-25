# Analysis: S5E12 | EDA+XGB - Competition Starter

**Author**: masayakawamata
**URL**: https://www.kaggle.com/code/masayakawamata/s5e12-eda-xgb-competition-starter
**Score**: 0.69896
**Votes**: 174
**Analysis Date**: 2025-12-25

## Executive Summary

This notebook implements a well-structured XGBoost baseline for the diabetes prediction competition, achieving 0.69896 public LB. The key innovation is using the external original diabetes dataset **not for data augmentation** (which risks covariate shift), but as an **external reference for feature engineering** via target encoding and frequency encoding. The approach includes robust CV-aware target encoding with smoothing, native categorical feature support in XGBoost, and memory optimization techniques.

## Reproducibility Assessment

**Overall Score**: HIGH

**Reason**: The notebook is exceptionally well-documented with clear markdown explanations, complete code snippets, and reproducible techniques using only standard libraries (pandas, numpy, xgboost, sklearn). All feature engineering steps are explicit, and the custom TargetEncoder class is fully implemented. No external dependencies on private datasets or pre-generated CSV files. The only hardware-specific element is the optional `device: 'cuda'` parameter which can be easily switched to CPU.

## Key Techniques

### 1. Feature Engineering

**Primary Innovation: External Dataset as Reference (Not Augmentation)**

The notebook makes a critical insight: the original diabetes dataset has different distributions than train/test (sharper peaks, shifts). Instead of concatenating (which introduces covariate shift), it uses the original data to create **statistical reference features**:

**Two Feature Types Created**:

1. **orig_mean_{feature}**: Target encoding from original dataset
   - Interpretation: Real-world diabetes probability for this category/value
   - Serves as a robust, leakage-free risk indicator

2. **orig_count_{feature}**: Frequency encoding from original dataset
   - Interpretation: How common this value is in medical records
   - Captures prevalence information

**Code Implementation**:

```python
ORIG = []

for col in BASE:  # BASE = all features except id and target
    # MEAN (Target Encoding from Original Dataset)
    mean_map = orig.groupby(col)[TARGET].mean()
    new_mean_col_name = f"orig_mean_{col}"
    mean_map.name = new_mean_col_name

    train = train.merge(mean_map, on=col, how='left')
    test = test.merge(mean_map, on=col, how='left')
    ORIG.append(new_mean_col_name)

    # COUNT (Frequency Encoding from Original Dataset)
    new_count_col_name = f"orig_count_{col}"
    count_map = orig.groupby(col).size().reset_index(name=new_count_col_name)

    train = train.merge(count_map, on=col, how='left')
    test = test.merge(count_map, on=col, how='left')
    ORIG.append(new_count_col_name)

print(f'{len(ORIG)} ORIG Features Created.')
```

**Filling Missing Values** (for unseen categories):

```python
for col in ORIG:
    if 'mean' in col:
        train[col] = train[col].fillna(orig[TARGET].mean())
        test[col] = test[col].fillna(orig[TARGET].mean())
    else:  # count features
        train[col] = train[col].fillna(0)
        test[col] = test[col].fillna(0)
```

**Impact**: Creates 2 × len(BASE) features (~40+ features if BASE has 20 features)

---

**Secondary Innovation: CV-Aware Target Encoding with Smoothing**

Applied to numerical features with > 2 unique values to avoid noise from binary features.

**Custom TargetEncoder Class Features**:
- Internal K-Fold CV to prevent leakage (out-of-fold encoding)
- Empirical Bayes smoothing for rare categories
- Multiple aggregation functions (mean, count, std, etc.)
- Proper handling of unseen categories

**Key Implementation Details**:

```python
class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_encode, aggs=['mean'], cv=5, smooth='auto', drop_original=False):
        self.cols_to_encode = cols_to_encode
        self.aggs = aggs
        self.cv = cv
        self.smooth = smooth  # 'auto' uses Empirical Bayes
        self.drop_original = drop_original
        self.mappings_ = {}
        self.global_stats_ = {}

    def fit_transform(self, X, y):
        """Uses internal CV to prevent leakage"""
        # First, fit on entire dataset for global mappings
        self.fit(X, y)

        # Initialize empty DataFrame for encoded features
        encoded_features = pd.DataFrame(index=X.index)

        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val = X.iloc[val_idx]

            temp_df_train = X_train.copy()
            temp_df_train['target'] = y_train

            for col in self.cols_to_encode:
                for agg_func in self.aggs:
                    new_col_name = f'TE_{col}_{agg_func}'

                    # Calculate fold-specific global stat
                    fold_global_stat = y_train.agg(agg_func)

                    # Calculate category-specific stats
                    mapping = temp_df_train.groupby(col)['target'].agg(agg_func)

                    # Apply smoothing only for 'mean' aggregation
                    if agg_func == 'mean':
                        counts = temp_df_train.groupby(col)['target'].count()

                        m = self.smooth
                        if self.smooth == 'auto':
                            # Empirical Bayes smoothing
                            variance_between = mapping.var()
                            avg_variance_within = temp_df_train.groupby(col)['target'].var().mean()
                            if variance_between > 0:
                                m = avg_variance_within / variance_between
                            else:
                                m = 0  # No smoothing if no variance

                        # Smoothing formula: (count * category_mean + m * global_mean) / (count + m)
                        smoothed_mapping = (counts * mapping + m * fold_global_stat) / (counts + m)
                        encoded_values = X_val[col].map(smoothed_mapping)
                    else:
                        encoded_values = X_val[col].map(mapping)

                    # Store encoded values for validation fold
                    encoded_features.loc[X_val.index, new_col_name] = encoded_values.fillna(fold_global_stat)

        # Merge with original DataFrame
        X_transformed = X.copy()
        for col in encoded_features.columns:
            X_transformed[col] = encoded_features[col]

        if self.drop_original:
            X_transformed.drop(columns=self.cols_to_encode, inplace=True)

        return X_transformed
```

**Usage in Training Loop**:

```python
# Select TE Columns (Exclude binary features)
TE_COLS = [col for col in NUMS if train[col].nunique() > 2]

# Apply during training
TE = TargetEncoder(cols_to_encode=TE_COLS, cv=5, smooth='auto',
                   aggs=['mean', 'count'], drop_original=False)
X_train = TE.fit_transform(X_train, y_train)
X_val = TE.transform(X_val)
X_test_fold = TE.transform(X_test_fold)
```

### 2. Preprocessing

**No Complex Preprocessing**:
- Dataset is clean (zero missing values)
- No outlier removal
- No scaling (tree-based model)

**Memory Optimization**:

```python
def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and col_type.name != 'category':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    return df

X = reduce_mem_usage(X)
test = reduce_mem_usage(test)
```

**Categorical Feature Handling** (XGBoost Native Support):

```python
for c in CATS:
    # 1. Factorize (convert to integers)
    combined = pd.concat([X_train[c], X_val[c], X_test_fold[c]])
    combined_encoded, _ = combined.factorize()

    # 2. Assign back to DataFrame
    X_train[c] = combined_encoded[:len(X_train)]
    X_val[c] = combined_encoded[len(X_train):len(X_train)+len(X_val)]
    X_test_fold[c] = combined_encoded[len(X_train)+len(X_val):]

    # 3. Cast to Category dtype for XGBoost native support
    X_train[c] = X_train[c].astype('category')
    X_val[c] = X_val[c].astype('category')
    X_test_fold[c] = X_test_fold[c].astype('category')
```

### 3. Model Configuration

**Model**: XGBoost with native categorical support

**Key Hyperparameters**:

```python
xgb_params = {
    'n_estimators': 20000,           # Large max, relies on early stopping
    'learning_rate': 0.01,            # Low learning rate for stability
    'max_depth': 4,                   # Shallow trees (default is 6)
    'subsample': 0.8,                 # Row sampling
    'colsample_bytree': 0.8,          # Column sampling
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'auc',             # ROC AUC for binary classification
    'device': 'cuda',                 # GPU acceleration (optional)
    'enable_categorical': True        # Native categorical feature support
}

model = XGBClassifier(**xgb_params)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=200,        # Stop if no improvement for 200 rounds
    verbose=500                       # Print every 500 rounds
)
```

**Notable Parameter Choices**:
- **max_depth=4**: Shallower than default (6) to prevent overfitting
- **learning_rate=0.01**: Much lower than default (0.3) for fine-grained learning
- **enable_categorical=True**: Leverages XGBoost's native categorical handling (introduced in v1.5+)
- **early_stopping_rounds=200**: Conservative stopping to ensure convergence

### 4. Validation Strategy

**CV Type**: Stratified 5-Fold Cross-Validation

**Implementation**:

```python
from sklearn.model_selection import StratifiedKFold

# Initialize arrays
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test))

# Stratified K-Fold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    # Split data
    X_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx]
    X_test_fold = test[FEATURES].copy()

    # Apply feature engineering (TE + categorical encoding)
    # ...

    # Train model
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=200)

    # Predict
    val_preds = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_preds
    test_preds += model.predict_proba(X_test_fold)[:, 1] / kf.get_n_splits()

    fold_score = roc_auc_score(y_val, val_preds)
    print(f"Fold {fold+1} AUC: {fold_score:.5f}")

print(f"OOF AUC: {roc_auc_score(y, oof_preds):.5f}")
```

**Key Features**:
- Stratified splits preserve class distribution
- Out-of-fold predictions for unbiased validation
- Test predictions averaged across 5 folds
- Memory management with `gc.collect()`

**No Sample Weighting**: All samples treated equally

### 5. External Dataset Usage

**Critical Insight**: Do NOT concatenate train + orig for data augmentation

**Reasoning** (from EDA):
- Original dataset shows distinct distributional differences (sharper peaks, shifts)
- Simply concatenating introduces covariate shift
- Train/Test distributions are nearly identical

**Strategy**: Use original data as **external reference** for feature engineering

**Implementation**:
1. Load orig dataset separately
2. Compute statistics on orig (target mean, counts per category)
3. Merge statistics as new features into train/test
4. Fill missing values (for categories not in orig) with global statistics

**Features Created**:
- `orig_mean_{feature}`: Target probability in original dataset
- `orig_count_{feature}`: Frequency in original dataset

**Code** (see Section 1 for full implementation):

```python
# Example for one feature
mean_map = orig.groupby('age')[TARGET].mean()
train['orig_mean_age'] = train['age'].map(mean_map).fillna(orig[TARGET].mean())
test['orig_mean_age'] = test['age'].map(mean_map).fillna(orig[TARGET].mean())

count_map = orig.groupby('age').size()
train['orig_count_age'] = train['age'].map(count_map).fillna(0)
test['orig_count_age'] = test['age'].map(count_map).fillna(0)
```

## Implementation Recommendations

### Priority 1: External Dataset Reference Features (HIGHEST VALUE)

**Why**: This is the key innovation that distinguishes this approach from naive concatenation. It respects distributional differences while extracting domain knowledge.

**How to Implement in MLA**:

Create preprocessing module: `external_dataset_reference.py`

```python
def fit_transform(train_df, val_df, test_df, config, orig_df=None):
    """
    Create reference features from external original dataset.

    Config:
        target_column: str (default: 'diagnosed_diabetes')
        feature_columns: list (default: all except id/target)
        aggs: list (default: ['mean', 'count'])
    """
    if orig_df is None:
        # If no orig dataset provided, skip
        return train_df, val_df, test_df, orig_df, {}

    target = config.get('target_column', 'diagnosed_diabetes')
    feature_cols = config.get('feature_columns', None)
    if feature_cols is None:
        feature_cols = [c for c in train_df.columns if c not in ['id', target]]

    aggs = config.get('aggs', ['mean', 'count'])

    new_features = []

    for col in feature_cols:
        if 'mean' in aggs:
            # Target encoding from orig
            mean_map = orig_df.groupby(col)[target].mean()
            new_col = f'orig_mean_{col}'
            train_df[new_col] = train_df[col].map(mean_map).fillna(orig_df[target].mean())
            val_df[new_col] = val_df[col].map(mean_map).fillna(orig_df[target].mean())
            test_df[new_col] = test_df[col].map(mean_map).fillna(orig_df[target].mean())
            new_features.append(new_col)

        if 'count' in aggs:
            # Frequency encoding from orig
            count_map = orig_df.groupby(col).size()
            new_col = f'orig_count_{col}'
            train_df[new_col] = train_df[col].map(count_map).fillna(0)
            val_df[new_col] = val_df[col].map(count_map).fillna(0)
            test_df[new_col] = test_df[col].map(count_map).fillna(0)
            new_features.append(new_col)

    state = {'new_features': new_features}
    return train_df, val_df, test_df, orig_df, state
```

**Template**: `preprocess/external-ref.yaml`

```yaml
chain: [external_dataset_reference]
config:
  aggs: [mean, count]
```

### Priority 2: CV-Aware Target Encoding with Smoothing

**Why**: Robust target encoding prevents leakage and handles rare categories gracefully. The implementation is production-ready with proper CV and smoothing.

**How to Implement in MLA**:

Create preprocessing module: `target_encoder_cv.py`

Copy the TargetEncoder class from the notebook (see Section 1 for full code).

**Usage in preprocessing**:

```python
def fit_transform(train_df, val_df, test_df, config, orig_df=None):
    """
    Apply CV-aware target encoding to numerical features.

    Config:
        target_column: str
        min_unique_values: int (default: 3, exclude binary features)
        cv_folds: int (default: 5)
        smooth: 'auto' or float
        aggs: list (default: ['mean', 'count'])
    """
    from your_target_encoder import TargetEncoder

    target = config.get('target_column', 'diagnosed_diabetes')
    min_unique = config.get('min_unique_values', 3)
    cv_folds = config.get('cv_folds', 5)
    smooth = config.get('smooth', 'auto')
    aggs = config.get('aggs', ['mean', 'count'])

    # Select numerical features with > min_unique values
    nums = train_df.select_dtypes(include=['int', 'float']).columns.tolist()
    te_cols = [c for c in nums if c != target and train_df[c].nunique() > min_unique]

    if not te_cols:
        return train_df, val_df, test_df, orig_df, {}

    # Fit on train
    te = TargetEncoder(cols_to_encode=te_cols, cv=cv_folds, smooth=smooth,
                       aggs=aggs, drop_original=False)
    train_df = te.fit_transform(train_df, train_df[target])
    val_df = te.transform(val_df)
    test_df = te.transform(test_df)

    state = {'te_cols': te_cols, 'aggs': aggs}
    return train_df, val_df, test_df, orig_df, state
```

**Template**: `preprocess/target-encode.yaml`

```yaml
chain: [target_encoder_cv]
config:
  min_unique_values: 3
  cv_folds: 5
  smooth: auto
  aggs: [mean, count]
```

### Priority 3: XGBoost Configuration with Native Categorical Support

**Why**: The hyperparameter choices are well-tuned for this dataset, and native categorical support simplifies preprocessing.

**How to Implement in MLA**:

**Template**: `model/xgb-categorical-shallow.yaml`

```yaml
model: xgboost_classifier
config:
  n_estimators: 20000
  learning_rate: 0.01
  max_depth: 4
  subsample: 0.8
  colsample_bytree: 0.8
  random_state: 42
  eval_metric: auc
  enable_categorical: true
  early_stopping_rounds: 200
  verbose: 500
```

**Model Implementation** (modify existing XGBoost model):

Ensure categorical features are properly encoded before passing to XGBoost:

```python
def train(train_df, val_df, config, artifacts=None):
    # Identify categorical columns
    cat_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Factorize and convert to category dtype
    for col in cat_cols:
        combined = pd.concat([train_df[col], val_df[col]])
        combined_encoded, _ = combined.factorize()
        train_df[col] = combined_encoded[:len(train_df)].astype('category')
        val_df[col] = combined_encoded[len(train_df):].astype('category')

    # Train XGBoost with enable_categorical=True
    model = XGBClassifier(**config)
    model.fit(
        train_df.drop(columns=['target']), train_df['target'],
        eval_set=[(val_df.drop(columns=['target']), val_df['target'])],
        early_stopping_rounds=config.get('early_stopping_rounds', 200),
        verbose=config.get('verbose', 100)
    )

    return model
```

## MLA Integration Notes

### Preprocessing Modules

**Module 1**: `external_dataset_reference.py`
- Requires: orig_df artifact
- Creates: orig_mean_* and orig_count_* features
- Dependencies: Must run after `external_dataset` module that loads orig

**Module 2**: `target_encoder_cv.py`
- Requires: train/val/test split
- Creates: TE_*_mean and TE_*_count features
- Dependencies: None (can run standalone)
- Note: Exclude binary features (nunique <= 2)

**Module 3**: `categorical_factorizer.py` (optional)
- Prepares categorical features for XGBoost native support
- Can be integrated into model training instead

### Model Template

**Template**: `xgb-categorical-shallow.yaml`
- Uses shallow trees (max_depth=4) to prevent overfitting
- Low learning rate (0.01) for stable convergence
- Native categorical support enabled

### Execution Order

```bash
# Pipeline
uv run python scripts/mla.py init -p playground-series-s5e12
uv run python scripts/mla.py eda -p playground-series-s5e12
uv run python scripts/mla.py preprocess -p playground-series-s5e12 -t external-dataset
uv run python scripts/mla.py preprocess -p playground-series-s5e12 -t external-ref
uv run python scripts/mla.py preprocess -p playground-series-s5e12 -t target-encode
uv run python scripts/mla.py model -p playground-series-s5e12 -t xgb-categorical-shallow
uv run python scripts/mla.py predict -p playground-series-s5e12
uv run python scripts/mla.py submit -p playground-series-s5e12
```

## Code Snippets for Reference

### Complete External Reference Feature Engineering

```python
# Load datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
orig = pd.read_csv('diabetes_dataset.csv')

TARGET = 'diagnosed_diabetes'
BASE = [col for col in train.columns if col not in ['id', TARGET]]

ORIG = []

for col in BASE:
    # Target encoding from original dataset
    mean_map = orig.groupby(col)[TARGET].mean()
    new_mean_col = f"orig_mean_{col}"
    train[new_mean_col] = train[col].map(mean_map).fillna(orig[TARGET].mean())
    test[new_mean_col] = test[col].map(mean_map).fillna(orig[TARGET].mean())
    ORIG.append(new_mean_col)

    # Frequency encoding from original dataset
    count_map = orig.groupby(col).size()
    new_count_col = f"orig_count_{col}"
    train[new_count_col] = train[col].map(count_map).fillna(0)
    test[new_count_col] = test[col].map(count_map).fillna(0)
    ORIG.append(new_count_col)

print(f'Created {len(ORIG)} features from original dataset')
```

### XGBoost Training with Native Categorical Support

```python
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Prepare categorical features
cat_cols = ['gender', 'smoker']  # example
for col in cat_cols:
    train[col], _ = train[col].factorize()
    train[col] = train[col].astype('category')
    test[col], _ = test[col].factorize()
    test[col] = test[col].astype('category')

# XGBoost parameters
params = {
    'n_estimators': 20000,
    'learning_rate': 0.01,
    'max_depth': 4,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'auc',
    'enable_categorical': True,
    'random_state': 42
}

# 5-Fold CV
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(kf.split(train, train[TARGET])):
    X_train, y_train = train.iloc[train_idx], train[TARGET].iloc[train_idx]
    X_val, y_val = train.iloc[val_idx], train[TARGET].iloc[val_idx]

    model = XGBClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=200,
              verbose=500)

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(test)[:, 1] / 5

print(f'OOF AUC: {roc_auc_score(train[TARGET], oof_preds):.5f}')
```

## Caveats and Limitations

### Dataset-Specific Assumptions

1. **Clean Data**: The notebook assumes zero missing values. If applied to other datasets, add missing value handling.

2. **Low Cardinality**: Features have low unique counts (discrete values). Target encoding works well here, but may behave differently on high-cardinality features.

3. **Balanced Classes**: No sample weighting or class balancing techniques used. May need adjustment for highly imbalanced datasets.

### Computational Requirements

1. **GPU Optional**: `device='cuda'` is used but not required. On CPU, training will be slower but still feasible.

2. **Memory Usage**: The memory optimization function is crucial for large datasets. Without it, memory usage could be 2-3x higher.

3. **Training Time**: With 20000 estimators and early stopping, each fold takes ~10-15 minutes on GPU, ~30-60 minutes on CPU.

### Reproducibility Considerations

1. **XGBoost Version**: Native categorical support requires XGBoost >= 1.5.0. Older versions will fail.

2. **Random Seed**: Fixed at 42 throughout. Changing seeds will produce slightly different results due to CV splits and model training randomness.

3. **Feature Order**: The TargetEncoder class uses internal CV, so results are deterministic given the same seed, but feature engineering order matters.

### Potential Issues

1. **Target Encoding Leakage**: The notebook correctly uses out-of-fold encoding, but if this is modified or simplified, leakage could occur.

2. **Overfitting Risk**: With 20000 estimators, there's risk of overfitting if early stopping fails. Monitor validation scores carefully.

3. **Original Dataset Dependency**: If the original dataset is not available or has different feature names, the external reference features cannot be created. The code should gracefully handle this case.

### What Might Not Generalize

1. **Shallow Trees (max_depth=4)**: This works for this dataset's discrete features, but continuous features or complex interactions may need deeper trees.

2. **Low Learning Rate (0.01)**: Optimized for this competition. Other datasets may converge faster with higher learning rates.

3. **No Feature Selection**: All features (including engineered ones) are used. On noisier datasets, feature selection may improve performance.

---

**End of Report**
