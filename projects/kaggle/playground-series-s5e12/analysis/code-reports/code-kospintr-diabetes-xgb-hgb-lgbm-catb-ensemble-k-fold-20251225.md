# Analysis: Diabetes | XGB/HGB/LGBM/CatB Ensemble | K-Fold

**Author**: Ákos Pintér
**URL**: https://www.kaggle.com/code/kospintr/diabetes-xgb-hgb-lgbm-catb-ensemble-k-fold
**Votes**: 67
**Analysis Date**: 2025-12-25

## Executive Summary

This notebook implements a robust ensemble approach combining XGBoost, HistGradientBoosting, LightGBM, and CatBoost with 5-fold cross-validation. The key innovation is the multi-label stratification strategy and sophisticated preprocessing pipeline that handles different feature types (normal, logarithmic, categorical) separately. Final predictions use median aggregation across all folds and models.

## Reproducibility Assessment

**Overall Score**: HIGH

**Reason**: Complete code with clear structure, all hyperparameters explicitly defined, uses standard libraries (sklearn, xgboost, lightgbm, catboost), and provides toggle flags for different configurations. The preprocessing pipeline is well-documented and the ensemble strategy is straightforward to replicate.

## Key Techniques

### 1. Feature Engineering

**Innovation**: Multi-label stratified splitting and logarithmic transformation of physical activity feature.

**Code snippet**:
```python
# Multi-label stratification
strat_cols = ['family_history_diabetes', 'cardiovascular_history', 'ethnicity', 'diagnosed_diabetes']
trainval['multicat'] = LabelEncoder().fit_transform(trainval[strat_cols].astype(str).agg('_'.join, axis=1))
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(sss.split(trainval, trainval['multicat']))

# Logarithmic transformation for physical activity
log_columns = ['physical_activity_minutes_per_week']
log_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('log_trans', FunctionTransformer(func=np.log, feature_names_out='one-to-one')),
    ('std_scaler', RobustScaler())
])
```

**Reproducibility**: HIGH

**Impact**: Multi-label stratification ensures balanced representation of key categorical features across train/validation splits, potentially improving CV reliability. Log transformation normalizes skewed physical activity distribution.

### 2. Preprocessing

**Innovation**: Feature-type-aware preprocessing pipeline with separate handling for numerical, logarithmic, categorical, and boolean features.

**Code snippet**:
```python
# Separate pipelines for different feature types
num_columns.remove('alcohol_consumption_per_week')
minmax_columns = ['alcohol_consumption_per_week']

stdscale_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaling', RobustScaler())
])

minmaxscale_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('minmax_scaling', MinMaxScaler())
])

ordinal_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

bool_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="most_frequent"))
])

preprocessing = ColumnTransformer([
    ("std_scaling", stdscale_pipeline, num_columns),
    ("minmax_scaling", minmaxscale_pipeline, minmax_columns),
    ("logstd_scaling", log_pipeline, log_columns),
    ("ordinal", ordinal_pipeline, cat_columns),
    ("bool", bool_pipeline, bool_columns)
])
```

**Reproducibility**: HIGH

**Impact**: Tailored preprocessing for each feature type (RobustScaler for most numericals, MinMaxScaler for alcohol consumption, log transformation for physical activity) likely improves model performance by respecting feature distributions.

### 3. Model Configuration

**Models**: Ensemble of XGBoost, HistGradientBoosting, LightGBM, CatBoost

**XGBoost Hyperparameters**:
```python
xgb = xgboost.XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=0.604,
    seed=0,
    device='gpu',
    eval_metric="auc",
    enable_categorical=True,
    early_stopping_rounds=200,
    learning_rate=0.3,
    lambda=0,
    alpha=10,
    colsample_bytree=0.5,
    subsample=0.9,
    max_depth=6
)
```

**HistGradientBoosting Hyperparameters**:
```python
hgbc = HistGradientBoostingClassifier(
    scoring='roc_auc',
    class_weight='balanced',
    random_state=42,
    max_iter=1000,
    n_iter_no_change=100
)
```

**LightGBM Hyperparameters**:
```python
lgbm = lightgbm.LGBMClassifier(
    objective='binary',
    metric='auc',
    is_unbalance=True,
    random_state=42,
    device='cpu',
    verbosity=-1,
    n_estimators=2000,
    learning_rate=0.04151567000333162,
    num_leaves=93,
    max_depth=3,
    min_child_samples=97,
    subsample=0.8336810469662667,
    colsample_bytree=0.5021699121748862,
    reg_alpha=0.015640727219830758,
    reg_lambda=1.374990603296636e-06
)
```

**CatBoost Hyperparameters**:
```python
catc = CatBoostClassifier(
    eval_metric='AUC',
    auto_class_weights='Balanced',
    random_state=123,
    task_type='CPU',
    verbose=False,
    n_estimators=12000,
    depth=3,
    learning_rate=0.01,
    use_best_model=True,
    early_stopping_rounds=300
)
```

**Reproducibility**: HIGH - All hyperparameters explicitly defined

**Impact**: Diverse ensemble with complementary model types. XGB uses high learning rate (0.3) with strong regularization (alpha=10). LGBM has very precise tuned parameters suggesting extensive HPO. CatBoost uses extremely high n_estimators (12000) with early stopping.

### 4. Validation Strategy

**Type**: Stratified 5-Fold with multi-label stratification

**Code snippet**:
```python
cv_gen = StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(
    train_prepared, train['multicat'] if EXTENDED_STRAT else train_labels
)

for i, (train_index, eval_index) in enumerate(cv_gen):
    X_train, X_eval = train_prepared[train_index], train_prepared[eval_index]
    y_train, y_eval = train_labels[train_index], train_labels[eval_index]
    pipeline_train = deepcopy(pipeline)
    pipeline_train.set_params(**param)
    if est_id in est_ids_w_earlystopping:
        eval_set = {'est__eval_set': [(X_eval, np.array(y_eval))]}
        pipeline_train.fit(X_train, np.array(y_train), **eval_set)
    else:
        pipeline_train.fit(X_train, np.array(y_train))
    globals()[f'model{i+1}_{EstimatorStr[est_id]}'] = pipeline_train
```

**Reproducibility**: HIGH

**Impact**: Stratification on multi-label combination ensures balanced splits. Early stopping with eval_set prevents overfitting. Each fold model is saved separately for final ensemble.

### 5. Ensemble Strategy

**Type**: Median aggregation across all folds and all models

**Code snippet**:
```python
# Make predictions for all folds and all estimators
for i in range(5):
    for est_id in est_ids:
        model_test = globals()[f'model{i+1}_{EstimatorStr[est_id]}']
        test_pred[f'pred{i+1}_{EstimatorStr[est_id]}'] = model_test.predict_proba(test_prepared)[:,1]

# Take median of all predictions (5 folds × 4 models = 20 predictions)
submission_df['diagnosed_diabetes'] = test_pred[
    [f'pred{i+1}_{EstimatorStr[est_id]}' for est_id in est_ids for i in range(5)]
].median(axis=1)
```

**Reproducibility**: HIGH

**Impact**: Median aggregation (rather than mean) is more robust to outlier predictions. Combining 20 predictions (5 folds × 4 models) provides strong ensemble diversity.

## Implementation Recommendations

### Priority 1 (Implement first):
- **Multi-label stratification**: Create composite stratification column from key categorical features (`family_history_diabetes`, `cardiovascular_history`, `ethnicity`) to ensure balanced CV folds. This is especially important for imbalanced datasets with multiple important categorical features.

### Priority 2:
- **Feature-type-aware preprocessing**: Implement separate pipelines for different feature distributions:
  - RobustScaler for most numerical features (resistant to outliers)
  - MinMaxScaler for bounded features like alcohol_consumption
  - Log transformation + RobustScaler for skewed features like physical_activity
  - OrdinalEncoder for categorical features (preserves ordinality if it exists)

### Priority 3:
- **Median ensemble aggregation**: Instead of mean averaging, use median of all fold predictions across all models. More robust to outlier models and complements the RobustScaler philosophy.

## MLA Integration Notes

**Preprocessing Module**: `diabetes_feature_aware_pipeline.py`
- Implement ColumnTransformer with separate pipelines for:
  - Standard numerical features: SimpleImputer(median) + RobustScaler
  - Bounded numerical features: SimpleImputer(median) + MinMaxScaler
  - Log-normal features: SimpleImputer(median) + Log + RobustScaler
  - Categorical features: SimpleImputer(most_frequent) + OrdinalEncoder(handle_unknown=-1)
  - Boolean features: SimpleImputer(most_frequent)
- Accept configuration to toggle PCA on numerical features

**Preprocessing Module**: `diabetes_multilabel_stratification.py`
- Create composite stratification column from specified categorical features
- Use LabelEncoder on concatenated string representation
- Return as artifact for use in CV splitting

**Model Template**: `ensemble-xgb-hgb-lgbm-catb.yaml`
```yaml
model: ensemble_boosting
config:
  models:
    - type: xgboost
      params:
        learning_rate: 0.3
        max_depth: 6
        alpha: 10
        lambda: 0
        colsample_bytree: 0.5
        subsample: 0.9
        enable_categorical: true
        early_stopping_rounds: 200
    - type: histgradientboosting
      params:
        max_iter: 1000
        n_iter_no_change: 100
        scoring: roc_auc
        class_weight: balanced
    - type: lightgbm
      params:
        n_estimators: 2000
        learning_rate: 0.041516
        num_leaves: 93
        max_depth: 3
        min_child_samples: 97
        subsample: 0.833681
        colsample_bytree: 0.502170
        reg_alpha: 0.015641
        reg_lambda: 0.000001
    - type: catboost
      params:
        n_estimators: 12000
        depth: 3
        learning_rate: 0.01
        use_best_model: true
        early_stopping_rounds: 300
  cv:
    n_folds: 5
    stratify_multicolumn: true
  ensemble:
    method: median
    aggregate_folds: true
```

## Code Snippets for Reference

```python
# Multi-label stratification implementation
def create_multilabel_strat(df, cols):
    """Create stratification column from multiple categorical features"""
    return LabelEncoder().fit_transform(
        df[cols].astype(str).agg('_'.join, axis=1)
    )

# Feature-type-aware preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler, MinMaxScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
import numpy as np

def build_preprocessing_pipeline(num_cols, minmax_cols, log_cols, cat_cols, bool_cols):
    """Build feature-type-aware preprocessing pipeline"""
    std_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('scaler', RobustScaler())
    ])

    minmax_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('scaler', MinMaxScaler())
    ])

    log_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('log', FunctionTransformer(func=np.log, feature_names_out='one-to-one')),
        ('scaler', RobustScaler())
    ])

    ordinal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="most_frequent")),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    bool_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="most_frequent"))
    ])

    return ColumnTransformer([
        ("std_scaling", std_pipeline, num_cols),
        ("minmax_scaling", minmax_pipeline, minmax_cols),
        ("log_scaling", log_pipeline, log_cols),
        ("ordinal", ordinal_pipeline, cat_cols),
        ("bool", bool_pipeline, bool_cols)
    ])

# Median ensemble aggregation
def median_ensemble_predict(models, X):
    """Aggregate predictions using median across all models and folds"""
    predictions = []
    for fold_models in models:
        for model in fold_models:
            predictions.append(model.predict_proba(X)[:, 1])
    return np.median(predictions, axis=0)
```

## Caveats and Limitations

- **External dataset handling**: Code includes flag to merge external diabetes dataset but disabled by default (`ADD_EXTERN_DATA = False`). If enabled, concatenates with train data and resets IDs, which may introduce data leakage if distributions differ.

- **Computational requirements**: Training 4 models × 5 folds = 20 total models. CatBoost with 12000 estimators is particularly expensive. Total training time not reported but likely 30-60+ minutes on CPU.

- **Feature selection**: No explicit feature selection performed. Relies on model's native feature importance. Some features may have very low importance (visible in feature importance plot) but are retained.

- **Hyperparameter origins**: LightGBM parameters are extremely precise (e.g., `learning_rate=0.04151567000333162`), suggesting they came from HPO (possibly Optuna). However, the HPO code is not included, making these parameters dataset-specific and potentially not transferable.

- **GPU dependency**: XGBoost configured with `device='gpu'` but will fall back to CPU if GPU unavailable. Performance difference may be significant for large datasets.

- **Class weight handling**: Different approaches across models:
  - XGB: `scale_pos_weight=0.604` (hardcoded)
  - HGB: `class_weight='balanced'` (automatic)
  - LGBM: `is_unbalance=True` (automatic)
  - CatBoost: `auto_class_weights='Balanced'` (automatic)

  The hardcoded value for XGB may not transfer to datasets with different class distributions.

- **Validation split**: Uses single 80/20 split for initial train/val separation. While 5-fold CV is used for training, the validation scores reported are on a single held-out set, which may have higher variance than CV scores.

- **Median vs Mean**: Using median aggregation assumes some model/fold predictions may be outliers. This is defensive but may underutilize consistently strong models if they deviate from the median.

## Dataset-Specific Insights

- **Physical activity feature**: Treated specially with log transformation + RobustScaler, suggesting right-skewed distribution with potential zero values (adds +1 minute when external data enabled).

- **Alcohol consumption**: Uses MinMaxScaler instead of RobustScaler, suggesting it's a bounded variable (likely 0 to some maximum).

- **Categorical encoding**: Uses OrdinalEncoder rather than OneHotEncoder, suggesting author believes categorical features have some inherent ordering or wants to keep dimensionality low. This works well for tree-based models but may not capture non-ordinal relationships.

- **Boolean features**: Minimal preprocessing (just imputation), suggesting they're already in 0/1 format and don't need scaling.
