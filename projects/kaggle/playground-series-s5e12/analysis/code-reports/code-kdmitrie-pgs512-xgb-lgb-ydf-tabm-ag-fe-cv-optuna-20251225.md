# Analysis: PGS512: XGB+LGB+YDF+TABM+AG: FE, CV & OPTUNA

**Author**: Konstantin Dmitriev
**URL**: https://www.kaggle.com/code/kdmitrie/pgs512-xgb-lgb-ydf-tabm-ag-fe-cv-optuna
**Votes**: 82
**Analysis Date**: 2025-12-25

## Executive Summary

This notebook implements a sophisticated ensemble approach combining 6+ different models (XGBoost, LightGBM, YDF, TabM, RealMLP, AutoGluon) with extensive feature engineering and hill-climbing ensemble optimization. The author performs adversarial validation to identify distribution differences between datasets and makes informed decisions about feature dropping and dataset merging.

## Reproducibility Assessment

**Overall Score**: MEDIUM

**Reason**: The notebook is well-structured with clear code and comprehensive implementation. However, many predictions are loaded from pre-computed files (`/kaggle/input/pgs512-predictions/`) rather than computed live, which means full reproduction requires running multiple training sessions. The core techniques (FE, adversarial validation, ensemble) are fully reproducible, but the complete pipeline would require significant compute time (4+ hours for AutoGluon alone). Optuna hyperparameter optimization is disabled by default (`CFG.optuna = False`), using pre-tuned parameters instead.

## Key Techniques

### 1. Feature Engineering

**Innovation**: Multi-layered feature engineering combining binning, digit extraction, rounding, count encoding, and target encoding with sophisticated dataset handling.

**Code snippet**:
```python
# Bin features - quantile-based discretization
def add_bin_features(self, numeric: list[str], q: int=5, suffix: str='_bin'):
    new_cols = []
    for col in numeric:
        self.df[col + suffix], _ = pd.qcut(self.df[col], q=q, labels=False,
                                           retbins=True, duplicates="drop")
        new_cols.append(col + suffix)
    return new_cols

# Digit features - extract individual digits from numeric values
def add_digit_features(self, numeric: list[str], suffix: str='_dig'):
    new_cols = []
    for col in numeric:
        sp = self.df[col].astype(str).str.split('.', expand=True).fillna('')
        b = max(sp[0].astype(str).str.len())
        a = max(sp[1].astype(str).str.len()) if 1 in sp.columns else 0

        for k in range(1 - b, a + 1):
            new_col = f'{col}{suffix}{k}'
            self.df[new_col] = ((self.df[col] * 10**k) % 10).fillna(-1).astype("int8")
            new_cols.append(new_col)
    return new_cols

# Round features
def add_round_features(self, numeric: list[str], round: list=[-1, 0], suffix: str='_round'):
    new_cols = []
    for col in numeric:
        for r in round:
            new_col = f'{col}{suffix}{r}'
            self.df[new_col] = self.df[col].round(r)
            new_cols.append(new_col)
    return new_cols

# Count encoding
def count_encoder_categoric(self, categoric: list[str], suffix: str='_cnt') -> list[str]:
    new_cols = []
    for col in categoric:
        col_cnt = self.df.groupby(col)[self.target].count().astype('int32')
        col_cnt.name = col + suffix
        new_cols.append(col_cnt.name)
        self.df = self.df.merge(col_cnt, on=col, how='left')
    return new_cols
```

**Reproducibility**: HIGH

**Impact**: The feature engineering creates multiple representations of the same data (binned, digit-level, rounded, count-encoded). This multi-view approach helps different models capture different patterns. Particularly innovative is the digit extraction which can capture fine-grained numerical patterns.

### 2. Adversarial Validation & Feature Selection

**Innovation**: Uses adversarial validation not just to detect distribution shift, but to identify specific features causing the shift and make informed decisions about dropping them.

**Code snippet**:
```python
def adversarial_validation(df1, df2):
    target = '_dataset'
    xgb_params = CFG.xgb_params7_optuna.copy()
    xgb_params['n_estimators'] = 1000
    k_fold_strategy = RepeatedStratifiedKFold(n_splits=2, n_repeats=1,
                                              random_state=CFG.kfold_random_state)

    df1 = df1.drop(CFG.target, axis=1)
    df2 = df2.drop(CFG.target, axis=1)
    df1[target] = 1.0
    df2[target] = 0.0
    df = pd.concat((df1, df2))

    cv = CV(k_fold_strategy, df, target=target, proba=True)
    _, oof_preds = cv.fit(get_model=lambda: XGBWrapper(xgb_params), provide_val_data=True)
    return roc_auc_score(df[target], oof_preds), cv

# Identify features causing distribution shift
oof_score_train_test, cv_tt = adversarial_validation(df_train, df_test)
oof_score_train_orig, cv_tro = adversarial_validation(df_train, df_orig)

# Extract feature importances to find problematic features
for name, model in [('train-test', cv_tt.models[0].model)]:
    dd = pd.DataFrame((zip(df_train.columns, model.feature_importances_)),
                      columns=['feature', 'importance'])
    print(dd.sort_values(by='importance', ascending=False)[:6])

# Results showed: physical_activity_minutes_per_week, triglycerides,
# cholesterol_total, alcohol_consumption_per_week cause shift
# -> Drop these features if CFG.drop_inhomogeneous_features = True
```

**Reproducibility**: HIGH

**Impact**: TRAIN<->TEST AUC = 0.62 (moderate shift), TRAIN<->ORIG AUC = 0.91 (severe shift). This analysis led to dropping 3-5 inhomogeneous features, which likely improved generalization by removing features that don't transfer well between train and test distributions.

### 3. Fold-Based Target Encoding

**Innovation**: Implements proper target encoding with sklearn's TargetEncoder to prevent leakage, applied on-the-fly during cross-validation folds.

**Code snippet**:
```python
from sklearn.preprocessing import TargetEncoder

def fold_transform_train(df, model) -> pd.DataFrame:
    if CFG.merge_original_dataset:
        df = pd.concat((df, df_orig), axis=0, ignore_index=True)

    model.te = {}
    for col in cat1 + cat2:  # Categorical columns
        model.te[col] = TargetEncoder(target_type='continuous', smooth=1,
                                     cv=10, shuffle=True, random_state=42)
        df[col] = model.te[col].fit_transform(df[col].to_frame(),
                                             df[CFG.target]).astype('float32')
    return df

def fold_transform_test(df, model) -> pd.DataFrame:
    for col in cat1 + cat2:
        df[col] = model.te[col].transform(df[col].to_frame()).astype('float32')
    return df

# Used in CV class
cv = CV(k_fold_strategy, df_train, target=CFG.target, proba=True,
        fold_transform_train=fold_transform_train,
        fold_transform_test=fold_transform_test)
```

**Reproducibility**: HIGH

**Impact**: Target encoding is one of the most powerful techniques for categorical features, especially with tree-based models. The proper implementation (fit on train fold, transform on validation fold) prevents leakage while capturing target-feature relationships.

### 4. Model Configuration - Optuna-Tuned Hyperparameters

**Model**: XGBoost (primary), LightGBM, YDF, TabM, RealMLP, AutoGluon

**Key Hyperparameters (XGBoost - Optuna tuned)**:
```python
xgb_params7_optuna = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.008559367757686604,
    'max_depth': 5,
    'subsample': 0.93,
    'colsample_bytree': 0.19,
    'seed': SEED,
    'device': 'cuda',
    'grow_policy': 'lossguide',
    'alpha': 2.0,  # L1 regularization
    'lambda': 0.73,  # L2 regularization
    'min_child_weight': 5,
    'max_bin': 512,
    'n_estimators': 20000,
    'early_stopping_rounds': 300
}
```

**Key Hyperparameters (LightGBM - Optuna tuned)**:
```python
lgb_params7_optuna = {
    'random_state': SEED,
    'n_estimators': 20_000,
    'metric': 'AUC',
    'objective': 'binary',
    'learning_rate': 0.005230191195442491,
    'max_depth': 3,
    'min_child_samples': 128,
    'subsample': 0.86,
    'colsample_bytree': 0.48,
    'num_leaves': 519,
    'reg_alpha': 0.28,
    'reg_lambda': 7.92,
    'max_bin': 202,
    'device': 'gpu'
}
```

**AutoGluon Configuration**:
```python
ag_presets = 'best_quality'
ag_time_limit = 4*3600  # 4 hours
# Later reduced to 3000s and 1200s for different variants
```

**Code snippet**:
```python
class XGBWrapper:
    def __init__(self, xgb_params: dict) -> None:
        self.model = xgb.XGBClassifier(**xgb_params)

    def fit(self, x, y, val_x, val_y):
        self.model.fit(x, y, eval_set=[(val_x, val_y)], verbose=300)

    def predict_proba(self, x):
        return self.model.predict_proba(x, iteration_range=(0, self.model.best_iteration + 1))
```

**Reproducibility**: HIGH (if using pre-tuned params), LOW (if re-running Optuna with 200 trials)

**Impact**: Hyperparameters are carefully tuned via Optuna with 200 trials. Notable: very low learning rate (0.0086 for XGB, 0.0052 for LGB) with many estimators (20k), high regularization (alpha=2.0), low colsample (0.19) suggesting emphasis on robust, well-regularized models.

### 5. Validation Strategy

**Type**: RepeatedStratifiedKFold (5 splits × 2 repeats = 10 folds total)

**Code snippet**:
```python
k_fold_strategy = RepeatedStratifiedKFold(
    n_splits=CFG.kfold_n_splits,     # 5
    n_repeats=CFG.kfold_n_repeats,   # 2
    random_state=CFG.kfold_random_state
)

class CV:
    def fit(self, get_model: Callable, provide_val_data: bool=False):
        self.models = []
        oof_preds = np.zeros_like(self.y)

        for fold, (train_index, test_index) in enumerate(self.k_fold_strategy.split(self.y, self.y)):
            model = get_model()

            x_train = self.fold_transform_train(self.df.iloc[train_index].copy(), model)
            y_train = x_train.pop(self.target)

            x_test = self.fold_transform_test(self.df.iloc[test_index].copy(), model)
            y_test = x_test.pop(self.target)

            if provide_val_data:
                model.fit(x_train, y_train, x_test, y_test)
            else:
                model.fit(x_train, y_train)

            oof_preds[test_index] += model.predict_proba(x_test)[:, 1]
            self.models.append(model)

        return self.models, oof_preds
```

**Reproducibility**: HIGH

**Impact**: Using repeated CV (2 repeats) provides more robust estimates than single 5-fold CV, at the cost of 2x compute time. Each model is trained 10 times, providing better generalization and more stable predictions for ensembling.

### 6. Ensemble Strategy - Hill Climbing Optimization

**Innovation**: Instead of simple averaging, uses iterative hill climbing to find optimal weights for combining 7 different model predictions (multiple XGB variants, LGB, YDF, TabM, RealMLP, 2 AutoGluon variants).

**Code snippet**:
```python
def hc(oof_preds, positive_weights=True,
       weights_choice=(-0.5, 0.5, -0.01, 0.01),
       initial_weights=None, n_epochs=20):
    oof_preds_np = np.array(oof_preds).T

    weights = np.zeros((len(test_preds), 1)) if initial_weights is None else initial_weights
    score = 0

    for _ in range(n_epochs):
        for p, _ in enumerate(oof_preds):
            for w in weights_choice:
                # Create new weights
                test_w = weights.copy()
                test_w[p] += w
                if positive_weights:
                    test_w = np.clip(test_w, 0, 100)

                # Make a prediction on OOF data
                oof_test_pred = oof_preds_np @ test_w
                oof_test_score = roc_auc_score(df_train[CFG.target], oof_test_pred)

                if oof_test_score > score:
                    weights, score = test_w, oof_test_score
    return weights, score

# Two-stage optimization: coarse then fine
weights1, score1 = hc(oof_preds, positive_weights=True,
                      weights_choice=(-0.5, 0.5, -0.01, 0.01))
weights2, score2 = hc(oof_preds, positive_weights=True,
                      weights_choice=(-0.5, 0.5, -0.001, 0.001),
                      initial_weights=weights1)

test_preds_hc = np.array(test_preds).T @ weights2
```

**Reproducibility**: HIGH

**Impact**: This greedy optimization finds better ensemble weights than simple averaging. The two-stage approach (coarse adjustment then fine-tuning) is efficient and effective. Model diversity (tree-based + neural models) combined with optimal weighting provides strong ensemble performance.

## Implementation Recommendations

### Priority 1 (Implement first):
**Adversarial Validation-Based Feature Selection**
- Justification: This is a unique and actionable insight. The notebook demonstrates that specific features (physical_activity_minutes_per_week, triglycerides, cholesterol_total) cause severe distribution shift between train/orig datasets. Implementing this as a preprocessing module would help identify and handle such features systematically.
- Implementation: Create `preprocess-adversarial-feature-selection.py` that identifies features with high importance in adversarial validation and optionally drops them.

### Priority 2:
**Multi-View Feature Engineering (Binning + Digits + Rounding)**
- Justification: The combination of binning, digit extraction, and rounding creates multiple complementary views of numeric features. This is particularly powerful for tree-based models.
- Implementation: Create `preprocess-multiview-features.py` combining these three transformations, with configurable parameters (num_bins, digit_depth, round_levels).

### Priority 3:
**Hill Climbing Ensemble Optimization**
- Justification: Better than simple averaging and easy to implement. Works on OOF predictions, so no risk of overfitting.
- Implementation: Add to model post-processing or create an ensemble module that loads multiple OOF predictions and optimizes weights.

## MLA Integration Notes

### Preprocessing Module: `adversarial_validation_feature_drop.py`
```python
# Fit adversarial model to identify distribution shift
# Extract feature importances
# Drop top-k features causing shift (configurable threshold)
# Parameters: threshold_auc (drop features if AV-AUC > X)
#            top_n_features (number of top importance features to drop)
```

### Preprocessing Module: `multiview_numeric_features.py`
```python
# For each numeric column:
#   - Add binned version (qcut with q bins)
#   - Add digit features (extract individual digits)
#   - Add rounded versions (multiple rounding levels)
# Parameters: num_bins=5, digit_extraction=True, round_levels=[-1, 0]
```

### Model Template: `ensemble-hill-climbing.yaml`
```yaml
model: ensemble_hill_climbing
config:
  base_models: [xgb_v1, xgb_v2, lgb_v1, ydf_v1]
  positive_weights: true
  n_epochs: 20
  weight_steps: [-0.5, 0.5, -0.01, 0.01, -0.001, 0.001]
```

### Preprocessing Module: `target_encoding_cv.py`
```python
# Proper CV-aware target encoding using sklearn.preprocessing.TargetEncoder
# Parameters: smooth=1, cv=10, shuffle=True
```

## Code Snippets for Reference

### Complete Feature Engineering Pipeline
```python
class FeatureEngineer:
    def __init__(self, df: pd.DataFrame, target: str):
        self.df = df
        self.target = target

    def engineer_all(self, numeric_cols, categoric_cols):
        # Step 1: Binning
        categoric_cols += self.add_bin_features(numeric_cols, q=5)

        # Step 2: Digit extraction
        categoric_cols += self.add_digit_features(numeric_cols)

        # Step 3: Rounding
        self.add_round_features(numeric_cols, round=[-1, 0])

        # Step 4: Label encode categoricals
        self.label_encoder_categoric(categoric_cols)

        # Step 5: Create numeric_as_categorical
        cat_from_num = self.label_encoder_numeric(numeric_cols, suffix='_cat')
        categoric_cols += cat_from_num

        # Step 6: Count encoding
        self.count_encoder_categoric(categoric_cols, suffix='_cnt')

        return self.df

# Usage in MLA framework
fe = FeatureEngineer(train_df, target='diagnosed_diabetes')
train_df = fe.engineer_all(numeric_cols, categoric_cols)
```

### Adversarial Validation Analysis
```python
def identify_shift_features(df_train, df_test, target_col, top_n=5):
    """Identify features causing distribution shift"""
    # Prepare data
    df_train_av = df_train.drop(target_col, axis=1)
    df_test_av = df_test.drop(target_col, axis=1)
    df_train_av['_dataset'] = 1
    df_test_av['_dataset'] = 0
    df_combined = pd.concat([df_train_av, df_test_av])

    # Train adversarial model
    from xgboost import XGBClassifier
    model = XGBClassifier(n_estimators=1000, learning_rate=0.01,
                          max_depth=5, random_state=42)
    X = df_combined.drop('_dataset', axis=1)
    y = df_combined['_dataset']
    model.fit(X, y)

    # Get feature importances
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return importances.head(top_n)
```

## Caveats and Limitations

### Dataset-Specific Assumptions:
- The adversarial validation identifies specific features (physical_activity, triglycerides, cholesterol) that may be unique to this diabetes dataset
- The digit extraction assumes numeric features have meaningful digit-level patterns (may not transfer to all competitions)
- Original dataset merging is disabled (`merge_original_dataset=False`) due to severe distribution shift - this decision is dataset-specific

### Computational Requirements:
- Full pipeline requires significant compute: 20,000 estimators × 10 folds × 6 models
- AutoGluon alone: 4 hours with best_quality preset
- Hill climbing ensemble: relatively cheap (minutes)
- Optuna hyperparameter tuning: disabled by default, would add many hours
- Total estimated time for full reproduction: 10+ hours on GPU

### Reproducibility Challenges:
- Many predictions loaded from pre-computed files rather than trained live
- Optuna results are pre-saved; re-running would yield slightly different hyperparameters
- AutoGluon with `best_quality` and dynamic stacking has some randomness despite seed setting
- Different GPU/CUDA versions may yield slightly different results for tree methods

### What Might Not Transfer:
- The specific features identified for dropping (inhomogeneous features) are diabetes-specific
- Digit extraction works well here but may not help for all numeric distributions
- The ensemble weights are optimized for this specific set of models - different model combinations would need re-optimization
- Target encoding with smooth=1, cv=10 may need adjustment for smaller datasets
- Repeated 5-fold (10 total) may be overkill for larger datasets (consider single 5-fold or 10-fold)

### MLA Framework Considerations:
- Hill climbing ensemble assumes all models produce OOF predictions in the same format
- Feature engineering creates many derived features - may need feature selection afterward
- The CV class with fold transforms is elegant but requires careful integration with MLA's preprocessing flow
- Multiple model variants (3 XGB versions, 2 AutoGluon versions) increase complexity
