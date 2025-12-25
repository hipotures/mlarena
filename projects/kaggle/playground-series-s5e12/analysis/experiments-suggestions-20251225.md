# Experiment Suggestions: Playground Series S5E12

**Generated**: 2025-12-25
**Based on**: Analysis of 8 top Kaggle notebooks (174-50 votes)
**Current Best Local CV**: 0.730 | **Current Best Public LB**: 0.703

---

## Executive Summary

After analyzing 8 top-performing notebooks from the diabetes prediction competition, several clear patterns emerge alongside unique innovations that remain underexplored. The user's current experiments have focused heavily on external dataset statistical aggregations (orig_mean/count features) and tail-based sample weighting, achieving local CV of 0.715-0.730. However, there are significant gaps in feature engineering approaches, model diversity, and validation strategies that could unlock further improvements.

**Key Findings**:
1. **Original dataset feature engineering** (mean/count aggregations) is well-established - 5 of 8 notebooks use it, including top performers
2. **Medical domain features** (cardiovascular ratios, lipid profiles, clinical thresholds) appear in 6 of 8 notebooks but are NOT in user's current pipeline
3. **Advanced validation strategies** (Repeated K-Fold, multi-label stratification) used by top performers but user relies on standard splits
4. **Model ensembling** with sophisticated weighting (Optuna, hill climbing) could provide 1-3% lift over single models
5. **Adversarial validation** for feature selection is a unique technique used by only 1 notebook (82 votes) - high-value, low-competition

**Current Pipeline Analysis**:
The user has extensively tested external dataset integration, binning, orig_stats features, and tail-based sample weighting. Missing areas include: medical domain features, ratio-based features, categorical target encoding (beyond simple encoding), feature-type-aware preprocessing, and advanced ensemble strategies.

---

## Top Patterns Across Leaderboard

### 1. Feature Engineering
**Most Common Approach**: External dataset statistical aggregations (5/8 notebooks)
- `orig_mean_{feature}`: Target encoding from original dataset
- `orig_count_{feature}`: Frequency encoding from original dataset
- **User Status**: ✅ IMPLEMENTED (diabetes-orig-stats)

**Second Most Common**: Medical domain features (6/8 notebooks)
- Cardiovascular: pulse_pressure, mean_arterial_pressure, rate_pressure_product
- Lipid ratios: LDL/HDL, Total/HDL, TG/HDL, lipid_burden
- Clinical thresholds: BMI categories (18.5/25/30), BP thresholds (130/80, 140/90)
- **User Status**: ❌ NOT IMPLEMENTED

### 2. Model Choice
**Most Popular**: Gradient Boosting ensembles (8/8 notebooks use GBDT)
- LightGBM: 7/8 notebooks
- XGBoost: 7/8 notebooks
- CatBoost: 6/8 notebooks
- **Ensemble Diversity**: 5/8 use 3+ models, 3/8 use 4+ models
- **User Status**: Likely using AutoGluon (includes GBDT ensemble)

### 3. Validation Strategy
**Most Common**: 5-Fold Stratified CV (6/8 notebooks)
- **Advanced**: Repeated 5-Fold (2 repeats = 10 total folds) in 2 notebooks
- **Specialized**: Multi-label stratification (combining multiple categorical features) in 1 notebook
- **User Status**: Unknown (likely standard AutoGluon CV)

### 4. Preprocessing
**Most Common**: Minimal preprocessing for tree models (6/8 notebooks)
- Label/Ordinal encoding for categoricals: 6/8
- Target encoding with CV-awareness: 3/8
- Feature-type-aware pipelines (separate scaling per distribution): 2/8
- **User Status**: Basic encoding implemented

### 5. Ensemble Strategy
**Most Common**: Simple averaging (4/8 notebooks)
- **Advanced**: Optuna weight optimization (2 notebooks - 51 and 82 votes)
- **Advanced**: Hill climbing optimization (1 notebook - 82 votes)
- **Advanced**: Stacking with disagreement features (1 notebook - 59 votes)
- **User Status**: Likely AutoGluon stacking

---

## Suggested Experiments (Ranked by Priority)

### Experiment 1: Medical Domain Feature Engineering [PRIORITY 1]

**Rationale**: 6 of 8 notebooks include medical domain features, yet this is completely absent from the user's current pipeline. These features leverage established clinical risk factors for diabetes (cardiovascular markers, lipid profiles) and have strong theoretical justification. Expected to provide 1-3% local CV improvement based on notebook comparisons.

**Expected Impact**: +2-3% local CV (based on rafanikitas, udayprattap notebooks)
**Difficulty**: LOW
**Source Notebooks**:
- rafanikitas (59 votes): Extensive cardiovascular + lipid features
- udayprattap (50 votes): Clinical BMI/BP categories, cholesterol ratios, medical risk score
- zhukovoleksiy (51 votes): Pulse pressure, MAP, BMI-waist interaction
- masayakawamata (174 votes): Implicitly via orig dataset encoding

**Implementation**:
- **Preprocessing template**: `preprocess-diabetes-medical-domain.yaml`
- **Model template**: Use existing `catboost-statonly-1h-noorig.yaml`
- **Estimated time budget**: 1-2 hours (preprocessing is fast, model training 1h)

**Key Details**:
```python
# Cardiovascular features (highest impact)
df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
df['mean_arterial_pressure'] = (df['systolic_bp'] + 2 * df['diastolic_bp']) / 3
df['rate_pressure_product'] = df['heart_rate'] * df['systolic_bp']

# Lipid profile ratios (well-established diabetes markers)
df['ldl_hdl_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1e-9)
df['tg_hdl_ratio'] = df['triglycerides'] / (df['hdl_cholesterol'] + 1e-9)
df['cholesterol_hdl_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1e-9)
df['non_hdl_cholesterol'] = df['cholesterol_total'] - df['hdl_cholesterol']
df['lipid_burden'] = df['ldl_hdl_ratio'] + df['tg_hdl_ratio'] + df['cholesterol_hdl_ratio']

# Clinical threshold categories (interpretable)
df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100], labels=[0,1,2,3])
df['bp_category'] = 0
df.loc[(df['systolic_bp'] >= 130) | (df['diastolic_bp'] >= 80), 'bp_category'] = 1
df.loc[(df['systolic_bp'] >= 140) | (df['diastolic_bp'] >= 90), 'bp_category'] = 2

# Medical risk score (weighted combination)
df['medical_risk'] = (
    df['family_history_diabetes'] * 0.3 +
    df['hypertension_history'] * 0.3 +
    df['cardiovascular_history'] * 0.4
)
```

**Caveats**:
- Medical thresholds are well-established but may need adjustment if orig dataset merging changes distributions
- Safe division needed for ratios (add small epsilon to denominators)

---

### Experiment 2: Multi-View Numeric Features (Binning + Digit Extraction + Rounding) [PRIORITY 1]

**Rationale**: This is a unique technique from kdmitrie's notebook (82 votes - highest in analysis set) that creates multiple complementary representations of numeric features. The digit extraction is particularly innovative - extracting individual digits can capture fine-grained patterns that binning alone misses. User has binning implemented but not digit extraction or coordinated rounding.

**Expected Impact**: +1-2% local CV (based on kdmitrie's feature importance analysis)
**Difficulty**: MEDIUM
**Source Notebooks**:
- kdmitrie (82 votes): Complete implementation with all three transformations

**Implementation**:
- **Preprocessing template**: `preprocess-multiview-numeric.yaml`
- **Model template**: Use existing model
- **Estimated time budget**: 2 hours (feature engineering adds complexity, training 1h)

**Key Details**:
```python
# 1. Quantile-based binning (user already has this but ensure q=5 uniformly)
for col in numeric_cols:
    df[col + '_bin'], _ = pd.qcut(df[col], q=5, labels=False, duplicates='drop')

# 2. Digit extraction (NOVEL - not in user's pipeline)
for col in numeric_cols:
    sp = df[col].astype(str).str.split('.', expand=True).fillna('')
    b = max(sp[0].str.len())  # digits before decimal
    a = max(sp[1].str.len()) if 1 in sp.columns else 0  # after decimal

    for k in range(1-b, a+1):
        df[f'{col}_dig{k}'] = ((df[col] * 10**k) % 10).fillna(-1).astype('int8')

# 3. Rounding at multiple levels (creates discrete versions)
for col in numeric_cols:
    df[f'{col}_round_neg1'] = df[col].round(-1)  # nearest 10
    df[f'{col}_round_0'] = df[col].round(0)      # nearest 1
```

**Caveats**:
- Creates many features (3x original numeric count) - may need feature selection afterward
- Digit extraction assumes meaningful digit-level patterns (works for medical measurements)
- Memory usage increases significantly

---

### Experiment 3: Ratio-Based Physiological Features [PRIORITY 1]

**Rationale**: mariusborel's notebook (57 votes) demonstrates that ratio features capturing relationships between activity, sleep, BMI, and diet can be highly effective. These are different from medical domain features - they capture lifestyle balance and efficiency metrics. Simple to implement with clear physiological interpretation.

**Expected Impact**: +1-2% local CV (mariusborel showed these in top feature importances)
**Difficulty**: LOW
**Source Notebooks**:
- mariusborel (57 votes): Comprehensive ratio features with physiological justification

**Implementation**:
- **Preprocessing template**: `preprocess-diabetes-ratios.yaml`
- **Model template**: Use existing model
- **Estimated time budget**: 1-2 hours

**Key Details**:
```python
# Activity/sleep balance
df['activity_per_sleep_hour'] = df['physical_activity_minutes_per_week'] / (df['sleep_hours_per_day'] + 1e-9)
df['sleep_screen_ratio'] = df['sleep_hours_per_day'] / (df['screen_time_hours_per_day'] + 1e-9)

# BMI efficiency/interaction
df['bmi_diet_ratio'] = df['bmi'] / (df['diet_score'] + 1e-9)
df['bmi_diastolic_product'] = df['bmi'] * df['diastolic_bp']

# Blood pressure relationships
df['diastolic_systolic_ratio'] = df['diastolic_bp'] / (df['systolic_bp'] + 1e-9)
df['pulse_pressure_bmi_ratio'] = (df['systolic_bp'] - df['diastolic_bp']) / (df['bmi'] + 1e-9)
```

**Caveats**:
- Requires safe division (add small epsilon to all denominators)
- Some ratios may have extreme values if denominators are very small (consider clipping)

---

### Experiment 4: CV-Aware Target Encoding with Smoothing [PRIORITY 2]

**Rationale**: 3 of 8 notebooks implement sophisticated target encoding (masayakawamata, zhukovoleksiy, kdmitrie) with proper out-of-fold strategy to prevent leakage. This is superior to simple label encoding used in user's pipeline. The key innovation is Bayesian smoothing which prevents overfitting to rare categories.

**Expected Impact**: +0.5-1.5% local CV (target encoding is known to be powerful for tree models)
**Difficulty**: MEDIUM
**Source Notebooks**:
- masayakawamata (174 votes - TOP PERFORMER): Custom TargetEncoder class with empirical Bayes smoothing
- zhukovoleksiy (51 votes): TargetEncoderOOF with configurable smoothing
- kdmitrie (82 votes): sklearn's TargetEncoder with cv=10

**Implementation**:
- **Preprocessing template**: `preprocess-target-encoding-cv.yaml`
- **Model template**: Use existing model
- **Estimated time budget**: 2-3 hours (requires careful OOF implementation)

**Key Details**:
```python
from sklearn.preprocessing import TargetEncoder

# Approach 1: Use sklearn's built-in (simplest)
te = TargetEncoder(target_type='continuous', smooth=1, cv=10,
                   shuffle=True, random_state=42)
for col in categorical_cols:
    df[f'TE_{col}'] = te.fit_transform(df[[col]], df[target])

# Approach 2: Custom implementation with auto smoothing (masayakawamata style)
# Uses empirical Bayes: smoothing = variance_within / variance_between
# See masayakawamata notebook for full implementation
```

**Caveats**:
- Must integrate with existing preprocessing chain (apply AFTER other transformations)
- Memory intensive if many categorical columns (creates duplicate encoded columns)
- Smoothing parameter (default=1 for sklearn, auto for custom) may need tuning

---

### Experiment 5: Adversarial Validation for Feature Selection [PRIORITY 2]

**Rationale**: This is a UNIQUE technique used by only kdmitrie (82 votes) - no other analyzed notebook uses it. The approach identifies features causing distribution shift between train/test or train/orig, then optionally drops them. In kdmitrie's analysis, features like physical_activity, triglycerides, cholesterol caused severe shift. This could explain why some user experiments plateau - they may be training on features that don't transfer to test set.

**Expected Impact**: +1-2% local CV and improved generalization (based on kdmitrie's AUC improvements)
**Difficulty**: MEDIUM
**Source Notebooks**:
- kdmitrie (82 votes): Full implementation with feature importance analysis

**Implementation**:
- **Preprocessing template**: `preprocess-adversarial-feature-selection.yaml`
- **Model template**: Use existing model
- **Estimated time budget**: 2-3 hours

**Key Details**:
```python
# Step 1: Train adversarial model to distinguish train from test
df_train_av = train.drop(target_col, axis=1)
df_test_av = test.drop(target_col, axis=1)
df_train_av['_dataset'] = 1
df_test_av['_dataset'] = 0
df_combined = pd.concat([df_train_av, df_test_av])

# Train XGBoost to predict which dataset each row came from
model = XGBClassifier(n_estimators=1000, learning_rate=0.01, max_depth=5)
model.fit(df_combined.drop('_dataset', axis=1), df_combined['_dataset'])
av_auc = roc_auc_score(df_combined['_dataset'], model.predict_proba(...)[:, 1])

# Step 2: Extract feature importances
importances = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Step 3: Drop top-k features causing shift (configurable threshold)
# kdmitrie found: physical_activity_minutes_per_week, triglycerides,
# cholesterol_total, alcohol_consumption_per_week caused shift
features_to_drop = importances.head(k)['feature'].tolist()
```

**Caveats**:
- Dropping features may hurt if they're genuinely predictive (trade-off between signal and shift)
- AV-AUC interpretation: 0.5 = no shift, 1.0 = perfect shift. Drop if > 0.8-0.9
- Run AV on both train/test AND train/orig to understand different shift patterns

---

### Experiment 6: Repeated Stratified K-Fold Validation [PRIORITY 2]

**Rationale**: kdmitrie (82 votes) uses Repeated 5-Fold (2 repeats = 10 total folds) which provides more robust CV estimates than single 5-fold. This is particularly important for ensemble optimization where you need stable OOF predictions. User's current local CV of 0.715-0.730 may have high variance - repeated CV would give confidence intervals.

**Expected Impact**: No direct score improvement, but +stability in CV estimates (reduces variance by ~30-40%)
**Difficulty**: LOW
**Source Notebooks**:
- kdmitrie (82 votes): RepeatedStratifiedKFold implementation

**Implementation**:
- **Model template**: Modify existing model to use RepeatedStratifiedKFold
- **Estimated time budget**: 2x current training time (doubles folds)

**Key Details**:
```python
from sklearn.model_selection import RepeatedStratifiedKFold

cv = RepeatedStratifiedKFold(
    n_splits=5,      # 5 folds
    n_repeats=2,     # repeat twice = 10 total folds
    random_state=42
)

# Benefit: More stable OOF predictions for ensembling
# Benefit: Better estimate of true generalization (confidence intervals)
# Cost: 2x training time
```

**Caveats**:
- Doubles training time (5 folds → 10 folds)
- Mainly valuable if ensembling multiple models or doing meta-learning
- May not be worth it for quick experiments, but essential for final submissions

---

### Experiment 7: Multi-Label Stratification [PRIORITY 2]

**Rationale**: kospintr (67 votes) uses multi-label stratification - creating a composite stratification column from multiple important categorical features (family_history_diabetes, cardiovascular_history, ethnicity). This ensures CV folds are balanced across multiple dimensions, not just the target. Particularly important for datasets with multiple important categorical features like this diabetes competition.

**Expected Impact**: +0.5-1% local CV through better fold balance (based on kospintr's results)
**Difficulty**: LOW
**Source Notebooks**:
- kospintr (67 votes): Implementation with 4 categorical features combined

**Implementation**:
- **Preprocessing template**: `preprocess-multilabel-stratification.yaml` (creates strat column)
- **Model template**: Modify CV to use multicat column for stratification
- **Estimated time budget**: 1-2 hours

**Key Details**:
```python
from sklearn.preprocessing import LabelEncoder

# Combine multiple categorical features into one stratification column
strat_cols = ['family_history_diabetes', 'cardiovascular_history',
              'ethnicity', 'gender']
train['multicat'] = LabelEncoder().fit_transform(
    train[strat_cols].astype(str).agg('_'.join, axis=1)
)

# Use for stratification in CV
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in cv.split(X, train['multicat']):
    # Train on balanced folds
```

**Caveats**:
- Creates many unique combinations (product of cardinalities) - some may have very few samples
- Only useful if categorical features have important interactions with target
- Requires integration with model's CV strategy

---

### Experiment 8: Polynomial + Interaction Features (Selective) [PRIORITY 3]

**Rationale**: udayprattap (50 votes) uses selective polynomial and interaction features (age*bmi, age*chol, bmi*chol, bmi^2, chol^2, age^2) scaled appropriately. This captures non-linear relationships that boosting may struggle with if trees aren't deep enough. User's pipeline doesn't include any polynomial features.

**Expected Impact**: +0.5-1% local CV (udayprattap showed improvement from v1 to v5)
**Difficulty**: LOW
**Source Notebooks**:
- udayprattap (50 votes): Selective interaction + polynomial features
- rafanikitas (59 votes): Similar approach with medical focus

**Implementation**:
- **Preprocessing template**: `preprocess-diabetes-interactions.yaml`
- **Model template**: Use existing model
- **Estimated time budget**: 1-2 hours

**Key Details**:
```python
# Interaction features (scaled to prevent magnitude issues)
df['age_bmi'] = df['age'] * df['bmi'] / 100
df['age_chol'] = df['age'] * df['cholesterol_total'] / 100
df['bmi_chol'] = df['bmi'] * df['cholesterol_total'] / 100
df['family_age'] = df['family_history_diabetes'] * df['age'] / 10

# Polynomial features (scaled)
df['bmi_squared'] = df['bmi'] ** 2 / 100
df['chol_squared'] = df['cholesterol_total'] ** 2 / 1000
df['age_squared'] = df['age'] ** 2 / 1000

# Age-medical interaction
df['age_map_risk'] = df['age'] * df['mean_arterial_pressure'] / 100
```

**Caveats**:
- Scaling is critical (divide by 100-1000) to keep magnitudes similar to original features
- Creates collinearity - may need L1/L2 regularization in models
- Not all interactions are meaningful - start with medically justified ones

---

### Experiment 9: Ensemble with Optuna Weight Optimization [PRIORITY 3]

**Rationale**: Two notebooks (zhukovoleksiy 51 votes, kdmitrie 82 votes with hill climbing variant) use Optuna to optimize ensemble weights rather than simple averaging. This can extract 0.5-1% additional performance from diverse models. User likely uses AutoGluon's built-in stacking, but custom weighting on OOF predictions could improve further.

**Expected Impact**: +0.5-1% local CV over simple averaging (based on both notebooks)
**Difficulty**: HIGH
**Source Notebooks**:
- zhukovoleksiy (51 votes): OptunaWeights class with 100 trials
- kdmitrie (82 votes): Hill climbing optimization (greedy search)

**Implementation**:
- **Model template**: Custom ensemble module (not easily templated)
- **Estimated time budget**: 4-6 hours (requires training multiple models + optimization)

**Key Details**:
```python
# Approach 1: Optuna (zhukovoleksiy style)
import optuna

def objective(trial, y_true, y_preds):
    weights = [trial.suggest_float(f'weight{i}', 1e-15, 1.0)
               for i in range(len(y_preds))]
    weighted_pred = np.average(np.array(y_preds).T, axis=1, weights=weights)
    return roc_auc_score(y_true, weighted_pred)

study = optuna.create_study(direction='maximize')
study.optimize(partial(objective, y_true=y_oof, y_preds=oof_preds_list),
               n_trials=100)

# Approach 2: Hill climbing (kdmitrie style - faster)
# Greedy search: iteratively adjust each weight, keep if improves score
# See kdmitrie notebook for full implementation
```

**Caveats**:
- Requires training multiple diverse models (XGB, LGB, CatBoost, etc.)
- Risk of overfitting to OOF predictions (use conservative n_trials ~50-100)
- Hill climbing is faster but may get stuck in local optima vs Optuna's TPE sampler

---

### Experiment 10: IQR-Based Outlier Clipping [PRIORITY 3]

**Rationale**: mariusborel (57 votes) uses IQR-based clipping (1.5*IQR threshold) rather than removal, which preserves all samples while mitigating extreme value influence. This is safer than aggressive outlier removal (which zhukovoleksiy uses and may hurt generalization). User's pipeline doesn't show explicit outlier handling.

**Expected Impact**: +0.5-1% local CV through improved robustness
**Difficulty**: LOW
**Source Notebooks**:
- mariusborel (57 votes): IQR clipping implementation
- zhukovoleksiy (51 votes): Aggressive quantile-based removal (anti-pattern to avoid)

**Implementation**:
- **Preprocessing template**: `preprocess-outlier-clip-iqr.yaml`
- **Model template**: Use existing model
- **Estimated time budget**: 1 hour

**Key Details**:
```python
from scipy.stats import iqr

for col in numeric_cols:
    if df[col].nunique() > 20:  # skip discrete features
        iqr_val = iqr(df[col])
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        lower_bound = q1 - 1.5 * iqr_val
        upper_bound = q3 + 1.5 * iqr_val
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
```

**Caveats**:
- Calculate bounds on TRAIN only, apply to train/val/test
- May clip genuine extreme values (medical outliers can be real)
- 1.5*IQR is standard but conservative (consider 3*IQR for medical data)

---

## Implementation Roadmap

### Phase 1: High-Value Quick Wins (1-2 weeks)
**Priority 1 experiments** that are low-hanging fruit:
1. Medical Domain Features (Exp #1) - 2 hours
2. Ratio-Based Features (Exp #3) - 2 hours
3. IQR Outlier Clipping (Exp #10) - 1 hour

**Expected Cumulative Impact**: +3-5% local CV
**Total Time**: ~5-7 hours

### Phase 2: Medium-Complexity Enhancements (2-3 weeks)
**Priority 2 experiments** requiring more integration:
1. CV-Aware Target Encoding (Exp #4) - 3 hours
2. Adversarial Validation (Exp #5) - 3 hours
3. Multi-Label Stratification (Exp #7) - 2 hours
4. Repeated K-Fold (Exp #6) - doubles training time but no code changes

**Expected Cumulative Impact**: +2-4% local CV on top of Phase 1
**Total Time**: ~10-15 hours + increased training time

### Phase 3: Advanced Techniques (3-4 weeks)
**Priority 3 experiments** requiring significant effort:
1. Multi-View Numeric Features (Exp #2) - 3 hours + feature selection
2. Polynomial/Interaction Features (Exp #8) - 2 hours
3. Ensemble Optimization (Exp #9) - 6+ hours

**Expected Cumulative Impact**: +1-3% local CV on top of Phases 1-2
**Total Time**: ~15-20 hours

### Recommended Starting Point
**Combine Experiments 1 + 3 + 10** in single preprocessing chain:
```bash
# Preprocessing chain
preprocess-outlier-clip-iqr →
preprocess-diabetes-medical-domain →
preprocess-diabetes-ratios →
[existing: preprocess-external-diabetes-gap] →
[existing: preprocess-diabetes-binning-gap] →
[existing: preprocess-diabetes-orig-stats-gap]

# Expected: +3-5% local CV improvement over current best
# Time: ~5-7 hours total (including 1h model training)
```

---

## Techniques to AVOID

Based on analysis of what DIDN'T work in notebooks:

1. **Aggressive Outlier Removal** (zhukovoleksiy removes top 50 per column)
   - Reason: Reduces sample size non-deterministically, may hurt generalization
   - Better alternative: IQR clipping (preserves all samples)

2. **Merging Original Dataset Directly into Training** (kdmitrie tried and rejected)
   - Reason: Severe distribution shift (AV-AUC = 0.91) causes poor generalization
   - Better alternative: Extract statistics from orig, don't concat

3. **Simple Threshold Optimization** (udayprattap v3 failed spectacularly: 0.57 public)
   - Reason: Overfits to validation set, doesn't transfer to test
   - Better alternative: Optimize model hyperparameters, not prediction threshold

4. **Too Many Engineered Features Without Selection** (udayprattap v2: 58 features → 0.641)
   - Reason: Overfitting through feature noise
   - Better alternative: Moderate feature count (30-40) with importance-based selection

5. **Label Encoding Only** (rv1922 baseline approach)
   - Reason: Misses categorical-target relationships
   - Better alternative: Target encoding with proper CV

---

## Notes on Current User Pipeline

**Strengths**:
- External dataset integration is well-implemented (orig_mean/count features)
- Binning approach appears solid
- Tail-based sample weighting shows experimentation with class imbalance

**Gaps**:
- No medical domain features (major opportunity - 6/8 notebooks use them)
- No ratio-based features (lifestyle balance metrics)
- Limited advanced categorical encoding (no target encoding with CV)
- No adversarial validation (could explain plateaus)
- Unknown model ensemble strategy (likely AutoGluon default)

**Recommended Focus**:
Phase 1 experiments (#1, #3, #10) fill the biggest gaps and have lowest implementation complexity.

---

## Competition-Specific Insights

### Dataset Characteristics (from notebooks)
- **Size**: ~700K train, ~300K test (large enough for complex features)
- **Class Imbalance**: 65/35 split (moderate, handled by most with class weights)
- **Missing Values**: Zero (clean dataset, no imputation needed)
- **Feature Types**: Mix of continuous, discrete, and categorical (11-20 features typically)
- **Distribution Shift**: Moderate between train/test (AV-AUC ~0.62), severe with orig (AV-AUC ~0.91)

### What's Working on Leaderboard
- Medical domain knowledge beats pure data-driven approaches
- Ensemble diversity matters more than individual model strength
- External dataset as reference (not training data) is consensus best practice
- Shallow trees with strong regularization (prevent overfitting to 700K samples)

### What's NOT Working
- Deep complex models (neural nets struggle vs GBDT - see zhukovoleksiy)
- Aggressive outlier removal
- Direct merging of orig dataset into training
- Over-engineering features without selection (diminishing returns after ~40 features)

---

**End of Report**
