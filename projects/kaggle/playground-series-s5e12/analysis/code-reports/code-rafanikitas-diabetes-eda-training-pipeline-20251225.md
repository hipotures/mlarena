# Analysis: Diabetes EDA|Training Pipeline

**Author**: Nikitas Rafael Nikitas
**URL**: https://www.kaggle.com/code/rafanikitas/diabetes-eda-training-pipeline
**Votes**: 59
**Analysis Date**: 2025-12-25

## Executive Summary

This notebook implements a comprehensive ensemble approach using LightGBM, XGBoost, and CatBoost with 2-level stacking. The key innovation is an extensive feature engineering pipeline creating 25+ derived features across cardiovascular, lipid, and lifestyle domains, combined with sophisticated meta-learning that incorporates disagreement features between base models.

## Reproducibility Assessment

**Overall Score**: HIGH

**Reason**: Complete code with well-defined functions, standard libraries (sklearn, LightGBM, XGBoost, CatBoost), and clear execution flow. All hyperparameters are explicit in the MODEL_CONFIG. Feature engineering function is fully reproducible. The only GPU dependency can be switched to CPU.

## Key Techniques

### 1. Feature Engineering

**Innovation**: Domain-driven feature creation covering cardiovascular risk (pulse pressure, mean arterial pressure), lipid profiles (LDL/HDL ratios, lipid burden), and lifestyle interactions. Creates 25+ features across multiple health domains.

**Code snippet**:
```python
def engineer_features(df):
    daily_physical_hours = (df["physical_activity_minutes_per_week"] / 60) / 7

    # Lifestyle features
    df["screen_activity_ratio"] = df["screen_time_hours_per_day"] / (daily_physical_hours + 1e-6)
    df["sleep_efficiency_pct"] = df["sleep_hours_per_day"] / (24 - df["screen_time_hours_per_day"] - (daily_physical_hours + 1e-6))

    # Cardiovascular features
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    df["pulse_pressure_ratio"] = (df["pulse_pressure"] / df["systolic_bp"])
    df["mean_arterial_pressure"] = (df["systolic_bp"] + 2 * df["diastolic_bp"]) / 3
    df["rate_pressure_product"] = df["heart_rate"] * df["systolic_bp"]

    # Lipid profile features
    df["ldl_hdl_ratio"] = df["ldl_cholesterol"] / (df["hdl_cholesterol"] + 1e-9)
    df["cholesterol_hdl_ratio"] = df["cholesterol_total"] / (df["hdl_cholesterol"] + 1e-9)
    df["non_hdl_cholesterol"] = df["cholesterol_total"] - df["hdl_cholesterol"]
    df["tg_hdl_ratio"] = df["triglycerides"] / (df["hdl_cholesterol"] + 1e-9)
    df["lipid_sum"] = df["cholesterol_total"] + df["triglycerides"]
    df["lipid_burden"] = (df["ldl_hdl_ratio"] + df["tg_hdl_ratio"] + df["cholesterol_hdl_ratio"])

    # Risk interaction features
    df["age_bmi_risk"] = df["age"] * df["bmi"]
    df["age_norm_activity"] = df["physical_activity_minutes_per_week"] / (df["age"] + 1)
    df["activity_bmi_diff"] = df["physical_activity_minutes_per_week"] - df["bmi"]
    df["age_map_risk"] = df["age"] * df["mean_arterial_pressure"]

    # Medical history aggregates
    df['risk_history'] = df['hypertension_history'] + df['cardiovascular_history']
    df['genetic_history'] = df['family_history_diabetes'] * df['bmi']
    df['activity_x_age'] = df['physical_activity_minutes_per_week'] * df['age']

    # Composite lifestyle risk score
    df["lifestyle_risk_score"] = (
        0.3 * df["bmi"] +
        0.2 * df["waist_to_hip_ratio"] +
        0.2 * df["screen_time_hours_per_day"] -
        0.2 * df["physical_activity_minutes_per_week"] -
        0.1 * df["sleep_hours_per_day"]
    )

    return df
```

**Reproducibility**: HIGH

**Impact**: Creates medically-informed features that capture complex relationships. Lipid ratios (LDL/HDL, TG/HDL) are well-established diabetes risk markers. Cardiovascular features (pulse pressure, mean arterial pressure) capture blood pressure dynamics beyond raw systolic/diastolic values.

### 2. Imbalance Handling

**Innovation**: Adaptive class weighting based on actual class distribution, integrated into all boosting models.

**Code snippet**:
```python
def train_all_models(X, y, model_config):
    pos_weight = (y == 0).sum() / (y == 1).sum()
    class_weights = {0: 1.0, 1: pos_weight}

    for name, cfg in model_config.items():
        ModelClass = cfg["class"]
        params = cfg["params"].copy()

        # Inject imbalance handling where needed
        if name in ["LightGBM", "XGBoost"]:
            params["scale_pos_weight"] = pos_weight

        if name == "CatBoost":
            params["class_weights"] = class_weights

        model = ModelClass(**params)
```

**Reproducibility**: HIGH

**Impact**: Automatically adjusts for class imbalance without manual tuning. Calculates positive class weight dynamically, ensuring models don't bias toward majority class.

### 3. Model Configuration

**Model Ensemble**: LightGBM + XGBoost + CatBoost with 2-level stacking

**Base Model Configurations**:

**LightGBM**:
```python
{
    "n_estimators": 7000,
    "learning_rate": 0.01,
    "subsample": 0.85,
    "num_leaves": 24,
    "colsample_bytree": 0.8,
    "device": "gpu",
    "eval_metric": "auc"
}
```

**XGBoost**:
```python
{
    "n_estimators": 2000,
    "learning_rate": 0.07,
    "max_depth": 4,
    "subsample": 0.75,
    "colsample_bytree": 0.8,
    "tree_method": "gpu_hist",
    "enable_categorical": True
}
```

**CatBoost**:
```python
{
    "iterations": 2000,
    "learning_rate": 0.08,
    "depth": 4,
    "task_type": "GPU",
    "eval_metric": "AUC"
}
```

**Meta-learner (LightGBM)**:
```python
lgb.LGBMClassifier(
    objective="binary",
    boosting_type="gbdt",
    n_estimators=5000,
    learning_rate=0.01,
    num_leaves=48,
    max_depth=4,
    min_child_samples=100
)
```

**Reproducibility**: HIGH (can switch GPU to CPU)

### 4. Validation Strategy

**Type**: Stratified 5-Fold CV for base models, Stratified 5-Fold CV for meta-model

**Code snippet**:
```python
def train_cv_model(model_name, model, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    oof_pred = np.zeros(len(X))
    fold_scores = []
    fold_models = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[valid_idx]

        # Model-specific fit logic with early stopping
        if model_name == "XGBoost":
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        elif model_name == "CatBoost":
            model.fit(X_train, y_train, cat_features=CATEGORY_COLUMNS,
                     eval_set=(X_val, y_val), verbose=False)
        else:  # LightGBM
            model.fit(X_train, y_train, categorical_feature=CATEGORY_COLUMNS,
                     eval_set=[(X_val, y_val)])

        val_pred_proba = model.predict_proba(X_val)[:, 1]
        oof_pred[valid_idx] = val_pred_proba

        fold_auc = roc_auc_score(y_val, val_pred_proba)
        fold_scores.append(fold_auc)
        fold_models.append(model)

    return oof_pred, fold_models[-1]
```

**Reproducibility**: HIGH

**Impact**: Proper OOF prediction collection enables unbiased meta-model training. Uses stratification to maintain class distribution across folds.

### 5. Ensemble Strategy - Two-Level Stacking

**Innovation**: Meta-features incorporate both base predictions AND disagreement signals (pairwise differences, spread, std).

**Code snippet**:
```python
def build_meta_features(oof_preds: dict) -> pd.DataFrame:
    """
    Build meta features from base model OOF predictions.
    Includes raw predictions + disagreement features.
    """
    X = pd.DataFrame(oof_preds)
    cols = X.columns.tolist()

    # Pairwise diffs
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            X[f"{cols[i]}_minus_{cols[j]}"] = X[cols[i]] - X[cols[j]]

    # Aggregate disagreement features
    X["pred_mean"] = X[cols].mean(axis=1)
    X["pred_std"] = X[cols].std(axis=1)
    X["pred_max"] = X[cols].max(axis=1)
    X["pred_min"] = X[cols].min(axis=1)
    X["pred_spread"] = X["pred_max"] - X["pred_min"]

    return X

def train_meta_model_cv(oof_preds, y, n_splits=5):
    X_meta = build_meta_features(oof_preds)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_meta = np.zeros(len(y))
    models = []

    for fold, (tr, va) in enumerate(skf.split(X_meta, y)):
        model = lgb.LGBMClassifier(
            n_estimators=5000, learning_rate=0.01, num_leaves=48,
            max_depth=4, min_child_samples=100
        )
        model.fit(X_meta.iloc[tr], y.iloc[tr],
                 eval_set=[(X_meta.iloc[va], y.iloc[va])],
                 callbacks=[lgb.early_stopping(200, verbose=False)])

        oof_meta[va] = model.predict_proba(X_meta.iloc[va])[:, 1]
        models.append(model)

    return models
```

**Reproducibility**: HIGH

**Impact**: Disagreement features help the meta-learner identify when base models are uncertain or conflicting, potentially improving edge-case predictions.

## Implementation Recommendations

### Priority 1 (Implement first):
- **Domain-driven feature engineering**: The cardiovascular and lipid ratio features are medically grounded and likely to transfer well. Implement `pulse_pressure`, `mean_arterial_pressure`, `ldl_hdl_ratio`, `tg_hdl_ratio`, and `cholesterol_hdl_ratio` first as these are well-established diabetes risk markers.

### Priority 2:
- **Adaptive class weighting**: Implement dynamic `scale_pos_weight` calculation in preprocessing module. This is a simple but effective technique that works across datasets with varying class distributions.

### Priority 3:
- **Stacking with disagreement features**: The meta-feature engineering with pairwise differences and spread metrics is sophisticated and may provide marginal gains. Worth testing after base ensemble is working.

## MLA Integration Notes

**Preprocessing Module**: `preprocess-diabetes-domain-features.py`
```python
def fit_transform(train_df, val_df, test_df, config, orig_df=None):
    """
    Creates cardiovascular, lipid, and lifestyle interaction features.
    Config options:
    - include_cardiovascular: bool (default True)
    - include_lipid: bool (default True)
    - include_lifestyle: bool (default True)
    - include_composite_score: bool (default True)
    """
    # Apply engineer_features() to all dataframes
    # Return (train, val, test, orig, state)
```

**Preprocessing Module**: `preprocess-diabetes-class-weights.py`
```python
def fit_transform(train_df, val_df, test_df, config, orig_df=None):
    """
    Calculate sample weights for imbalanced classes.
    Returns sample_weight artifact for model consumption.
    """
    pos_weight = (train_df[TARGET] == 0).sum() / (train_df[TARGET] == 1).sum()
    sample_weights = train_df[TARGET].map({0: 1.0, 1: pos_weight})

    # Save to artifacts
    artifacts = {'sample_weight': sample_weights}
    return (train_df, val_df, test_df, orig_df, {}, artifacts)
```

**Model Template**: `stacking-lgb-xgb-cat.yaml`
```yaml
model: stacking_ensemble
config:
  base_models:
    - lgbm:
        n_estimators: 7000
        learning_rate: 0.01
        num_leaves: 24
        subsample: 0.85
        colsample_bytree: 0.8
    - xgboost:
        n_estimators: 2000
        learning_rate: 0.07
        max_depth: 4
        subsample: 0.75
        colsample_bytree: 0.8
        enable_categorical: true
    - catboost:
        iterations: 2000
        learning_rate: 0.08
        depth: 4
  meta_model:
    type: lightgbm
    n_estimators: 5000
    learning_rate: 0.01
    num_leaves: 48
    max_depth: 4
    early_stopping_rounds: 200
  meta_features:
    include_pairwise_diffs: true
    include_disagreement_stats: true
```

## Code Snippets for Reference

**Complete cardiovascular feature set (ready to use)**:
```python
# Cardiovascular risk features
df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
df["pulse_pressure_ratio"] = df["pulse_pressure"] / df["systolic_bp"]
df["mean_arterial_pressure"] = (df["systolic_bp"] + 2 * df["diastolic_bp"]) / 3
df["rate_pressure_product"] = df["heart_rate"] * df["systolic_bp"]
df["age_map_risk"] = df["age"] * df["mean_arterial_pressure"]

# Lipid profile features
df["ldl_hdl_ratio"] = df["ldl_cholesterol"] / (df["hdl_cholesterol"] + 1e-9)
df["cholesterol_hdl_ratio"] = df["cholesterol_total"] / (df["hdl_cholesterol"] + 1e-9)
df["tg_hdl_ratio"] = df["triglycerides"] / (df["hdl_cholesterol"] + 1e-9)
df["non_hdl_cholesterol"] = df["cholesterol_total"] - df["hdl_cholesterol"]
df["lipid_burden"] = df["ldl_hdl_ratio"] + df["tg_hdl_ratio"] + df["cholesterol_hdl_ratio"]

# Risk interactions
df["age_bmi_risk"] = df["age"] * df["bmi"]
df["genetic_history"] = df["family_history_diabetes"] * df["bmi"]
df["risk_history"] = df["hypertension_history"] + df["cardiovascular_history"]
```

## Caveats and Limitations

### Dataset-specific assumptions:
- Assumes all cholesterol and blood pressure columns are present and non-missing
- Lifestyle risk score weights (0.3, 0.2, etc.) may be tuned for this specific competition
- Sleep efficiency calculation assumes 24-hour constraint holds exactly

### Computational requirements:
- GPU training specified for all models (can be switched to CPU but slower)
- 7000 estimators for LightGBM may require 10-15 minutes on CPU
- Stacking adds training time (5-fold CV for meta-model on top of base OOF)

### What might not transfer:
- `sleep_efficiency_pct` formula assumes specific column names and may produce invalid values if screen time + physical activity > 24 hours
- The composite `lifestyle_risk_score` weights are not justified and may need retuning
- Disagreement features in meta-learning may only help if base models are sufficiently diverse
- GPU requirements make it less portable; ensure CPU fallback is implemented

### Medical context:
- Features like pulse pressure, mean arterial pressure, and lipid ratios are well-established in medical literature as diabetes risk factors
- The approach is domain-informed rather than purely data-driven, which is a strength for this competition
- However, some interaction features (e.g., `activity_bmi_diff`) lack clear medical interpretation

### EDA insights worth noting:
- Author notes that genetic history (family_history_diabetes) shows diabetes rate jumping from 58% to 87%
- Age, BMI, and waist-to-hip ratio identified as strong risk factors through visual analysis
- Non-smokers are majority (~70%), which may affect smoking-related feature importance
