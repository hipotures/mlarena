"""
Experiment 4: Stacking Ensemble

Strategy:
- Level 1: LightGBM, XGBoost, CatBoost (diverse base models)
- Level 2: Logistic Regression (meta-learner)
- Out-of-fold predictions to prevent overfitting
- Probability calibration (Isotonic Regression)
- Uses Tier 2 features (includes Tier 1 + encodings)

Expected improvement: +0.01-0.03 AUC
Time budget: ~1-1.5 hours
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PREP_DIR = PROJECT_ROOT / "code" / "preprocessing"
sys.path.insert(0, str(PREP_DIR))

from fe_tier2_encoding import add_tier2_features  # noqa: E402

from kaggle_tools.config_models import ModelConfig  # noqa: E402

EXPERIMENT_NAME = "exp04_stacking_ensemble"

# Global storage for models
_BASE_MODELS = {}
_META_MODEL = None
_ENCODING_ARTIFACTS = {}


def get_default_config() -> Dict[str, Any]:
    """Configuration for stacking ensemble."""
    return {
        "hyperparameters": {
            "n_folds": 5,  # CV folds for out-of-fold predictions
            "early_stopping_rounds": 50,
            "base_model_iterations": 500,  # Iterations per base model
        },
        "model": {
            "eval_metric": "roc_auc",
            "problem_type": "binary",
        },
        "preprocessing": {
            "feature_set": "tier2_encoding",
            "include_tier1": True,
        }
    }


def preprocess(train_df: pd.DataFrame, config: ModelConfig, is_train: bool = True) -> pd.DataFrame:
    """Apply Tier 2 feature engineering."""
    global _ENCODING_ARTIFACTS

    

    target_col = config.dataset.target

    print(f"[{EXPERIMENT_NAME}] Applying Tier 2 features (includes Tier 1 + encodings)...")

    if is_train:
        enriched, _, artifacts = add_tier2_features(
            train_df=train_df,
            test_df=None,
            target_col=target_col,
            include_tier1=True
        )
        _ENCODING_ARTIFACTS = artifacts
    else:
        # Apply stored encoders for test data
        from fe_tier1 import add_tier1_features

        enriched = add_tier1_features(train_df)

        # Apply encoders
        if 'target_encoder' in _ENCODING_ARTIFACTS:
            enriched = _ENCODING_ARTIFACTS['target_encoder'].transform(enriched)

        if 'woe_encoder' in _ENCODING_ARTIFACTS:
            enriched = _ENCODING_ARTIFACTS['woe_encoder'].transform(enriched)

        # Polynomial and interactions (manual application)
        # ... (similar to exp02)

    return enriched


def train_base_model_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> lgb.Booster:
    """Train LightGBM base model."""
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 7,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'class_weight': 'balanced',
        'verbosity': -1,
        'random_state': 42,
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ]
    )

    return model


def train_base_model_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> xgb.Booster:
    """Train XGBoost base model."""
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate': 0.05,
        'max_depth': 6,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'scale_pos_weight': 4,  # Handle imbalance (80/20 ratio)
        'random_state': 42,
        'verbosity': 0,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dval, 'valid')],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    return model


def train_base_model_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> cb.CatBoost:
    """Train CatBoost base model."""
    # Identify categorical features
    cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    model = cb.CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        loss_function='Logloss',
        eval_metric='AUC',
        auto_class_weights='Balanced',
        cat_features=cat_features,
        random_state=42,
        verbose=False,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False,
    )

    return model


def generate_oof_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 5
) -> Tuple[pd.DataFrame, Dict[str, List]]:
    """
    Generate out-of-fold predictions from base models.

    Returns:
        Tuple of (oof_predictions_df, trained_models_dict)
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Initialize OOF predictions
    oof_lgbm = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))
    oof_catboost = np.zeros(len(X))

    # Store models for each fold
    models_lgbm = []
    models_xgb = []
    models_catboost = []

    print(f"[{EXPERIMENT_NAME}] Generating out-of-fold predictions with {n_folds} folds...")

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n  Fold {fold_idx + 1}/{n_folds}")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Train LightGBM
        print("    Training LightGBM...")
        lgbm_model = train_base_model_lgbm(X_train, y_train, X_val, y_val)
        oof_lgbm[val_idx] = lgbm_model.predict(X_val)
        models_lgbm.append(lgbm_model)

        # Train XGBoost
        print("    Training XGBoost...")
        xgb_model = train_base_model_xgb(X_train, y_train, X_val, y_val)
        dval = xgb.DMatrix(X_val)
        oof_xgb[val_idx] = xgb_model.predict(dval)
        models_xgb.append(xgb_model)

        # Train CatBoost
        print("    Training CatBoost...")
        catboost_model = train_base_model_catboost(X_train, y_train, X_val, y_val)
        oof_catboost[val_idx] = catboost_model.predict_proba(X_val)[:, 1]
        models_catboost.append(catboost_model)

        # Fold scores
        lgbm_score = roc_auc_score(y_val, oof_lgbm[val_idx])
        xgb_score = roc_auc_score(y_val, oof_xgb[val_idx])
        catboost_score = roc_auc_score(y_val, oof_catboost[val_idx])

        print(f"    LightGBM AUC: {lgbm_score:.5f}")
        print(f"    XGBoost AUC: {xgb_score:.5f}")
        print(f"    CatBoost AUC: {catboost_score:.5f}")

    # Calculate overall OOF scores
    lgbm_oof_score = roc_auc_score(y, oof_lgbm)
    xgb_oof_score = roc_auc_score(y, oof_xgb)
    catboost_oof_score = roc_auc_score(y, oof_catboost)

    print(f"\n[{EXPERIMENT_NAME}] Overall out-of-fold scores:")
    print(f"  LightGBM: {lgbm_oof_score:.5f}")
    print(f"  XGBoost: {xgb_oof_score:.5f}")
    print(f"  CatBoost: {catboost_oof_score:.5f}")

    # Create OOF predictions dataframe
    oof_df = pd.DataFrame({
        'lgbm_pred': oof_lgbm,
        'xgb_pred': oof_xgb,
        'catboost_pred': oof_catboost,
    })

    models = {
        'lgbm': models_lgbm,
        'xgb': models_xgb,
        'catboost': models_catboost,
    }

    return oof_df, models


def train(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    config: ModelConfig,
    artifacts: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict, Dict[str, Any]]:
    """
    Train stacking ensemble.

    Returns:
        Tuple of (models_dict, metadata)
    """
    global _BASE_MODELS, _META_MODEL

    

    target_col = config.dataset.target
    hparams = config.hyperparameters

    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]

    print(f"\n{'='*60}")
    print(f"[{EXPERIMENT_NAME}] Stacking Ensemble Training")
    print(f"{'='*60}")
    print(f"Training samples: {len(X)}")
    print(f"Features: {len(X.columns)}")
    print(f"CV folds: {getattr(hparams, 'n_folds', 5)}")
    print(f"{'='*60}\n")

    # Step 1: Train base models and generate OOF predictions
    oof_predictions, base_models = generate_oof_predictions(
        X, y, n_folds=getattr(hparams, 'n_folds', 5)
    )

    # Step 2: Train meta-model on OOF predictions
    print(f"\n[{EXPERIMENT_NAME}] Training meta-model (Logistic Regression)...")
    meta_model = LogisticRegression(
        penalty='l2',
        C=1.0,
        class_weight='balanced',
        random_state=42,
        max_iter=1000,
    )

    meta_model.fit(oof_predictions, y)

    # Evaluate meta-model
    meta_pred = meta_model.predict_proba(oof_predictions)[:, 1]
    meta_score = roc_auc_score(y, meta_pred)

    print(f"[{EXPERIMENT_NAME}] Meta-model ROC-AUC: {meta_score:.5f}")

    # Step 3: Calibrate meta-model
    print(f"[{EXPERIMENT_NAME}] Calibrating meta-model (Isotonic Regression)...")
    calibrated_meta = CalibratedClassifierCV(
        meta_model,
        method='isotonic',
        cv=5
    )
    calibrated_meta.fit(oof_predictions, y)

    calibrated_pred = calibrated_meta.predict_proba(oof_predictions)[:, 1]
    calibrated_score = roc_auc_score(y, calibrated_pred)

    print(f"[{EXPERIMENT_NAME}] Calibrated meta-model ROC-AUC: {calibrated_score:.5f}")

    # Store models
    _BASE_MODELS = base_models
    _META_MODEL = calibrated_meta

    metadata = {
        "experiment": EXPERIMENT_NAME,
        "base_models_oof_lgbm": roc_auc_score(y, oof_predictions['lgbm_pred']),
        "base_models_oof_xgb": roc_auc_score(y, oof_predictions['xgb_pred']),
        "base_models_oof_catboost": roc_auc_score(y, oof_predictions['catboost_pred']),
        "meta_model_score": meta_score,
        "calibrated_meta_score": calibrated_score,
        "n_folds": getattr(hparams, 'n_folds', 5),
        "feature_count": len(X.columns),
    }

    print(f"\n[{EXPERIMENT_NAME}] Training completed!")
    print(f"  Final stacked model ROC-AUC: {calibrated_score:.5f}")

    return {"base_models": base_models, "meta_model": calibrated_meta}, metadata


def predict(
    model: Dict,
    test_df: pd.DataFrame,
    config: ModelConfig,
    artifacts: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Generate predictions using stacked ensemble."""
    

    target_col = config.dataset.target
    X_test = test_df.drop(columns=[target_col], errors='ignore')

    print(f"[{EXPERIMENT_NAME}] Generating ensemble predictions on {len(X_test)} samples...")

    base_models = model.get('base_models', _BASE_MODELS)
    meta_model = model.get('meta_model', _META_MODEL)

    # Get predictions from each base model (average across folds)
    lgbm_preds = np.mean([m.predict(X_test) for m in base_models['lgbm']], axis=0)
    xgb_preds = np.mean([m.predict(xgb.DMatrix(X_test)) for m in base_models['xgb']], axis=0)
    catboost_preds = np.mean([m.predict_proba(X_test)[:, 1] for m in base_models['catboost']], axis=0)

    # Create meta-features
    meta_features = pd.DataFrame({
        'lgbm_pred': lgbm_preds,
        'xgb_pred': xgb_preds,
        'catboost_pred': catboost_preds,
    })

    # Get final predictions from meta-model
    pred_proba = meta_model.predict_proba(meta_features)[:, 1]

    print(f"[{EXPERIMENT_NAME}] Predictions generated. Mean: {pred_proba.mean():.4f}")

    # Create submission DataFrame
    id_col = config.dataset.id_column
    target_col = config.dataset.target

    submission_df = pd.DataFrame({
        id_col: test_df[id_col],
        target_col: pred_proba
    })

    return submission_df


__all__ = ['get_default_config', 'preprocess', 'train', 'predict']
