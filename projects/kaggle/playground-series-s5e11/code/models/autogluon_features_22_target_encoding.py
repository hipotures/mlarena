"""
AutoGluon variant #22: fe21 features + Target Encoding with Cross-Validation.
Implements proper target encoding to prevent data leakage.
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import sys

MODEL_DIR = Path(__file__).parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import autogluon_features_21 as base_model
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from autogluon.tabular import TabularPredictor
from kaggle_tools.config_models import ModelConfig

VARIANT_NAME = "feature-set-22-target-encoding"

def get_default_config() -> Dict[str, Any]:
    return base_model.get_default_config()

class TargetEncoder:
    """Target encoder with cross-validation to prevent leakage."""
    
    def __init__(self, n_splits: int = 5, smoothing: float = 1.0):
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.global_mean = None
        self.category_means = {}
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series, cols: list) -> pd.DataFrame:
        """Fit and transform with CV to prevent leakage."""
        X = X.copy()
        
        # Calculate global mean for smoothing
        self.global_mean = y.mean()
        
        # Initialize encoded columns
        for col in cols:
            X[f'{col}_target_encoded'] = 0.0
        
        # Stratified K-Fold for proper CV
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        for train_idx, val_idx in skf.split(X, y):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            
            # Calculate target encoding for each categorical column
            for col in cols:
                # Calculate mean target for each category in training fold
                category_stats = y_train_fold.groupby(X_train_fold[col]).agg(['mean', 'count'])
                
                # Apply smoothing (similar to Bayesian average)
                smoothed_means = {}
                for cat in category_stats.index:
                    cat_mean = category_stats.loc[cat, 'mean']
                    cat_count = category_stats.loc[cat, 'count']
                    
                    # Bayesian average with smoothing
                    smoothed_mean = (
                        (cat_mean * cat_count + self.global_mean * self.smoothing) /
                        (cat_count + self.smoothing)
                    )
                    smoothed_means[cat] = smoothed_mean
                
                # Apply to validation fold
                X.loc[val_idx, f'{col}_target_encoded'] = (
                    X.loc[val_idx, col].map(smoothed_means).fillna(self.global_mean)
                )
        
        # Store final encoding for test set (using all training data)
        for col in cols:
            category_stats = y.groupby(X[col]).agg(['mean', 'count'])
            self.category_means[col] = {}
            
            for cat in category_stats.index:
                cat_mean = category_stats.loc[cat, 'mean']
                cat_count = category_stats.loc[cat, 'count']
                
                smoothed_mean = (
                    (cat_mean * cat_count + self.global_mean * self.smoothing) /
                    (cat_count + self.smoothing)
                )
                self.category_means[col][cat] = smoothed_mean
        
        return X
    
    def transform(self, X: pd.DataFrame, cols: list) -> pd.DataFrame:
        """Transform test data using stored encodings."""
        X = X.copy()
        
        for col in cols:
            X[f'{col}_target_encoded'] = (
                X[col].map(self.category_means.get(col, {})).fillna(self.global_mean)
            )
        
        return X


# Global encoder instance and WoE mappings
target_encoder = None
woe_mappings: Dict[str, Dict[Any, float]] = {}

def _engineer_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Apply fe21 features plus target encoding."""
    global target_encoder, woe_mappings
    
    # First apply all fe21 features
    enriched = base_model._engineer_features(df)
    
    # Target encoding for high-cardinality categorical variables
    categorical_cols = [
        'grade_subgrade',
        'loan_purpose',
        'employment_status',
        'home_ownership'
    ]
    
    # Filter to only existing columns
    cols_to_encode = [col for col in categorical_cols if col in enriched.columns]
    
    if is_train:
        # For training: fit and transform with CV
        if 'loan_paid_back' in enriched.columns:
            target_encoder = TargetEncoder(n_splits=5, smoothing=1.0)
            enriched = target_encoder.fit_transform(
                enriched, 
                enriched['loan_paid_back'], 
                cols_to_encode
            )
    else:
        # For test: use stored encodings
        if target_encoder is not None:
            enriched = target_encoder.transform(enriched, cols_to_encode)
        else:
            # Fallback if encoder not fitted
            for col in cols_to_encode:
                enriched[f'{col}_target_encoded'] = 0.0
    
    # Weight of Evidence encoding for binary target
    # This is similar to target encoding but uses log odds
    if is_train and 'loan_paid_back' in enriched.columns:
        # Cast target to float once to avoid type issues during aggregation
        target = enriched['loan_paid_back'].fillna(0).astype(float)
        total_positive = target.sum()
        total_negative = len(target) - total_positive
        total_positive = max(total_positive, 0.5)
        total_negative = max(total_negative, 0.5)

        woe_mappings = {}
        for col in cols_to_encode:
            woe_dict = {}
            for cat in enriched[col].unique():
                mask = enriched[col] == cat
                n_positive = target.loc[mask].sum()
                n_negative = mask.sum() - n_positive
                
                # Avoid log(0) by adding smoothing
                n_positive = max(n_positive, 0.5)
                n_negative = max(n_negative, 0.5)
                
                woe = np.log((n_positive / total_positive) / (n_negative / total_negative))
                woe_dict[cat] = woe
            
            woe_mappings[col] = woe_dict
            enriched[f'{col}_woe'] = enriched[col].map(woe_dict).fillna(0)
    else:
        # Apply stored WoE mappings for inference to keep feature set consistent
        for col in cols_to_encode:
            mapping = woe_mappings.get(col, {})
            enriched[f'{col}_woe'] = enriched[col].map(mapping).fillna(0)
    
    return enriched

def prepare_artifacts(train_df: pd.DataFrame, config: ModelConfig):
    """Initialize shared artifacts dict for train/predict phases."""
    return {}

def preprocess(df: pd.DataFrame, config: ModelConfig, is_train: bool = True) -> pd.DataFrame:
    """Apply feature engineering with target encoding."""
    return _engineer_features(df, is_train=is_train)

def train(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> Tuple[TabularPredictor, Dict[str, Any]]:
    """Train with target-encoded features."""
    print(f"[{VARIANT_NAME}] Training with fe21 + Target Encoding (CV-protected).")
    print(f"[{VARIANT_NAME}] Total features: {len(train_df.columns)}")

    # Store encoder in artifacts for prediction phase
    if artifacts is None:
        artifacts = {}
    artifacts['target_encoder'] = target_encoder
    artifacts['woe_mappings'] = woe_mappings

    model, summary = base_model.train(train_df, val_df, config, artifacts)

    # Add encoder to summary for tracking
    summary['has_target_encoding'] = True
    summary['encoded_features'] = ['grade_subgrade', 'loan_purpose', 'employment_status', 'home_ownership']

    return model, summary

def predict(
    model: TabularPredictor,
    test_df: pd.DataFrame,
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> pd.DataFrame:
    """Generate predictions using stored encoder."""
    global target_encoder, woe_mappings

    # Restore encoder from artifacts
    if artifacts:
        if 'target_encoder' in artifacts:
            target_encoder = artifacts['target_encoder']
        if 'woe_mappings' in artifacts:
            woe_mappings = artifacts['woe_mappings']

    # Re-preprocess test_df with target encoding + WoE to match training features
    test_df = _engineer_features(test_df, is_train=False)

    return base_model.predict(model, test_df, config, artifacts)
