"""
AutoGluon variant #17: rich baseline + log of total income and EMI estimate.
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import sys

MODEL_DIR = Path(__file__).parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import autogluon_features_rich_baseline as base_model
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from kaggle_tools.config_models import ModelConfig

VARIANT_NAME = "feature-set-17"

def get_default_config() -> Dict[str, Any]:
    return base_model.get_default_config()

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = base_model._engineer_features(df)
    
    # New features for this variant
    # 1. Log of Total Income
    if 'annual_income' in enriched.columns and 'co_applicant_income' in enriched.columns:
        total_income = enriched['annual_income'] + enriched['co_applicant_income']
    else:
        total_income = enriched.get('annual_income', pd.Series(0, index=df.index))
    enriched['log_total_income'] = np.log1p(total_income)

    # 2. EMI Estimate
    monthly_rate = enriched['interest_rate'] / 12 / 100
    n_payments = enriched.get('loan_term_months', 36) # Assume 36 months if not available
    
    # Handle cases where rate is -1 (which would lead to 0 in denominator)
    monthly_rate_plus_1 = 1 + monthly_rate
    denominator = np.power(monthly_rate_plus_1, n_payments) - 1
    
    # Avoid division by zero
    # Using np.divide to handle it safely
    emi = np.divide(
        enriched['loan_amount'] * monthly_rate * np.power(monthly_rate_plus_1, n_payments),
        denominator,
        out=np.zeros_like(enriched['loan_amount'], dtype=float),
        where=(denominator!=0)
    )
    enriched['EMI_estimate'] = emi

    return enriched

def preprocess(df: pd.DataFrame, config: ModelConfig, is_train: bool = True) -> pd.DataFrame:
    return _engineer_features(df)

def train(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> Tuple[TabularPredictor, Dict[str, Any]]:
    print(f"[{VARIANT_NAME}] Training with log_total_income + EMI_estimate.")
    return base_model.train(train_df, val_df, config, artifacts)

def predict(
    model: TabularPredictor,
    test_df: pd.DataFrame,
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> pd.DataFrame:
    return base_model.predict(model, test_df, config, artifacts)
