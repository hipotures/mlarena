"""
AutoGluon variant #18: rich baseline + income after EMI and income credit power.
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

VARIANT_NAME = "feature-set-18"

def get_default_config() -> Dict[str, Any]:
    return base_model.get_default_config()

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = base_model._engineer_features(df)
    
    # --- Dependencies for new features ---
    monthly_income = enriched['annual_income'] / 12
    
    # EMI Estimate calculation
    monthly_rate = enriched['interest_rate'] / 12 / 100
    n_payments = enriched.get('loan_term_months', 36)
    monthly_rate_plus_1 = 1 + monthly_rate
    denominator = np.power(monthly_rate_plus_1, n_payments) - 1
    emi = np.divide(
        enriched['loan_amount'] * monthly_rate * np.power(monthly_rate_plus_1, n_payments),
        denominator,
        out=np.zeros_like(enriched['loan_amount'], dtype=float),
        where=(denominator!=0)
    )

    # --- New features for this variant ---
    enriched['income_after_EMI'] = monthly_income - emi
    enriched['income_credit_power'] = (enriched['annual_income'] * enriched['credit_score']) / 100000

    return enriched

def preprocess(df: pd.DataFrame, config: ModelConfig, is_train: bool = True) -> pd.DataFrame:
    return _engineer_features(df)

def train(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> Tuple[TabularPredictor, Dict[str, Any]]:
    print(f"[{VARIANT_NAME}] Training with income_after_EMI + income_credit_power.")
    return base_model.train(train_df, val_df, config, artifacts)

def predict(
    model: TabularPredictor,
    test_df: pd.DataFrame,
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> pd.DataFrame:
    return base_model.predict(model, test_df, config, artifacts)
