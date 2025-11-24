"""
AutoGluon variant #21: Ultimate combination of best features from fe17, fe20, fe13, fe15.
Combines the top-performing feature sets discovered through systematic testing.
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

VARIANT_NAME = "feature-set-21-ultimate"

def get_default_config() -> Dict[str, Any]:
    """Extended time for better performance."""
    return {
        "hyperparameters": {
            "presets": "best_quality",
            "time_limit": 10800,  # 3 hours for best results
            "num_bag_folds": 8,   # More folds for stability
            "num_stack_levels": 2, # Deeper stacking
            "use_gpu": False,
            "excluded_models": ["NN_TORCH"],  # Skip slow neural networks
        },
        "model": {
            "leaderboard_rows": 30,
        },
    }

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Combine all best-performing features from experiments."""
    
    # Start with rich baseline
    enriched = base_model._engineer_features(df)
    
    # === Features from fe17 (BEST: 0.92356) ===
    # 1. Log of Total Income
    if 'annual_income' in enriched.columns and 'co_applicant_income' in enriched.columns:
        total_income = enriched['annual_income'] + enriched['co_applicant_income']
    else:
        total_income = enriched.get('annual_income', pd.Series(0, index=df.index))
    enriched['log_total_income'] = np.log1p(total_income)
    
    # 2. EMI Estimate (Equated Monthly Installment)
    monthly_rate = enriched['interest_rate'] / 12 / 100
    n_payments = enriched.get('loan_term_months', 36)  # Default 36 months
    
    monthly_rate_plus_1 = 1 + monthly_rate
    denominator = np.power(monthly_rate_plus_1, n_payments) - 1
    
    emi = np.divide(
        enriched['loan_amount'] * monthly_rate * np.power(monthly_rate_plus_1, n_payments),
        denominator,
        out=np.zeros_like(enriched['loan_amount'], dtype=float),
        where=(denominator != 0)
    )
    enriched['EMI_estimate'] = emi
    
    # === Features from fe20 (2nd BEST: 0.92353) ===
    # Student flag
    if 'employment_status' in enriched.columns:
        enriched['is_student_flag'] = (enriched['employment_status'] == 'Student').astype(int)
    else:
        enriched['is_student_flag'] = 0
    
    # Debt consolidation flag
    if 'loan_purpose' in enriched.columns:
        enriched['purpose_is_debt_consolidation_flag'] = (
            enriched['loan_purpose'] == 'Debt Consolidation'
        ).astype(int)
    else:
        enriched['purpose_is_debt_consolidation_flag'] = 0
    
    # === Features from fe13 (3rd BEST: 0.92331) ===
    enriched['monthly_income'] = enriched['annual_income'] / 12
    enriched['monthly_debt'] = enriched['monthly_income'] * enriched['debt_to_income_ratio']
    
    # === Features from fe15 (4th BEST: 0.92330) ===
    if 'grade_subgrade' in enriched.columns:
        enriched['grade_char'] = enriched['grade_subgrade'].str[0]
        enriched['subgrade_numeric'] = pd.to_numeric(
            enriched['grade_subgrade'].str[1:], errors='coerce'
        ).fillna(0)
        
        grade_map = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}
        enriched['grade_numeric'] = enriched['grade_char'].map(grade_map).fillna(0)
    else:
        enriched['subgrade_numeric'] = 0
        enriched['grade_numeric'] = 0
    
    # === Additional high-value features based on analysis ===
    # Payment risk ratio (30% income rule)
    enriched['payment_risk_ratio'] = emi / (enriched['monthly_income'] * 0.3)
    enriched['payment_risk_ratio'] = enriched['payment_risk_ratio'].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)
    
    # Advanced risk score
    enriched['risk_score_v2'] = (
        (enriched['credit_score'] / 850) * 
        (1 / (enriched['debt_to_income_ratio'] + 0.1)) * 
        np.log1p(enriched['annual_income'])
    )
    
    # Loan burden index
    safe_denominator = enriched['annual_income'] * (1 - enriched['debt_to_income_ratio'] + 0.01)
    enriched['loan_burden_index'] = enriched['loan_amount'] / safe_denominator
    
    # Income-credit power interaction
    enriched['income_credit_power'] = enriched['annual_income'] * (enriched['credit_score'] / 850)
    
    # High-risk categorical flags
    enriched['high_dti_flag'] = (enriched['debt_to_income_ratio'] > 0.4).astype(int)
    enriched['low_credit_flag'] = (enriched['credit_score'] < 600).astype(int)
    enriched['high_interest_flag'] = (enriched['interest_rate'] > 15).astype(int)
    
    # Combined risk indicator
    enriched['risk_flags_sum'] = (
        enriched['high_dti_flag'] + 
        enriched['low_credit_flag'] + 
        enriched['high_interest_flag'] +
        enriched['is_student_flag']
    )
    
    return enriched

def preprocess(df: pd.DataFrame, config: ModelConfig, is_train: bool = True) -> pd.DataFrame:
    """Apply ultimate feature engineering."""
    return _engineer_features(df)

def train(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> Tuple[TabularPredictor, Dict[str, Any]]:
    """Train with extended time and optimized hyperparameters."""
    print(f"[{VARIANT_NAME}] Training ULTIMATE model with best features from all experiments.")
    print(f"[{VARIANT_NAME}] Total features: {len(train_df.columns)}")
    print(f"[{VARIANT_NAME}] Using 3-hour training with 8-fold CV and 2-level stacking.")
    
    return base_model.train(train_df, val_df, config, artifacts)

def predict(
    model: TabularPredictor,
    test_df: pd.DataFrame,
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> pd.DataFrame:
    """Generate predictions."""
    return base_model.predict(model, test_df, config, artifacts)
