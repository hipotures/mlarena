"""
AutoGluon variant #16: rich baseline + grade/interest rate interaction and income per capita.
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

VARIANT_NAME = "feature-set-16"

def get_default_config() -> Dict[str, Any]:
    return base_model.get_default_config()

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = base_model._engineer_features(df)
    
    # New features for this variant
    # 1. Difference between interest rate and the average for its grade
    if 'grade_letter' in enriched.columns:
        grade_avg_interest = enriched.groupby('grade_letter')['interest_rate'].transform('mean')
        enriched['grade_x_int_rate_diff'] = enriched['interest_rate'] - grade_avg_interest
    else:
        enriched['grade_x_int_rate_diff'] = 0

    # 2. Income per capita
    # First, ensure Total_Income exists (it's not in the base EDA file)
    if 'annual_income' in enriched.columns and 'co_applicant_income' in enriched.columns:
        total_income = enriched['annual_income'] + enriched['co_applicant_income']
    else:
        total_income = enriched.get('annual_income', pd.Series(0, index=df.index))

    # Clean up Dependents column
    if 'dependents' in enriched.columns:
        dependents_numeric = pd.to_numeric(enriched['dependents'].astype(str).str.replace('+', ''), errors='coerce').fillna(0)
    else:
        dependents_numeric = pd.Series(0, index=df.index)
        
    enriched['income_per_capita'] = base_model._safe_ratio(total_income, dependents_numeric + 1)


    return enriched

def preprocess(df: pd.DataFrame, config: ModelConfig, is_train: bool = True) -> pd.DataFrame:
    return _engineer_features(df)

def train(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> Tuple[TabularPredictor, Dict[str, Any]]:
    print(f"[{VARIANT_NAME}] Training with grade_x_int_rate_diff + income_per_capita.")
    return base_model.train(train_df, val_df, config, artifacts)

def predict(
    model: TabularPredictor,
    test_df: pd.DataFrame,
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> pd.DataFrame:
    return base_model.predict(model, test_df, config, artifacts)
