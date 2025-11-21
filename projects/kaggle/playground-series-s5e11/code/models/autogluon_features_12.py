"""
AutoGluon variant #12: rich baseline + loan to income velocity and risk adjusted return.
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import sys
import numpy as np

MODEL_DIR = Path(__file__).parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import autogluon_features_rich_baseline as base_model
import pandas as pd
from autogluon.tabular import TabularPredictor
from kaggle_tools.config_models import ModelConfig

VARIANT_NAME = "feature-set-12"

def get_default_config() -> Dict[str, Any]:
    return base_model.get_default_config()

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = base_model._engineer_features(df)
    
    # New features for this variant
    enriched['loan_to_income_velocity'] = enriched['loan_amount'] / np.sqrt(enriched['annual_income'] + 1)
    enriched['risk_adjusted_return'] = (enriched['interest_rate'] * 100) / (enriched['credit_score'] / 10)

    return enriched

def preprocess(df: pd.DataFrame, config: ModelConfig, is_train: bool = True) -> pd.DataFrame:
    return _engineer_features(df)

def train(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> Tuple[TabularPredictor, Dict[str, Any]]:
    print(f"[{VARIANT_NAME}] Training with loan_to_income_velocity + risk_adjusted_return.")
    return base_model.train(train_df, val_df, config, artifacts)

def predict(
    model: TabularPredictor,
    test_df: pd.DataFrame,
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> pd.DataFrame:
    return base_model.predict(model, test_df, config, artifacts)
