"""
AutoGluon variant #20: rich baseline + student and debt consolidation flags.
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

VARIANT_NAME = "feature-set-20"

def get_default_config() -> Dict[str, Any]:
    return base_model.get_default_config()

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = base_model._engineer_features(df)
    
    # New features for this variant
    if 'employment_status' in enriched.columns:
        enriched['is_student_flag'] = (enriched['employment_status'] == 'Student').astype(int)
    else:
        enriched['is_student_flag'] = 0

    if 'loan_purpose' in enriched.columns:
        enriched['purpose_is_debt_consolidation_flag'] = (enriched['loan_purpose'] == 'Debt Consolidation').astype(int)
    else:
        enriched['purpose_is_debt_consolidation_flag'] = 0

    return enriched

def preprocess(df: pd.DataFrame, config: ModelConfig, is_train: bool = True) -> pd.DataFrame:
    return _engineer_features(df)

def train(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> Tuple[TabularPredictor, Dict[str, Any]]:
    print(f"[{VARIANT_NAME}] Training with is_student_flag + purpose_is_debt_consolidation_flag.")
    return base_model.train(train_df, val_df, config, artifacts)

def predict(
    model: TabularPredictor,
    test_df: pd.DataFrame,
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> pd.DataFrame:
    return base_model.predict(model, test_df, config, artifacts)
