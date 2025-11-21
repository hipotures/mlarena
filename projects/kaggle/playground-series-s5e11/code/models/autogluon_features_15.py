"""
AutoGluon variant #15: rich baseline + numeric grade and subgrade features.
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
from autogluon.tabular import TabularPredictor
from kaggle_tools.config_models import ModelConfig

VARIANT_NAME = "feature-set-15"

def get_default_config() -> Dict[str, Any]:
    return base_model.get_default_config()

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = base_model._engineer_features(df)
    
    # New features for this variant
    if 'grade_subgrade' in enriched.columns:
        enriched['grade_char'] = enriched['grade_subgrade'].str[0]
        enriched['subgrade_numeric'] = pd.to_numeric(enriched['grade_subgrade'].str[1:], errors='coerce').fillna(0)

        grade_map = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}
        enriched['grade_numeric'] = enriched['grade_char'].map(grade_map).fillna(0)
    else:
        # Create dummy columns if the source is not there to avoid errors
        enriched['subgrade_numeric'] = 0
        enriched['grade_numeric'] = 0


    return enriched

def preprocess(df: pd.DataFrame, config: ModelConfig, is_train: bool = True) -> pd.DataFrame:
    return _engineer_features(df)

def train(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> Tuple[TabularPredictor, Dict[str, Any]]:
    print(f"[{VARIANT_NAME}] Training with grade_numeric + subgrade_numeric.")
    return base_model.train(train_df, val_df, config, artifacts)

def predict(
    model: TabularPredictor,
    test_df: pd.DataFrame,
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> pd.DataFrame:
    return base_model.predict(model, test_df, config, artifacts)
