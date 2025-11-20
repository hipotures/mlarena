"""
AutoGluon variant #4: base EDA features + purpose/employment and gender/marital interactions.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from pathlib import Path
import sys

MODEL_DIR = Path(__file__).parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import autogluon_eda_features as base_model
import pandas as pd
from autogluon.tabular import TabularPredictor

from kaggle_tools.config_models import ModelConfig

VARIANT_NAME = "feature-set-04"


def get_default_config() -> Dict[str, Any]:
    return {
        "hyperparameters": {
            "presets": "best_quality",
            "time_limit": 3600,
            "use_gpu": False,
        },
        "model": {
            "leaderboard_rows": 20,
        },
    }


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = base_model._engineer_features(df)

    enriched["purpose_employment"] = (
        enriched["loan_purpose"].astype(str) + "_" + enriched["employment_status"].astype(str)
    )
    enriched["gender_marital"] = enriched["gender"].astype(str) + "_" + enriched["marital_status"].astype(str)

    return enriched


def preprocess(df: pd.DataFrame, config: ModelConfig, is_train: bool = True) -> pd.DataFrame:
    return _engineer_features(df)


def train(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> Tuple[TabularPredictor, Dict[str, Any]]:
    print(f"[{VARIANT_NAME}] Training with purpose_employment + gender_marital.")
    return base_model.train(train_df, val_df, config, artifacts)


def predict(
    model: TabularPredictor,
    test_df: pd.DataFrame,
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> pd.DataFrame:
    return base_model.predict(model, test_df, config, artifacts)
