"""
AutoGluon variant #2: base EDA features + income bands and leverage vs. credit.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from pathlib import Path
import sys

MODEL_DIR = Path(__file__).parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import autogluon_eda_features as base_model
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

from kaggle_tools.config_models import ModelConfig

VARIANT_NAME = "feature-set-02"


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

    enriched["loan_to_credit_ratio"] = base_model._safe_ratio(enriched["loan_amount"], enriched["credit_score"])
    enriched["income_bucket"] = pd.cut(
        enriched["annual_income"],
        bins=[0, 25000, 40000, 60000, 90000, np.inf],
        labels=["<=25k", "25-40k", "40-60k", "60-90k", "90k+"],
        include_lowest=True,
    ).astype(str)

    return enriched


def preprocess(df: pd.DataFrame, config: ModelConfig, is_train: bool = True) -> pd.DataFrame:
    return _engineer_features(df)


def train(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> Tuple[TabularPredictor, Dict[str, Any]]:
    print(f"[{VARIANT_NAME}] Training with loan_to_credit_ratio + income_bucket.")
    return base_model.train(train_df, val_df, config, artifacts)


def predict(
    model: TabularPredictor,
    test_df: pd.DataFrame,
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> pd.DataFrame:
    return base_model.predict(model, test_df, config, artifacts)
