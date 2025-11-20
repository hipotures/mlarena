"""
AutoGluon variant #3: base EDA features + loan amount bands and ordered credit buckets.
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

VARIANT_NAME = "feature-set-03"


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

    enriched["loan_amount_bucket"] = pd.cut(
        enriched["loan_amount"],
        bins=[0, 5000, 10000, 20000, 40000, 80000, np.inf],
        labels=["<=5k", "5-10k", "10-20k", "20-40k", "40-80k", "80k+"],
        include_lowest=True,
    ).astype(str)
    enriched["credit_score_bucket_numeric"] = pd.Categorical(
        enriched["credit_score_bucket"],
        categories=base_model.CS_BUCKETS["labels"],
        ordered=True,
    ).codes

    return enriched


def preprocess(df: pd.DataFrame, config: ModelConfig, is_train: bool = True) -> pd.DataFrame:
    return _engineer_features(df)


def train(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> Tuple[TabularPredictor, Dict[str, Any]]:
    print(f"[{VARIANT_NAME}] Training with loan_amount_bucket + credit_score_bucket_numeric.")
    return base_model.train(train_df, val_df, config, artifacts)


def predict(
    model: TabularPredictor,
    test_df: pd.DataFrame,
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> pd.DataFrame:
    return base_model.predict(model, test_df, config, artifacts)
