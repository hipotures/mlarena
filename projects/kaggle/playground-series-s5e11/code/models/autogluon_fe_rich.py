"""
AutoGluon model with enriched feature engineering (fe_rich).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from pathlib import Path
import sys

MODEL_DIR = Path(__file__).parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import autogluon_eda_features as base_model  # noqa: E402
from autogluon.tabular import TabularPredictor  # noqa: E402
from kaggle_tools.config_models import ModelConfig  # noqa: E402

PREP_DIR = Path(__file__).parent.parent / "preprocessing"
if str(PREP_DIR) not in sys.path:
    sys.path.insert(0, str(PREP_DIR))

from fe_rich import add_features  # noqa: E402

VARIANT_NAME = "feature-set-rich"


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


def preprocess(train_df, config: ModelConfig, is_train: bool = True):
    return add_features(train_df)


def train(
    train_df,
    val_df: Optional[Any],
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> Tuple[TabularPredictor, Dict[str, Any]]:
    print(f"[{VARIANT_NAME}] Training with enriched feature set.")
    return base_model.train(train_df, val_df, config, artifacts)


def predict(
    model: TabularPredictor,
    test_df,
    config: ModelConfig,
    artifacts: Optional[Any] = None,
):
    return base_model.predict(model, test_df, config, artifacts)
