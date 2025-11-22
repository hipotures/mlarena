"""
Consolidated baseline with all successful feature engineering variants from 01-10.
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

VARIANT_NAME = "feature-set-rich-baseline"


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


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    safe_den = den.replace(0, pd.NA)
    ratio = num / safe_den
    return ratio.replace([np.inf, -np.inf], pd.NA).fillna(0)


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = base_model._engineer_features(df)

    # Features from fe01
    enriched["interest_burden_ratio"] = base_model._safe_ratio(
        enriched["interest_rate"] * enriched["loan_amount"],
        enriched["annual_income"],
    )
    enriched["dti_interest_product"] = enriched["debt_to_income_ratio"] * enriched["interest_rate"]

    # Features from fe02
    enriched["loan_to_credit_ratio"] = base_model._safe_ratio(enriched["loan_amount"], enriched["credit_score"])
    enriched["income_bucket"] = pd.cut(
        enriched["annual_income"],
        bins=[0, 25000, 40000, 60000, 90000, np.inf],
        labels=["<=25k", "25-40k", "40-60k", "60-90k", "90k+"],
        include_lowest=True,
    ).astype(str)

    # Features from fe03
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

    # Features from fe04
    enriched["purpose_employment"] = (
        enriched["loan_purpose"].astype(str) + "_" + enriched["employment_status"].astype(str)
    )
    enriched["gender_marital"] = enriched["gender"].astype(str) + "_" + enriched["marital_status"].astype(str)

    # Features from fe05
    enriched["grade_interest_score"] = enriched["grade_letter_score"] * enriched["interest_rate"]
    enriched["loan_to_income_bucket"] = pd.cut(
        enriched["loan_to_income_ratio"],
        bins=[0, 0.05, 0.1, 0.2, 0.35, 0.5, float("inf")],
        labels=["<=5%", "5-10%", "10-20%", "20-35%", "35-50%", "50%+"],
        include_lowest=True,
    ).astype(str)

    # Features from fe06
    enriched["dti_squared"] = np.square(enriched["debt_to_income_ratio"])
    enriched["interest_rate_squared"] = np.square(enriched["interest_rate"])

    # Features from fe07
    high_risk_purposes = {"Medical", "Vacation", "Other"}
    stable_employment = {"Employed", "Self-employed"}
    enriched["high_risk_purpose_flag"] = enriched["loan_purpose"].isin(high_risk_purposes).astype(int)
    enriched["stable_employment_flag"] = enriched["employment_status"].isin(stable_employment).astype(int)

    # Features from fe09
    residual_income = enriched["annual_income"] * (1 - enriched["debt_to_income_ratio"])
    enriched["residual_income"] = residual_income
    enriched["residual_income_ratio"] = base_model._safe_ratio(residual_income, enriched["loan_amount"])

    # Features from fe10
    enriched["interest_rate_to_credit_ratio"] = base_model._safe_ratio(
        enriched["interest_rate"],
        enriched["credit_score"],
    )
    enriched["low_income_high_rate_flag"] = (
        (enriched["annual_income"] < 30000) & (enriched["interest_rate"] > 15)
    ).astype(int)

    return enriched


def preprocess(df: pd.DataFrame, config: ModelConfig, is_train: bool = True) -> pd.DataFrame:
    return _engineer_features(df)


def train(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> Tuple[TabularPredictor, Dict[str, Any]]:
    print(f"[{VARIANT_NAME}] Training with {len(train_df.columns)} total features.")
    # We can reuse the base model's train and predict as the core logic is the same.
    return base_model.train(train_df, val_df, config, artifacts)


def predict(
    model: TabularPredictor,
    test_df: pd.DataFrame,
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> pd.DataFrame:
    return base_model.predict(model, test_df, config, artifacts)
