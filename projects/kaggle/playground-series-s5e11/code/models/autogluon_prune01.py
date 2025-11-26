"""
AutoGluon prune01:
- Base = autogluon_baseline_pruned (minimal FE + iterative pruning)
- Adds 10 best-performing engineered features pulled from top variants
  (fe17, fe20, fe13, fe15, fe21 combo).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import sys

MODEL_DIR = Path(__file__).parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

import autogluon_baseline_pruned as base_model
from kaggle_tools.config_models import ModelConfig

VARIANT_NAME = "prune01"


def get_default_config() -> Dict[str, Any]:
    """Reuse base config (template will override to quick-test-xgb-pruned)."""
    return base_model.get_default_config()


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the top 10 features from the best-performing FE experiments."""
    enriched = base_model._engineer_features(df)

    # fe17: total income magnitude + EMI estimate
    total_income = enriched["annual_income"] + enriched.get("co_applicant_income", 0)
    enriched["log_total_income"] = np.log1p(total_income)

    monthly_rate = enriched["interest_rate"] / 12 / 100
    n_payments = (
        enriched["loan_term_months"].fillna(36)
        if "loan_term_months" in enriched
        else pd.Series(36, index=enriched.index)
    )
    monthly_rate_plus_1 = 1 + monthly_rate
    denominator = np.power(monthly_rate_plus_1, n_payments) - 1
    emi = np.divide(
        enriched["loan_amount"] * monthly_rate * np.power(monthly_rate_plus_1, n_payments),
        denominator,
        out=np.zeros_like(enriched["loan_amount"], dtype=float),
        where=denominator != 0,
    )
    enriched["EMI_estimate"] = emi

    # fe20: risk flags from employment/purpose
    enriched["is_student_flag"] = (
        (enriched["employment_status"] == "Student").astype(int)
        if "employment_status" in enriched.columns
        else 0
    )
    enriched["purpose_is_debt_consolidation_flag"] = (
        (enriched["loan_purpose"] == "Debt Consolidation").astype(int)
        if "loan_purpose" in enriched.columns
        else 0
    )

    # fe13: monthly scale of income/debt
    monthly_income = enriched["annual_income"] / 12
    enriched["monthly_income"] = monthly_income
    enriched["monthly_debt"] = monthly_income * enriched["debt_to_income_ratio"]

    # fe15: numeric grade / subgrade strength
    if "grade_subgrade" in enriched.columns:
        grade_series = enriched["grade_subgrade"].astype(str)
        enriched["grade_numeric"] = grade_series.str[0].map(
            {"A": 7, "B": 6, "C": 5, "D": 4, "E": 3, "F": 2, "G": 1}
        ).fillna(0)
        enriched["subgrade_numeric"] = pd.to_numeric(
            grade_series.str[1:],
            errors="coerce",
        ).fillna(0)
    else:
        enriched["grade_numeric"] = 0
        enriched["subgrade_numeric"] = 0

    # fe21 combo: repayment stress + blended risk score
    income_for_payment = monthly_income * 0.3
    enriched["payment_risk_ratio"] = np.divide(
        emi,
        income_for_payment,
        out=np.zeros_like(emi, dtype=float),
        where=income_for_payment != 0,
    )
    enriched["payment_risk_ratio"] = (
        enriched["payment_risk_ratio"].replace([np.inf, -np.inf], 0).fillna(0)
    )

    credit_score = enriched["credit_score"].fillna(0)
    dti_adj = enriched["debt_to_income_ratio"].fillna(0)
    enriched["risk_score_v2"] = (
        (credit_score / 850.0)
        * (1.0 / (dti_adj + 0.1))
        * np.log1p(enriched["annual_income"].fillna(0))
    )

    return enriched


def preprocess(df: pd.DataFrame, config: ModelConfig, is_train: bool = True) -> pd.DataFrame:
    """Apply feature engineering before AutoGluon handles pruning."""
    return _engineer_features(df)


def train(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> Tuple[TabularPredictor, Dict[str, Any]]:
    print(f"[{VARIANT_NAME}] Training with baseline-pruned + 10 top engineered features.")
    return base_model.train(train_df, val_df, config, artifacts)


def predict(
    model: TabularPredictor,
    test_df: pd.DataFrame,
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> pd.DataFrame:
    """Generate predictions using the same engineered feature set."""
    return base_model.predict(model, test_df, config, artifacts)
