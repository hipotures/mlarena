"""
Feature engineering bundle for playground-series-s5e11.

Adds domain-inspired ratios, payment burden proxies, and stability flags
on top of the existing AutoGluon EDA features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Reuse helpers from the existing engineered feature set.
MODEL_DIR = Path(__file__).parent.parent / "models"
import sys  # noqa: E402

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import autogluon_eda_features as base_model  # noqa: E402


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """Safe division with zero/inf handling."""
    safe_den = den.replace(0, pd.NA)
    ratio = num / safe_den
    return ratio.replace([np.inf, -np.inf], pd.NA).fillna(0)


def _approx_monthly_payment(loan_amount: pd.Series, annual_rate_pct: pd.Series, months: int) -> pd.Series:
    """Approximate annuity monthly payment; fall back to linear if rate is tiny."""
    monthly_rate = (annual_rate_pct / 100) / 12
    # Avoid division by zero for near-zero rates.
    tiny = monthly_rate.abs() < 1e-9
    annuity = loan_amount * monthly_rate * (1 + monthly_rate) ** months / ((1 + monthly_rate) ** months - 1)
    linear = loan_amount / months
    return np.where(tiny, linear, annuity)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich dataframe with domain features.

    Assumes columns: annual_income, debt_to_income_ratio, credit_score,
    loan_amount, interest_rate, loan_purpose, employment_status,
    education_level, marital_status, grade_subgrade (strings).
    """
    enriched = df.copy()
    enriched = base_model._engineer_features(enriched)

    # Basic derived incomes and burdens.
    enriched["monthly_income"] = enriched["annual_income"] / 12
    enriched["loan_to_income"] = _safe_ratio(enriched["loan_amount"], enriched["annual_income"])
    enriched["remaining_income"] = enriched["annual_income"] * (1 - enriched["debt_to_income_ratio"])

    # Monthly payment approximation (assume 36 months if term unknown).
    default_term_months = 36
    if "loan_term_months" in enriched.columns:
        term = enriched["loan_term_months"].fillna(default_term_months)
    else:
        term = default_term_months
    monthly_payment = _approx_monthly_payment(enriched["loan_amount"], enriched["interest_rate"], term)
    enriched["payment_income_ratio"] = _safe_ratio(monthly_payment, enriched["monthly_income"])
    enriched["income_after_payment"] = enriched["monthly_income"] - monthly_payment

    # Risk composites.
    enriched["interest_to_credit"] = _safe_ratio(enriched["interest_rate"], enriched["credit_score"])
    enriched["credit_risk_composite"] = enriched["credit_score"] / (enriched["debt_to_income_ratio"] + 1e-3)
    enriched["loan_cost_indicator"] = enriched["loan_amount"] * enriched["interest_rate"]

    # Purpose/employment stability flags.
    high_risk_purpose = {"Medical", "Vacation", "Other"}
    stable_employment = {"Employed", "Self-employed", "Retired"}
    enriched["purpose_group"] = np.where(
        enriched["loan_purpose"].isin(["Debt consolidation"]),
        "debt_consolidation",
        "other",
    )
    enriched["employment_stability_flag"] = enriched["employment_status"].isin(stable_employment).astype(int)
    enriched["high_risk_purpose_flag"] = enriched["loan_purpose"].isin(high_risk_purpose).astype(int)

    return enriched


__all__ = ["add_features"]
