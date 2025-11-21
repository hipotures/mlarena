"""
Tier 1 Feature Engineering: Log transforms, DTI features, and critical ratios.

Expected improvement: +0.03-0.05 AUC
Based on Claude analysis documentation recommendations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import PowerTransformer


def _safe_ratio(num: pd.Series, den: pd.Series, fill_value: float = 0.0) -> pd.Series:
    """Safe division with zero/inf handling."""
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = num / den
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    return ratio.fillna(fill_value)


def _approx_monthly_payment(
    loan_amount: pd.Series,
    annual_rate_pct: pd.Series,
    months: int = 36
) -> pd.Series:
    """
    Calculate monthly payment using annuity formula.

    Formula: M = P * [r(1+r)^n] / [(1+r)^n - 1]
    where P=loan amount, r=monthly rate, n=number of months
    """
    monthly_rate = (annual_rate_pct / 100) / 12

    # For very small rates, use linear approximation
    tiny_rate = monthly_rate.abs() < 1e-9

    # Annuity formula
    factor = (1 + monthly_rate) ** months
    annuity = loan_amount * monthly_rate * factor / (factor - 1)

    # Linear fallback for zero/tiny rates
    linear = loan_amount / months

    return np.where(tiny_rate, linear, annuity)


def add_tier1_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Tier 1 critical features from documentation analysis.

    Features added:
    1. Log transformations (income, loan_amount) - reduces skewness
    2. Yeo-Johnson transformations - handles zeros and negatives
    3. DTI-based features - payment capacity, monthly debt
    4. Critical ratios - loan_to_income, payment_to_income
    5. Income derivatives - remaining income, income after payment
    6. Interest cost metrics - total interest, interest to income
    """
    enriched = df.copy()

    # ============================================================
    # 1. LOG TRANSFORMATIONS (for right-skewed distributions)
    # ============================================================
    enriched['log_annual_income'] = np.log1p(enriched['annual_income'])
    enriched['log_loan_amount'] = np.log1p(enriched['loan_amount'])

    # Square root for DTI (less aggressive than log)
    if enriched['debt_to_income_ratio'].min() >= 0:
        enriched['sqrt_dti'] = np.sqrt(enriched['debt_to_income_ratio'])

    # ============================================================
    # 2. YEO-JOHNSON TRANSFORMATION (more robust than Box-Cox)
    # ============================================================
    # Only for highly skewed features
    skewed_features = []
    for col in ['annual_income', 'loan_amount']:
        if col in enriched.columns:
            skewness = stats.skew(enriched[col].dropna())
            if abs(skewness) > 1.0:  # Highly skewed
                skewed_features.append(col)

    if skewed_features:
        pt = PowerTransformer(method='yeo-johnson', standardize=False)
        for col in skewed_features:
            enriched[f'{col}_yj'] = pt.fit_transform(enriched[[col]])

    # ============================================================
    # 3. INCOME DERIVATIVES
    # ============================================================
    enriched['monthly_income'] = enriched['annual_income'] / 12

    # Income after debt obligations
    enriched['monthly_debt'] = enriched['monthly_income'] * enriched['debt_to_income_ratio']
    enriched['remaining_income'] = enriched['monthly_income'] - enriched['monthly_debt']
    enriched['remaining_income_annual'] = enriched['annual_income'] * (1 - enriched['debt_to_income_ratio'])

    # Payment capacity (key metric from documentation)
    enriched['payment_capacity'] = enriched['remaining_income']

    # ============================================================
    # 4. LOAN TO INCOME RATIOS (critical predictors)
    # ============================================================
    enriched['loan_to_income_ratio'] = _safe_ratio(
        enriched['loan_amount'],
        enriched['annual_income']
    )

    # Alternative velocity metric (loan / sqrt(income))
    enriched['loan_to_income_velocity'] = _safe_ratio(
        enriched['loan_amount'],
        np.sqrt(enriched['annual_income'])
    )

    # ============================================================
    # 5. MONTHLY PAYMENT AND PAYMENT RATIOS
    # ============================================================
    # Assume 36 months if term not specified
    monthly_payment = _approx_monthly_payment(
        enriched['loan_amount'],
        enriched['interest_rate'],
        months=36
    )
    enriched['monthly_payment_est'] = monthly_payment

    # Payment to income ratio (DTI equivalent for new loan)
    enriched['payment_income_ratio'] = _safe_ratio(
        monthly_payment,
        enriched['monthly_income']
    )

    # Income after new payment
    enriched['income_after_payment'] = enriched['monthly_income'] - monthly_payment

    # Payment burden as percentage
    enriched['payment_burden_pct'] = _safe_ratio(
        monthly_payment,
        enriched['monthly_income']
    ) * 100

    # ============================================================
    # 6. INTEREST COST METRICS
    # ============================================================
    # Total interest over loan life (simplified: assume 3 years)
    enriched['total_interest_cost'] = enriched['loan_amount'] * enriched['interest_rate'] / 100 * 3

    # Interest to income ratio
    enriched['interest_to_income'] = _safe_ratio(
        enriched['total_interest_cost'],
        enriched['annual_income']
    )

    # Interest burden (annual)
    enriched['annual_interest_cost'] = enriched['loan_amount'] * enriched['interest_rate'] / 100
    enriched['interest_burden_ratio'] = _safe_ratio(
        enriched['annual_interest_cost'],
        enriched['annual_income']
    )

    # ============================================================
    # 7. COMBINED DTI (existing + new loan)
    # ============================================================
    # What would DTI be after taking this loan?
    new_monthly_obligation = monthly_payment
    total_monthly_debt = enriched['monthly_debt'] + new_monthly_obligation
    enriched['combined_dti'] = _safe_ratio(
        total_monthly_debt,
        enriched['monthly_income']
    )

    # Disposable income after all obligations
    enriched['disposable_income'] = enriched['monthly_income'] - total_monthly_debt

    # ============================================================
    # 8. CREDIT SCORE NORMALIZED METRICS
    # ============================================================
    # Normalize credit score to 0-1 range (assuming max 850)
    enriched['credit_score_norm'] = enriched['credit_score'] / 850.0

    # Credit-adjusted loan amount (higher score = can handle more)
    enriched['credit_adjusted_loan'] = enriched['loan_amount'] / (enriched['credit_score'] + 1)

    # ============================================================
    # 9. RISK FLAGS
    # ============================================================
    # High DTI flag (>40% is risky)
    enriched['high_dti_flag'] = (enriched['debt_to_income_ratio'] > 0.40).astype(int)

    # High combined DTI flag
    enriched['high_combined_dti_flag'] = (enriched['combined_dti'] > 0.50).astype(int)

    # Low remaining income flag
    enriched['low_remaining_income_flag'] = (enriched['remaining_income'] < 1000).astype(int)

    # High loan to income flag (>3x annual income)
    enriched['high_loan_to_income_flag'] = (enriched['loan_to_income_ratio'] > 3.0).astype(int)

    return enriched


def preprocess_tier1(train_df: pd.DataFrame, test_df: pd.DataFrame = None):
    """
    Apply Tier 1 feature engineering to train and test sets.

    Returns:
        If test_df is None: enriched train_df
        Otherwise: (enriched_train, enriched_test)
    """
    train_enriched = add_tier1_features(train_df)

    if test_df is not None:
        test_enriched = add_tier1_features(test_df)
        return train_enriched, test_enriched

    return train_enriched


__all__ = ['add_tier1_features', 'preprocess_tier1']
