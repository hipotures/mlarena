"""
Tier 2 Feature Engineering: Advanced encodings and interactions.

Features:
- Target Encoding with CV (prevents leakage)
- Weight of Evidence (WoE) encoding
- Polynomial features (degree 2)
- Cross-feature interactions
- Risk-adjusted metrics

Expected improvement: +0.02-0.04 AUC
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings('ignore', category=RuntimeWarning)


class TargetEncoderCV:
    """
    Target Encoding with cross-validation to prevent leakage.

    Uses out-of-fold encoding for training and global mean for test.
    Includes smoothing to regularize rare categories.
    """

    def __init__(self, cols: List[str], smoothing: float = 10.0, min_samples: int = 20):
        self.cols = cols
        self.smoothing = smoothing
        self.min_samples = min_samples
        self.global_means_ = {}
        self.encodings_ = {}

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, n_folds: int = 5) -> pd.DataFrame:
        """
        Fit and transform training data using out-of-fold encoding.

        Args:
            X: Training features
            y: Target variable
            n_folds: Number of CV folds

        Returns:
            Transformed DataFrame with encoded columns
        """
        X_encoded = X.copy()
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        for col in self.cols:
            # Calculate global mean for this column
            self.global_means_[col] = y.mean()

            # Initialize encoded column
            X_encoded[f'{col}_te'] = 0.0

            # Out-of-fold encoding
            for train_idx, val_idx in skf.split(X, y):
                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]

                # Calculate category statistics on training fold
                category_stats = y_train.groupby(X_train[col]).agg(['sum', 'count'])

                # Smoothed encoding
                encoding = self._smooth_encoding(
                    category_stats['sum'],
                    category_stats['count'],
                    self.global_means_[col]
                )

                # Apply to validation fold
                X_encoded.loc[val_idx, f'{col}_te'] = X.loc[val_idx, col].map(encoding).fillna(
                    self.global_means_[col]
                )

            # Store global encoding for transform()
            global_stats = y.groupby(X[col]).agg(['sum', 'count'])
            self.encodings_[col] = self._smooth_encoding(
                global_stats['sum'],
                global_stats['count'],
                self.global_means_[col]
            )

        return X_encoded

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform test data using global encodings."""
        X_encoded = X.copy()

        for col in self.cols:
            X_encoded[f'{col}_te'] = X[col].map(self.encodings_[col]).fillna(
                self.global_means_[col]
            )

        return X_encoded

    def _smooth_encoding(
        self,
        sum_values: pd.Series,
        count_values: pd.Series,
        global_mean: float
    ) -> pd.Series:
        """
        Apply smoothing to target encoding.

        Formula: (sum + smoothing * global_mean) / (count + smoothing)
        """
        smoothed = (sum_values + self.smoothing * global_mean) / (count_values + self.smoothing)
        return smoothed


class WoEEncoder:
    """
    Weight of Evidence (WoE) Encoder.

    WoE = ln(% Good / % Bad)
    Commonly used in credit scoring.
    """

    def __init__(self, cols: List[str], epsilon: float = 1e-5):
        self.cols = cols
        self.epsilon = epsilon  # Small value to avoid log(0)
        self.woe_mappings_ = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'WoEEncoder':
        """Calculate WoE for each category."""
        n_good = (y == 1).sum()
        n_bad = (y == 0).sum()

        for col in self.cols:
            woe_dict = {}

            for category in X[col].unique():
                mask = X[col] == category
                good = (y[mask] == 1).sum()
                bad = (y[mask] == 0).sum()

                # Calculate distribution percentages
                pct_good = (good + self.epsilon) / (n_good + self.epsilon)
                pct_bad = (bad + self.epsilon) / (n_bad + self.epsilon)

                # WoE = ln(% Good / % Bad)
                woe = np.log(pct_good / pct_bad)
                woe_dict[category] = woe

            self.woe_mappings_[col] = woe_dict

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply WoE encoding."""
        X_encoded = X.copy()

        for col in self.cols:
            X_encoded[f'{col}_woe'] = X[col].map(self.woe_mappings_[col]).fillna(0)

        return X_encoded

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


def add_tier2_features(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    target_col: str = 'loan_paid_back',
    include_tier1: bool = True
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Dict]:
    """
    Add Tier 2 advanced encoding and interaction features.

    Args:
        train_df: Training data (with target)
        test_df: Test data (optional)
        target_col: Name of target column
        include_tier1: Whether to include Tier 1 features first

    Returns:
        Tuple of (enriched_train, enriched_test, artifacts)
        artifacts contains fitted encoders for later use
    """
    artifacts = {}

    # Start with copies
    train_enriched = train_df.copy()
    test_enriched = test_df.copy() if test_df is not None else None

    # Optionally add Tier 1 features first
    if include_tier1:
        from fe_tier1 import add_tier1_features
        train_enriched = add_tier1_features(train_enriched)
        if test_enriched is not None:
            test_enriched = add_tier1_features(test_enriched)

    # Separate target
    y_train = train_enriched[target_col]
    X_train = train_enriched.drop(columns=[target_col])
    X_test = test_enriched.drop(columns=[target_col], errors='ignore') if test_enriched is not None else None

    # ============================================================
    # 1. TARGET ENCODING (high-cardinality categoricals)
    # ============================================================
    target_encode_cols = ['grade_subgrade', 'loan_purpose']
    target_encode_cols = [c for c in target_encode_cols if c in X_train.columns]

    if target_encode_cols:
        print(f"Applying Target Encoding to: {target_encode_cols}")
        te = TargetEncoderCV(cols=target_encode_cols, smoothing=10.0)
        X_train = te.fit_transform(X_train, y_train, n_folds=5)

        if X_test is not None:
            X_test = te.transform(X_test)

        artifacts['target_encoder'] = te

    # ============================================================
    # 2. WEIGHT OF EVIDENCE ENCODING
    # ============================================================
    woe_cols = ['loan_purpose', 'employment_status', 'education_level']
    woe_cols = [c for c in woe_cols if c in X_train.columns]

    if woe_cols:
        print(f"Applying WoE Encoding to: {woe_cols}")
        woe = WoEEncoder(cols=woe_cols)
        X_train = woe.fit_transform(X_train, y_train)

        if X_test is not None:
            X_test = woe.transform(X_test)

        artifacts['woe_encoder'] = woe

    # ============================================================
    # 3. POLYNOMIAL FEATURES (degree 2, key variables only)
    # ============================================================
    poly_cols = ['credit_score', 'debt_to_income_ratio', 'annual_income']
    poly_cols = [c for c in poly_cols if c in X_train.columns]

    if poly_cols:
        print(f"Creating polynomial features (degree 2) for: {poly_cols}")
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)

        poly_train = poly.fit_transform(X_train[poly_cols])
        poly_names = poly.get_feature_names_out(poly_cols)

        # Add selected polynomial features
        for i, name in enumerate(poly_names):
            # Skip original features (already present)
            if '^' in name or ' ' in name:  # Only interactions and powers
                X_train[f'poly_{name}'] = poly_train[:, i]

                if X_test is not None:
                    poly_test = poly.transform(X_test[poly_cols])
                    X_test[f'poly_{name}'] = poly_test[:, i]

        artifacts['poly_transformer'] = poly

    # ============================================================
    # 4. CROSS-FEATURE INTERACTIONS
    # ============================================================
    # Income × Credit Score (creditworthiness power)
    if 'annual_income' in X_train.columns and 'credit_score' in X_train.columns:
        X_train['income_credit_power'] = (X_train['annual_income'] * X_train['credit_score']) / 100000

        if X_test is not None:
            X_test['income_credit_power'] = (X_test['annual_income'] * X_test['credit_score']) / 100000

    # Loan Amount × Interest Rate (total cost indicator)
    if 'loan_amount' in X_train.columns and 'interest_rate' in X_train.columns:
        X_train['loan_cost_indicator'] = X_train['loan_amount'] * X_train['interest_rate']

        if X_test is not None:
            X_test['loan_cost_indicator'] = X_test['loan_amount'] * X_test['interest_rate']

    # Credit Score × DTI (risk composite)
    if 'credit_score' in X_train.columns and 'debt_to_income_ratio' in X_train.columns:
        X_train['credit_risk_score'] = X_train['credit_score'] / (X_train['debt_to_income_ratio'] + 0.01)

        if X_test is not None:
            X_test['credit_risk_score'] = X_test['credit_score'] / (X_test['debt_to_income_ratio'] + 0.01)

    # ============================================================
    # 5. RISK-ADJUSTED RETURN
    # ============================================================
    # Interest rate adjusted by credit quality
    if 'interest_rate' in X_train.columns and 'credit_score' in X_train.columns:
        X_train['risk_adjusted_return'] = (X_train['interest_rate'] * 100) / (X_train['credit_score'] / 10)

        if X_test is not None:
            X_test['risk_adjusted_return'] = (X_test['interest_rate'] * 100) / (X_test['credit_score'] / 10)

    # ============================================================
    # 6. GRADE-BASED FEATURES
    # ============================================================
    # Split grade_subgrade into grade and subgrade_num
    if 'grade_subgrade' in X_train.columns:
        # Extract letter grade (A, B, C, etc.)
        X_train['grade'] = X_train['grade_subgrade'].str[0]

        # Map to numeric (A=7, B=6, ..., G=1)
        grade_map = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}
        X_train['grade_numeric'] = X_train['grade'].map(grade_map).fillna(0)

        # Extract subgrade number
        X_train['subgrade_num'] = X_train['grade_subgrade'].str[1:].astype(float)

        if X_test is not None:
            X_test['grade'] = X_test['grade_subgrade'].str[0]
            X_test['grade_numeric'] = X_test['grade'].map(grade_map).fillna(0)
            X_test['subgrade_num'] = X_test['grade_subgrade'].str[1:].astype(float)

    # Combine train and test
    train_enriched = pd.concat([X_train, y_train], axis=1)
    test_enriched = X_test if X_test is not None else None

    print(f"Tier 2 features added. Train shape: {train_enriched.shape}")
    if test_enriched is not None:
        print(f"Test shape: {test_enriched.shape}")

    return train_enriched, test_enriched, artifacts


__all__ = ['add_tier2_features', 'TargetEncoderCV', 'WoEEncoder']
