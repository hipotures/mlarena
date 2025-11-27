"""
Risky transformers for cv_stage (must be fitted per-fold to avoid leakage).

These transformers depend on the target variable or training data distribution
in ways that would leak validation information if fitted on the entire train set.

CRITICAL: These MUST be fitted per-fold within cross-validation loops.
"""

from typing import List, Optional

import pandas as pd
import numpy as np

from kaggle_tools.preprocessing.base import BaseTransformer


class TargetEncodingTransformer(BaseTransformer):
    """
    Target encoding with smoothing (mean encoding).

    WARNING: MUST be fitted per-fold to avoid leakage!

    This transformer replaces categorical values with the mean of the target
    variable for that category. Without per-fold fitting, validation folds
    would see their own target values in the encoding.
    """

    def __init__(
        self,
        columns: List[str],
        target: str,
        smoothing: float = 1.0,
        min_samples_leaf: int = 1,
    ):
        """
        Initialize target encoding transformer.

        Args:
            columns: Categorical columns to encode
            target: Target column name
            smoothing: Smoothing parameter (higher = more regularization)
            min_samples_leaf: Minimum samples for category (else use global mean)
        """
        super().__init__()
        self.columns = columns
        self.target = target
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self._encodings: dict = {}
        self._global_mean: float = 0.0

    def fit(self, df: pd.DataFrame) -> "TargetEncodingTransformer":
        """
        Fit target encoding on TRAIN fold only.

        CRITICAL: This must be called on train_fold, NOT on full train set.

        Args:
            df: Training fold DataFrame (must contain target column)

        Returns:
            self
        """
        if self.target not in df.columns:
            raise ValueError(f"Target column '{self.target}' not found in DataFrame")

        self._global_mean = df[self.target].mean()

        for col in self.columns:
            if col not in df.columns:
                continue

            # Compute smoothed target encoding
            stats = df.groupby(col)[self.target].agg(["mean", "count"])

            # Smoothing formula: (count * mean + smoothing * global_mean) / (count + smoothing)
            smoothed_means = (
                stats["count"] * stats["mean"] + self.smoothing * self._global_mean
            ) / (stats["count"] + self.smoothing)

            # Filter by min_samples_leaf
            mask = stats["count"] >= self.min_samples_leaf
            smoothed_means = smoothed_means[mask]

            self._encodings[col] = smoothed_means.to_dict()

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform using fitted encodings.

        Args:
            df: DataFrame to transform (can be train or validation fold)

        Returns:
            DataFrame with target-encoded columns
        """
        self._check_fitted()
        result = df.copy()

        for col, encoding_map in self._encodings.items():
            if col in result.columns:
                # Map categories to encoded values, use global mean for unseen categories
                result[f"{col}_te"] = result[col].map(encoding_map).fillna(self._global_mean)

        return result


class FrequencyEncodingTransformer(BaseTransformer):
    """
    Frequency encoding (replace category with its frequency).

    Risky because: Category frequencies differ between train/val if data is not i.i.d.
    """

    def __init__(self, columns: List[str], normalize: bool = True):
        """
        Initialize frequency encoding transformer.

        Args:
            columns: Categorical columns to encode
            normalize: If True, use relative frequency (0-1), else absolute counts
        """
        super().__init__()
        self.columns = columns
        self.normalize = normalize
        self._freq_maps: dict = {}

    def fit(self, df: pd.DataFrame) -> "FrequencyEncodingTransformer":
        """
        Compute frequency of each category.

        Args:
            df: Training fold DataFrame

        Returns:
            self
        """
        for col in self.columns:
            if col in df.columns:
                self._freq_maps[col] = df[col].value_counts(normalize=self.normalize).to_dict()

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply frequency encoding.

        Args:
            df: DataFrame to transform

        Returns:
            DataFrame with frequency-encoded columns
        """
        self._check_fitted()
        result = df.copy()

        for col, freq_map in self._freq_maps.items():
            if col in result.columns:
                # Map categories to frequencies, use 0 for unseen categories
                result[f"{col}_freq"] = result[col].map(freq_map).fillna(0)

        return result


class TargetAggregatesTransformer(BaseTransformer):
    """
    Create aggregated statistics of target grouped by categorical features.

    WARNING: MUST be fitted per-fold!

    Example: For each value of 'category_A', compute [mean, std, min, max] of target.
    """

    def __init__(
        self,
        group_cols: List[str],
        target: str,
        agg_funcs: List[str] = None,
    ):
        """
        Initialize target aggregates transformer.

        Args:
            group_cols: Columns to group by
            target: Target column name
            agg_funcs: Aggregation functions (default: ["mean", "std", "min", "max"])
        """
        super().__init__()
        self.group_cols = group_cols
        self.target = target
        self.agg_funcs = agg_funcs or ["mean", "std", "min", "max"]
        self._agg_maps: dict = {}

    def fit(self, df: pd.DataFrame) -> "TargetAggregatesTransformer":
        """
        Compute target aggregates per group.

        Args:
            df: Training fold DataFrame

        Returns:
            self
        """
        if self.target not in df.columns:
            raise ValueError(f"Target column '{self.target}' not found")

        for col in self.group_cols:
            if col not in df.columns:
                continue

            agg_stats = df.groupby(col)[self.target].agg(self.agg_funcs)
            self._agg_maps[col] = agg_stats.to_dict("index")

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply target aggregates.

        Args:
            df: DataFrame to transform

        Returns:
            DataFrame with aggregate features
        """
        self._check_fitted()
        result = df.copy()

        for col, agg_map in self._agg_maps.items():
            if col not in result.columns:
                continue

            for agg_func in self.agg_funcs:
                feature_name = f"{col}_target_{agg_func}"
                result[feature_name] = result[col].map(
                    lambda x: agg_map.get(x, {}).get(agg_func, np.nan)
                )

        return result


class WeightOfEvidenceTransformer(BaseTransformer):
    """
    Weight of Evidence (WoE) encoding for binary classification.

    WARNING: MUST be fitted per-fold!

    WoE = ln(P(X|Y=1) / P(X|Y=0))
    """

    def __init__(self, columns: List[str], target: str, epsilon: float = 1e-10):
        """
        Initialize WoE transformer.

        Args:
            columns: Categorical columns to encode
            target: Target column (must be binary: 0/1)
            epsilon: Small constant to avoid division by zero
        """
        super().__init__()
        self.columns = columns
        self.target = target
        self.epsilon = epsilon
        self._woe_maps: dict = {}

    def fit(self, df: pd.DataFrame) -> "WeightOfEvidenceTransformer":
        """
        Compute WoE for each category.

        Args:
            df: Training fold DataFrame

        Returns:
            self
        """
        if self.target not in df.columns:
            raise ValueError(f"Target column '{self.target}' not found")

        # Check binary target
        unique_targets = df[self.target].unique()
        if not set(unique_targets).issubset({0, 1}):
            raise ValueError(f"Target must be binary (0/1), got: {unique_targets}")

        n_pos = (df[self.target] == 1).sum()
        n_neg = (df[self.target] == 0).sum()

        for col in self.columns:
            if col not in df.columns:
                continue

            woe_map = {}
            for category in df[col].unique():
                mask = df[col] == category
                n_pos_cat = (df.loc[mask, self.target] == 1).sum()
                n_neg_cat = (df.loc[mask, self.target] == 0).sum()

                # Compute WoE with epsilon smoothing
                pos_rate = (n_pos_cat + self.epsilon) / (n_pos + self.epsilon)
                neg_rate = (n_neg_cat + self.epsilon) / (n_neg + self.epsilon)

                woe_map[category] = np.log(pos_rate / neg_rate)

            self._woe_maps[col] = woe_map

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply WoE encoding.

        Args:
            df: DataFrame to transform

        Returns:
            DataFrame with WoE-encoded columns
        """
        self._check_fitted()
        result = df.copy()

        for col, woe_map in self._woe_maps.items():
            if col in result.columns:
                result[f"{col}_woe"] = result[col].map(woe_map).fillna(0.0)

        return result
