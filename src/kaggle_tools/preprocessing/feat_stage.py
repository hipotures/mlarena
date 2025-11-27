"""
Safe transformers for feat_stage (no data leakage).

These transformers can be fitted on the entire train set because they don't
depend on the target variable in ways that would leak validation information.
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from kaggle_tools.preprocessing.base import BaseTransformer


class LogTransformer(BaseTransformer):
    """
    Apply log(1+x) transformation to numeric columns.

    Safe because: Transformation is independent of target variable.
    """

    def __init__(self, columns: Optional[List[str]] = None):
        """
        Initialize log transformer.

        Args:
            columns: List of columns to transform. If None, auto-detect numeric columns.
        """
        super().__init__()
        self.columns = columns
        self._fitted_columns: List[str] = []

    def fit(self, df: pd.DataFrame) -> "LogTransformer":
        """
        Identify numeric columns to transform.

        Args:
            df: Training DataFrame

        Returns:
            self
        """
        if self.columns is None:
            # Auto-detect numeric columns
            self._fitted_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self._fitted_columns = [col for col in self.columns if col in df.columns]

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply log(1+x) transformation.

        Args:
            df: DataFrame to transform

        Returns:
            DataFrame with log-transformed columns
        """
        self._check_fitted()
        result = df.copy()

        for col in self._fitted_columns:
            if col in result.columns:
                # Clip to avoid log(negative)
                result[f"{col}_log"] = np.log1p(result[col].clip(lower=0))

        return result


class PolynomialFeaturesTransformer(BaseTransformer):
    """
    Create polynomial interaction features.

    Safe because: Interactions are mathematical operations, not target-dependent.
    """

    def __init__(self, columns: List[str], degree: int = 2):
        """
        Initialize polynomial features transformer.

        Args:
            columns: List of column names to create interactions from
            degree: Polynomial degree (2 = pairwise interactions)
        """
        super().__init__()
        self.columns = columns
        self.degree = degree

    def fit(self, df: pd.DataFrame) -> "PolynomialFeaturesTransformer":
        """
        Validate columns exist.

        Args:
            df: Training DataFrame

        Returns:
            self
        """
        missing = [col for col in self.columns if col not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create polynomial interaction features.

        Args:
            df: DataFrame to transform

        Returns:
            DataFrame with interaction features
        """
        self._check_fitted()
        result = df.copy()

        if self.degree >= 2:
            # Create pairwise interactions
            for i, col1 in enumerate(self.columns):
                for col2 in self.columns[i + 1 :]:
                    if col1 in result.columns and col2 in result.columns:
                        result[f"{col1}_x_{col2}"] = result[col1] * result[col2]

        if self.degree >= 3:
            # Create squared terms
            for col in self.columns:
                if col in result.columns:
                    result[f"{col}_squared"] = result[col] ** 2

        return result


class StandardScalerTransformer(BaseTransformer):
    """
    Standard scaling (z-score normalization).

    Safe because: Mean/std are computed on train set and applied to both train/test.
    """

    def __init__(self, columns: Optional[List[str]] = None):
        """
        Initialize standard scaler.

        Args:
            columns: Columns to scale. If None, auto-detect numeric columns.
        """
        super().__init__()
        self.columns = columns
        self._means: dict = {}
        self._stds: dict = {}

    def fit(self, df: pd.DataFrame) -> "StandardScalerTransformer":
        """
        Compute mean and std for each column.

        Args:
            df: Training DataFrame

        Returns:
            self
        """
        if self.columns is None:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols = [col for col in self.columns if col in df.columns]

        for col in cols:
            self._means[col] = df[col].mean()
            self._stds[col] = df[col].std()

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply z-score normalization.

        Args:
            df: DataFrame to transform

        Returns:
            Scaled DataFrame
        """
        self._check_fitted()
        result = df.copy()

        for col, mean in self._means.items():
            if col in result.columns:
                std = self._stds[col]
                if std > 0:
                    result[f"{col}_scaled"] = (result[col] - mean) / std
                else:
                    result[f"{col}_scaled"] = 0.0

        return result


class BinningTransformer(BaseTransformer):
    """
    Bin numeric features into discrete categories.

    Safe because: Binning thresholds are computed on train set only.
    """

    def __init__(self, column: str, n_bins: int = 10, strategy: str = "quantile"):
        """
        Initialize binning transformer.

        Args:
            column: Column to bin
            n_bins: Number of bins
            strategy: "quantile" or "uniform"
        """
        super().__init__()
        self.column = column
        self.n_bins = n_bins
        self.strategy = strategy
        self._bin_edges: Optional[np.ndarray] = None

    def fit(self, df: pd.DataFrame) -> "BinningTransformer":
        """
        Compute bin edges.

        Args:
            df: Training DataFrame

        Returns:
            self
        """
        if self.column not in df.columns:
            raise ValueError(f"Column '{self.column}' not found in DataFrame")

        values = df[self.column].dropna().values

        if self.strategy == "quantile":
            quantiles = np.linspace(0, 1, self.n_bins + 1)
            self._bin_edges = np.quantile(values, quantiles)
        else:  # uniform
            self._bin_edges = np.linspace(values.min(), values.max(), self.n_bins + 1)

        # Ensure unique edges
        self._bin_edges = np.unique(self._bin_edges)

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply binning.

        Args:
            df: DataFrame to transform

        Returns:
            DataFrame with binned column
        """
        self._check_fitted()
        result = df.copy()

        if self.column in result.columns:
            result[f"{self.column}_binned"] = pd.cut(
                result[self.column],
                bins=self._bin_edges,
                labels=False,
                include_lowest=True,
                duplicates="drop",
            )

        return result
