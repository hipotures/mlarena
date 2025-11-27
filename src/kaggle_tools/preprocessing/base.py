"""
Base transformer interface for feature engineering with data leakage protection.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd


class BaseTransformer(ABC):
    """
    Base class for feature transformers.

    All transformers must implement fit() and transform() methods.
    This ensures consistent interface for both feat_stage (global) and
    cv_stage (per-fold) transformers.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize transformer with optional configuration.

        Args:
            config: Dictionary with transformer-specific parameters
        """
        self.config = config or {}
        self.fitted = False

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseTransformer":
        """
        Fit transformer on training data.

        This method should learn any necessary statistics from the data
        (e.g., mean/std for normalization, category mappings for encoding).

        Args:
            df: Training DataFrame

        Returns:
            self for method chaining
        """
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted parameters.

        This method should apply the transformation learned during fit().
        It must work on both train and test data.

        Args:
            df: DataFrame to transform

        Returns:
            Transformed DataFrame
        """
        pass

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step (convenience method).

        Args:
            df: DataFrame to fit and transform

        Returns:
            Transformed DataFrame
        """
        self.fit(df)
        return self.transform(df)

    def _check_fitted(self):
        """Raise error if transformer not fitted."""
        if not self.fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before transform()"
            )
