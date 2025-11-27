"""
Feature engineering module with data leakage protection.

This module provides a two-stage feature engineering pipeline:
- feat_stage: Global transformers fitted on entire train set (SAFE)
- cv_stage: Per-fold transformers fitted on train folds only (RISKY but powerful)
"""

from kaggle_tools.preprocessing.base import BaseTransformer
from kaggle_tools.preprocessing.pipeline import FeaturePipeline

__all__ = [
    "BaseTransformer",
    "FeaturePipeline",
]
