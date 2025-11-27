"""
Model ensembling and stacking module.

This module provides:
- Blending: Weighted and rank averaging
- Meta-learning: Stacking with meta-models
- Calibration: Probability calibration for better predictions
"""

from kaggle_tools.stacking.blender import WeightedBlender, RankBlender
from kaggle_tools.stacking.meta_learner import MetaLearner
from kaggle_tools.stacking.calibration import CalibratedPredictor

__all__ = [
    "WeightedBlender",
    "RankBlender",
    "MetaLearner",
    "CalibratedPredictor",
]
