"""
Module collection.

Importing this package triggers module registration via decorators.
"""

# Order of imports is not critical because registration is decorator-driven.
from mlarena.modules.eda import EDAModule  # noqa: F401
from mlarena.modules.preprocess import PreprocessModule  # noqa: F401
from mlarena.modules.feat import FeatureModule  # noqa: F401
from mlarena.modules.model import ModelModule  # noqa: F401
from mlarena.modules.predict import PredictModule  # noqa: F401
from mlarena.modules.tune import TuneModule  # noqa: F401
from mlarena.modules.stack import StackModule  # noqa: F401
from mlarena.modules.submit import SubmitModule  # noqa: F401
from mlarena.modules.fetch_score import FetchScoreModule  # noqa: F401

__all__ = [
    "EDAModule",
    "PreprocessModule",
    "FeatureModule",
    "ModelModule",
    "PredictModule",
    "TuneModule",
    "StackModule",
    "SubmitModule",
    "FetchScoreModule",
]
