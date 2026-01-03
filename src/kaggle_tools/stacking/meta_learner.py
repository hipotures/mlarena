"""
Meta-learning (stacking) with out-of-fold predictions.

Stacking trains a meta-model on out-of-fold predictions from base models,
which can learn optimal combination weights and non-linear interactions.
"""

from typing import Any, List, Optional

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from rich.console import Console

console = Console()


class MetaLearner:
    """
    Meta-model stacking using out-of-fold predictions.

    This is more sophisticated than simple blending because the meta-model
    can learn non-linear combinations and feature interactions.

    Example:
        >>> from sklearn.linear_model import Ridge
        >>>
        >>> meta = MetaLearner(meta_model_class=Ridge)
        >>>
        >>> # Fit on out-of-fold predictions
        >>> meta.fit(oof_predictions, y_train)
        >>>
        >>> # Predict on test set
        >>> final_preds = meta.predict(test_predictions)
    """

    def __init__(
        self,
        meta_model_class: type = LogisticRegression,
        meta_model_params: Optional[dict] = None,
    ):
        """
        Initialize meta-learner.

        Args:
            meta_model_class: Meta-model class (e.g., LogisticRegression, Ridge, XGBoost)
            meta_model_params: Parameters for meta-model initialization
        """
        self.meta_model_class = meta_model_class
        self.meta_model_params = meta_model_params or {}
        self.meta_model = None

    def fit(
        self,
        oof_predictions: List[pd.DataFrame],
        y_train: pd.Series,
    ) -> "MetaLearner":
        """
        Train meta-model on out-of-fold predictions.

        Args:
            oof_predictions: List of OOF prediction DataFrames from base models
            y_train: True target values

        Returns:
            self
        """
        # Stack OOF predictions horizontally
        X_meta = pd.concat(oof_predictions, axis=1)

        # Instantiate and train meta-model
        self.meta_model = self.meta_model_class(**self.meta_model_params)
        self.meta_model.fit(X_meta, y_train)

        # Compute meta-model CV score
        cv_scores = cross_val_score(
            self.meta_model,
            X_meta,
            y_train,
            cv=5,
            scoring="roc_auc",
        )

        console.print(f"[green]✓[/green] [bold]Meta-model CV score:[/bold] [yellow]{cv_scores.mean():.4f}[/yellow] (+/- {cv_scores.std():.4f})")

        return self

    def predict(self, test_predictions: List[pd.DataFrame]) -> pd.Series:
        """
        Generate final ensemble prediction using meta-model.

        Args:
            test_predictions: List of test prediction DataFrames from base models

        Returns:
            Final predictions
        """
        if self.meta_model is None:
            raise RuntimeError("Meta-model not trained. Call fit() first.")

        # Stack test predictions
        X_meta_test = pd.concat(test_predictions, axis=1)

        # Predict with meta-model
        if hasattr(self.meta_model, "predict_proba"):
            preds = self.meta_model.predict_proba(X_meta_test)[:, 1]
        else:
            preds = self.meta_model.predict(X_meta_test)

        return pd.Series(preds, index=X_meta_test.index)
