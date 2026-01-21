"""
Probability calibration for better-calibrated predictions.

Well-calibrated predictions are important for:
- Ensembling (better blending)
- Decision thresholds (accurate probability estimates)
- Log loss metrics
"""

import pandas as pd


class CalibratedPredictor:
    """
    Calibrate predictions using isotonic regression or Platt scaling.

    Example:
        >>> calibrator = CalibratedPredictor(method="isotonic")
        >>>
        >>> # Fit on validation predictions
        >>> calibrator.fit(val_predictions, y_val)
        >>>
        >>> # Calibrate test predictions
        >>> calibrated = calibrator.transform(test_predictions)
    """

    def __init__(self, method: str = "isotonic"):
        """
        Initialize calibrator.

        Args:
            method: Calibration method ("isotonic" or "sigmoid")
        """
        if method not in ["isotonic", "sigmoid"]:
            raise ValueError(f"Unknown method: {method}. Use 'isotonic' or 'sigmoid'")

        self.method = method
        self.calibrator = None

    def fit(
        self,
        predictions: pd.Series,
        y_true: pd.Series,
    ) -> "CalibratedPredictor":
        """
        Fit calibration model.

        Args:
            predictions: Uncalibrated predictions (probabilities)
            y_true: True labels

        Returns:
            self
        """
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression

        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
        else:  # sigmoid (Platt scaling)
            self.calibrator = LogisticRegression()

        # Reshape for sklearn
        X = (
            predictions.values.reshape(-1, 1)
            if self.method == "sigmoid"
            else predictions.values
        )
        y = y_true.values

        self.calibrator.fit(X, y)

        return self

    def transform(self, predictions: pd.Series) -> pd.Series:
        """
        Apply calibration to predictions.

        Args:
            predictions: Uncalibrated predictions

        Returns:
            Calibrated predictions
        """
        if self.calibrator is None:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")

        X = (
            predictions.values.reshape(-1, 1)
            if self.method == "sigmoid"
            else predictions.values
        )

        if self.method == "sigmoid":
            calibrated = self.calibrator.predict_proba(X)[:, 1]
        else:
            calibrated = self.calibrator.predict(X)

        return pd.Series(calibrated, index=predictions.index)
