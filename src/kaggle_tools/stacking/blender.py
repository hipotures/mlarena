"""
Simple blending methods: weighted averaging and rank averaging.

These methods combine predictions from multiple models without
requiring a meta-model training step.
"""

from typing import List, Optional

import numpy as np
import pandas as pd


class WeightedBlender:
    """
    Weighted average of predictions.

    Simple but effective ensemble method. Weights can be:
    - Uniform (equal weights)
    - Performance-based (proportional to CV scores)
    - Manual (user-defined)

    Example:
        >>> blender = WeightedBlender(weights=[0.4, 0.3, 0.3])
        >>> predictions = [model1_preds, model2_preds, model3_preds]
        >>> blended = blender.blend(predictions)
    """

    def __init__(self, weights: Optional[List[float]] = None):
        """
        Initialize weighted blender.

        Args:
            weights: List of weights (must sum to 1.0). If None, use uniform weights.
        """
        self.weights = weights

    def blend(
        self, predictions: List[pd.Series], weights: Optional[List[float]] = None
    ) -> pd.Series:
        """
        Blend predictions using weighted average.

        Args:
            predictions: List of prediction Series from different models
            weights: Optional weights to use for blending. If omitted, uses
                self.weights or defaults to uniform.

        Returns:
            Blended predictions as Series

        Raises:
            ValueError: If predictions list is empty or weights don't sum to 1.0
        """
        if not predictions:
            raise ValueError("Predictions list cannot be empty")

        n_models = len(predictions)

        # Use uniform weights if not provided
        # Choose weights in priority order: explicit arg -> self.weights -> uniform
        if weights is None:
            if self.weights is None:
                weights = np.ones(n_models) / n_models
            else:
                weights = np.array(self.weights)
        else:
            weights = np.array(weights)

        # Validate weights
        if len(weights) != n_models:
            raise ValueError(
                f"Number of weights ({len(weights)}) must match "
                f"number of predictions ({n_models})"
            )

            if not np.isclose(weights.sum(), 1.0):
                raise ValueError(f"Weights must sum to 1.0, got: {weights.sum()}")

        # Weighted average
        result = pd.Series(0.0, index=predictions[0].index)

        for pred, weight in zip(predictions, weights):
            result += weight * pred

        return result


class RankBlender:
    """
    Rank averaging (more robust to outliers than weighted averaging).

    Each model's predictions are converted to ranks, then averaged,
    then converted back to scores.

    Example:
        >>> blender = RankBlender()
        >>> predictions = [model1_preds, model2_preds, model3_preds]
        >>> blended = blender.blend(predictions)
    """

    def __init__(self, method: str = "average"):
        """
        Initialize rank blender.

        Args:
            method: Ranking method ("average", "min", "max", "dense", "ordinal")
        """
        self.method = method

    def blend(self, predictions: List[pd.Series]) -> pd.Series:
        """
        Blend predictions using rank averaging.

        Args:
            predictions: List of prediction Series from different models

        Returns:
            Blended predictions (as ranks)
        """
        if not predictions:
            raise ValueError("Predictions list cannot be empty")

        # Convert each prediction to ranks
        ranked_predictions = []

        for pred in predictions:
            ranks = pred.rank(method=self.method, ascending=True)
            ranked_predictions.append(ranks)

        # Average ranks
        avg_ranks = pd.concat(ranked_predictions, axis=1).mean(axis=1)

        return avg_ranks


class PowerBlender:
    """
    Weighted power averaging: (w1*p1^k + w2*p2^k + ...)^(1/k).

    Power parameter k can help emphasize confident predictions:
    - k > 1: Emphasizes higher values
    - k < 1: Smooths predictions
    - k = 1: Reduces to weighted average
    - k = 2: Quadratic mean (RMS-like)

    Example:
        >>> blender = PowerBlender(weights=[0.5, 0.5], power=2.0)
        >>> predictions = [model1_preds, model2_preds]
        >>> blended = blender.blend(predictions)
    """

    def __init__(self, weights: Optional[List[float]] = None, power: float = 2.0):
        """
        Initialize power blender.

        Args:
            weights: List of weights (must sum to 1.0). If None, use uniform weights.
            power: Power parameter (k)
        """
        self.weights = weights
        self.power = power

    def blend(
        self,
        predictions: List[pd.Series],
        power: Optional[float] = None,
        weights: Optional[List[float]] = None,
    ) -> pd.Series:
        """
        Blend predictions using power averaging.

        Args:
            predictions: List of prediction Series
            power: Optional override for power parameter
            weights: Optional weights to use for blending

        Returns:
            Blended predictions
        """
        if not predictions:
            raise ValueError("Predictions list cannot be empty")

        n_models = len(predictions)

        # Use uniform weights if not provided
        # Choose weights and power
        if weights is None:
            if self.weights is None:
                weights = np.ones(n_models) / n_models
            else:
                weights = np.array(self.weights)
        else:
            weights = np.array(weights)
        power = self.power if power is None else power

        if len(weights) != n_models:
            raise ValueError(
                f"Number of weights ({len(weights)}) must match "
                f"number of predictions ({n_models})"
            )

        if not np.isclose(weights.sum(), 1.0):
            raise ValueError(f"Weights must sum to 1.0, got: {weights.sum()}")

        # Power average: (sum(w_i * p_i^k))^(1/k)
        result = pd.Series(0.0, index=predictions[0].index)

        for pred, weight in zip(predictions, weights):
            result += weight * (pred**power)

        result = result ** (1.0 / power)

        return result
