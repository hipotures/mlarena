"""
Cross-validation objective functions for Optuna.

This module provides CVObjective class that runs K-fold cross-validation
for each Optuna trial with early stopping and pruning support.
"""

from typing import Callable, Any
import numpy as np
import pandas as pd

import optuna
from sklearn.model_selection import KFold, StratifiedKFold


class CVObjective:
    """
    Cross-validation objective for Optuna with early stopping and pruning.

    This objective function runs K-fold CV for each trial and reports
    intermediate scores to enable pruning of unpromising trials.

    Example:
        >>> from xgboost import XGBClassifier
        >>> from sklearn.metrics import roc_auc_score
        >>>
        >>> objective = CVObjective(
        ...     model_class=XGBClassifier,
        ...     train_df=train_df,
        ...     target_col="target",
        ...     param_space_fn=lambda trial: xgboost_param_space(trial, config),
        ...     metric_fn=roc_auc_score,
        ...     cv_folds=5,
        ...     early_stopping_rounds=50,
        ...     random_seed=42,
        ... )
        >>>
        >>> study.optimize(objective, n_trials=100)
    """

    def __init__(
        self,
        model_class: type,
        train_df: pd.DataFrame,
        target_col: str,
        param_space_fn: Callable,
        metric_fn: Callable,
        cv_folds: int = 5,
        early_stopping_rounds: int = 50,
        random_seed: int = 42,
        stratified: bool = True,
        use_gpu: bool = False,
    ):
        """
        Initialize CV objective.

        Args:
            model_class: Model class to instantiate (e.g., XGBClassifier)
            train_df: Training DataFrame
            target_col: Target column name
            param_space_fn: Function that takes trial and returns hyperparameters
            metric_fn: Metric function (higher is better)
            cv_folds: Number of CV folds
            early_stopping_rounds: Early stopping rounds for model training
            random_seed: Random seed for reproducibility
            stratified: Use stratified K-fold (for classification)
            use_gpu: Enable GPU for training
        """
        self.model_class = model_class
        self.train_df = train_df
        self.target_col = target_col
        self.param_space_fn = param_space_fn
        self.metric_fn = metric_fn
        self.cv_folds = cv_folds
        self.early_stopping_rounds = early_stopping_rounds
        self.random_seed = random_seed
        self.stratified = stratified
        self.use_gpu = use_gpu

        # Prepare data
        self.X = train_df.drop(columns=[target_col])
        self.y = train_df[target_col]

    def __call__(self, trial: optuna.Trial) -> float:
        """
        Optuna objective: returns mean CV score.

        Args:
            trial: Optuna trial object

        Returns:
            Mean cross-validation score

        Raises:
            optuna.TrialPruned: If trial should be pruned
        """
        # Sample hyperparameters
        params = self.param_space_fn(trial)

        # K-fold cross-validation
        if self.stratified:
            kfold = StratifiedKFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=self.random_seed,
            )
        else:
            kfold = KFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=self.random_seed,
            )

        cv_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(self.X, self.y)):
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]

            # Instantiate model with trial hyperparameters
            model = self.model_class(**params)

            # Train model with early stopping
            try:
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=self.early_stopping_rounds,
                    verbose=False,
                )
            except TypeError:
                # Some models don't support eval_set
                model.fit(X_train, y_train)

            # Predict and score
            if hasattr(model, "predict_proba"):
                # Classification: use probabilities
                preds = model.predict_proba(X_val)
                if preds.ndim == 2 and preds.shape[1] == 2:
                    # Binary classification: take positive class probability
                    preds = preds[:, 1]
            else:
                # Regression: use predictions
                preds = model.predict(X_val)

            score = self.metric_fn(y_val, preds)
            cv_scores.append(score)

            # Report intermediate value for pruning
            trial.report(score, fold_idx)

            # Prune trial if not promising
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Return mean CV score
        mean_score = np.mean(cv_scores)
        return mean_score
