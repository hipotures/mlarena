"""
XGBoost model with Optuna hyperparameter tuning.

This template demonstrates the Optuna system integration:
- Automatic study creation/resumption with SQLite storage
- Cross-validation with early stopping
- Best parameters caching for reproducibility
- Integration with FeaturePipeline
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import json

import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from kaggle_tools.config_models import ModelConfig
from kaggle_tools.optuna import StudyManager, CVObjective, xgboost_param_space


def get_default_config() -> Dict[str, Any]:
    """Return default configuration for this model."""
    return {
        "hyperparameters": {
            "preset": "quick",  # References configs/presets/quick.yaml
        },
        "model": {
            "n_estimators": 1000,
            "early_stopping_rounds": 50,
        },
    }


def train(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    """
    Train XGBoost using Optuna-tuned hyperparameters.

    Workflow:
    1. Check if Optuna study already completed (resume support)
    2. If not, run Optuna hyperparameter search
    3. Train final model on full train set with best params
    4. Return model and summary statistics

    Args:
        train_df: Training DataFrame
        val_df: Optional validation DataFrame
        config: Model configuration
        artifacts: Optional artifacts (not used in this template)

    Returns:
        Tuple of (trained_model, summary_dict)
    """
    # Prepare data
    X_train = train_df.drop(columns=[config.dataset.target])
    y_train = train_df[config.dataset.target]

    eval_set = None
    if val_df is not None:
        X_val = val_df.drop(columns=[config.dataset.target])
        y_val = val_df[config.dataset.target]
        eval_set = [(X_val, y_val)]

    # Setup Optuna artifacts directory
    optuna_dir = config.system.artifact_dir / "optuna"
    optuna_dir.mkdir(parents=True, exist_ok=True)
    best_params_file = optuna_dir / "best_params.json"

    # Check if study already completed
    if best_params_file.exists():
        print(f"Loading cached best params from {best_params_file}")
        with open(best_params_file) as f:
            best_params = json.load(f)
    else:
        print("Running Optuna hyperparameter search...")

        # Create study manager
        study_manager = StudyManager(config.optuna)
        study = study_manager.create_or_load_study()

        # Create CV objective
        objective = CVObjective(
            model_class=xgb.XGBClassifier,
            train_df=train_df,
            target_col=config.dataset.target,
            param_space_fn=lambda trial: xgboost_param_space(
                trial, config.optuna.param_space["xgboost"]
            ),
            metric_fn=roc_auc_score,
            cv_folds=config.optuna.cv_folds,
            early_stopping_rounds=config.optuna.early_stopping_rounds,
            random_seed=config.system.random_seed,
        )

        # Run optimization
        study.optimize(
            objective,
            n_trials=config.optuna.n_trials,
            timeout=config.optuna.timeout,
            n_jobs=config.optuna.n_jobs,
        )

        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value

        print(f"\nBest CV score: {best_value:.5f}")
        print(f"Best parameters: {best_params}")

        # Save for reproducibility
        with open(best_params_file, "w") as f:
            json.dump(best_params, f, indent=2)

        # Export trials
        study_manager.export_trials_dataframe(optuna_dir / "trials.csv")

    # Train final model with best params
    print("\nTraining final model with best parameters...")

    model = xgb.XGBClassifier(
        **best_params,
        n_estimators=config.model.get("n_estimators", 1000),
        random_state=config.system.random_seed,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        early_stopping_rounds=config.model.get("early_stopping_rounds", 50),
        verbose=True,
    )

    # Compute local CV score
    from sklearn.model_selection import cross_val_score

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=config.optuna.cv_folds,
        scoring="roc_auc",
    )
    local_cv = float(cv_scores.mean())
    cv_std = float(cv_scores.std())

    print(f"\nFinal model CV score: {local_cv:.5f} (+/- {cv_std:.5f})")

    summary = {
        "local_cv": local_cv,
        "cv_std": cv_std,
        "best_params": best_params,
        "cv_scores": cv_scores.tolist(),
    }

    return model, summary


def predict(
    model: xgb.XGBClassifier,
    test_df: pd.DataFrame,
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Generate predictions on test set.

    Args:
        model: Trained XGBoost model
        test_df: Test DataFrame
        config: Model configuration
        artifacts: Optional artifacts (not used)

    Returns:
        DataFrame with predictions
    """
    X_test = test_df.drop(columns=[config.dataset.id_column], errors="ignore")

    submission = pd.DataFrame()
    submission[config.dataset.id_column] = test_df[config.dataset.id_column]

    if config.dataset.submission_probas:
        submission[config.dataset.target] = model.predict_proba(X_test)[:, 1]
    else:
        submission[config.dataset.target] = model.predict(X_test)

    return submission
