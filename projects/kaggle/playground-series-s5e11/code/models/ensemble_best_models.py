"""
Ensemble model: Weighted average of best performing models.
Compatible with the generic ML runner.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import glob

import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor

from kaggle_tools.config_models import ModelConfig


def get_default_config() -> Dict[str, Any]:
    """
    Default configuration for ensemble model.

    The 'models' list specifies which models to ensemble:
    - name: identifier for the model
    - path: glob pattern to find model (relative to project root)
    - weight: ensemble weight (will be normalized)
    - public_score: optional, for reference
    """
    return {
        "hyperparameters": {
            "presets": None,  # Not used for ensemble
            "time_limit": None,  # Not used for ensemble
            "use_gpu": False,
        },
        "model": {
            "ensemble_method": "weighted_average",  # or 'rank_average' or 'power_average'
            "calibration": False,  # Apply probability calibration (requires val_df)
            "power": 2.0,  # For power_average method
            "models": [
                {
                    "name": "fe17_log_EMI",
                    "path": "experiments/*/artifacts/best-cpu-fe17-autogluon_features_17",
                    "weight": 0.30,
                    "public_score": 0.92356
                },
                {
                    "name": "fe20_student_debt",
                    "path": "experiments/*/artifacts/best-cpu-fe20-autogluon_features_20",
                    "weight": 0.25,
                    "public_score": 0.92353
                },
                {
                    "name": "original_best",
                    "path": "AutogluonModels",
                    "weight": 0.45,
                    "public_score": 0.92434
                }
            ],
        },
    }


def _find_model_path(pattern: str, project_root: Path) -> Optional[Path]:
    """Find the most recent model matching the pattern."""
    search_pattern = str(project_root / pattern)
    matches = glob.glob(search_pattern)

    if matches:
        # Return most recent
        return Path(max(matches, key=lambda x: Path(x).stat().st_mtime))
    return None


def _load_models(config: ModelConfig) -> Dict[str, TabularPredictor]:
    """Load all models for ensemble."""
    models = {}
    model_configs = config.model.get("models", [])

    if not model_configs:
        raise ValueError("No models specified in config.model['models']")

    for model_cfg in model_configs:
        model_name = model_cfg["name"]
        model_path = _find_model_path(model_cfg["path"], config.system.project_root)

        if model_path and model_path.exists():
            try:
                predictor = TabularPredictor.load(str(model_path))
                models[model_name] = predictor
                print(f"✓ Loaded {model_name} from {model_path}")
            except Exception as e:
                print(f"✗ Failed to load {model_name}: {e}")
        else:
            print(f"✗ Model not found: {model_name} (pattern: {model_cfg['path']})")

    return models


def _drop_ignored(df: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    """Drop ignored columns."""
    drop_cols = set(config.dataset.ignored_columns + [config.dataset.id_column])
    drop_cols.discard(config.dataset.target)
    return df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")


def _weighted_average_ensemble(
    predictions: Dict[str, np.ndarray],
    weights: Dict[str, float]
) -> np.ndarray:
    """Compute weighted average of predictions."""
    total_weight = sum(weights.values())
    ensemble_pred = np.zeros_like(next(iter(predictions.values())))

    for model_name, pred in predictions.items():
        weight = weights.get(model_name, 0.0)
        ensemble_pred += (weight / total_weight) * pred

    return ensemble_pred


def _rank_average_ensemble(predictions: Dict[str, np.ndarray]) -> np.ndarray:
    """Average of rank-transformed predictions."""
    from scipy.stats import rankdata

    ranks = {}
    for model_name, pred in predictions.items():
        ranks[model_name] = rankdata(pred) / len(pred)

    # Simple average of ranks
    ensemble_rank = np.mean(list(ranks.values()), axis=0)
    return ensemble_rank


def _power_average_ensemble(
    predictions: Dict[str, np.ndarray],
    weights: Dict[str, float],
    power: float = 2.0
) -> np.ndarray:
    """Power-weighted average (emphasizes confident predictions)."""
    total_weight = sum(weights.values())

    # Apply power transformation
    powered_preds = {}
    for model_name, pred in predictions.items():
        # Map to [0.01, 0.99] to avoid numerical issues
        clipped = np.clip(pred, 0.01, 0.99)
        powered_preds[model_name] = np.power(clipped, power)

    # Weighted average
    ensemble_pred = np.zeros_like(next(iter(predictions.values())))
    for model_name, pred in powered_preds.items():
        weight = weights.get(model_name, 0.0)
        ensemble_pred += (weight / total_weight) * pred

    # Inverse power transformation
    ensemble_pred = np.power(ensemble_pred, 1/power)
    return ensemble_pred


def _calibrate_probabilities(
    train_preds: np.ndarray,
    train_labels: np.ndarray,
    test_preds: np.ndarray
) -> np.ndarray:
    """Apply isotonic regression calibration."""
    from sklearn.isotonic import IsotonicRegression

    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(train_preds, train_labels)

    return calibrator.predict(test_preds)


def train(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    'Train' ensemble by loading pre-trained models.

    Returns:
        model: Dict containing loaded models and configuration
        summary: Dict with ensemble info
    """
    print(f"[Ensemble] Loading models for ensemble...")

    # Load all models
    models = _load_models(config)

    if len(models) < 2:
        raise ValueError(f"Need at least 2 models for ensemble, found {len(models)}")

    print(f"[Ensemble] Successfully loaded {len(models)} models")

    # Store models and config for prediction
    ensemble_model = {
        "models": models,
        "config": config.model,
        "val_df": val_df,  # Store for calibration
    }

    # Calculate diversity if validation data available
    if val_df is not None:
        try:
            from scipy.stats import pearsonr

            features = _drop_ignored(val_df, config)
            val_predictions = {}

            for model_name, model in models.items():
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(features, as_pandas=False, as_multiclass=False)
                    if isinstance(pred, np.ndarray) and pred.ndim > 1 and pred.shape[1] > 1:
                        pred = pred[:, 1]
                    else:
                        pred = pred.flatten() if isinstance(pred, np.ndarray) else np.array(pred).flatten()
                else:
                    pred = model.predict(features, as_pandas=False)
                    pred = pred.flatten() if isinstance(pred, np.ndarray) else np.array(pred).flatten()
                val_predictions[model_name] = pred

            # Calculate average pairwise correlation
            model_names = list(val_predictions.keys())
            n_models = len(model_names)
            correlations = []

            for i in range(n_models):
                for j in range(i+1, n_models):
                    corr, _ = pearsonr(val_predictions[model_names[i]], val_predictions[model_names[j]])
                    correlations.append(corr)
                    print(f"  Correlation {model_names[i]} vs {model_names[j]}: {corr:.4f}")

            avg_corr = np.mean(correlations) if correlations else 0.0
            diversity = 1.0 - avg_corr

            print(f"\n[Ensemble] Average correlation: {avg_corr:.4f}")
            print(f"[Ensemble] Diversity score: {diversity:.4f}")
        except Exception as e:
            print(f"[Ensemble] Could not calculate diversity: {e}")
            diversity = None
    else:
        diversity = None

    summary = {
        "n_models": len(models),
        "model_names": list(models.keys()),
        "ensemble_method": config.model.get("ensemble_method", "weighted_average"),
        "diversity": diversity,
        "local_cv": None,  # Ensemble doesn't have its own CV score
    }

    return ensemble_model, summary


def predict(
    model: Dict[str, Any],
    test_df: pd.DataFrame,
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Generate ensemble predictions.

    Args:
        model: Dict containing loaded models and config
        test_df: Test data
        config: Model configuration
        artifacts: Unused

    Returns:
        DataFrame with predictions
    """
    models = model["models"]
    ensemble_config = model["config"]
    val_df = model.get("val_df")

    # Prepare features
    features = _drop_ignored(test_df, config)

    # Generate predictions from each model
    predictions = {}
    weights = {}
    model_configs = ensemble_config.get("models", [])

    for model_cfg in model_configs:
        model_name = model_cfg["name"]

        if model_name in models:
            predictor = models[model_name]

            # Generate predictions
            if hasattr(predictor, 'predict_proba'):
                pred = predictor.predict_proba(features, as_pandas=False, as_multiclass=False)
                # Get positive class probability
                if isinstance(pred, np.ndarray) and pred.ndim > 1 and pred.shape[1] > 1:
                    pred = pred[:, 1]
                else:
                    pred = pred.flatten() if isinstance(pred, np.ndarray) else np.array(pred).flatten()
            else:
                pred = predictor.predict(features, as_pandas=False)
                pred = pred.flatten() if isinstance(pred, np.ndarray) else np.array(pred).flatten()

            predictions[model_name] = pred
            weights[model_name] = model_cfg["weight"]

            print(f"[Ensemble] Generated predictions from {model_name}: shape={pred.shape}, "
                  f"min={pred.min():.4f}, max={pred.max():.4f}, mean={pred.mean():.4f}")

    if len(predictions) < 2:
        raise ValueError(f"Need at least 2 model predictions for ensemble, got {len(predictions)}")

    # Apply ensemble method
    method = ensemble_config.get("ensemble_method", "weighted_average")

    if method == "weighted_average":
        ensemble_pred = _weighted_average_ensemble(predictions, weights)
    elif method == "rank_average":
        ensemble_pred = _rank_average_ensemble(predictions)
    elif method == "power_average":
        power = ensemble_config.get("power", 2.0)
        ensemble_pred = _power_average_ensemble(predictions, weights, power)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    print(f"[Ensemble] Method: {method}")
    print(f"[Ensemble] Combined predictions: min={ensemble_pred.min():.4f}, "
          f"max={ensemble_pred.max():.4f}, mean={ensemble_pred.mean():.4f}")

    # Apply calibration if requested and validation data available
    if ensemble_config.get("calibration", False) and val_df is not None:
        print("[Ensemble] Applying probability calibration...")

        try:
            val_features = _drop_ignored(val_df, config)

            # Generate validation predictions for calibration
            val_predictions = []
            for model_name in predictions.keys():
                predictor = models[model_name]
                if hasattr(predictor, 'predict_proba'):
                    val_pred = predictor.predict_proba(val_features, as_pandas=False, as_multiclass=False)
                    if isinstance(val_pred, np.ndarray) and val_pred.ndim > 1 and val_pred.shape[1] > 1:
                        val_pred = val_pred[:, 1]
                    else:
                        val_pred = val_pred.flatten() if isinstance(val_pred, np.ndarray) else np.array(val_pred).flatten()
                else:
                    val_pred = predictor.predict(val_features, as_pandas=False)
                    val_pred = val_pred.flatten() if isinstance(val_pred, np.ndarray) else np.array(val_pred).flatten()
                val_predictions.append(val_pred)

            # Average validation predictions
            val_ensemble = np.mean(val_predictions, axis=0)
            val_labels = val_df[config.dataset.target].values

            # Calibrate
            ensemble_pred = _calibrate_probabilities(val_ensemble, val_labels, ensemble_pred)
            print("[Ensemble] Calibration applied successfully")
        except Exception as e:
            print(f"[Ensemble] Calibration failed: {e}")

    # Create submission DataFrame
    submission = pd.DataFrame()
    submission[config.dataset.id_column] = test_df[config.dataset.id_column]

    # Ensure probabilities are in valid range
    if config.dataset.submission_probas:
        ensemble_pred = np.clip(ensemble_pred, 0.001, 0.999)

    submission[config.dataset.target] = ensemble_pred

    return submission
