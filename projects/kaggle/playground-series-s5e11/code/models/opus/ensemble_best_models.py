"""
Final Ensemble: Weighted average of best performing models.
Combines predictions from fe17, fe20, fe21, fe22, and original best model.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor

# Model paths and weights based on public scores
ENSEMBLE_CONFIG = {
    "models": [
        {
            "name": "fe17_log_EMI",
            "path": "experiments/*/artifacts/best-cpu-fe17-autogluon_features_17",
            "weight": 0.30,  # Best single: 0.92356
            "public_score": 0.92356
        },
        {
            "name": "fe20_student_debt",
            "path": "experiments/*/artifacts/best-cpu-fe20-autogluon_features_20", 
            "weight": 0.25,  # 2nd best: 0.92353
            "public_score": 0.92353
        },
        {
            "name": "fe21_ultimate",
            "path": "experiments/*/artifacts/best-cpu-fe21-autogluon_features_21",
            "weight": 0.20,  # Expected high score
            "public_score": None  # To be determined
        },
        {
            "name": "fe22_target_encoded",
            "path": "experiments/*/artifacts/best-cpu-fe22-autogluon_features_22",
            "weight": 0.15,  # Target encoding boost
            "public_score": None  # To be determined
        },
        {
            "name": "original_best",
            "path": "experiments/*/artifacts/on-best_quality",
            "weight": 0.10,  # Original best: 0.92434
            "public_score": 0.92434
        }
    ],
    "ensemble_method": "weighted_average",  # Can also use 'rank_average' or 'power_average'
    "calibration": True  # Apply probability calibration
}

def find_model_path(pattern: str, project_root: Path) -> Optional[Path]:
    """Find the most recent model matching the pattern."""
    import glob
    
    search_pattern = str(project_root / pattern)
    matches = glob.glob(search_pattern)
    
    if matches:
        # Return most recent
        return Path(max(matches, key=lambda x: Path(x).stat().st_mtime))
    return None

def load_models(project_root: Path) -> Dict[str, TabularPredictor]:
    """Load all models for ensemble."""
    models = {}
    
    for model_cfg in ENSEMBLE_CONFIG["models"]:
        model_path = find_model_path(model_cfg["path"], project_root)
        
        if model_path and model_path.exists():
            try:
                predictor = TabularPredictor.load(str(model_path))
                models[model_cfg["name"]] = predictor
                print(f"✓ Loaded {model_cfg['name']} from {model_path}")
            except Exception as e:
                print(f"✗ Failed to load {model_cfg['name']}: {e}")
        else:
            print(f"✗ Model not found: {model_cfg['name']}")
    
    return models

def weighted_average_ensemble(
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

def rank_average_ensemble(predictions: Dict[str, np.ndarray]) -> np.ndarray:
    """Average of rank-transformed predictions."""
    from scipy.stats import rankdata
    
    ranks = {}
    for model_name, pred in predictions.items():
        ranks[model_name] = rankdata(pred) / len(pred)
    
    # Simple average of ranks
    ensemble_rank = np.mean(list(ranks.values()), axis=0)
    return ensemble_rank

def power_average_ensemble(
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

def calibrate_probabilities(
    train_preds: np.ndarray,
    train_labels: np.ndarray,
    test_preds: np.ndarray
) -> np.ndarray:
    """Apply isotonic regression calibration."""
    from sklearn.isotonic import IsotonicRegression
    
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(train_preds, train_labels)
    
    return calibrator.predict(test_preds)

def create_ensemble_predictions(
    test_df: pd.DataFrame,
    project_root: Path,
    validation_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Create ensemble predictions from multiple models."""
    
    # Load all models
    models = load_models(project_root)
    
    if len(models) < 2:
        raise ValueError(f"Need at least 2 models for ensemble, found {len(models)}")
    
    # Generate predictions from each model
    predictions = {}
    weights = {}
    
    for model_cfg in ENSEMBLE_CONFIG["models"]:
        model_name = model_cfg["name"]
        
        if model_name in models:
            model = models[model_name]
            
            # Generate predictions
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(test_df, as_pandas=False)
                # Get positive class probability
                if pred.shape[1] > 1:
                    pred = pred[:, 1]
                else:
                    pred = pred.flatten()
            else:
                pred = model.predict(test_df, as_pandas=False)
            
            predictions[model_name] = pred
            weights[model_name] = model_cfg["weight"]
            
            print(f"Generated predictions from {model_name}: shape={pred.shape}")
    
    # Apply ensemble method
    method = ENSEMBLE_CONFIG["ensemble_method"]
    
    if method == "weighted_average":
        ensemble_pred = weighted_average_ensemble(predictions, weights)
    elif method == "rank_average":
        ensemble_pred = rank_average_ensemble(predictions)
    elif method == "power_average":
        ensemble_pred = power_average_ensemble(predictions, weights)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
    
    print(f"Ensemble method: {method}")
    print(f"Ensemble predictions: min={ensemble_pred.min():.4f}, "
          f"max={ensemble_pred.max():.4f}, mean={ensemble_pred.mean():.4f}")
    
    # Apply calibration if requested and validation data available
    if ENSEMBLE_CONFIG["calibration"] and validation_df is not None:
        print("Applying probability calibration...")
        
        # Generate validation predictions for calibration
        val_predictions = []
        for model_name in predictions.keys():
            model = models[model_name]
            if hasattr(model, 'predict_proba'):
                val_pred = model.predict_proba(validation_df, as_pandas=False)
                if val_pred.shape[1] > 1:
                    val_pred = val_pred[:, 1]
            else:
                val_pred = model.predict(validation_df, as_pandas=False)
            val_predictions.append(val_pred)
        
        # Average validation predictions
        val_ensemble = np.mean(val_predictions, axis=0)
        val_labels = validation_df['loan_paid_back'].values
        
        # Calibrate
        ensemble_pred = calibrate_probabilities(val_ensemble, val_labels, ensemble_pred)
        print("Calibration applied successfully")
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'id': test_df['id'],
        'loan_paid_back': ensemble_pred
    })
    
    # Ensure probabilities are in valid range
    submission['loan_paid_back'] = submission['loan_paid_back'].clip(0.001, 0.999)
    
    return submission

def analyze_ensemble_diversity(predictions: Dict[str, np.ndarray]):
    """Analyze diversity among ensemble members."""
    from scipy.stats import pearsonr
    
    model_names = list(predictions.keys())
    n_models = len(model_names)
    
    print("\n=== Ensemble Diversity Analysis ===")
    print("Correlation matrix between models:")
    
    corr_matrix = np.zeros((n_models, n_models))
    for i, name1 in enumerate(model_names):
        for j, name2 in enumerate(model_names):
            if i <= j:
                corr, _ = pearsonr(predictions[name1], predictions[name2])
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
                
                if i < j:
                    print(f"  {name1} vs {name2}: {corr:.4f}")
    
    avg_correlation = (corr_matrix.sum() - n_models) / (n_models * (n_models - 1))
    print(f"\nAverage pairwise correlation: {avg_correlation:.4f}")
    print(f"Diversity score (1 - avg_corr): {1 - avg_correlation:.4f}")
    
    return corr_matrix

if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analyze-diversity", action="store_true")
    
    args = parser.parse_args()
    
    # Load test data
    test_df = pd.read_csv(args.test_data)
    
    # Load validation data if provided
    val_df = None
    if args.validation_data:
        val_df = pd.read_csv(args.validation_data)
    
    # Create ensemble predictions
    submission = create_ensemble_predictions(
        test_df, 
        args.project_root,
        validation_df=val_df
    )
    
    # Save submission
    submission.to_csv(args.output, index=False)
    print(f"\n✓ Ensemble submission saved to: {args.output}")
    
    # Analyze diversity if requested
    if args.analyze_diversity:
        # Would need to extract predictions dict from create_ensemble_predictions
        print("\nRun with individual model predictions for diversity analysis")
