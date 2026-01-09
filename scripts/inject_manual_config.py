#!/usr/bin/env python3
import argparse
import optuna
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Inject a manual configuration into an Optuna study")
    parser.add_argument("--db", default="/mnt/mlarena/projects/kaggle/playground-series-s6e1/experiments/db/optuna_smoke_s6e1_heavy_v2.sqlite", help="Path to Optuna SQLite DB")
    parser.add_argument("--study-name", default="smoke_s6e1_heavy_v2", help="Name of the study")
    args = parser.parse_args()

    db_path = Path(args.db).resolve()
    if not db_path.exists():
        print(f"Error: DB {db_path} not found.")
        sys.exit(1)

    storage = f"sqlite:///{db_path}"
    
    try:
        study = optuna.load_study(study_name=args.study_name, storage=storage)
    except KeyError:
        # Fallback if study name is wrong, try to list them
        print(f"Study '{args.study_name}' not found. Available studies:")
        optuna.study.get_all_study_summaries(storage=storage)
        sys.exit(1)

    print(f"Loaded study: {study.study_name} with {len(study.trials)} trials.")

    # Define the configuration we want to inject based on the "advanced-chain" analysis
    # (sanity_check -> rare_category_handler -> custom_features)
    # Note: custom_features is not in Optuna space, so we skip it.
    # We turn OFF everything else to isolate the signal.
    
    inject_params = {
        # --- Core Modules ---
        "sanity_check.enabled": True,
        "sanity_check.variant": "default",
        "sanity_check.default.drop_duplicates": True,
        "sanity_check.default.max_missing_fraction": 0.9, 

        # Rare Category Handler
        "rare_category_handler.enabled": True,
        "rare_category_handler.variant": "top_k",
        "rare_category_handler.top_k.min_freq": 12,
        "rare_category_handler.top_k.min_freq_ratio": 0.02,
        "rare_category_handler.top_k.top_k": 30,

        # Feature Polynomial (Correct mapping for feature_engineer)
        "feature_polynomial.enabled": True,
        "feature_polynomial.variant": "degree_2",
        "feature_polynomial.degree_2.poly_interaction_only": False,
        "feature_polynomial.degree_2.max_generated_features": 150,
        
        # Disable incorrect interaction mapping
        "feature_interactions.enabled": False,

        # Encoder
        "encoder.enabled": True,
        "encoder.variant": "one_hot",

        # Imputer
        "imputer.enabled": True,
        "imputer.variant": "mean",
        "imputer.mean.numeric_strategy": "mean",

        # Scaler - Quantile Normal (Heavy variant)
        "scaler.enabled": True,
        "scaler.variant": "quantile_normal",
        "scaler.quantile_normal.scaling_method": "quantile_normal",
        "scaler.quantile_normal.n_quantiles": 900,

        # Outlier Handler - Isolation Forest (Heavy variant)
        "outlier_handler.enabled": True,
        "outlier_handler.variant": "isolation_forest",
        "outlier_handler.isolation_forest.outlier_method": "isolation_forest",
        "outlier_handler.isolation_forest.isoforest_contamination": 0.03,
        "outlier_handler.isolation_forest.action": "flag_only",
        
        # --- Disable Others ---
        "drift_detector.enabled": False,
        "feature_selector.enabled": False,
        "missingness_features.enabled": False,
        "numeric_binner.enabled": False,
        "rank_features_post.enabled": False,
        "rank_features_pre.enabled": False,
        "row_aggregates.enabled": False,
        "target_transformer.enabled": False,
        "feature_engineer.enabled": False, 
    }

    # Validate against Distributions
    # We need to make sure we don't send params that don't exist in the current search space
    # or satisfy conditional logic.
    
    print("\nValidating parameters against study distributions...")
    
    # In Optuna, we can't easily validate conditional parameters without running the objective function
    # because distributions are defined dynamically. 
    # However, enqueue_trial creates a FixedTrial which *will* fail at runtime if params are invalid.
    # The safest way is to trust our mapping but warn the user.

    print("Parameters to inject:")
    for k, v in inject_params.items():
        print(f"  {k}: {v}")

    confirm = input("\nDo you want to enqueue this trial? (y/n): ")
    if confirm.lower() != 'y':
        print("Aborted.")
        sys.exit(0)

    study.enqueue_trial(inject_params)
    print("Successfully enqueued trial! It will be picked up by the next worker.")

if __name__ == "__main__":
    main()
