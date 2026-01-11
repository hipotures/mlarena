#!/usr/bin/env python3
import argparse
import optuna
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Inject manual config for trial 5912 variant")
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
        print(f"Study '{args.study_name}' not found.")
        sys.exit(1)

    print(f"Loaded study: {study.study_name}")

    # Params from trial 5912, with target_transformer DISABLED
    inject_params = {
    "adversarial_validation.enabled": False,
    "categorical_cross.enabled": False,
    "categorical_encoder.enabled": False,
    "clustering_features.enabled": True,
    "clustering_features.kmeans_id.n_clusters": 45.0,
    "clustering_features.variant": 'kmeans_id',
    "dimensionality_reducer.enabled": False,
    "drift_detector.enabled": True,
    "drift_detector.psi_drop.drift_metric": 'psi',
    "drift_detector.psi_drop.max_drop_fraction": 0.1,
    "drift_detector.psi_drop.max_psi": 0.2,
    "drift_detector.variant": 'psi_drop',
    "encoder.enabled": True,
    "encoder.one_hot.drop_first": False,
    "encoder.one_hot.encoding_method": 'one_hot',
    "encoder.variant": 'one_hot',
    "feature_interactions.enabled": False,
    "feature_polynomial.degree_2.max_generated_features": 25.0,
    "feature_polynomial.degree_2.poly_interaction_only": False,
    "feature_polynomial.enabled": True,
    "feature_polynomial.variant": 'degree_2',
    "imputer.most_frequent.numeric_strategy": 'most_frequent',
    "imputer.variant": 'most_frequent',
    "missingness_features.enabled": True,
    "missingness_features.row_stats_only.add_row_missing_ratio": True,
    "missingness_features.row_stats_only.cap_row_missing_count": 25.0,
    "missingness_features.variant": 'row_stats_only',
    "numeric_binner.enabled": True,
    "numeric_binner.kmeans_ordinal.encode": 'ordinal',
    "numeric_binner.kmeans_ordinal.n_bins": 25.0,
    "numeric_binner.kmeans_ordinal.strategy": 'kmeans',
    "numeric_binner.variant": 'kmeans_ordinal',
    "outlier_handler.enabled": False,
    "rank_features_post.enabled": False,
    "rank_features_pre.enabled": False,
    "rare_category_handler.enabled": True,
    "rare_category_handler.top_k.min_freq": 19.0,
    "rare_category_handler.top_k.min_freq_ratio": 0.015,
    "rare_category_handler.top_k.top_k": 10.0,
    "rare_category_handler.variant": 'top_k',
    "row_aggregates.enabled": False,
    "sanity_check.default.drop_duplicates": False,
    "sanity_check.default.max_missing_fraction": 0.97,
    "sanity_check.enabled": True,
    "sanity_check.variant": 'default',
    "scaler.enabled": True,
    "scaler.quantile_normal.n_quantiles": 800.0,
    "scaler.quantile_normal.scaling_method": 'quantile_normal',
    "scaler.variant": 'quantile_normal',
    
    # --- MODIFIED: DISABLED TARGET TRANSFORMER ---
    "target_transformer.enabled": False,
    }

    print("Injecting parameters for 5912 (modified)...")
    study.enqueue_trial(inject_params)
    print("Success! Trial enqueued.")

if __name__ == "__main__":
    main()
