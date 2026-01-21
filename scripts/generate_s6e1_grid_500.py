import yaml
import os
from pathlib import Path

PROJECT_PATH = "projects/kaggle/playground-series-s6e1"
MODEL_DIR = f"{PROJECT_PATH}/templates/model"
PREPROCESS_DIR = f"{PROJECT_PATH}/templates/preprocess"

# Grid ranges (centered around Top 1)
nf_list = [0.3, 0.4, 0.5, 0.6]
mf_list = [18, 19, 20, 21, 22]
ic_list = [0.01, 0.02, 0.03, 0.04, 0.05]
adt_list = [25, 30, 35, 40, 45]

# Base model config
model_base = {
    "model": "autogluon_baseline",
    "config": {"random_state": 42},
    "preset": "medium",
    "time_limit": 600,
    "use_gpu": True,
    "seed": 42,
    "included_model_types": ["XGB"],
    "fit_args": {
        "save_space": True,
        "fit_weighted_ensemble": False,
        "auto_stack": False,
        "use_bag_holdout": True,
        "save_bag_folds": True,
        "num_bag_folds": 5,
        "num_bag_sets": 1,
        "num_cpus": 6
    }
}

current_idx = 100

for nf in nf_list:
    for mf in mf_list:
        for ic in ic_list:
            for adt in adt_list:
                suffix = f"test-20250118_{current_idx}"
                
                # 1. Model Template
                model_config = model_base.copy()
                model_config["preprocess_template"] = suffix
                model_config["experiment_id"] = suffix
                with open(f"{MODEL_DIR}/{suffix}.yaml", "w") as f:
                    yaml.dump(model_config, f, sort_keys=False)
                
                # 2. Preprocess Main Template (load_orig removed)
                preprocess_main = {
                    "chain": [
                        "sanity_check",
                        "train_fraction_fast",
                        "imputer_basic",
                        f"{suffix}-rare_category_handler",
                        f"{suffix}-categorical_encoder",
                        f"{suffix}-outlier_handler",
                        "test-20250118_01-scaler",
                        "test-20250118_01-feature_interactions",
                        "test-20250118_01-feature_polynomial",
                        f"{suffix}-feature_selector"
                    ]
                }
                with open(f"{PREPROCESS_DIR}/{suffix}.yaml", "w") as f:
                    yaml.dump(preprocess_main, f, sort_keys=False)
                
                # 3. Sub-modules
                # Rare Category
                with open(f"{PREPROCESS_DIR}/{suffix}-rare_category_handler.yaml", "w") as f:
                    yaml.dump({
                        "module": "rare_category_handler",
                        "cache": True,
                        "config": {
                            "min_freq": mf,
                            "min_freq_ratio": 0.02,
                            "top_k": None,
                            "rare_label": "__RARE__",
                            "detect_id_like_columns": True,
                            "id_unique_fraction_threshold": 0.95
                        }
                    }, f, sort_keys=False)
                
                # Categorical Encoder
                with open(f"{PREPROCESS_DIR}/{suffix}-categorical_encoder.yaml", "w") as f:
                    yaml.dump({
                        "module": "categorical_encoder",
                        "cache": True,
                        "config": {
                            "max_cardinality": 100,
                            "enable_auto_detect": True,
                            "auto_detect_threshold": adt,
                            "use_original_features_only": False
                        }
                    }, f, sort_keys=False)
                
                # Outlier Handler
                with open(f"{PREPROCESS_DIR}/{suffix}-outlier_handler.yaml", "w") as f:
                    yaml.dump({
                        "module": "outlier_handler",
                        "cache": True,
                        "config": {
                            "outlier_method": "isolation_forest",
                            "isoforest_contamination": ic,
                            "action": "flag_only",
                            "random_state": 42
                        }
                    }, f, sort_keys=False)
                
                # Feature Selector
                with open(f"{PREPROCESS_DIR}/{suffix}-feature_selector.yaml", "w") as f:
                    yaml.dump({
                        "module": "feature_selector",
                        "cache": True,
                        "config": {
                            "selection_method": "mi",
                            "n_features": nf,
                            "max_drop_fraction": 0.5,
                            "random_state": 42,
                            "protect_cb_features": True
                        }
                    }, f, sort_keys=False)
                
                current_idx += 1
                if current_idx >= 600:
                    break
            if current_idx >= 600: break
        if current_idx >= 600: break
    if current_idx >= 600: break

print(f"Regenerated 500 experiments (100-599) without load_orig.")
