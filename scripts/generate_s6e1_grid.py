import yaml
import os
from pathlib import Path

PROJECT_PATH = "projects/kaggle/playground-series-s6e1"
MODEL_DIR = f"{PROJECT_PATH}/templates/model"
PREPROCESS_DIR = f"{PROJECT_PATH}/templates/preprocess"

# Grid parameters
n_features_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
min_freq_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Base model config (from mcts_s6e1_009_eval_001.yaml)
model_base = {
    "model": "autogluon_baseline",
    "config": {"random_state": 42},
    "preset": "medium",
    "time_limit": 600,
    "use_gpu": True,
    "seed": 42,
    "included_model_types": ["XGB", "GBM"],
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

start_idx = 100
current_idx = start_idx

for nf in n_features_list:
    for mf in min_freq_list:
        suffix = f"test-20250118_{current_idx}"
        
        # 1. Model Template
        model_config = model_base.copy()
        model_config["preprocess_template"] = suffix
        with open(f"{MODEL_DIR}/{suffix}.yaml", "w") as f:
            yaml.dump(model_config, f, sort_keys=False)
            
        # 2. Preprocess Main Template
        preprocess_main = {
            "chain": [
                "load_orig_exam_score",
                "imputer_basic",
                f"{suffix}-rare_category_handler",
                "test-20250118_01-categorical_encoder",
                "test-20250118_01-outlier_handler",
                "test-20250118_01-scaler",
                "test-20250118_01-feature_interactions",
                "test-20250118_01-feature_polynomial",
                f"{suffix}-feature_selector"
            ]
        }
        with open(f"{PREPROCESS_DIR}/{suffix}.yaml", "w") as f:
            yaml.dump(preprocess_main, f, sort_keys=False)
            
        # 3. Rare Category Handler Template
        rare_config = {
            "module": "rare_category_handler",
            "cache": True,
            "config": {
                "min_freq": mf,
                "min_freq_ratio": 0.02,
                "top_k": None, # Set to null so min_freq works
                "rare_label": "__RARE__",
                "detect_id_like_columns": True,
                "id_unique_fraction_threshold": 0.95
            }
        }
        with open(f"{PREPROCESS_DIR}/{suffix}-rare_category_handler.yaml", "w") as f:
            yaml.dump(rare_config, f, sort_keys=False)
            
        # 4. Feature Selector Template
        selector_config = {
            "module": "feature_selector",
            "cache": True,
            "config": {
                "selection_method": "mi",
                "n_features": nf,
                "max_drop_fraction": 0.5,
                "random_state": 42,
                "protect_cb_features": True
            }
        }
        with open(f"{PREPROCESS_DIR}/{suffix}-feature_selector.yaml", "w") as f:
            yaml.dump(selector_config, f, sort_keys=False)
            
        current_idx += 1

print(f"Generated {current_idx - start_idx} experiments (100-199)")
