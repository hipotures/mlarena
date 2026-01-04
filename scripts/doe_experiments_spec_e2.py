"""
Specification of 48 DOE experiments (E36-E83) for playground-series-s6e1.
Focus: Combinations of top performers (TargetSmooth, CatBoost, Log1p) + Tuning.
"""

def get_sanity_check_module(timestamp):
    return {
        f"{timestamp}_sanity_check": {
            "module": "sanity_check",
            "cache": True,
            "config": {
                "check_missing": True,
                "check_duplicates": True,
                "check_target": True,
                "min_unique_fraction": 0.0,
                "max_missing_fraction": 0.95,
                "drop_duplicates": True,
            }
        }
    }

def get_encoder_module(timestamp, method, config_overrides=None):
    config = {"encoding_method": method, "handle_unknown": "ignore"}
    if config_overrides:
        config.update(config_overrides)
    return {
        f"{timestamp}_encoder_{method}": {
            "module": "encoder",
            "cache": True,
            "config": config
        }
    }

def get_target_transform_module(timestamp, method):
    return {
        f"{timestamp}_target_{method}": {
            "module": "target_transformer",
            "cache": True,
            "config": {"method": method}
        }
    }

def get_rare_module(timestamp, ratio):
    label_suffix = str(ratio).replace(".", "")
    return {
        f"{timestamp}_rare_{label_suffix}": {
            "module": "rare_category_handler",
            "cache": True,
            "config": {
                "min_freq_ratio": ratio,
                "rare_label": "__RARE__"
            }
        }
    }

def get_outlier_module(timestamp):
    return {
        f"{timestamp}_outlier_iqr": {
            "module": "outlier_handler",
            "cache": True,
            "config": {
                "method": "iqr",
                "action": "clip",
                "iqr_multiplier": 1.5
            }
        }
    }

def get_datetime_module(timestamp):
    return {
        f"{timestamp}_datetime": {
            "module": "datetime_handler",
            "cache": True,
            "config": {}
        }
    }

EXPERIMENTS = []

# Timestamps start from 20260103180000, increment by 1 min
base_ts = 20260103180000

def add_exp(exp_id, name, description, modules_list):
    global base_ts
    timestamp = str(base_ts)
    base_ts += 100
    
    # Prefix module names with timestamp to ensure uniqueness
    prefixed_modules = []
    modules_dict = {}
    
    for mod_func, args in modules_list:
        mod_dict = mod_func(timestamp, *args)
        key = list(mod_dict.keys())[0]
        prefixed_modules.append(key)
        modules_dict.update(mod_dict)
        
    EXPERIMENTS.append({
        "exp_id": exp_id,
        "timestamp": timestamp,
        "name": name,
        "description": description,
        "chain": prefixed_modules,
        "modules": modules_dict
    })

# --- Group 1: Target Transform Combinations (Log1p & Sqrt) ---
# E30 was log1p + target_oof
add_exp("E36", "log1p_smooth", "Log1p + Target Smooth", [
    (get_sanity_check_module, []),
    (get_target_transform_module, ["log"]),
    (get_encoder_module, ["target_mean", {"target_encoding_smoothing": 10.0, "target_encoding_min_samples": 100}])
])
add_exp("E37", "log1p_cat", "Log1p + CatBoost", [
    (get_sanity_check_module, []),
    (get_target_transform_module, ["log"]),
    (get_encoder_module, ["catboost"])
])
add_exp("E38", "log1p_onehot", "Log1p + OneHot", [
    (get_sanity_check_module, []),
    (get_target_transform_module, ["log"]),
    (get_encoder_module, ["one_hot", {"max_cardinality": 50}])
])
add_exp("E39", "sqrt_smooth", "Sqrt + Target Smooth", [
    (get_sanity_check_module, []),
    (get_target_transform_module, ["sqrt"]),
    (get_encoder_module, ["target_mean", {"target_encoding_smoothing": 10.0, "target_encoding_min_samples": 100}])
])
add_exp("E40", "sqrt_cat", "Sqrt + CatBoost", [
    (get_sanity_check_module, []),
    (get_target_transform_module, ["sqrt"]),
    (get_encoder_module, ["catboost"])
])
add_exp("E41", "sqrt_oof", "Sqrt + Target OOF", [
    (get_sanity_check_module, []),
    (get_target_transform_module, ["sqrt"]),
    (get_encoder_module, ["target_mean", {"target_encoding_smoothing": 1.0, "target_encoding_min_samples": 1}])
])
add_exp("E42", "sqrt_onehot", "Sqrt + OneHot", [
    (get_sanity_check_module, []),
    (get_target_transform_module, ["sqrt"]),
    (get_encoder_module, ["one_hot", {"max_cardinality": 50}])
])

# --- Group 2: Rare Category (0.01 & 0.005) ---
add_exp("E43", "rare01_smooth", "Rare 0.01 + Target Smooth", [
    (get_sanity_check_module, []),
    (get_rare_module, [0.01]),
    (get_encoder_module, ["target_mean", {"target_encoding_smoothing": 10.0, "target_encoding_min_samples": 100}])
])
add_exp("E44", "rare01_cat", "Rare 0.01 + CatBoost", [
    (get_sanity_check_module, []),
    (get_rare_module, [0.01]),
    (get_encoder_module, ["catboost"])
])
add_exp("E45", "rare01_oof", "Rare 0.01 + Target OOF", [
    (get_sanity_check_module, []),
    (get_rare_module, [0.01]),
    (get_encoder_module, ["target_mean", {"target_encoding_smoothing": 1.0, "target_encoding_min_samples": 1}])
])
add_exp("E46", "rare01_onehot", "Rare 0.01 + OneHot", [
    (get_sanity_check_module, []),
    (get_rare_module, [0.01]),
    (get_encoder_module, ["one_hot", {"max_cardinality": 50}])
])
add_exp("E47", "rare005_smooth", "Rare 0.005 + Target Smooth", [
    (get_sanity_check_module, []),
    (get_rare_module, [0.005]),
    (get_encoder_module, ["target_mean", {"target_encoding_smoothing": 10.0, "target_encoding_min_samples": 100}])
])
add_exp("E48", "rare005_cat", "Rare 0.005 + CatBoost", [
    (get_sanity_check_module, []),
    (get_rare_module, [0.005]),
    (get_encoder_module, ["catboost"])
])
add_exp("E49", "rare005_oof", "Rare 0.005 + Target OOF", [
    (get_sanity_check_module, []),
    (get_rare_module, [0.005]),
    (get_encoder_module, ["target_mean", {"target_encoding_smoothing": 1.0, "target_encoding_min_samples": 1}])
])
add_exp("E50", "rare005_onehot", "Rare 0.005 + OneHot", [
    (get_sanity_check_module, []),
    (get_rare_module, [0.005]),
    (get_encoder_module, ["one_hot", {"max_cardinality": 50}])
])

# --- Group 3: Outlier Handling ---
add_exp("E51", "outlier_smooth", "Outlier + Target Smooth", [
    (get_sanity_check_module, []),
    (get_outlier_module, []),
    (get_encoder_module, ["target_mean", {"target_encoding_smoothing": 10.0, "target_encoding_min_samples": 100}])
])
add_exp("E52", "outlier_cat", "Outlier + CatBoost", [
    (get_sanity_check_module, []),
    (get_outlier_module, []),
    (get_encoder_module, ["catboost"])
])
add_exp("E53", "outlier_oof", "Outlier + Target OOF", [
    (get_sanity_check_module, []),
    (get_outlier_module, []),
    (get_encoder_module, ["target_mean", {"target_encoding_smoothing": 1.0, "target_encoding_min_samples": 1}])
])
add_exp("E54", "outlier_onehot", "Outlier + OneHot", [
    (get_sanity_check_module, []),
    (get_outlier_module, []),
    (get_encoder_module, ["one_hot", {"max_cardinality": 50}])
])

# --- Group 4: Hybrid Chains (Rare/Outlier + Log1p) ---
add_exp("E55", "rare01_log_smooth", "Rare01 + Log1p + Smooth", [
    (get_sanity_check_module, []),
    (get_rare_module, [0.01]),
    (get_target_transform_module, ["log"]),
    (get_encoder_module, ["target_mean", {"target_encoding_smoothing": 10.0, "target_encoding_min_samples": 100}])
])
add_exp("E56", "rare01_log_cat", "Rare01 + Log1p + CatBoost", [
    (get_sanity_check_module, []),
    (get_rare_module, [0.01]),
    (get_target_transform_module, ["log"]),
    (get_encoder_module, ["catboost"])
])
add_exp("E57", "out_log_smooth", "Outlier + Log1p + Smooth", [
    (get_sanity_check_module, []),
    (get_outlier_module, []),
    (get_target_transform_module, ["log"]),
    (get_encoder_module, ["target_mean", {"target_encoding_smoothing": 10.0, "target_encoding_min_samples": 100}])
])
add_exp("E58", "out_log_cat", "Outlier + Log1p + CatBoost", [
    (get_sanity_check_module, []),
    (get_outlier_module, []),
    (get_target_transform_module, ["log"]),
    (get_encoder_module, ["catboost"])
])
add_exp("E59", "rare_out_log_smooth", "Rare + Out + Log + Smooth", [
    (get_sanity_check_module, []),
    (get_rare_module, [0.01]),
    (get_outlier_module, []),
    (get_target_transform_module, ["log"]),
    (get_encoder_module, ["target_mean", {"target_encoding_smoothing": 10.0, "target_encoding_min_samples": 100}])
])
add_exp("E60", "rare_out_log_cat", "Rare + Out + Log + Cat", [
    (get_sanity_check_module, []),
    (get_rare_module, [0.01]),
    (get_outlier_module, []),
    (get_target_transform_module, ["log"]),
    (get_encoder_module, ["catboost"])
])

# --- Group 5: New Encoders & Tuning ---
add_exp("E61", "glmm_enc", "GLMM Encoder", [
    (get_sanity_check_module, []),
    (get_encoder_module, ["glmm"])
])
add_exp("E62", "loo_enc", "LeaveOneOut Encoder", [
    (get_sanity_check_module, []),
    (get_encoder_module, ["leave_one_out"])
])
add_exp("E63", "smooth_s5", "Target Smooth (s=5)", [
    (get_sanity_check_module, []),
    (get_encoder_module, ["target_mean", {"target_encoding_smoothing": 5.0, "target_encoding_min_samples": 100}])
])
add_exp("E64", "smooth_s20", "Target Smooth (s=20)", [
    (get_sanity_check_module, []),
    (get_encoder_module, ["target_mean", {"target_encoding_smoothing": 20.0, "target_encoding_min_samples": 100}])
])
add_exp("E65", "smooth_m50", "Target Smooth (min=50)", [
    (get_sanity_check_module, []),
    (get_encoder_module, ["target_mean", {"target_encoding_smoothing": 10.0, "target_encoding_min_samples": 50}])
])
add_exp("E66", "smooth_m200", "Target Smooth (min=200)", [
    (get_sanity_check_module, []),
    (get_encoder_module, ["target_mean", {"target_encoding_smoothing": 10.0, "target_encoding_min_samples": 200}])
])
add_exp("E67", "oof_s5", "Target OOF (s=5)", [
    (get_sanity_check_module, []),
    (get_encoder_module, ["target_mean", {"target_encoding_smoothing": 5.0, "target_encoding_min_samples": 1}])
])
add_exp("E68", "oof_m10", "Target OOF (min=10)", [
    (get_sanity_check_module, []),
    (get_encoder_module, ["target_mean", {"target_encoding_smoothing": 1.0, "target_encoding_min_samples": 10}])
])
add_exp("E69", "cat_a1", "CatBoost (a=1)", [
    (get_sanity_check_module, []),
    (get_encoder_module, ["catboost", {"a": 1}])
])
add_exp("E70", "cat_a10", "CatBoost (a=10)", [
    (get_sanity_check_module, []),
    (get_encoder_module, ["catboost", {"a": 10}])
])

# --- Group 6: Advanced Combinations ---
add_exp("E71", "rare_log_glmm", "Rare + Log + GLMM", [
    (get_sanity_check_module, []),
    (get_rare_module, [0.01]),
    (get_target_transform_module, ["log"]),
    (get_encoder_module, ["glmm"])
])
add_exp("E72", "rare_log_loo", "Rare + Log + LOO", [
    (get_sanity_check_module, []),
    (get_rare_module, [0.01]),
    (get_target_transform_module, ["log"]),
    (get_encoder_module, ["leave_one_out"])
])
add_exp("E73", "rare005_log_smooth", "Rare005 + Log + Smooth", [
    (get_sanity_check_module, []),
    (get_rare_module, [0.005]),
    (get_target_transform_module, ["log"]),
    (get_encoder_module, ["target_mean", {"target_encoding_smoothing": 10.0, "target_encoding_min_samples": 100}])
])
add_exp("E74", "rare005_log_cat", "Rare005 + Log + Cat", [
    (get_sanity_check_module, []),
    (get_rare_module, [0.005]),
    (get_target_transform_module, ["log"]),
    (get_encoder_module, ["catboost"])
])
add_exp("E75", "out_sqrt_smooth", "Out + Sqrt + Smooth", [
    (get_sanity_check_module, []),
    (get_outlier_module, []),
    (get_target_transform_module, ["sqrt"]),
    (get_encoder_module, ["target_mean", {"target_encoding_smoothing": 10.0, "target_encoding_min_samples": 100}])
])
add_exp("E76", "out_sqrt_cat", "Out + Sqrt + Cat", [
    (get_sanity_check_module, []),
    (get_outlier_module, []),
    (get_target_transform_module, ["sqrt"]),
    (get_encoder_module, ["catboost"])
])
add_exp("E77", "rare_out_smooth", "Rare + Out + Smooth", [
    (get_sanity_check_module, []),
    (get_rare_module, [0.01]),
    (get_outlier_module, []),
    (get_encoder_module, ["target_mean", {"target_encoding_smoothing": 10.0, "target_encoding_min_samples": 100}])
])
add_exp("E78", "rare_out_cat", "Rare + Out + Cat", [
    (get_sanity_check_module, []),
    (get_rare_module, [0.01]),
    (get_outlier_module, []),
    (get_encoder_module, ["catboost"])
])
add_exp("E79", "date_smooth", "Datetime + Smooth", [
    (get_sanity_check_module, []),
    (get_datetime_module, []),
    (get_encoder_module, ["target_mean", {"target_encoding_smoothing": 10.0, "target_encoding_min_samples": 100}])
])
add_exp("E80", "date_cat", "Datetime + Cat", [
    (get_sanity_check_module, []),
    (get_datetime_module, []),
    (get_encoder_module, ["catboost"])
])
add_exp("E81", "date_log_smooth", "Date + Log + Smooth", [
    (get_sanity_check_module, []),
    (get_datetime_module, []),
    (get_target_transform_module, ["log"]),
    (get_encoder_module, ["target_mean", {"target_encoding_smoothing": 10.0, "target_encoding_min_samples": 100}])
])
add_exp("E82", "date_log_cat", "Date + Log + Cat", [
    (get_sanity_check_module, []),
    (get_datetime_module, []),
    (get_target_transform_module, ["log"]),
    (get_encoder_module, ["catboost"])
])
add_exp("E83", "kitchen_sink_v2", "All (Rare,Out,Log,Date,Smooth)", [
    (get_sanity_check_module, []),
    (get_rare_module, [0.01]),
    (get_outlier_module, []),
    (get_target_transform_module, ["log"]),
    (get_datetime_module, []),
    (get_encoder_module, ["target_mean", {"target_encoding_smoothing": 10.0, "target_encoding_min_samples": 100}])
])
