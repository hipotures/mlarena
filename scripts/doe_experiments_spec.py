"""
Full specification of 36 DOE experiments for playground-series-s6e1.
"""

def get_sanity_check_module(timestamp):
    """Standard sanity check module."""
    return {
        f"{timestamp}_sanity_check": {
            "module": "sanity_check",
            "cache": True,
            "config": {
                "check_missing": True,
                "check_duplicates": True,
                "check_target": True,
                "min_unique_fraction": 0.0,  # CRITICAL: Disable auto-drop of low-cardinality columns
                "max_missing_fraction": 0.95,
                "drop_duplicates": True,
            }
        }
    }

# Full experiment specifications
EXPERIMENTS = [
    # E00: Baseline
    {
        "exp_id": "E00",
        "timestamp": "20260101120000",
        "name": "baseline",
        "parent": None,
        "description": "Baseline: sanity_check only",
        "chain": ["20260101120000_sanity_check"],
        "modules": get_sanity_check_module("20260101120000")
    },

    # E01: One-hot encoding
    {
        "exp_id": "E01",
        "timestamp": "20260101120100",
        "name": "onehot",
        "parent": "E00",
        "description": "Encoding: ordinal → one-hot",
        "chain": ["20260101120100_sanity_check", "20260101120100_encoder_onehot"],
        "modules": {
            **get_sanity_check_module("20260101120100"),
            "20260101120100_encoder_onehot": {
                "module": "encoder",
                "cache": True,
                "config": {
                    "encoding_method": "one_hot",
                    "max_cardinality": 50,
                    "drop_first": False,
                    "handle_unknown": "ignore",
                }
            }
        }
    },

    # E02: Target encoding (OOF)
    {
        "exp_id": "E02",
        "timestamp": "20260101120200",
        "name": "target_enc",
        "parent": "E00",
        "description": "Encoding: ordinal → target_mean (5-fold OOF)",
        "chain": ["20260101120200_sanity_check", "20260101120200_encoder_target_oof"],
        "modules": {
            **get_sanity_check_module("20260101120200"),
            "20260101120200_encoder_target_oof": {
                "module": "encoder",
                "cache": True,
                "config": {
                    "encoding_method": "target_mean",
                    "target_encoding_smoothing": 1.0,
                    "target_encoding_min_samples": 1,
                    "handle_unknown": "ignore",
                }
            }
        }
    },

    # E03: Custom features
    {
        "exp_id": "E03",
        "timestamp": "20260101120300",
        "name": "custom_feat",
        "parent": "E00",
        "description": "Add: custom features (5 interactions)",
        "chain": ["20260101120300_sanity_check", "20260101120300_custom_features"],
        "modules": {
            **get_sanity_check_module("20260101120300"),
            "20260101120300_custom_features": {
                "module": "custom_features",
                "cache": True,
                "config": {}
            }
        }
    },

    # E04: Polynomial features (degree=2, numeric only)
    {
        "exp_id": "E04",
        "timestamp": "20260101120400",
        "name": "poly2",
        "parent": "E00",
        "description": "Add: polynomial features (degree=2, numeric only)",
        "chain": ["20260101120400_sanity_check", "20260101120400_feature_engineer_poly2"],
        "modules": {
            **get_sanity_check_module("20260101120400"),
            "20260101120400_feature_engineer_poly2": {
                "module": "feature_engineer",
                "cache": True,
                "config": {
                    "poly_degree": 2,
                    "poly_interaction_only": False,
                    "poly_include_bias": False,
                    "poly_columns": ["study_hours", "class_attendance", "sleep_hours"],
                }
            }
        }
    },

    # E05: Standard scaling
    {
        "exp_id": "E05",
        "timestamp": "20260101120500",
        "name": "standard_scale",
        "parent": "E00",
        "description": "Add: standard scaling (numeric)",
        "chain": ["20260101120500_sanity_check", "20260101120500_scaler_standard"],
        "modules": {
            **get_sanity_check_module("20260101120500"),
            "20260101120500_scaler_standard": {
                "module": "scaler",
                "cache": True,
                "config": {
                    "method": "standard",
                }
            }
        }
    },

    # E06: Robust scaling
    {
        "exp_id": "E06",
        "timestamp": "20260101120600",
        "name": "robust_scale",
        "parent": "E00",
        "description": "Add: robust scaling (numeric)",
        "chain": ["20260101120600_sanity_check", "20260101120600_scaler_robust"],
        "modules": {
            **get_sanity_check_module("20260101120600"),
            "20260101120600_scaler_robust": {
                "module": "scaler",
                "cache": True,
                "config": {
                    "method": "robust",
                }
            }
        }
    },

    # E07: Log1p target transform
    {
        "exp_id": "E07",
        "timestamp": "20260101120700",
        "name": "log1p_target",
        "parent": "E00",
        "description": "Add: log1p target transform",
        "chain": ["20260101120700_sanity_check", "20260101120700_target_transformer_log1p"],
        "modules": {
            **get_sanity_check_module("20260101120700"),
            "20260101120700_target_transformer_log1p": {
                "module": "target_transformer",
                "cache": True,
                "config": {
                    "method": "log",
                }
            }
        }
    },

    # E08: Sqrt target transform
    {
        "exp_id": "E08",
        "timestamp": "20260101120800",
        "name": "sqrt_target",
        "parent": "E00",
        "description": "Add: sqrt target transform",
        "chain": ["20260101120800_sanity_check", "20260101120800_target_transformer_sqrt"],
        "modules": {
            **get_sanity_check_module("20260101120800"),
            "20260101120800_target_transformer_sqrt": {
                "module": "target_transformer",
                "cache": True,
                "config": {
                    "method": "sqrt",
                }
            }
        }
    },

    # E09: Rare category handler (0.01)
    {
        "exp_id": "E09",
        "timestamp": "20260101120900",
        "name": "rare_001",
        "parent": "E00",
        "description": "Add: rare category handler (0.01)",
        "chain": ["20260101120900_sanity_check", "20260101120900_rare_handler_001"],
        "modules": {
            **get_sanity_check_module("20260101120900"),
            "20260101120900_rare_handler_001": {
                "module": "rare_category_handler",
                "cache": True,
                "config": {
                    "min_freq_ratio": 0.01,  # Fixed: use ratio instead of absolute freq
                    "rare_label": "__RARE__",
                }
            }
        }
    },

    # E10: Rare category handler (0.005)
    {
        "exp_id": "E10",
        "timestamp": "20260101121000",
        "name": "rare_0005",
        "parent": "E00",
        "description": "Add: rare category handler (0.005)",
        "chain": ["20260101121000_sanity_check", "20260101121000_rare_handler_0005"],
        "modules": {
            **get_sanity_check_module("20260101121000"),
            "20260101121000_rare_handler_0005": {
                "module": "rare_category_handler",
                "cache": True,
                "config": {
                    "min_freq_ratio": 0.005,  # Fixed: use ratio instead of absolute freq
                    "rare_label": "__RARE__",
                }
            }
        }
    },

    # E11-E20: Additional single-step experiments
    # (Simplified versions - can be expanded later)

    # E21: Chain A1 (target_enc + custom_feat)
    {
        "exp_id": "E21",
        "timestamp": "20260101122100",
        "name": "chain_a1",
        "parent": "E02",
        "description": "Chain A: target_enc + custom_feat",
        "chain": [
            "20260101122100_sanity_check",
            "20260101122100_encoder_target_oof",
            "20260101122100_custom_features"
        ],
        "modules": {
            **get_sanity_check_module("20260101122100"),
            "20260101122100_encoder_target_oof": {
                "module": "encoder",
                "cache": True,
                "config": {
                    "encoding_method": "target_mean",
                    "target_encoding_smoothing": 1.0,
                    "target_encoding_min_samples": 1,
                    "handle_unknown": "ignore",
                }
            },
            "20260101122100_custom_features": {
                "module": "custom_features",
                "cache": True,
                "config": {}
            }
        }
    },

    # E22-E35: Additional chain experiments
    # These will be generated programmatically below
]

# Additional experiments (E11-E20 and remaining chains E22-E35)
# Generated programmatically to avoid repetition

def add_remaining_experiments():
    """Add remaining 25 experiments programmatically."""

    # E11: Variance threshold
    EXPERIMENTS.append({
        "exp_id": "E11",
        "timestamp": "20260101121100",
        "name": "var_thresh",
        "parent": "E00",
        "description": "Add: variance threshold (0.01)",
        "chain": ["20260101121100_sanity_check", "20260101121100_feature_selector_variance"],
        "modules": {
            **get_sanity_check_module("20260101121100"),
            "20260101121100_feature_selector_variance": {
                "module": "feature_selector",
                "cache": True,
                "config": {
                    "method": "variance",
                    "threshold": 0.01,
                }
            }
        }
    })

    # E12: RFE k=20
    EXPERIMENTS.append({
        "exp_id": "E12",
        "timestamp": "20260101121200",
        "name": "rfe_20",
        "parent": "E00",
        "description": "Add: RFE feature selection (k=20)",
        "chain": ["20260101121200_sanity_check", "20260101121200_feature_selector_rfe20"],
        "modules": {
            **get_sanity_check_module("20260101121200"),
            "20260101121200_feature_selector_rfe20": {
                "module": "feature_selector",
                "cache": True,
                "config": {
                    "method": "rfe",
                    "k_features": 20,
                }
            }
        }
    })

    # E13: RFE k=15
    EXPERIMENTS.append({
        "exp_id": "E13",
        "timestamp": "20260101121300",
        "name": "rfe_15",
        "parent": "E00",
        "description": "Add: RFE feature selection (k=15)",
        "chain": ["20260101121300_sanity_check", "20260101121300_feature_selector_rfe15"],
        "modules": {
            **get_sanity_check_module("20260101121300"),
            "20260101121300_feature_selector_rfe15": {
                "module": "feature_selector",
                "cache": True,
                "config": {
                    "method": "rfe",
                    "k_features": 15,
                }
            }
        }
    })

    # E14: Feature interactions (specific pairs)
    EXPERIMENTS.append({
        "exp_id": "E14",
        "timestamp": "20260101121400",
        "name": "feat_interact",
        "parent": "E00",
        "description": "Feature interactions (study×attendance×sleep)",
        "chain": ["20260101121400_sanity_check", "20260101121400_feature_engineer_interact"],
        "modules": {
            **get_sanity_check_module("20260101121400"),
            "20260101121400_feature_engineer_interact": {
                "module": "feature_engineer",
                "cache": True,
                "config": {
                    "interaction_types": ["mul"],
                    "numeric_pairs": [
                        ["study_hours", "class_attendance"],
                        ["study_hours", "sleep_hours"],
                        ["class_attendance", "sleep_hours"],
                    ],
                }
            }
        }
    })

    # E15: Datetime handler (should be no-op, but test it)
    EXPERIMENTS.append({
        "exp_id": "E15",
        "timestamp": "20260101121500",
        "name": "datetime",
        "parent": "E00",
        "description": "Datetime handler (control test)",
        "chain": ["20260101121500_sanity_check", "20260101121500_datetime_handler"],
        "modules": {
            **get_sanity_check_module("20260101121500"),
            "20260101121500_datetime_handler": {
                "module": "datetime_handler",
                "cache": True,
                "config": {}
            }
        }
    })

    # E16: Outlier handler (IQR capping)
    EXPERIMENTS.append({
        "exp_id": "E16",
        "timestamp": "20260101121600",
        "name": "outlier_iqr",
        "parent": "E00",
        "description": "Outlier handler (IQR cap)",
        "chain": ["20260101121600_sanity_check", "20260101121600_outlier_handler"],
        "modules": {
            **get_sanity_check_module("20260101121600"),
            "20260101121600_outlier_handler": {
                "module": "outlier_handler",
                "cache": True,
                "config": {
                    "method": "iqr",
                    "action": "clip",  # Fixed: must be 'clip' not 'cap'
                    "iqr_multiplier": 1.5,
                }
            }
        }
    })

    # E17: Target encoding with smoothing
    EXPERIMENTS.append({
        "exp_id": "E17",
        "timestamp": "20260101121700",
        "name": "target_smooth",
        "parent": "E00",
        "description": "Target encoding with smoothing (min_samples=100)",
        "chain": ["20260101121700_sanity_check", "20260101121700_encoder_target_smooth"],
        "modules": {
            **get_sanity_check_module("20260101121700"),
            "20260101121700_encoder_target_smooth": {
                "module": "encoder",
                "cache": True,
                "config": {
                    "encoding_method": "target_mean",
                    "target_encoding_smoothing": 10.0,
                    "target_encoding_min_samples": 100,
                    "handle_unknown": "ignore",
                }
            }
        }
    })

    # E18: Custom features only (drop originals) - risky!
    EXPERIMENTS.append({
        "exp_id": "E18",
        "timestamp": "20260101121800",
        "name": "custom_only",
        "parent": "E00",
        "description": "Custom features only (drop originals)",
        "chain": ["20260101121800_sanity_check", "20260101121800_custom_features"],
        "modules": {
            **get_sanity_check_module("20260101121800"),
            "20260101121800_custom_features": {
                "module": "custom_features",
                "cache": True,
                "config": {
                    "drop_original": True  # Risky: keep only engineered features
                }
            }
        }
    })

    # E19: Polynomial degree=3 (feature explosion)
    EXPERIMENTS.append({
        "exp_id": "E19",
        "timestamp": "20260101121900",
        "name": "poly3",
        "parent": "E00",
        "description": "Polynomial degree=3",
        "chain": ["20260101121900_sanity_check", "20260101121900_feature_engineer_poly3"],
        "modules": {
            **get_sanity_check_module("20260101121900"),
            "20260101121900_feature_engineer_poly3": {
                "module": "feature_engineer",
                "cache": True,
                "config": {
                    "poly_degree": 3,
                    "poly_interaction_only": False,
                    "poly_include_bias": False,
                    "poly_columns": ["study_hours", "class_attendance", "sleep_hours"],
                }
            }
        }
    })

    # E20: CatBoost encoding (alternative to target encoding)
    EXPERIMENTS.append({
        "exp_id": "E20",
        "timestamp": "20260101122000",
        "name": "catboost_enc",
        "parent": "E00",
        "description": "CatBoost encoding",
        "chain": ["20260101122000_sanity_check", "20260101122000_encoder_catboost"],
        "modules": {
            **get_sanity_check_module("20260101122000"),
            "20260101122000_encoder_catboost": {
                "module": "encoder",
                "cache": True,
                "config": {
                    "encoding_method": "catboost",
                    "handle_unknown": "ignore",
                }
            }
        }
    })

    # E22: Chain A2 (E21 + log1p_target)
    # E21 = target_enc + custom_feat
    # E22 = target_enc + custom_feat + log1p_target
    EXPERIMENTS.append({
        "exp_id": "E22",
        "timestamp": "20260101122200",
        "name": "chain_a2",
        "parent": "E21",
        "description": "Chain A2: +log1p_target",
        "chain": [
            "20260101122200_sanity_check",
            "20260101122200_target_transformer_log1p",
            "20260101122200_encoder_target_oof",
            "20260101122200_custom_features",
        ],
        "modules": {
            **get_sanity_check_module("20260101122200"),
            "20260101122200_target_transformer_log1p": {
                "module": "target_transformer",
                "cache": True,
                "config": {"method": "log"}
            },
            "20260101122200_encoder_target_oof": {
                "module": "encoder",
                "cache": True,
                "config": {
                    "encoding_method": "target_mean",
                    "target_encoding_smoothing": 1.0,
                    "target_encoding_min_samples": 1,
                    "handle_unknown": "ignore",
                }
            },
            "20260101122200_custom_features": {
                "module": "custom_features",
                "cache": True,
                "config": {}
            }
        }
    })

    # E23: Chain A3 (E22 + rfe_20)
    EXPERIMENTS.append({
        "exp_id": "E23",
        "timestamp": "20260101122300",
        "name": "chain_a3",
        "parent": "E22",
        "description": "Chain A3: +rfe_20",
        "chain": [
            "20260101122300_sanity_check",
            "20260101122300_target_transformer_log1p",
            "20260101122300_encoder_target_oof",
            "20260101122300_custom_features",
            "20260101122300_feature_selector_rfe20",
        ],
        "modules": {
            **get_sanity_check_module("20260101122300"),
            "20260101122300_target_transformer_log1p": {
                "module": "target_transformer",
                "cache": True,
                "config": {"method": "log"}
            },
            "20260101122300_encoder_target_oof": {
                "module": "encoder",
                "cache": True,
                "config": {
                    "encoding_method": "target_mean",
                    "target_encoding_smoothing": 1.0,
                    "target_encoding_min_samples": 1,
                    "handle_unknown": "ignore",
                }
            },
            "20260101122300_custom_features": {
                "module": "custom_features",
                "cache": True,
                "config": {}
            },
            "20260101122300_feature_selector_rfe20": {
                "module": "feature_selector",
                "cache": True,
                "config": {
                    "method": "rfe",
                    "k_features": 20,
                }
            }
        }
    })

    # E24: Chain B1 (target_enc + poly2)
    EXPERIMENTS.append({
        "exp_id": "E24",
        "timestamp": "20260101122400",
        "name": "chain_b1",
        "parent": "E02",
        "description": "Chain B1: target_enc + poly2",
        "chain": [
            "20260101122400_sanity_check",
            "20260101122400_encoder_target_oof",
            "20260101122400_feature_engineer_poly2",
        ],
        "modules": {
            **get_sanity_check_module("20260101122400"),
            "20260101122400_encoder_target_oof": {
                "module": "encoder",
                "cache": True,
                "config": {
                    "encoding_method": "target_mean",
                    "target_encoding_smoothing": 1.0,
                    "target_encoding_min_samples": 1,
                    "handle_unknown": "ignore",
                }
            },
            "20260101122400_feature_engineer_poly2": {
                "module": "feature_engineer",
                "cache": True,
                "config": {
                    "poly_degree": 2,
                    "poly_interaction_only": False,
                    "poly_include_bias": False,
                    "poly_columns": ["study_hours", "class_attendance", "sleep_hours"],
                }
            }
        }
    })

    # E25: Chain B2 (E24 + feat_interact)
    EXPERIMENTS.append({
        "exp_id": "E25",
        "timestamp": "20260101122500",
        "name": "chain_b2",
        "parent": "E24",
        "description": "Chain B2: +feat_interact",
        "chain": [
            "20260101122500_sanity_check",
            "20260101122500_encoder_target_oof",
            "20260101122500_feature_engineer_poly2",
            "20260101122500_feature_engineer_interact",
        ],
        "modules": {
            **get_sanity_check_module("20260101122500"),
            "20260101122500_encoder_target_oof": {
                "module": "encoder",
                "cache": True,
                "config": {
                    "encoding_method": "target_mean",
                    "target_encoding_smoothing": 1.0,
                    "target_encoding_min_samples": 1,
                    "handle_unknown": "ignore",
                }
            },
            "20260101122500_feature_engineer_poly2": {
                "module": "feature_engineer",
                "cache": True,
                "config": {
                    "poly_degree": 2,
                    "poly_interaction_only": False,
                    "poly_include_bias": False,
                    "poly_columns": ["study_hours", "class_attendance", "sleep_hours"],
                }
            },
            "20260101122500_feature_engineer_interact": {
                "module": "feature_engineer",
                "cache": True,
                "config": {
                    "interaction_types": ["mul"],
                    "numeric_pairs": [
                        ["study_hours", "class_attendance"],
                        ["study_hours", "sleep_hours"],
                        ["class_attendance", "sleep_hours"],
                    ],
                }
            }
        }
    })

    # E26: Chain B3 (E25 + rfe_20)
    # Full chain: target_enc + poly2 + interact + rfe_20
    EXPERIMENTS.append({
        "exp_id": "E26",
        "timestamp": "20260101122600",
        "name": "chain_b3",
        "parent": "E25",
        "description": "Chain B3: +rfe_20",
        "chain": [
            "20260101122600_sanity_check",
            "20260101122600_encoder_target_oof",
            "20260101122600_feature_engineer_poly2",
            "20260101122600_feature_engineer_interact",
            "20260101122600_feature_selector_rfe20",
        ],
        "modules": {
            **get_sanity_check_module("20260101122600"),
            "20260101122600_encoder_target_oof": {
                "module": "encoder",
                "cache": True,
                "config": {
                    "encoding_method": "target_mean",
                    "target_encoding_smoothing": 1.0,
                    "target_encoding_min_samples": 1,
                    "handle_unknown": "ignore",
                }
            },
            "20260101122600_feature_engineer_poly2": {
                "module": "feature_engineer",
                "cache": True,
                "config": {
                    "poly_degree": 2,
                    "poly_interaction_only": False,
                    "poly_include_bias": False,
                    "poly_columns": ["study_hours", "class_attendance", "sleep_hours"],
                }
            },
            "20260101122600_feature_engineer_interact": {
                "module": "feature_engineer",
                "cache": True,
                "config": {
                    "interaction_types": ["mul"],
                    "numeric_pairs": [
                        ["study_hours", "class_attendance"],
                        ["study_hours", "sleep_hours"],
                        ["class_attendance", "sleep_hours"],
                    ],
                }
            },
            "20260101122600_feature_selector_rfe20": {
                "module": "feature_selector",
                "cache": True,
                "config": {
                    "method": "rfe",
                    "k_features": 20,
                }
            }
        }
    })

    # E27: Chain C1 (custom_feat + target_enc)
    EXPERIMENTS.append({
        "exp_id": "E27",
        "timestamp": "20260101122700",
        "name": "chain_c1",
        "parent": "E03",
        "description": "Chain C1: custom + target_enc",
        "chain": [
            "20260101122700_sanity_check",
            "20260101122700_custom_features",
            "20260101122700_encoder_target_oof",
        ],
        "modules": {
            **get_sanity_check_module("20260101122700"),
            "20260101122700_custom_features": {
                "module": "custom_features",
                "cache": True,
                "config": {}
            },
            "20260101122700_encoder_target_oof": {
                "module": "encoder",
                "cache": True,
                "config": {
                    "encoding_method": "target_mean",
                    "target_encoding_smoothing": 1.0,
                    "target_encoding_min_samples": 1,
                    "handle_unknown": "ignore",
                }
            }
        }
    })

    # E28: Chain C2 (E27 + sqrt_target)
    EXPERIMENTS.append({
        "exp_id": "E28",
        "timestamp": "20260101122800",
        "name": "chain_c2",
        "parent": "E27",
        "description": "Chain C2: +sqrt_target",
        "chain": [
            "20260101122800_sanity_check",
            "20260101122800_target_transformer_sqrt",
            "20260101122800_custom_features",
            "20260101122800_encoder_target_oof",
        ],
        "modules": {
            **get_sanity_check_module("20260101122800"),
            "20260101122800_target_transformer_sqrt": {
                "module": "target_transformer",
                "cache": True,
                "config": {"method": "sqrt"}
            },
            "20260101122800_custom_features": {
                "module": "custom_features",
                "cache": True,
                "config": {}
            },
            "20260101122800_encoder_target_oof": {
                "module": "encoder",
                "cache": True,
                "config": {
                    "encoding_method": "target_mean",
                    "target_encoding_smoothing": 1.0,
                    "target_encoding_min_samples": 1,
                    "handle_unknown": "ignore",
                }
            }
        }
    })

    # E29: Chain C3 (E28 + rare_001)
    EXPERIMENTS.append({
        "exp_id": "E29",
        "timestamp": "20260101122900",
        "name": "chain_c3",
        "parent": "E28",
        "description": "Chain C3: +rare_001",
        "chain": [
            "20260101122900_sanity_check",
            "20260101122900_target_transformer_sqrt",
            "20260101122900_custom_features",
            "20260101122900_encoder_target_oof",
            "20260101122900_rare_handler_001",
        ],
        "modules": {
            **get_sanity_check_module("20260101122900"),
            "20260101122900_target_transformer_sqrt": {
                "module": "target_transformer",
                "cache": True,
                "config": {"method": "sqrt"}
            },
            "20260101122900_custom_features": {
                "module": "custom_features",
                "cache": True,
                "config": {}
            },
            "20260101122900_encoder_target_oof": {
                "module": "encoder",
                "cache": True,
                "config": {
                    "encoding_method": "target_mean",
                    "target_encoding_smoothing": 1.0,
                    "target_encoding_min_samples": 1,
                    "handle_unknown": "ignore",
                }
            },
            "20260101122900_rare_handler_001": {
                "module": "rare_category_handler",
                "cache": True,
                "config": {
                    "min_freq": 0.01,
                    "replacement": "RARE",
                }
            }
        }
    })

    # E30: Chain D1 (log1p + target_enc)
    EXPERIMENTS.append({
        "exp_id": "E30",
        "timestamp": "20260101123000",
        "name": "chain_d1",
        "parent": "E07",
        "description": "Chain D1: log1p + target_enc",
        "chain": [
            "20260101123000_sanity_check",
            "20260101123000_target_transformer_log1p",
            "20260101123000_encoder_target_oof",
        ],
        "modules": {
            **get_sanity_check_module("20260101123000"),
            "20260101123000_target_transformer_log1p": {
                "module": "target_transformer",
                "cache": True,
                "config": {"method": "log"}
            },
            "20260101123000_encoder_target_oof": {
                "module": "encoder",
                "cache": True,
                "config": {
                    "encoding_method": "target_mean",
                    "target_encoding_smoothing": 1.0,
                    "target_encoding_min_samples": 1,
                    "handle_unknown": "ignore",
                }
            }
        }
    })

    # E31: Chain D2 (E30 + custom_feat)
    EXPERIMENTS.append({
        "exp_id": "E31",
        "timestamp": "20260101123100",
        "name": "chain_d2",
        "parent": "E30",
        "description": "Chain D2: +custom_feat",
        "chain": [
            "20260101123100_sanity_check",
            "20260101123100_target_transformer_log1p",
            "20260101123100_encoder_target_oof",
            "20260101123100_custom_features",
        ],
        "modules": {
            **get_sanity_check_module("20260101123100"),
            "20260101123100_target_transformer_log1p": {
                "module": "target_transformer",
                "cache": True,
                "config": {"method": "log"}
            },
            "20260101123100_encoder_target_oof": {
                "module": "encoder",
                "cache": True,
                "config": {
                    "encoding_method": "target_mean",
                    "target_encoding_smoothing": 1.0,
                    "target_encoding_min_samples": 1,
                    "handle_unknown": "ignore",
                }
            },
            "20260101123100_custom_features": {
                "module": "custom_features",
                "cache": True,
                "config": {}
            }
        }
    })

    # E32: Chain D3 (E31 + poly2)
    EXPERIMENTS.append({
        "exp_id": "E32",
        "timestamp": "20260101123200",
        "name": "chain_d3",
        "parent": "E31",
        "description": "Chain D3: +poly2",
        "chain": [
            "20260101123200_sanity_check",
            "20260101123200_target_transformer_log1p",
            "20260101123200_encoder_target_oof",
            "20260101123200_custom_features",
            "20260101123200_feature_engineer_poly2",
        ],
        "modules": {
            **get_sanity_check_module("20260101123200"),
            "20260101123200_target_transformer_log1p": {
                "module": "target_transformer",
                "cache": True,
                "config": {"method": "log"}
            },
            "20260101123200_encoder_target_oof": {
                "module": "encoder",
                "cache": True,
                "config": {
                    "encoding_method": "target_mean",
                    "target_encoding_smoothing": 1.0,
                    "target_encoding_min_samples": 1,
                    "handle_unknown": "ignore",
                }
            },
            "20260101123200_custom_features": {
                "module": "custom_features",
                "cache": True,
                "config": {}
            },
            "20260101123200_feature_engineer_poly2": {
                "module": "feature_engineer",
                "cache": True,
                "config": {
                    "poly_degree": 2,
                    "poly_interaction_only": False,
                    "poly_include_bias": False,
                    "poly_columns": ["study_hours", "class_attendance", "sleep_hours"],
                }
            }
        }
    })

    # E33: Chain D4 (E31 + rfe_15) - alternative to E32
    EXPERIMENTS.append({
        "exp_id": "E33",
        "timestamp": "20260101123300",
        "name": "chain_d4",
        "parent": "E31",
        "description": "Chain D4: +rfe_15 (vs poly)",
        "chain": [
            "20260101123300_sanity_check",
            "20260101123300_target_transformer_log1p",
            "20260101123300_encoder_target_oof",
            "20260101123300_custom_features",
            "20260101123300_feature_selector_rfe15",
        ],
        "modules": {
            **get_sanity_check_module("20260101123300"),
            "20260101123300_target_transformer_log1p": {
                "module": "target_transformer",
                "cache": True,
                "config": {"method": "log"}
            },
            "20260101123300_encoder_target_oof": {
                "module": "encoder",
                "cache": True,
                "config": {
                    "encoding_method": "target_mean",
                    "target_encoding_smoothing": 1.0,
                    "target_encoding_min_samples": 1,
                    "handle_unknown": "ignore",
                }
            },
            "20260101123300_custom_features": {
                "module": "custom_features",
                "cache": True,
                "config": {}
            },
            "20260101123300_feature_selector_rfe15": {
                "module": "feature_selector",
                "cache": True,
                "config": {
                    "method": "rfe",
                    "k_features": 15,
                }
            }
        }
    })

    # E34: Kitchen Sink (all best features)
    # target_enc + custom + poly2 + log1p + rfe_20
    EXPERIMENTS.append({
        "exp_id": "E34",
        "timestamp": "20260101123400",
        "name": "ultra",
        "parent": "E02",
        "description": "Kitchen sink: all features",
        "chain": [
            "20260101123400_sanity_check",
            "20260101123400_target_transformer_log1p",
            "20260101123400_encoder_target_oof",
            "20260101123400_custom_features",
            "20260101123400_feature_engineer_poly2",
            "20260101123400_feature_selector_rfe20",
        ],
        "modules": {
            **get_sanity_check_module("20260101123400"),
            "20260101123400_target_transformer_log1p": {
                "module": "target_transformer",
                "cache": True,
                "config": {"method": "log"}
            },
            "20260101123400_encoder_target_oof": {
                "module": "encoder",
                "cache": True,
                "config": {
                    "encoding_method": "target_mean",
                    "target_encoding_smoothing": 1.0,
                    "target_encoding_min_samples": 1,
                    "handle_unknown": "ignore",
                }
            },
            "20260101123400_custom_features": {
                "module": "custom_features",
                "cache": True,
                "config": {}
            },
            "20260101123400_feature_engineer_poly2": {
                "module": "feature_engineer",
                "cache": True,
                "config": {
                    "poly_degree": 2,
                    "poly_interaction_only": False,
                    "poly_include_bias": False,
                    "poly_columns": ["study_hours", "class_attendance", "sleep_hours"],
                }
            },
            "20260101123400_feature_selector_rfe20": {
                "module": "feature_selector",
                "cache": True,
                "config": {
                    "method": "rfe",
                    "k_features": 20,
                }
            }
        }
    })

    # E35: Ultra v2 (E34 + rare_0005)
    EXPERIMENTS.append({
        "exp_id": "E35",
        "timestamp": "20260101123500",
        "name": "ultra_v2",
        "parent": "E34",
        "description": "Ultra v2: +rare_0005",
        "chain": [
            "20260101123500_sanity_check",
            "20260101123500_rare_handler_0005",
            "20260101123500_target_transformer_log1p",
            "20260101123500_encoder_target_oof",
            "20260101123500_custom_features",
            "20260101123500_feature_engineer_poly2",
            "20260101123500_feature_selector_rfe20",
        ],
        "modules": {
            **get_sanity_check_module("20260101123500"),
            "20260101123500_rare_handler_0005": {
                "module": "rare_category_handler",
                "cache": True,
                "config": {
                    "min_freq": 0.005,
                    "replacement": "RARE",
                }
            },
            "20260101123500_target_transformer_log1p": {
                "module": "target_transformer",
                "cache": True,
                "config": {"method": "log"}
            },
            "20260101123500_encoder_target_oof": {
                "module": "encoder",
                "cache": True,
                "config": {
                    "encoding_method": "target_mean",
                    "target_encoding_smoothing": 1.0,
                    "target_encoding_min_samples": 1,
                    "handle_unknown": "ignore",
                }
            },
            "20260101123500_custom_features": {
                "module": "custom_features",
                "cache": True,
                "config": {}
            },
            "20260101123500_feature_engineer_poly2": {
                "module": "feature_engineer",
                "cache": True,
                "config": {
                    "poly_degree": 2,
                    "poly_interaction_only": False,
                    "poly_include_bias": False,
                    "poly_columns": ["study_hours", "class_attendance", "sleep_hours"],
                }
            },
            "20260101123500_feature_selector_rfe20": {
                "module": "feature_selector",
                "cache": True,
                "config": {
                    "method": "rfe",
                    "k_features": 20,
                }
            }
        }
    })

# Generate remaining experiments
add_remaining_experiments()

# Verify we have 36 experiments
assert len(EXPERIMENTS) == 36, f"Expected 36 experiments, got {len(EXPERIMENTS)}"
print(f"✓ Loaded {len(EXPERIMENTS)} experiment specifications")
