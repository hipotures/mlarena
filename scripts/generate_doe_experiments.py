"""
Generate 36 DOE experiment templates for playground-series-s6e1.

This script creates:
- 36 model templates (in templates/model/)
- 36 chain templates (in templates/preprocess/)
- ~100-150 module templates (in templates/preprocess/)

Based on the approved DOE plan for single-change experiments.
"""

from pathlib import Path
import yaml
import sys

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from doe_experiments_spec import EXPERIMENTS

TEMPLATES_DIR = PROJECT_ROOT / "projects" / "kaggle" / "playground-series-s6e1" / "templates"
MODEL_DIR = TEMPLATES_DIR / "model"
PREPROCESS_DIR = TEMPLATES_DIR / "preprocess"

# Create directories if they don't exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PREPROCESS_DIR.mkdir(parents=True, exist_ok=True)

# EXPERIMENTS is now imported from doe_experiments_spec.py

def generate_model_template(exp):
    """Generate model template content."""
    timestamp = exp["timestamp"]
    name = exp["name"]

    return {
        "model": "autogluon_baseline",
        "preprocess_template": f"{timestamp}_{name}",
        "config": {
            "preset": "best",
            "time_limit": 3600,
            "use_gpu": False,
            "included_model_types": ["GBM", "XGB", "CAT"],
            "random_state": 42,
            "eval_metric": "root_mean_squared_error",
            "hyperparameters": {
                "hyperparameter_tune_kwargs": {
                    "num_trials": 50,
                    "scheduler": "local",
                    "searcher": "auto",
                },
                "search_space": {
                    "GBM": {
                        "learning_rate": [0.01, 0.3, "log"],
                        "num_leaves": [20, 150, "int"],
                        "min_data_in_leaf": [10, 50, "int"],
                        "feature_fraction": [0.6, 1.0],
                        "bagging_fraction": [0.6, 1.0],
                        "bagging_freq": [1, 7, "int"],
                        "lambda_l1": [1e-5, 10.0, "log"],
                        "lambda_l2": [1e-5, 10.0, "log"],
                        "num_boost_round": [500, 3000, "int"],
                    },
                    "XGB": {
                        "learning_rate": [0.01, 0.3, "log"],
                        "max_depth": [3, 8, "int"],
                        "min_child_weight": [1, 7, "int"],
                        "subsample": [0.6, 1.0],
                        "colsample_bytree": [0.6, 1.0],
                        "gamma": [0.0, 2.0],
                        "reg_alpha": [1e-5, 10.0, "log"],
                        "reg_lambda": [1e-5, 10.0, "log"],
                        "n_estimators": [500, 3000, "int"],
                    },
                    "CAT": {
                        "learning_rate": [0.01, 0.3, "log"],
                        "depth": [4, 10, "int"],
                        "l2_leaf_reg": [1.0, 30.0],
                        "bagging_temperature": [0.0, 1.5],
                        "random_strength": [0.0001, 10.0, "log"],
                        "border_count": [32, 255, "int"],
                        "iterations": [500, 3000, "int"],
                    },
                },
            },
        }
    }

def generate_chain_template(exp):
    """Generate chain template content."""
    return {
        "chain": exp["chain"]
    }

def write_template(filepath, content):
    """Write template to YAML file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        yaml.dump(content, f, default_flow_style=False, sort_keys=False)
    print(f"✓ Created: {filepath.relative_to(PROJECT_ROOT)}")

def main():
    """Generate all experiment templates."""
    print("=" * 80)
    print("Generating DOE Experiment Templates for playground-series-s6e1")
    print("=" * 80)
    print()

    total_files = 0

    for exp in EXPERIMENTS:
        exp_id = exp["exp_id"]
        timestamp = exp["timestamp"]
        name = exp["name"]

        print(f"[{exp_id}] Generating templates for {name}...")

        # 1. Generate model template
        model_template = generate_model_template(exp)
        model_file = MODEL_DIR / f"{timestamp}_{name}.yaml"
        write_template(model_file, model_template)
        total_files += 1

        # 2. Generate chain template
        chain_template = generate_chain_template(exp)
        chain_file = PREPROCESS_DIR / f"{timestamp}_{name}.yaml"
        write_template(chain_file, chain_template)
        total_files += 1

        # 3. Generate module templates
        for module_name, module_content in exp["modules"].items():
            module_file = PREPROCESS_DIR / f"{module_name}.yaml"
            write_template(module_file, module_content)
            total_files += 1

        print()

    print("=" * 80)
    print(f"✓ Successfully generated {total_files} template files!")
    print(f"  - Model templates: {MODEL_DIR}")
    print(f"  - Preprocess templates: {PREPROCESS_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
