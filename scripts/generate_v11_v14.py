import yaml
import os
from pathlib import Path

PROJECT = "playground-series-s6e1"
BASE_NAME = "test_c_01_0306"
ROOT = Path(f"projects/kaggle/{PROJECT}")

# Definicje nowych wariantów MI
new_variants = {
    "v11": {"selection_method": "mi", "n_features": 0.3},
    "v12": {"selection_method": "mi", "n_features": 0.4},
    "v13": {"selection_method": "mi", "n_features": 0.6},
    "v14": {"selection_method": "mi", "n_features": 0.7},
}

def generate():
    # Load base model template
    with open(ROOT / f"templates/model/{BASE_NAME}.yaml", "r") as f:
        base_model = yaml.safe_load(f)

    for v_id, config in new_variants.items():
        # A. Create Selector Template
        selector_name = f"{BASE_NAME}-feature_selector_{v_id}"
        selector_path = ROOT / f"templates/preprocess/{selector_name}.yaml"
        selector_payload = {
            "module": "feature_selector",
            "cache": True,
            "config": config
        }
        with open(selector_path, "w") as f:
            yaml.dump(selector_payload, f, sort_keys=False)

        # B. Create Preprocess Chain Template
        chain_name = f"{BASE_NAME}_{v_id}"
        chain_path = ROOT / f"templates/preprocess/{chain_name}.yaml"
        chain_payload = {
            "chain": [
                "mcts",
                f"{BASE_NAME}-rare_category_handler",
                f"{BASE_NAME}-categorical_encoder",
                f"{BASE_NAME}-outlier_handler",
                f"{BASE_NAME}-scaler",
                f"{BASE_NAME}-feature_engineer",
                selector_name
            ]
        }
        with open(chain_path, "w") as f:
            yaml.dump(chain_payload, f, sort_keys=False)

        # C. Create Model Template
        model_name = f"{BASE_NAME}_{v_id}"
        model_path = ROOT / f"templates/model/{model_name}.yaml"
        model_payload = base_model.copy()
        model_payload["preprocess_template"] = chain_name
        with open(model_path, "w") as f:
            yaml.dump(model_payload, f, sort_keys=False)

        print(f"Generated {v_id}: model={model_name}, chain={chain_name}")

if __name__ == "__main__":
    generate()
