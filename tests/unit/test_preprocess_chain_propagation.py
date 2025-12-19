import json
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import pytest
from mlarena.core.experiment import ExperimentState
from mlarena.core.module import ModuleContext
from mlarena.modules.preprocess import PreprocessModule

# Helper to create a dummy project structure
def _make_project(tmp_path):
    project_root = tmp_path / "projects" / "kaggle" / "demo"
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy data
    train = pd.DataFrame({"id": [1, 2], "target": [0, 1], "col1": [10, 20]})
    test = pd.DataFrame({"id": [3, 4], "col1": [30, 40]})
    
    train.to_csv(data_dir / "train.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)

    # Create config files
    (project_root / "code" / "utils").mkdir(parents=True, exist_ok=True)
    (project_root / "code" / "utils" / "__init__.py").write_text("")
    (project_root / "code" / "utils" / "config.py").write_text(
        "\n".join([
            "from pathlib import Path",
            "PROJECT_ROOT = Path(__file__).parent.parent.parent",
            "DATA_DIR = PROJECT_ROOT / 'data'",
            "TRAIN_PATH = DATA_DIR / 'train.csv'",
            "TEST_PATH = DATA_DIR / 'test.csv'",
            "TARGET_COLUMN = 'target'",
            "ID_COLUMN = 'id'",
            "IGNORED_COLUMNS = []",
            "AUTOGLUON_PROBLEM_TYPE = 'binary'",
            "AUTOGLUON_EVAL_METRIC = 'roc_auc'",
        ])
    )
    return project_root

# Helper to mock template loading
def _patch_templates(monkeypatch, template_dict):
    class StubLoader:
        def __init__(self, *a, **k): pass
        def load(self, name): return template_dict.get(name, {})

    monkeypatch.setattr("mlarena.modules.preprocess.TemplateLoader", StubLoader)
    
    # Patch load_templates used in can_run and execute via sys.modules injection
    # This works because preprocess.py does 'from template_loader import load_templates'
    def mock_load_templates(*args, **kwargs):
        return template_dict, []
    
    mock_module = SimpleNamespace(load_templates=mock_load_templates)
    monkeypatch.setitem(__import__("sys").modules, "template_loader", mock_module)

# Mock config loading
def _patch_config(monkeypatch, project_root):
    cfg = SimpleNamespace(
        PROJECT_ROOT=project_root,
        DATA_DIR=project_root / "data",
        TRAIN_PATH=project_root / "data" / "train.csv",
        TEST_PATH=project_root / "data" / "test.csv",
        TARGET_COLUMN="target",
        ID_COLUMN="id",
        IGNORED_COLUMNS=[],
        AUTOGLUON_PROBLEM_TYPE="binary",
        AUTOGLUON_EVAL_METRIC="roc_auc",
    )
    monkeypatch.setattr("mlarena.modules.preprocess.load_project_config", lambda root: cfg)
    monkeypatch.setattr("mlarena.modules.preprocess.data_paths", lambda config: (config.TRAIN_PATH, config.TEST_PATH))

def test_chain_propagates_custom_state(monkeypatch, tmp_path):
    """
    Verify that custom_module_state (e.g. weights_path) generated in Step 1
    is propagated to the payload of Step 2 in a preprocessing chain.
    """
    # Setup
    project_root = _make_project(tmp_path)
    _patch_config(monkeypatch, project_root)
    
    # Define templates
    templates = {
        "step1_gen_weights": {
            "module": "mock_weight_gen", # We will mock this module
        },
        "step2_consumer": {
            "drop_columns": [], # Basic template, no custom module
        }
    }
    _patch_templates(monkeypatch, templates)

    # Mock the custom module that generates weights
    class MockWeightGen:
        def fit_transform(self, train_df, val_df, test_df, config, orig_df=None):
            # Simulate generating weights
            weights_path = str(Path(config["_artifact_dir"]) / "weights.csv")
            # Return 5-tuple: train, val, test, orig, state
            state = {"weights_path": weights_path, "some_metric": 0.95}
            return train_df, val_df, test_df, orig_df, state

    # Patch the dynamic module loader to return our mock
    def mock_load_preprocessing_module(self, module_name):
        if module_name == "mock_weight_gen":
            return MockWeightGen()
        raise ValueError(f"Unknown module {module_name}")

    monkeypatch.setattr(PreprocessModule, "_load_preprocessing_module", mock_load_preprocessing_module)

    # --- EXECUTE STEP 1 ---
    chain_id = "pre-chain-test"
    step1_id = "0-step1_gen_weights"
    
    # Manually create the experiment state/dir for step 1
    state1 = ExperimentState.load_or_create(
        project_root, "demo", experiment_id=f"{chain_id}/{step1_id}"
    )
    
    ctx1 = ModuleContext(
        project_name="demo",
        project_root=project_root,
        experiment_id=state1.experiment_id,
        experiment_dir=state1.experiment_dir,
        artifact_dir=state1.experiment_dir / "artifacts" / "preprocess",
        cli_args={},
        state=state1,
        config_module=None,
    )
    
    module1 = PreprocessModule(ctx1)
    module1.set_invocation_params({
        "preprocess_template": "step1_gen_weights",
        "chain_exp_id": chain_id,
        "input_source": None # First step
    })
    
    result1 = module1.execute()
    
    assert result1.success
    assert "custom_module_state" in result1.payload
    assert result1.payload["custom_module_state"]["some_metric"] == 0.95
    
    # Save state to disk so Step 2 can read it (simulating pipeline behavior)
    state1.complete_module("preprocess", result1.payload)
    state1.save()

    # --- EXECUTE STEP 2 ---
    step2_id = "1-step2_consumer"
    
    state2 = ExperimentState.load_or_create(
        project_root, "demo", experiment_id=f"{chain_id}/{step2_id}"
    )
    
    ctx2 = ModuleContext(
        project_name="demo",
        project_root=project_root,
        experiment_id=state2.experiment_id,
        experiment_dir=state2.experiment_dir,
        artifact_dir=state2.experiment_dir / "artifacts" / "preprocess",
        cli_args={},
        state=state2,
        config_module=None,
    )
    
    module2 = PreprocessModule(ctx2)
    module2.set_invocation_params({
        "preprocess_template": "step2_consumer",
        "chain_exp_id": chain_id,
        "input_source": step1_id # Points to Step 1
    })
    
    result2 = module2.execute()
    
    assert result2.success
    
    # --- VERIFY PROPAGATION ---
    # The payload of Step 2 should contain the custom state from Step 1
    assert "custom_module_state" in result2.payload
    custom_state = result2.payload["custom_module_state"]
    
    # Check if weights_path propagated
    assert "weights_path" in custom_state
    assert "some_metric" in custom_state
    assert custom_state["some_metric"] == 0.95
    
    # Verify the path points to step1's artifact dir (approximately, checking substring)
    assert "0-step1_gen_weights" in custom_state["weights_path"]
