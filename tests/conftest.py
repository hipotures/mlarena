import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

# Ensure local src/ is importable without installation
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mlarena.core.experiment import ExperimentState
from mlarena.core.module import ModuleContext
from mlarena.core.conf import GlobalConfig


@pytest.fixture(autouse=True)
def _isolate_tests(monkeypatch):
    """Ensure complete test isolation - clean registry, cwd, and modules."""
    import os
    from mlarena.core.registry import ModuleRegistry

    # Store original state
    original_cwd = os.getcwd()

    # Clean module registry before each test
    ModuleRegistry.clear()

    # Clear cached project modules
    for name in list(sys.modules.keys()):
        if name.startswith("utils.config") or name == "template_loader":
            sys.modules.pop(name, None)

    # Clean DummyPredictor store
    DummyPredictor.store.clear()

    yield

    # Reset after test
    try:
        os.chdir(original_cwd)
    except:
        pass  # Ignore if directory was deleted

    # Clean up again
    ModuleRegistry.clear()


class DummyPredictor:
    """Lightweight stand-in for AutoGluon TabularPredictor used in tests."""

    store = {}

    def __init__(
        self,
        label=None,
        problem_type=None,
        eval_metric=None,
        path=None,
        sample_weight=None,
        **kwargs,
    ):
        self.label = label
        self.problem_type = problem_type
        self.eval_metric = eval_metric
        self.sample_weight = sample_weight
        self.extra_init_kwargs = kwargs
        self.path = Path(path) if path else Path(".")
        self.model_best = "best"
        DummyPredictor.store[str(self.path)] = self

    def fit(self, df, presets=None, time_limit=None, ag_args_fit=None, hyperparameters=None, **kwargs):
        self.df = df.copy()
        self.presets = presets
        self.time_limit = time_limit
        self.ag_args_fit = ag_args_fit
        self.hyperparameters = hyperparameters
        self.extra_fit_kwargs = kwargs
        return self

    def leaderboard(self, silent=True):  # pragma: no cover - trivial
        return pd.DataFrame([{"model": "best", "score": 1.0}])

    def info(self):  # pragma: no cover - trivial
        return {"best_model": "best"}

    def predict(self, df):  # pragma: no cover - trivial
        return pd.Series([0] * len(df))

    def predict_proba(self, df, as_multiclass=False):  # pragma: no cover - trivial
        return pd.Series([0.5] * len(df))

    def model_info(self, model):  # pragma: no cover - trivial
        return {"val_score": 1.0, "name": model}

    @classmethod
    def load(cls, path):  # pragma: no cover - trivial
        key = str(Path(path))
        if key in cls.store:
            return cls.store[key]
        inst = cls(path=path)
        cls.store[key] = inst
        return inst


@pytest.fixture()
def mock_autogluon(monkeypatch):
    ag_tabular = SimpleNamespace(TabularPredictor=DummyPredictor)
    space_mod = SimpleNamespace(
        Real=lambda *a, **k: ("real", a, k),
        Int=lambda *a, **k: ("int", a, k),
        Categorical=lambda *a, **k: ("cat", a, k),
    )
    ag_common = SimpleNamespace(space=space_mod)
    ag = SimpleNamespace(tabular=ag_tabular, common=ag_common)
    monkeypatch.setitem(sys.modules, "autogluon", ag)
    monkeypatch.setitem(sys.modules, "autogluon.tabular", ag_tabular)
    monkeypatch.setitem(sys.modules, "autogluon.common", ag_common)
    monkeypatch.setitem(sys.modules, "autogluon.common.space", space_mod)
    return DummyPredictor


@pytest.fixture()
def project_root(tmp_path):
    root = tmp_path / "projects" / "kaggle" / "demo"
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture()
def experiment_state(project_root):
    return ExperimentState.load_or_create(project_root, "demo")


@pytest.fixture()
def context_factory(project_root):
    def _factory(module_name: str = "module", state: ExperimentState | None = None, config_module=None, config=None):
        st = state or ExperimentState.load_or_create(project_root, "demo")
        artifact_dir = st.experiment_dir / "artifacts" / module_name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        conf = config or GlobalConfig(project="demo")
        return ModuleContext(
            project_name="demo",
            project_root=project_root,
            experiment_id=st.experiment_id,
            experiment_dir=st.experiment_dir,
            artifact_dir=artifact_dir,
            cli_args={},
            state=st,
            config_module=config_module,
            config=conf,
        )

    return _factory


@pytest.fixture()
def sample_config_with_data(project_root):
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train = pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4],
            "feature": [0.1, 0.2, 0.3, 0.4],
            "target": [0, 1, 0, 1],
        }
    )
    test = pd.DataFrame({"PassengerId": [10, 11], "feature": [0.15, 0.35]})
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    return SimpleNamespace(
        PROJECT_ROOT=project_root,
        DATA_DIR=data_dir,
        TRAIN_PATH=train_path,
        TEST_PATH=test_path,
        TARGET_COLUMN="target",
        ID_COLUMN="PassengerId",
        AUTOGLUON_PROBLEM_TYPE="binary",
        AUTOGLUON_EVAL_METRIC="roc_auc",
        AUTOGLUON_PRESET="medium",
        AUTOGLUON_TIME_LIMIT=10,
        SUBMISSION_PROBAS=True,
        COMPETITION_NAME="demo-comp",
        IGNORED_COLUMNS=[],
    )
