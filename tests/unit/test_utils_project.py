import importlib
from pathlib import Path
import sys
import types
import importlib.util

from mlarena.utils.project import load_project_config, data_paths


def _write_config(project_root: Path, body: str) -> None:
    code_dir = project_root / "code" / "utils"
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / "__init__.py").write_text("")
    (code_dir / "config.py").write_text(body)


def test_load_project_config_from_module(tmp_path, monkeypatch):
    project_root = tmp_path / "proj"
    project_root.mkdir(parents=True, exist_ok=True)
    train_path = project_root / "data" / "train.csv"
    test_path = project_root / "data" / "test.csv"
    _write_config(
        project_root,
        "\n".join(
            [
                "from pathlib import Path",
                f"PROJECT_ROOT = Path('{project_root}')",
                f"TRAIN_PATH = Path('{train_path}')",
                f"TEST_PATH = Path('{test_path}')",
                "TARGET_COLUMN = 'survived'",
                "ID_COLUMN = 'passenger_id'",
                "AUTOGLUON_PROBLEM_TYPE = 'binary'",
                "AUTOGLUON_EVAL_METRIC = 'roc_auc'",
                "IGNORED_COLUMNS = ['ticket']",
            ]
        ),
    )

    monkeypatch.setattr(sys, "path", [str(project_root / "code")])
    monkeypatch.setattr("mlarena.utils.project.importlib.import_module", importlib.import_module)
    sys.modules.pop("utils.config", None)
    sys.modules.pop("utils", None)
    importlib.invalidate_caches()

    # Pre-load config module explicitly to avoid interference from other paths
    spec = importlib.util.spec_from_file_location("utils.config", project_root / "code" / "utils" / "config.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["utils"] = types.ModuleType("utils")
    sys.modules["utils.config"] = module
    spec.loader.exec_module(module)

    cfg = load_project_config(project_root)
    assert cfg.TARGET_COLUMN == "survived"
    assert cfg.ID_COLUMN == "passenger_id"
    assert cfg.AUTOGLUON_PROBLEM_TYPE == "binary"
    train, test = data_paths(cfg)
    assert train == train_path
    assert test == test_path


def test_load_project_config_fallback(tmp_path, monkeypatch):
    project_root = tmp_path / "proj"
    project_root.mkdir(parents=True, exist_ok=True)

    # Force fallback by making import fail
    def _fail_import(name):
        raise ModuleNotFoundError()

    monkeypatch.setattr("mlarena.utils.project.importlib.import_module", lambda name: _fail_import(name))

    cfg = load_project_config(project_root)
    assert cfg.TARGET_COLUMN == "target"
    assert cfg.ID_COLUMN == "id"
    train, test = data_paths(cfg)
    assert train.name == "train.csv"
    assert test.name == "test.csv"
    assert train.parent == project_root / "data"


def test_data_paths_default_to_data_dir(tmp_path):
    cfg = types.SimpleNamespace(DATA_DIR=tmp_path, TRAIN_PATH=None, TEST_PATH=None)
    train, test = data_paths(cfg)
    assert train == tmp_path / "train.csv"
    assert test == tmp_path / "test.csv"
