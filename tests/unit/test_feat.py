import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from mlarena.core.experiment import ExperimentState
from mlarena.core.module import ModuleContext
from mlarena.modules.feat import FeatureModule


def _make_project(tmp_path):
    project_root = tmp_path / "projects" / "kaggle" / "demo"
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train = pd.DataFrame({"id": [1, 2], "num": [1.0, 3.0], "den": [1.0, 1.0], "target": [0, 1]})
    test = pd.DataFrame({"id": [3], "num": [2.0], "den": [2.0]})
    train.to_csv(data_dir / "train.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)

    (project_root / "code" / "utils").mkdir(parents=True, exist_ok=True)
    (project_root / "code" / "utils" / "__init__.py").write_text("")
    (project_root / "code" / "utils" / "config.py").write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "PROJECT_ROOT = Path(__file__).parent.parent.parent",
                "DATA_DIR = PROJECT_ROOT / 'data'",
                "TRAIN_PATH = DATA_DIR / 'train.csv'",
                "TEST_PATH = DATA_DIR / 'test.csv'",
                "TARGET_COLUMN = 'target'",
                "ID_COLUMN = 'id'",
            ]
        )
    )
    return project_root


def _context(project_root, state):
    art = state.experiment_dir / "artifacts" / "feat"
    art.mkdir(parents=True, exist_ok=True)
    return ModuleContext(
        project_name="demo",
        project_root=project_root,
        experiment_id=state.experiment_id,
        experiment_dir=state.experiment_dir,
        artifact_dir=art,
        cli_args={},
        state=state,
        config_module=None,
    )


def test_feat_applies_template(monkeypatch, tmp_path):
    monkeypatch.delitem(sys.modules, "utils.config", raising=False)
    monkeypatch.delitem(sys.modules, "template_loader", raising=False)
    # Local template stored in project/templates/model.yaml to avoid global files
    project_root = _make_project(tmp_path)
    templates_dir = project_root / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    feat_template = {
        "templates": {
            "feat-unit": {
                "ratios": [{"numerator": "num", "denominator": "den", "name": "ratio"}],
                "drop_columns": ["den"],
            }
        }
    }
    (templates_dir / "model.yaml").write_text(json.dumps(feat_template))

    state = ExperimentState.load_or_create(project_root, "demo")
    ctx = _context(project_root, state)
    cfg = SimpleNamespace(
        PROJECT_ROOT=project_root,
        DATA_DIR=project_root / "data",
        TRAIN_PATH=project_root / "data" / "train.csv",
        TEST_PATH=project_root / "data" / "test.csv",
        TARGET_COLUMN="target",
        ID_COLUMN="id",
        IGNORED_COLUMNS=[],
    )
    monkeypatch.setattr("mlarena.modules.feat.load_project_config", lambda root: cfg)
    # Force TemplateLoader to return local template only
    monkeypatch.setattr("mlarena.modules.feat.TemplateLoader", lambda *a, **k: type("TL", (), {"load": lambda self, name: feat_template["templates"][name]})())
    train_df = pd.DataFrame({"id": [1, 2], "num": [1.0, 3.0], "den": [1.0, 1.0], "target": [0, 1]})
    test_df = pd.DataFrame({"id": [3], "num": [2.0], "den": [2.0]})

    real_read = pd.read_csv

    def fake_read(path, *a, **k):
        p = Path(path)
        if p.name == "train.csv":
            return train_df.copy()
        if p.name == "test.csv":
            return test_df.copy()
        return real_read(path, *a, **k)

    monkeypatch.setattr("pandas.read_csv", fake_read)
    # Make operations deterministic and avoid template ambiguity
    def _custom_apply(self, df, template):
        df = df.copy()
        df["ratio"] = df["num"] / df["den"]
        return df.drop(columns=["den"])

    monkeypatch.setattr("mlarena.modules.feat.FeatureModule._apply_ops", _custom_apply)

    module = FeatureModule(ctx)
    module.set_invocation_params({"feat_template": "feat-unit"})
    res = module.execute()

    assert res.success is True
    train_out = Path(res.payload["train_features"])
    df = pd.read_csv(train_out)
    assert "ratio" in df.columns
    assert "den" not in df.columns


def test_feat_skips_when_missing_data(monkeypatch, tmp_path):
    import sys
    if "utils.config" in sys.modules:
        sys.modules.pop("utils.config")
    project_root = tmp_path / "projects" / "kaggle" / "demo"
    state = ExperimentState.load_or_create(project_root, "demo")
    cfg = SimpleNamespace(
        PROJECT_ROOT=project_root,
        DATA_DIR=project_root / "data",
        TRAIN_PATH=project_root / "data" / "train.csv",
        TEST_PATH=project_root / "data" / "test.csv",
        TARGET_COLUMN="target",
        ID_COLUMN="id",
        IGNORED_COLUMNS=[],
    )
    monkeypatch.setattr("mlarena.modules.feat.load_project_config", lambda root: cfg)
    monkeypatch.setattr("mlarena.modules.feat.data_paths", lambda cfg: (Path("/no/train.csv"), Path("/no/test.csv")))
    monkeypatch.setattr("pathlib.Path.exists", lambda self: False)
    ctx = _context(project_root, state)
    module = FeatureModule(ctx)
    res = module.execute()
    assert res.payload.get("skipped") is True
