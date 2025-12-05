import json
import pytest

from mlarena.core.config import TemplateLoader, load_pipeline_def


def test_pipeline_warnings_for_unknown_module(project_root):
    cfg_dir = project_root / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "pipeline_default.json").write_text('{"name": "default", "modules": ["model", "unknown_mod"]}')

    _, warnings = load_pipeline_def("default", project_root=project_root)
    assert any("unknown_mod" in w for w in warnings)


def test_pipeline_validation_errors(project_root):
    cfg_dir = project_root / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "pipeline_bad.json").write_text('["bad"]')

    with pytest.raises(ValueError):
        load_pipeline_def("bad", project_root=project_root)


def test_template_loader_reads_json(project_root):
    cfg_dir = project_root / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    tpl = {"preset": "high", "time_limit": 5}
    (cfg_dir / "unit.json").write_text(json.dumps(tpl))

    loader = TemplateLoader(project_root)
    loaded = loader.load("unit")
    assert loaded["preset"] == "high"
    assert loaded["time_limit"] == 5


def test_template_loader_returns_empty_when_missing(project_root):
    loader = TemplateLoader(project_root)
    assert loader.load("does-not-exist") == {}
