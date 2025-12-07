import json
import pytest
import yaml

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


def test_template_loader_reads_yaml(project_root):
    """Test that TemplateLoader correctly reads YAML templates."""
    tpl_dir = project_root / "templates"
    tpl_dir.mkdir(parents=True, exist_ok=True)

    # Write YAML template following the actual format
    tpl_data = {
        "templates": {
            "unit-test": {
                "preset": "high",
                "time_limit": 5
            }
        }
    }
    (tpl_dir / "model.yaml").write_text(yaml.dump(tpl_data))

    loader = TemplateLoader(project_root, template_type="model")
    loaded = loader.load("unit-test")
    assert loaded["preset"] == "high"
    assert loaded["time_limit"] == 5


def test_template_loader_returns_empty_when_missing(project_root):
    loader = TemplateLoader(project_root)
    assert loader.load("does-not-exist") == {}
