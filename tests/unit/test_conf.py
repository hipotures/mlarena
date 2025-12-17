import pytest
from pathlib import Path
from mlarena.core.conf import ConfigBuilder, GlobalConfig
from omegaconf import OmegaConf

def test_config_builder_basic(tmp_path):
    repo_root = tmp_path
    project_name = "test_project"
    project_root = repo_root / "projects" / "kaggle" / project_name
    project_root.mkdir(parents=True)
    
    builder = ConfigBuilder(project_name, repo_root)
    config = builder.build([])
    
    assert config.project == project_name
    assert config.common.seed == 42  # default value

def test_config_builder_overrides(tmp_path):
    repo_root = tmp_path
    project_name = "test_project"
    project_root = repo_root / "projects" / "kaggle" / project_name
    project_root.mkdir(parents=True)
    
    builder = ConfigBuilder(project_name, repo_root)
    
    # Test flat override
    config = builder.build(["common.seed=123"])
    assert config.common.seed == 123
    
    # Test nested module override
    config = builder.build(["model.time_limit=100"])
    assert config.model["time_limit"] == 100

def test_config_builder_profile_merge(tmp_path):
    repo_root = tmp_path
    project_name = "test_project"
    project_root = repo_root / "projects" / "kaggle" / project_name
    project_root.mkdir(parents=True)
    
    # Create a global profile
    profiles_dir = repo_root / "src" / "mlarena" / "templates" / "profiles"
    profiles_dir.mkdir(parents=True)
    (profiles_dir / "test_prof.yaml").write_text("common:\n  time_limit: 60\n  preset: fast")
    
    builder = ConfigBuilder(project_name, repo_root)
    
    # Load profile and check fallback
    config = builder.build(["profile=test_prof"])
    assert config.common.time_limit == 60
    assert config.common.preset == "fast"
    
    # Profile + CLI override (CLI should win)
    config = builder.build(["profile=test_prof", "common.time_limit=120"])
    assert config.common.time_limit == 120

def test_config_builder_project_yaml(tmp_path):
    repo_root = tmp_path
    project_name = "test_project"
    project_root = repo_root / "projects" / "kaggle" / project_name
    project_root.mkdir(parents=True)
    
    # Create project-specific config.yaml
    (project_root / "config.yaml").write_text("common:\n  seed: 999\nmodel:\n  time_limit: 500")
    
    builder = ConfigBuilder(project_name, repo_root)
    config = builder.build([])
    
    assert config.common.seed == 999
    assert config.model["time_limit"] == 500
    
    # Project YAML + CLI override
    config = builder.build(["common.seed=111"])
    assert config.common.seed == 111

def test_config_builder_conflict_order(tmp_path):
    """Test the priority order: base < profile < project yaml < CLI."""
    repo_root = tmp_path
    project_name = "test_project"
    project_root = repo_root / "projects" / "kaggle" / project_name
    project_root.mkdir(parents=True)
    
    # 1. Global Profile
    profiles_dir = repo_root / "src" / "mlarena" / "templates" / "profiles"
    profiles_dir.mkdir(parents=True)
    (profiles_dir / "smoke.yaml").write_text("common:\n  time_limit: 60")
    
    # 2. Project YAML
    (project_root / "config.yaml").write_text("common:\n  time_limit: 120")
    
    builder = ConfigBuilder(project_name, repo_root)
    
    # Profile vs Project YAML (Project wins)
    config = builder.build(["profile=smoke"])
    assert config.common.time_limit == 120
    
    # CLI vs everything (CLI wins)
    config = builder.build(["profile=smoke", "common.time_limit=300"])
    assert config.common.time_limit == 300
