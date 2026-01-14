"""
Unified configuration system for MLArena.

Combines OmegaConf for hierarchical merging and CLI overrides with 
Pydantic for type validation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import os

from omegaconf import OmegaConf, DictConfig
from pydantic import BaseModel, Field, ConfigDict

class CommonConfig(BaseModel):
    """Shared parameters used as fallbacks by modules."""
    seed: int = 42
    time_limit: Optional[int] = None
    use_gpu: Optional[bool] = None
    preset: Optional[str] = None

class GlobalConfig(BaseModel):
    """The root configuration tree."""
    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        populate_by_name=True,
    )

    # Core metadata
    project: str
    experiment_id: Optional[str] = None
    profile: Optional[str] = None
    
    # Auto-flow & Global settings
    model_template: str = "baseline"
    preprocess_template: Optional[str] = None
    wait_seconds: int = 30
    skip_submit: bool = False
    skip_git: bool = False
    
    # Execution control
    force: bool = False
    json_output: bool = False
    mcts_live: bool = False
    lock: bool = False  # Create overwrite.lock after successful completion
    skip_deps: bool = False
    show_payload: bool = False
    
    # Sections
    common: CommonConfig = Field(default_factory=CommonConfig)
    
    # Modules - dynamic sections
    init: Dict[str, Any] = Field(default_factory=dict)
    eda: Dict[str, Any] = Field(default_factory=dict)
    preprocess: Dict[str, Any] = Field(default_factory=dict)
    preprocess_tune: Dict[str, Any] = Field(default_factory=dict, alias="preprocess-tune")
    feat: Dict[str, Any] = Field(default_factory=dict)
    tune: Dict[str, Any] = Field(default_factory=dict)
    model: Dict[str, Any] = Field(default_factory=dict)
    predict: Dict[str, Any] = Field(default_factory=dict)
    stack: Dict[str, Any] = Field(default_factory=dict)
    submit: Dict[str, Any] = Field(default_factory=dict)
    fetch_score: Dict[str, Any] = Field(default_factory=dict, alias="fetch-score")

    # Internal runtime state (not usually set via CLI)
    project_root: Optional[Path] = None


class ConfigBuilder:
    """Handles the hierarchical merging of configuration layers."""

    def __init__(self, project_name: str, repo_root: Path):
        self.project_name = project_name
        self.repo_root = repo_root
        self.project_root = repo_root / "projects" / "kaggle" / project_name
        
    def _get_profile_path(self, profile_name: str) -> Path:
        """Resolve profile name to a YAML path."""
        # Try project-local profile first, then global
        local_path = self.project_root / "templates" / "profiles" / f"{profile_name}.yaml"
        if local_path.exists():
            return local_path
            
        global_path = self.repo_root / "src" / "mlarena" / "templates" / "profiles" / f"{profile_name}.yaml"
        return global_path

    def build(self, cli_overrides: List[str]) -> GlobalConfig:
        """
        Build the final configuration by merging layers:
        1. Base Defaults (Structured Config)
        2. Profile (if requested in CLI via 'profile=name')
        3. Project config (projects/kaggle/name/config.yaml)
        4. CLI Overrides (e.g., 'model.time_limit=300')
        """
        # 1. Start with base dict from Pydantic defaults
        base_conf = OmegaConf.create(GlobalConfig(project=self.project_name).model_dump())
        
        # 2. Parse CLI first to check for 'profile' or other early-decision flags
        # We don't merge it yet, just peek
        cli_conf = OmegaConf.from_cli(cli_overrides)
        
        layers = [base_conf]
        
        # 3. Load profile if specified
        profile_name = cli_conf.get("profile")
        if profile_name:
            profile_path = self._get_profile_path(profile_name)
            if profile_path.exists():
                layers.append(OmegaConf.load(profile_path))
            else:
                # Fallback for built-in smoke/dev if files don't exist yet
                if profile_name == "smoke":
                    layers.append(OmegaConf.create({"common": {"time_limit": 60, "preset": "medium", "use_gpu": False}}))
                elif profile_name == "dev":
                    layers.append(OmegaConf.create({"common": {"time_limit": 300, "preset": "high", "use_gpu": False}}))

        # 4. Load project-specific config.yaml
        project_yaml = self.project_root / "config.yaml"
        if project_yaml.exists():
            layers.append(OmegaConf.load(project_yaml))
            
        # 5. Add CLI overrides last (highest priority)
        layers.append(cli_conf)
        
        # 6. Merge all layers
        merged_conf = OmegaConf.merge(*layers)
        
        # 6b. Enforce quiet flags if json_output is true
        if merged_conf.get("json_output"):
            # Ensure module sections exist
            for section in ["preprocess", "model"]:
                if section not in merged_conf:
                    merged_conf[section] = {}
            
            merged_conf.preprocess.quiet_preprocess_panel = True
            merged_conf.model.quiet_model_panel = True
            merged_conf.model.verbosity = 0
            
            # Also common parameters
            if "common" not in merged_conf:
                merged_conf.common = {}
            # Some models might use verbosity from common
            merged_conf.common.verbosity = 0

        # 7. Convert to Pydantic for validation
        container = OmegaConf.to_container(merged_conf, resolve=True)
        config = GlobalConfig(**container)
        
        # Set runtime state
        config.project_root = self.project_root
        
        return config

def get_config(project: str, overrides: List[str] = None) -> GlobalConfig:
    """Convenience helper to get config for a project."""
    from mlarena.cli.main import REPO_ROOT
    builder = ConfigBuilder(project, REPO_ROOT)
    return builder.build(overrides or [])
