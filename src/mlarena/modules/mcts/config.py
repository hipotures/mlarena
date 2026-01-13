from __future__ import annotations
from pathlib import Path
from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
import yaml

class MultiFidelityConfig(BaseModel):
    enable: bool = True
    levels: List[Dict[str, Any]] = Field(default_factory=list)
    promotion: Dict[str, Any] = Field(default_factory=dict)

class PruningConfig(BaseModel):
    enable: bool = True
    incumbent_margin: float = 0.0

class PenaltiesConfig(BaseModel):
    features_lambda: float = 0.0
    time_lambda: float = 0.0

class TemplatesConfig(BaseModel):
    retention: Literal["best", "top_k", "all"] = "best"
    retain_top_k: int = 20
    retain_fidelities: List[str] = Field(default_factory=lambda: ["F2"])
    retain_failures: bool = True
    ephemeral_fidelities: List[str] = Field(default_factory=lambda: ["F0", "F1"])

class DedupeConfig(BaseModel):
    enable: bool = True
    strategy: str = "unique_signature"

class ParallelismConfig(BaseModel):
    workers: int = 1
    virtual_loss: float = 1.0

class MCTSConfig(BaseModel):
    # Storage & Identity
    storage_url: str = "sqlite:///experiments/db/mcts.db"
    study_name: Optional[str] = None
    direction: Literal["minimize", "maximize"] = "maximize"
    resume_policy: Literal["strict", "force"] = "strict"
    stale_running_trials: Literal["fail", "requeue"] = "fail"

    # Search Budget & Logic
    budget: int = 80
    max_depth: int = 9
    selection_policy: Literal["uct", "puct"] = "puct"
    exploration_weight: float = 1.414
    prior_policy: Literal["uniform", "heuristic", "surrogate"] = "uniform"

    # Progressive Widening
    expansion_width: float = 2.0  # k
    expansion_alpha: float = 0.5 # alpha
    seed: int = 42

    # Execution
    root_mode: Literal["no_preprocess", "harness_only"] = "harness_only"
    executor: Literal["cli", "task_queue"] = "cli"
    json_output: bool = True
    debug: bool = True

    # Execution Details
    model_verbosity: int = 2
    model_cleanup: bool = True
    cleanup_processed: bool = False

    # Feasibility
    allow_heavy_steps: bool = True
    allow_heavy_variants: bool = True

    # Sub-configs
    multi_fidelity: MultiFidelityConfig = Field(default_factory=MultiFidelityConfig)
    pruning: PruningConfig = Field(default_factory=PruningConfig)
    penalties: PenaltiesConfig = Field(default_factory=PenaltiesConfig)
    templates: TemplatesConfig = Field(default_factory=TemplatesConfig)
    parallelism: ParallelismConfig = Field(default_factory=ParallelismConfig)
    dedupe: DedupeConfig = Field(default_factory=DedupeConfig)

    @field_validator("direction")
    def validate_direction(cls, v):
        if v not in ("minimize", "maximize"):
            raise ValueError("direction must be 'minimize' or 'maximize'")
        return v

def load_mcts_config(path: Path) -> MCTSConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    data = yaml.safe_load(path.read_text()) or {}
    mcts_data = data.get("mcts", {})
    
    # If study_name is not in the yaml, we might need to handle it or expect it to be passed/injected
    # For now, let's assume validation will fail if it's missing, unless we provide a default
    # But study_name is mandatory in the model.
    
    return MCTSConfig(**mcts_data)
