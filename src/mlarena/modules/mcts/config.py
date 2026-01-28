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


class EvaluationConfig(BaseModel):
    model: Optional[str] = None


class OracleConfig(BaseModel):
    enabled: bool = False
    model_path: Optional[str] = None
    dry_run: bool = True
    pruning_threshold: float = 0.20
    eps: float = 0.0
    max_actions: int = (
        0  # 0 = disabled (use threshold), >0 = take top K (ignore threshold)
    )
    use_prior_in_puct: bool = False


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

    # Group-specific weights for tree models (XGB/LGBM/CatBoost)
    # Monotonic transformations (scaler, rank_features) are invariant for tree models
    group_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "scaler": 0.2,  # Very low priority - mostly invariant for tree models
            "rank_features": 0.2,  # Very low priority - mostly invariant for tree models
        }
    )

    # Progressive Widening
    expansion_width: float = 2.0  # k
    expansion_alpha: float = 0.5  # alpha
    seed: int = 42
    lookahead: int = 3  # Limit valid next steps to immediate N neighbors
    randomize_order: bool = False  # Enable random topological sort for searched steps

    # 2nd-layer PW: number of parameter samples per (operator=step+variant) at a node
    # m_params(op) = max(1, floor(param_expansion_width * N_op^param_expansion_alpha))
    param_expansion_width: float = 1.0
    param_expansion_alpha: float = 0.5
    # hard cap to avoid blow-up
    param_expansion_max_samples: int = 16

    # Execution
    root_mode: Literal["no_preprocess", "harness_only"] = "harness_only"
    executor: Literal["cli", "task_queue"] = "cli"
    json_output: bool = True
    debug: bool = True
    # Failure penalty used during backprop (None => auto-scale from baseline)
    fail_penalty: Optional[float] = None

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
    oracle: OracleConfig = Field(default_factory=OracleConfig)
    penalties: PenaltiesConfig = Field(default_factory=PenaltiesConfig)
    templates: TemplatesConfig = Field(default_factory=TemplatesConfig)
    parallelism: ParallelismConfig = Field(default_factory=ParallelismConfig)
    dedupe: DedupeConfig = Field(default_factory=DedupeConfig)
    evaluation: Optional[EvaluationConfig] = None

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

    # If evaluation is not in mcts section, try to get it from root level
    # This allows evaluation config to be shared between MCTS, RANT, and regular tuning
    if "evaluation" not in mcts_data and "evaluation" in data:
        mcts_data["evaluation"] = data["evaluation"]

    return MCTSConfig(**mcts_data)
