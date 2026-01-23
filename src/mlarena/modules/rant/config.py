from __future__ import annotations
from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator
import yaml


class RefinementConfig(BaseModel):
    """Configuration for parameter refinement (neighborhood generation)."""
    grid_size: int = 20
    log_grid_size: int = 20
    allow_two_param_mutations: bool = False


class RuntimeConfig(BaseModel):
    """Configuration for runtime behavior (driver/worker mode)."""
    executor: Literal["cli", "task_queue"] = "cli"
    role: Literal["driver", "worker", "driver_worker"] = "driver_worker"
    poll_interval_sec: float = 2.0
    max_idle_polls: int = 60
    storage_timeout_sec: int = 60
    kill_on_interrupt: bool = True


class FailureConfig(BaseModel):
    """Configuration for failure handling."""
    max_sampling_attempts_per_insert: int = 200
    max_total_sampling_attempts_per_round: int = 50000
    max_consecutive_runtime_failures: int = 20
    stale_running_sec: int = 7200  # 2 hours
    stale_running_policy: Literal["requeue", "fail"] = "requeue"
    max_retries_per_trial: int = 1


class RANTConfig(BaseModel):
    """Main RANT configuration.

    RANT = Random AND Targeted Search
    Hybrid hyperparameter optimizer combining:
    - Random exploration (N trials per round)
    - Global top-K selection (from entire study history)
    - Local refinement (P children per parent, parameters only)
    """

    # Storage & Identity
    storage_url: str = "sqlite:///experiments/db/rant.db"
    study_name: Optional[str] = None
    direction: Literal["minimize", "maximize"] = "maximize"

    # Search Budget & Logic
    initial_random_trials: int = 50  # N: random trials per round
    top_k: int = 10  # K: number of parents selected from global history
    children_per_parent: int = 5  # P: refinement children per parent
    rounds_per_run: int = 1  # Number of rounds to execute per run

    # Pipeline Generation
    min_core_len: int = 3
    max_core_len: int = 9
    seed: int = 42

    # Feasibility
    allow_heavy_steps: bool = True
    allow_heavy_variants: bool = True

    # Diversity Control
    max_parents_per_architecture: int = 1  # Max parents with same architecture_signature

    # Sub-configs
    refinement: RefinementConfig = Field(default_factory=RefinementConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    failure: FailureConfig = Field(default_factory=FailureConfig)

    # Debugging
    debug: bool = False

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, v):
        if v not in ("minimize", "maximize"):
            raise ValueError("direction must be 'minimize' or 'maximize'")
        return v

    @field_validator("initial_random_trials", "top_k", "children_per_parent")
    @classmethod
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError("Must be positive")
        return v

    @field_validator("min_core_len", "max_core_len")
    @classmethod
    def validate_core_len(cls, v):
        if v < 1:
            raise ValueError("Core length must be at least 1")
        return v

    @field_validator("max_core_len")
    @classmethod
    def validate_max_core_len(cls, v, info):
        # info.data contains previously validated fields
        min_len = info.data.get("min_core_len")
        if min_len is not None and v < min_len:
            raise ValueError(f"max_core_len ({v}) must be >= min_core_len ({min_len})")
        return v


def load_rant_config(path: Path | str) -> RANTConfig:
    """Load RANT configuration from YAML file.

    Args:
        path: Path to YAML config file (typically mla_super_chain.yaml)

    Returns:
        RANTConfig: Validated configuration object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    data = yaml.safe_load(path.read_text()) or {}
    rant_data = data.get("rant", {})

    return RANTConfig(**rant_data)
