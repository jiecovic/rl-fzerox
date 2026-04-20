# src/rl_fzerox/core/config/schema_models/training.py
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from rl_fzerox.core.domain.training_algorithms import (
    DEFAULT_TRAIN_ALGORITHM,
    TRAIN_ALGORITHM_SAC,
    TrainAlgorithmName,
)


class TrainConfig(BaseModel):
    """Training settings for the current run."""

    model_config = ConfigDict(extra="forbid")

    algorithm: TrainAlgorithmName = DEFAULT_TRAIN_ALGORITHM
    vec_env: Literal["dummy", "subproc"] = "dummy"
    num_envs: PositiveInt = 1
    total_timesteps: PositiveInt = 1_000_000
    n_steps: PositiveInt = 1_024
    n_epochs: PositiveInt = 10
    batch_size: PositiveInt = 256
    learning_rate: PositiveFloat = 3e-4
    gamma: float = Field(default=0.99, gt=0.0, le=1.0)
    gae_lambda: float = Field(default=0.95, gt=0.0, le=1.0)
    clip_range: PositiveFloat = 0.2
    ent_coef: NonNegativeFloat | Literal["auto"] = 0.01
    vf_coef: PositiveFloat = 0.5
    max_grad_norm: PositiveFloat = 0.5
    buffer_size: PositiveInt = 1_000_000
    learning_starts: NonNegativeInt = 100
    tau: PositiveFloat = Field(default=0.005, le=1.0)
    train_freq: PositiveInt = 1
    gradient_steps: PositiveInt = 1
    target_update_interval: PositiveInt = 1
    target_entropy: float | Literal["auto"] = "auto"
    optimize_memory_usage: bool = False
    verbose: int = Field(default=0, ge=0, le=2)
    device: str = "auto"
    save_freq: PositiveInt = 1_000
    output_root: Path = Path("local/runs")
    run_name: str = "ppo_cnn"
    init_run_dir: Path | None = None
    init_artifact: Literal["latest", "best", "final"] = "latest"

    @model_validator(mode="after")
    def _validate_algorithm_specific_values(self) -> TrainConfig:
        if self.ent_coef == "auto" and self.algorithm != TRAIN_ALGORITHM_SAC:
            raise ValueError("train.ent_coef=auto is only supported with train.algorithm=sac")
        return self
