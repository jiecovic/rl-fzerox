# src/rl_fzerox/core/config/schema_models/training.py
from __future__ import annotations

from collections.abc import Mapping
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
    TRAINING_ALGORITHMS,
    TrainAlgorithmName,
)

ResumeArtifact = Literal["latest", "best", "final"]
ResumeMode = Literal["weights_only", "full_model"]


class TrainConfig(BaseModel):
    """Training settings for the current run."""

    model_config = ConfigDict(extra="forbid")

    algorithm: TrainAlgorithmName = TRAINING_ALGORITHMS.default
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
    resume_run_dir: Path | None = None
    resume_artifact: ResumeArtifact = "latest"
    resume_mode: ResumeMode = "weights_only"

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_resume_fields(cls, data: object) -> object:
        # V4 LEGACY SHIM: old configs used init_* for weights-only warm starts.
        # Keep the translation isolated so fresh manifests only persist resume_*.
        if not isinstance(data, Mapping):
            return data

        missing = object()
        values = {str(key): value for key, value in data.items()}
        legacy_run_dir = values.pop("init_run_dir", missing)
        legacy_artifact = values.pop("init_artifact", missing)
        if legacy_run_dir is not missing and "resume_run_dir" not in values:
            values["resume_run_dir"] = legacy_run_dir
        if legacy_artifact is not missing and "resume_artifact" not in values:
            values["resume_artifact"] = legacy_artifact
        return values

    @model_validator(mode="after")
    def _validate_algorithm_specific_values(self) -> TrainConfig:
        if self.ent_coef == "auto" and self.algorithm not in TRAINING_ALGORITHMS.sac_family:
            raise ValueError("train.ent_coef=auto is only supported with SAC-family algorithms")
        if self.resume_run_dir is None and self.resume_mode == "full_model":
            raise ValueError("train.resume_mode=full_model requires train.resume_run_dir")
        return self
