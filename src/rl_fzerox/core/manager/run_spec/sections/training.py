# src/rl_fzerox/core/manager/run_spec/sections/training.py
"""Training-optimizer section of the manager-owned run-spec model."""

from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
)


class ManagedTrainConfig(BaseModel):
    """Training knobs exposed by the first run-manager slice."""

    model_config = ConfigDict(extra="forbid")

    num_envs: PositiveInt = 10
    total_timesteps: PositiveInt = 50_000_000
    n_steps: PositiveInt = 2_048
    n_epochs: PositiveInt = 3
    batch_size: PositiveInt = 1_024
    learning_rate: PositiveFloat = 8.5e-5
    gamma: float = Field(default=0.995, gt=0.0, le=1.0)
    gae_lambda: float = Field(default=0.95, gt=0.0, le=1.0)
    clip_range: PositiveFloat = 0.19
    clip_range_vf: PositiveFloat | None = None
    ent_coef: NonNegativeFloat = 0.01
    vf_coef: PositiveFloat = 0.5
    max_grad_norm: PositiveFloat = 0.5
    normalize_advantage: bool = True
    target_kl: PositiveFloat | None = None
    stats_window_size: PositiveInt = 100
    checkpoint_every_rollouts: PositiveInt = 5
    save_latest_checkpoint: bool = True
    save_best_checkpoint: bool = True
    save_recent_checkpoints: bool = False
    recent_checkpoint_limit: PositiveInt | None = 5
