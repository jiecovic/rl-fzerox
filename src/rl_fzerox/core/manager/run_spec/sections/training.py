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


class ManagedTrainActorRegularizationConfig(BaseModel):
    """Optional policy-side actor regularization exposed by the run manager."""

    model_config = ConfigDict(extra="forbid")

    grounded_pitch_neutral_loss_weight: NonNegativeFloat = 0.0
    pitch_std_cap_loss_weight: NonNegativeFloat = 0.0
    grounded_pitch_std_cap: PositiveFloat = 0.35
    airborne_pitch_std_cap: PositiveFloat = 0.8
    steer_std_cap_loss_weight: NonNegativeFloat = 0.0
    steer_std_cap: PositiveFloat = 1.0
    steer_signed_balance_loss_weight: NonNegativeFloat = 0.0
    steer_signed_balance_deadzone: float = Field(default=0.2, ge=0.0, le=1.0)
    lean_signed_balance_loss_weight: NonNegativeFloat = 0.0
    lean_signed_balance_deadzone: float = Field(default=0.1, ge=0.0, le=1.0)


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
    entropy_group_weights: dict[str, NonNegativeFloat] = Field(default_factory=dict)
    actor_regularization: ManagedTrainActorRegularizationConfig = Field(
        default_factory=ManagedTrainActorRegularizationConfig
    )
    stats_window_size: PositiveInt = 100
    checkpoint_every_rollouts: PositiveInt = 5
    save_latest_checkpoint: bool = True
    save_best_checkpoint: bool = True
    save_recent_checkpoints: bool = False
    recent_checkpoint_limit: PositiveInt | None = 5
