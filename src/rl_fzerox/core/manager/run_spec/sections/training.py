# src/rl_fzerox/core/manager/run_spec/sections/training.py
"""Training-optimizer section of the manager-owned run-spec model."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

DEFAULT_ENTROPY_COEFFICIENT = 0.01
ACTION_ENTROPY_GROUP_KEYS = (
    "steer",
    "drive",
    "gas",
    "air_brake",
    "boost",
    "lean",
    "lean_left",
    "lean_right",
    "spin",
    "pitch",
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
    ent_coef: NonNegativeFloat = Field(default=DEFAULT_ENTROPY_COEFFICIENT, exclude=True)
    vf_coef: PositiveFloat = 0.5
    max_grad_norm: PositiveFloat = 0.5
    normalize_advantage: bool = True
    target_kl: PositiveFloat | None = None
    entropy_group_weights: dict[str, NonNegativeFloat] = Field(default_factory=dict, exclude=True)
    entropy_coefficients: dict[str, NonNegativeFloat] = Field(
        default_factory=lambda: default_entropy_coefficients()
    )
    actor_regularization: ManagedTrainActorRegularizationConfig = Field(
        default_factory=ManagedTrainActorRegularizationConfig
    )
    stats_window_size: PositiveInt = 100
    checkpoint_every_rollouts: PositiveInt = 5
    save_latest_checkpoint: bool = True
    save_best_checkpoint: bool = True
    save_recent_checkpoints: bool = False
    recent_checkpoint_limit: PositiveInt | None = 5

    @model_validator(mode="before")
    @classmethod
    def _normalize_entropy_coefficients(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        if "entropy_coefficients" not in data:
            ent_coef = _non_negative_float(
                data.get("ent_coef"),
                default=DEFAULT_ENTROPY_COEFFICIENT,
            )
            raw_weights = data.get("entropy_group_weights")
            weights = raw_weights if isinstance(raw_weights, Mapping) else {}
            data["entropy_coefficients"] = {
                key: ent_coef * _non_negative_float(weights.get(key), default=1.0)
                for key in ACTION_ENTROPY_GROUP_KEYS
            }
        return data


def default_entropy_coefficients() -> dict[str, float]:
    return {key: DEFAULT_ENTROPY_COEFFICIENT for key in ACTION_ENTROPY_GROUP_KEYS}


def _non_negative_float(value: object, *, default: float) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool) and value >= 0.0:
        return float(value)
    return default
