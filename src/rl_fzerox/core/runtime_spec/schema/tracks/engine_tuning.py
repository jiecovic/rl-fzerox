# src/rl_fzerox/core/runtime_spec/schema/tracks/engine_tuning.py
"""Adaptive engine-tuning runtime schema.

This config controls reset-time engine-slider sampling for a selected training
target. The bandit backend is the production path; Gaussian-process and
MLP-ensemble fields remain schema-supported experimental backends and are
serialized only when their backend is active.
"""

from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from rl_fzerox.core.domain.engine import ENGINE_SLIDER
from rl_fzerox.core.engine_tuning.types import (
    ENGINE_TUNER_DEFAULTS,
    EngineTunerBackend,
    EngineTunerObjective,
    engine_bucket_candidates,
)


class AdaptiveEngineTuningConfig(BaseModel):
    """Reset-time adaptive engine-setting sampler configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    min_raw_value: NonNegativeInt = Field(default=ENGINE_SLIDER.min_step, le=ENGINE_SLIDER.max_step)
    max_raw_value: NonNegativeInt = Field(
        default=ENGINE_SLIDER.max_step,
        le=ENGINE_SLIDER.max_step,
    )
    backend: EngineTunerBackend = ENGINE_TUNER_DEFAULTS.backend
    objective: EngineTunerObjective = ENGINE_TUNER_DEFAULTS.objective
    reward_fingerprint: str | None = None
    bucket_raw_values: tuple[NonNegativeInt, ...] = ENGINE_TUNER_DEFAULTS.bandit_bucket_raw_values
    safe_finish_rate_threshold: float = Field(
        default=ENGINE_TUNER_DEFAULTS.safe_finish_rate_threshold,
        ge=0.0,
        le=1.0,
    )
    min_finish_rate_observations: NonNegativeInt = (
        ENGINE_TUNER_DEFAULTS.min_finish_rate_observations
    )
    stat_decay: float = Field(default=ENGINE_TUNER_DEFAULTS.stat_decay, gt=0.0, lt=1.0)
    prior_finish_time_seconds: PositiveFloat = ENGINE_TUNER_DEFAULTS.prior_finish_time_seconds
    exploration_scale: NonNegativeFloat = ENGINE_TUNER_DEFAULTS.exploration_seconds
    ensemble_members: PositiveInt = ENGINE_TUNER_DEFAULTS.mlp_ensemble_members
    randomized_prior_seconds: NonNegativeFloat = ENGINE_TUNER_DEFAULTS.mlp_randomized_prior_seconds
    hidden_dim: PositiveInt = ENGINE_TUNER_DEFAULTS.mlp_hidden_dim
    training_steps: PositiveInt = ENGINE_TUNER_DEFAULTS.mlp_training_steps
    learning_rate: PositiveFloat = ENGINE_TUNER_DEFAULTS.mlp_learning_rate
    bootstrap_keep_probability: float = Field(
        default=ENGINE_TUNER_DEFAULTS.mlp_bootstrap_keep_probability,
        gt=0.0,
        le=1.0,
    )
    warmup_successes: PositiveInt = ENGINE_TUNER_DEFAULTS.mlp_warmup_successes
    observation_noise_seconds: NonNegativeFloat = ENGINE_TUNER_DEFAULTS.observation_noise_seconds
    curve_lengthscale_raw: PositiveFloat = ENGINE_TUNER_DEFAULTS.curve_lengthscale_raw
    uniform_exploration: float = Field(
        default=ENGINE_TUNER_DEFAULTS.uniform_exploration,
        ge=0.0,
        le=1.0,
    )
    greedy_plateau_tolerance_seconds: NonNegativeFloat = (
        ENGINE_TUNER_DEFAULTS.greedy_plateau_tolerance_seconds
    )

    @model_serializer(mode="wrap")
    def _serialize_engine_tuning(self, handler: SerializerFunctionWrapHandler) -> object:
        data = handler(self)
        if isinstance(data, dict) and self.backend != "bandit":
            data.pop("objective", None)
            data.pop("reward_fingerprint", None)
            data.pop("bucket_raw_values", None)
            data.pop("safe_finish_rate_threshold", None)
            data.pop("min_finish_rate_observations", None)
        if isinstance(data, dict) and self.backend == "bandit":
            data.pop("greedy_plateau_tolerance_seconds", None)
        if isinstance(data, dict) and self.backend != "gaussian_process":
            data.pop("stat_decay", None)
            data.pop("observation_noise_seconds", None)
            data.pop("curve_lengthscale_raw", None)
        if isinstance(data, dict) and self.backend not in {"bandit", "gaussian_process"}:
            data.pop("exploration_scale", None)
        if isinstance(data, dict) and self.backend != "mlp_ensemble":
            data.pop("ensemble_members", None)
            data.pop("randomized_prior_seconds", None)
            data.pop("hidden_dim", None)
            data.pop("training_steps", None)
            data.pop("learning_rate", None)
            data.pop("bootstrap_keep_probability", None)
            data.pop("warmup_successes", None)
        return data

    @model_validator(mode="after")
    def _validate_engine_range(self) -> AdaptiveEngineTuningConfig:
        if self.min_raw_value > self.max_raw_value:
            raise ValueError("engine_tuning.min_raw_value must be <= max_raw_value")
        if self.enabled and self.backend == "bandit":
            engine_bucket_candidates(bucket_raw_values=self.bucket_raw_values)
        return self
