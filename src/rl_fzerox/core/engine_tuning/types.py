# src/rl_fzerox/core/engine_tuning/types.py
"""Shared engine-tuning types and score helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from rl_fzerox.core.domain.engine_setting import (
    ENGINE_SLIDER,
    centered_engine_slider_buckets,
    engine_percent_to_slider_step,
    engine_slider_steps,
    validate_engine_slider_bucket_values,
)
from rl_fzerox.core.domain.engine_tuning import EngineTunerBackend, EngineTunerObjective


@dataclass(frozen=True, slots=True)
class EngineTunerBucketDefaults:
    """Default explicit bucket list generated for the bandit tuner."""

    side_count: int = 5
    min_raw_value: int = ENGINE_SLIDER.min_step
    max_raw_value: int = ENGINE_SLIDER.max_step

    @property
    def raw_values(self) -> tuple[int, ...]:
        return centered_engine_slider_buckets(
            minimum=self.min_raw_value,
            maximum=self.max_raw_value,
            side_count=self.side_count,
        )


@dataclass(frozen=True, slots=True)
class EngineTunerDefaults:
    """Default scale values for adaptive engine tuning."""

    backend: EngineTunerBackend = "bandit"
    objective: EngineTunerObjective = "finish_time"
    bandit_buckets: EngineTunerBucketDefaults = field(default_factory=EngineTunerBucketDefaults)
    safe_finish_rate_threshold: float = 0.9
    stat_decay: float = 0.995
    prior_finish_time_seconds: float = 200.0
    exploration_seconds: float = 30.0
    mlp_ensemble_members: int = 5
    mlp_randomized_prior_seconds: float = 30.0
    mlp_hidden_dim: int = 32
    mlp_training_steps: int = 48
    mlp_learning_rate: float = 0.004
    mlp_bootstrap_keep_probability: float = 0.8
    mlp_warmup_successes: int = 32
    observation_noise_seconds: float = 1.5
    curve_lengthscale_raw: float = float(engine_percent_to_slider_step(12))
    uniform_exploration: float = 0.05
    greedy_plateau_tolerance_seconds: float = 1.0

    @property
    def bandit_bucket_raw_values(self) -> tuple[int, ...]:
        """Return the default centered bandit bucket list."""

        return self.bandit_buckets.raw_values


ENGINE_TUNER_DEFAULTS = EngineTunerDefaults()


@dataclass(frozen=True, slots=True)
class EngineTunerCommonSettings:
    """Backend-independent knobs for one adaptive engine-tuning run."""

    min_raw_value: int = ENGINE_SLIDER.min_step
    max_raw_value: int = ENGINE_SLIDER.max_step
    prior_finish_time_seconds: float = ENGINE_TUNER_DEFAULTS.prior_finish_time_seconds
    uniform_exploration: float = ENGINE_TUNER_DEFAULTS.uniform_exploration


@dataclass(frozen=True, slots=True)
class BanditEngineTunerSettings(EngineTunerCommonSettings):
    """Static knobs used by the aggregate bandit backend."""

    backend: Literal["bandit"] = "bandit"
    objective: EngineTunerObjective = ENGINE_TUNER_DEFAULTS.objective
    reward_fingerprint: str | None = None
    bucket_raw_values: tuple[int, ...] = ENGINE_TUNER_DEFAULTS.bandit_bucket_raw_values
    exploration_seconds: float = ENGINE_TUNER_DEFAULTS.exploration_seconds
    safe_finish_rate_threshold: float = ENGINE_TUNER_DEFAULTS.safe_finish_rate_threshold


@dataclass(frozen=True, slots=True)
class GaussianProcessEngineTunerSettings(EngineTunerCommonSettings):
    """Static knobs used only by the experimental Gaussian-process backend."""

    backend: Literal["gaussian_process"] = "gaussian_process"
    greedy_plateau_tolerance_seconds: float = ENGINE_TUNER_DEFAULTS.greedy_plateau_tolerance_seconds
    stat_decay: float = ENGINE_TUNER_DEFAULTS.stat_decay
    exploration_seconds: float = ENGINE_TUNER_DEFAULTS.exploration_seconds
    observation_noise_seconds: float = ENGINE_TUNER_DEFAULTS.observation_noise_seconds
    curve_lengthscale_raw: float = ENGINE_TUNER_DEFAULTS.curve_lengthscale_raw


@dataclass(frozen=True, slots=True)
class MlpEnsembleEngineTunerSettings(EngineTunerCommonSettings):
    """Static knobs used only by the experimental MLP ensemble backend."""

    backend: Literal["mlp_ensemble"] = "mlp_ensemble"
    greedy_plateau_tolerance_seconds: float = ENGINE_TUNER_DEFAULTS.greedy_plateau_tolerance_seconds
    ensemble_members: int = ENGINE_TUNER_DEFAULTS.mlp_ensemble_members
    randomized_prior_seconds: float = ENGINE_TUNER_DEFAULTS.mlp_randomized_prior_seconds
    hidden_dim: int = ENGINE_TUNER_DEFAULTS.mlp_hidden_dim
    training_steps: int = ENGINE_TUNER_DEFAULTS.mlp_training_steps
    learning_rate: float = ENGINE_TUNER_DEFAULTS.mlp_learning_rate
    bootstrap_keep_probability: float = ENGINE_TUNER_DEFAULTS.mlp_bootstrap_keep_probability
    warmup_successes: int = ENGINE_TUNER_DEFAULTS.mlp_warmup_successes


EngineTunerSettings: TypeAlias = (
    BanditEngineTunerSettings | GaussianProcessEngineTunerSettings | MlpEnsembleEngineTunerSettings
)


@dataclass(frozen=True, slots=True)
class EngineTuningContext:
    """Stable identity for a family of engine-setting attempts."""

    course_key: str
    vehicle_id: str

    @property
    def key(self) -> str:
        return f"{self.course_key}|{self.vehicle_id}"


@dataclass(frozen=True, slots=True)
class EngineTuningChoice:
    """One reset-time engine choice plus diagnostic fields."""

    context: EngineTuningContext
    engine_setting_raw_value: int
    sampled_score: float
    mean_score: float
    finish_count: int
    estimated_finish_time_ms: int
    best_finish_time_ms: int | None
    score_count: int = 0
    best_score: float | None = None


@dataclass(frozen=True, slots=True)
class EngineTuningCandidateEstimate:
    """Estimated reset-time selection probability for one engine value."""

    engine_setting_raw_value: int
    probability: float
    mean_score: float
    uncertainty_score: float
    estimated_finish_time_ms: int
    finish_count: int = 0
    best_finish_time_ms: int | None = None
    score_count: int = 0
    best_score: float | None = None


@dataclass(frozen=True, slots=True)
class EngineTuningEpisodeOutcome:
    """Default-baseline episode result used to score one engine-setting sample.

    Failed episodes intentionally still contribute to completion, finish-rate,
    and return statistics. Finish-time objectives only score completed races.
    """

    context: EngineTuningContext
    engine_setting_raw_value: int
    completion_fraction: float
    finished: bool
    race_time_ms: int | None = None
    finish_position: int | None = None
    total_racers: int | None = None
    episode_return: float | None = None


def engine_candidates(*, minimum: int, maximum: int) -> tuple[int, ...]:
    """Return inclusive integer values clamped to game slider steps 0..128."""

    return engine_slider_steps(minimum=minimum, maximum=maximum)


def engine_bucket_candidates(
    *,
    bucket_raw_values: tuple[int, ...],
) -> tuple[int, ...]:
    """Return validated bandit engine bucket candidates."""

    return validate_engine_slider_bucket_values(bucket_raw_values)


def finish_time_score(race_time_ms: int) -> float:
    """Return a higher-is-better score in negative seconds."""

    return -(max(1.0, float(race_time_ms)) * 0.001)


def finish_time_ms_from_score(score: float) -> int:
    """Return a positive finish-time estimate from a negative-seconds score."""

    return max(1, int(round(max(0.001, -float(score)) * 1000.0)))


def successful_finish_time_ms(outcome: EngineTuningEpisodeOutcome) -> int | None:
    """Return a successful race time, ignoring failed or malformed outcomes."""

    if not outcome.finished or outcome.race_time_ms is None or outcome.race_time_ms <= 0:
        return None
    return int(outcome.race_time_ms)
