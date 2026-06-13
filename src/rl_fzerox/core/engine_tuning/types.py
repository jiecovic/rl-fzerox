# src/rl_fzerox/core/engine_tuning/types.py
"""Shared engine-tuning types and score helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

EngineTunerBackend = Literal["gaussian_process", "mlp_ensemble"]


@dataclass(frozen=True, slots=True)
class EngineTunerDefaults:
    """Default scale values for adaptive engine tuning."""

    backend: EngineTunerBackend = "gaussian_process"
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
    curve_lengthscale_raw: float = 12.0
    uniform_exploration: float = 0.05
    greedy_plateau_tolerance_seconds: float = 1.0


ENGINE_TUNER_DEFAULTS = EngineTunerDefaults()


@dataclass(frozen=True, slots=True)
class EngineTunerCommonSettings:
    """Backend-independent knobs for one adaptive engine-tuning run."""

    min_raw_value: int = 0
    max_raw_value: int = 100
    prior_finish_time_seconds: float = ENGINE_TUNER_DEFAULTS.prior_finish_time_seconds
    uniform_exploration: float = ENGINE_TUNER_DEFAULTS.uniform_exploration
    greedy_plateau_tolerance_seconds: float = ENGINE_TUNER_DEFAULTS.greedy_plateau_tolerance_seconds


@dataclass(frozen=True, slots=True)
class GaussianProcessEngineTunerSettings(EngineTunerCommonSettings):
    """Static knobs used only by the Gaussian-process backend."""

    backend: Literal["gaussian_process"] = "gaussian_process"
    stat_decay: float = ENGINE_TUNER_DEFAULTS.stat_decay
    exploration_seconds: float = ENGINE_TUNER_DEFAULTS.exploration_seconds
    observation_noise_seconds: float = ENGINE_TUNER_DEFAULTS.observation_noise_seconds
    curve_lengthscale_raw: float = ENGINE_TUNER_DEFAULTS.curve_lengthscale_raw


@dataclass(frozen=True, slots=True)
class MlpEnsembleEngineTunerSettings(EngineTunerCommonSettings):
    """Static knobs used only by the MLP ensemble backend."""

    backend: Literal["mlp_ensemble"] = "mlp_ensemble"
    ensemble_members: int = ENGINE_TUNER_DEFAULTS.mlp_ensemble_members
    randomized_prior_seconds: float = ENGINE_TUNER_DEFAULTS.mlp_randomized_prior_seconds
    hidden_dim: int = ENGINE_TUNER_DEFAULTS.mlp_hidden_dim
    training_steps: int = ENGINE_TUNER_DEFAULTS.mlp_training_steps
    learning_rate: float = ENGINE_TUNER_DEFAULTS.mlp_learning_rate
    bootstrap_keep_probability: float = ENGINE_TUNER_DEFAULTS.mlp_bootstrap_keep_probability
    warmup_successes: int = ENGINE_TUNER_DEFAULTS.mlp_warmup_successes


EngineTunerSettings: TypeAlias = GaussianProcessEngineTunerSettings | MlpEnsembleEngineTunerSettings


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


@dataclass(frozen=True, slots=True)
class EngineTuningCandidateEstimate:
    """Estimated reset-time selection probability for one engine value."""

    engine_setting_raw_value: int
    probability: float
    mean_score: float
    uncertainty_score: float
    estimated_finish_time_ms: int
    finish_count: int
    best_finish_time_ms: int | None


@dataclass(frozen=True, slots=True)
class EngineTuningEpisodeOutcome:
    """Episode result used to score one successful engine-setting sample."""

    context: EngineTuningContext
    engine_setting_raw_value: int
    completion_fraction: float
    finished: bool
    race_time_ms: int | None = None
    finish_position: int | None = None
    total_racers: int | None = None


def engine_candidates(*, minimum: int, maximum: int) -> tuple[int, ...]:
    """Return inclusive integer engine values clamped to the game's raw 0-100 range."""

    lower = max(0, min(100, int(minimum)))
    upper = max(0, min(100, int(maximum)))
    if lower > upper:
        raise ValueError(f"engine tuning min_raw_value exceeds max_raw_value: {lower} > {upper}")
    return tuple(range(lower, upper + 1))


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
