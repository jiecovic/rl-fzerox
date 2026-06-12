# src/rl_fzerox/core/engine_tuning/types.py
"""Shared engine-tuning types and score helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EngineTunerBackend = Literal["gaussian_process", "mlp_ensemble"]


@dataclass(frozen=True, slots=True)
class EngineTunerDefaults:
    """Default scale values for adaptive engine tuning."""

    backend: EngineTunerBackend = "gaussian_process"
    stat_decay: float = 0.995
    prior_finish_time_seconds: float = 200.0
    exploration_seconds: float = 30.0
    observation_noise_seconds: float = 1.5
    curve_lengthscale_raw: float = 12.0
    uniform_exploration: float = 0.05


ENGINE_TUNER_DEFAULTS = EngineTunerDefaults()


@dataclass(frozen=True, slots=True)
class EngineTunerSettings:
    """Static knobs for one adaptive engine-tuning run."""

    min_raw_value: int = 0
    max_raw_value: int = 100
    backend: EngineTunerBackend = ENGINE_TUNER_DEFAULTS.backend
    stat_decay: float = ENGINE_TUNER_DEFAULTS.stat_decay
    prior_finish_time_seconds: float = ENGINE_TUNER_DEFAULTS.prior_finish_time_seconds
    exploration_seconds: float = ENGINE_TUNER_DEFAULTS.exploration_seconds
    observation_noise_seconds: float = ENGINE_TUNER_DEFAULTS.observation_noise_seconds
    curve_lengthscale_raw: float = ENGINE_TUNER_DEFAULTS.curve_lengthscale_raw
    uniform_exploration: float = ENGINE_TUNER_DEFAULTS.uniform_exploration


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
