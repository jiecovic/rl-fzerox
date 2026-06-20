# src/rl_fzerox/core/engine_tuning/config.py
"""Configuration adapters for adaptive engine tuning."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import TYPE_CHECKING

from rl_fzerox.core.engine_tuning.types import (
    BanditEngineTunerSettings,
    EngineTunerSettings,
    GaussianProcessEngineTunerSettings,
    MlpEnsembleEngineTunerSettings,
)

if TYPE_CHECKING:
    from rl_fzerox.core.runtime_spec.schema import AdaptiveEngineTuningConfig


@dataclass(frozen=True, slots=True)
class EngineTuningTimebase:
    """Timing constants needed to derive timeout priors from emulator frames."""

    native_fps: float = 60.0
    uncertainty_horizon_fraction: float = 0.15


ENGINE_TUNING_TIMEBASE = EngineTuningTimebase()


def engine_tuner_settings(config: AdaptiveEngineTuningConfig) -> EngineTunerSettings:
    """Convert runtime schema into ordered tuner settings."""

    if config.backend == "bandit":
        return BanditEngineTunerSettings(
            min_raw_value=config.min_raw_value,
            max_raw_value=config.max_raw_value,
            prior_finish_time_seconds=config.prior_finish_time_seconds,
            uniform_exploration=config.uniform_exploration,
            objective=config.objective,
            reward_fingerprint=config.reward_fingerprint,
            bucket_raw_values=config.bucket_raw_values,
            exploration_seconds=float(config.exploration_scale),
            safe_finish_rate_threshold=config.safe_finish_rate_threshold,
        )
    if config.backend == "mlp_ensemble":
        return MlpEnsembleEngineTunerSettings(
            min_raw_value=config.min_raw_value,
            max_raw_value=config.max_raw_value,
            prior_finish_time_seconds=config.prior_finish_time_seconds,
            uniform_exploration=config.uniform_exploration,
            greedy_plateau_tolerance_seconds=config.greedy_plateau_tolerance_seconds,
            ensemble_members=config.ensemble_members,
            randomized_prior_seconds=config.randomized_prior_seconds,
            hidden_dim=config.hidden_dim,
            training_steps=config.training_steps,
            learning_rate=config.learning_rate,
            bootstrap_keep_probability=config.bootstrap_keep_probability,
            warmup_successes=config.warmup_successes,
        )
    return GaussianProcessEngineTunerSettings(
        min_raw_value=config.min_raw_value,
        max_raw_value=config.max_raw_value,
        stat_decay=config.stat_decay,
        prior_finish_time_seconds=config.prior_finish_time_seconds,
        exploration_seconds=float(config.exploration_scale),
        observation_noise_seconds=config.observation_noise_seconds,
        curve_lengthscale_raw=config.curve_lengthscale_raw,
        uniform_exploration=config.uniform_exploration,
        greedy_plateau_tolerance_seconds=config.greedy_plateau_tolerance_seconds,
    )


def engine_tuning_episode_horizon_prior_seconds(
    *,
    max_episode_steps: int,
    action_repeat: int,
) -> float:
    """Return a timeout prior in seconds from the native episode horizon."""

    repeat = max(1, int(action_repeat))
    policy_decision_cap = ceil(max(1, int(max_episode_steps)) / repeat)
    horizon_frames = policy_decision_cap * repeat
    return horizon_frames / ENGINE_TUNING_TIMEBASE.native_fps


def engine_tuning_uncertainty_scale_seconds(*, prior_finish_time_seconds: float) -> float:
    """Return an internal uncertainty scale from the episode timeout horizon."""

    return max(
        1.0,
        float(prior_finish_time_seconds) * ENGINE_TUNING_TIMEBASE.uncertainty_horizon_fraction,
    )
