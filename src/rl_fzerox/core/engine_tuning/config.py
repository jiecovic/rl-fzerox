# src/rl_fzerox/core/engine_tuning/config.py
"""Configuration adapters for adaptive engine tuning."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from rl_fzerox.core.engine_tuning.tuner import EngineTunerSettings
from rl_fzerox.core.runtime_spec.schema import AdaptiveEngineTuningConfig


@dataclass(frozen=True, slots=True)
class EngineTuningTimebase:
    """Timing constants needed to derive timeout priors from emulator frames."""

    native_fps: float = 60.0


ENGINE_TUNING_TIMEBASE = EngineTuningTimebase()


def engine_tuner_settings(config: AdaptiveEngineTuningConfig) -> EngineTunerSettings:
    """Convert runtime schema into ordered tuner settings."""

    return EngineTunerSettings(
        min_raw_value=config.min_raw_value,
        max_raw_value=config.max_raw_value,
        backend=config.backend,
        stat_decay=config.stat_decay,
        prior_finish_time_seconds=config.prior_finish_time_seconds,
        exploration_seconds=float(config.exploration_scale),
        observation_noise_seconds=config.observation_noise_seconds,
        curve_lengthscale_raw=config.curve_lengthscale_raw,
        uniform_exploration=config.uniform_exploration,
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
