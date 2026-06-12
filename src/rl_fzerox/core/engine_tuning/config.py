# src/rl_fzerox/core/engine_tuning/config.py
"""Configuration adapters for adaptive engine tuning."""

from __future__ import annotations

from rl_fzerox.core.engine_tuning.bandit import EngineBanditSettings
from rl_fzerox.core.runtime_spec.schema import AdaptiveEngineTuningConfig


def engine_bandit_settings(config: AdaptiveEngineTuningConfig) -> EngineBanditSettings:
    """Convert runtime schema into bandit settings."""

    return EngineBanditSettings(
        min_raw_value=config.min_raw_value,
        max_raw_value=config.max_raw_value,
        bin_size=config.bin_size,
        stat_decay=config.stat_decay,
        prior_mean=config.prior_mean,
        prior_strength=config.prior_strength,
        exploration_scale=config.exploration_scale,
        uniform_exploration=config.uniform_exploration,
        completion_weight=config.completion_weight,
        finish_bonus=config.finish_bonus,
        position_weight=config.position_weight,
    )
