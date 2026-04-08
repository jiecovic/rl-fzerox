# src/rl_fzerox/core/envs/rewards/__init__.py
from collections.abc import Callable

from rl_fzerox.core.config.schema import RewardConfig
from rl_fzerox.core.envs.rewards.common import (
    RewardStep,
    RewardSummaryConfig,
    RewardTracker,
)
from rl_fzerox.core.envs.rewards.race_v2 import RaceV2RewardTracker, RaceV2RewardWeights

RewardTrackerFactory = Callable[..., RewardTracker]
DEFAULT_REWARD_NAME = "race_v2"
REWARD_TRACKER_REGISTRY: dict[str, RewardTrackerFactory] = {
    DEFAULT_REWARD_NAME: RaceV2RewardTracker,
}


def build_reward_tracker(
    config: RewardConfig | None = None,
    *,
    max_episode_steps: int = 12_000,
) -> RewardTracker:
    """Construct one registered reward tracker by name."""

    resolved_config = config or RewardConfig()
    factory = REWARD_TRACKER_REGISTRY.get(resolved_config.name)
    if factory is None:
        raise ValueError(f"Unsupported reward profile: {resolved_config.name!r}")
    weights = RaceV2RewardWeights(
        time_penalty_per_frame=resolved_config.time_penalty_per_frame,
        milestone_distance=resolved_config.milestone_distance,
        milestone_bonus=resolved_config.milestone_bonus,
        lap_1_completion_bonus=resolved_config.lap_1_completion_bonus,
        lap_2_completion_bonus=resolved_config.lap_2_completion_bonus,
        final_lap_completion_bonus=resolved_config.final_lap_completion_bonus,
        lap_position_scale=resolved_config.lap_position_scale,
        remaining_lap_penalty=resolved_config.remaining_lap_penalty,
        energy_loss_epsilon=resolved_config.energy_loss_epsilon,
        energy_loss_penalty_scale=resolved_config.energy_loss_penalty_scale,
        energy_gain_reward_scale=resolved_config.energy_gain_reward_scale,
        collision_recoil_penalty=resolved_config.collision_recoil_penalty,
        spinning_out_penalty=resolved_config.spinning_out_penalty,
        terminal_failure_base_penalty=resolved_config.terminal_failure_base_penalty,
        stuck_truncation_base_penalty=resolved_config.stuck_truncation_base_penalty,
        wrong_way_truncation_base_penalty=resolved_config.wrong_way_truncation_base_penalty,
        timeout_truncation_base_penalty=resolved_config.timeout_truncation_base_penalty,
        finish_position_scale=resolved_config.finish_position_scale,
    )
    return factory(weights=weights, max_episode_steps=max_episode_steps)


def reward_tracker_names() -> tuple[str, ...]:
    """Return the registered reward tracker names in insertion order."""

    return tuple(REWARD_TRACKER_REGISTRY)


__all__ = [
    "DEFAULT_REWARD_NAME",
    "RaceV2RewardTracker",
    "RaceV2RewardWeights",
    "REWARD_TRACKER_REGISTRY",
    "RewardStep",
    "RewardSummaryConfig",
    "RewardTracker",
    "RewardTrackerFactory",
    "build_reward_tracker",
    "reward_tracker_names",
]
