# src/rl_fzerox/core/envs/rewards/__init__.py
from collections.abc import Callable

from rl_fzerox.core.envs.rewards.common import (
    RewardStep,
    RewardSummaryConfig,
    RewardTracker,
)
from rl_fzerox.core.envs.rewards.race_v2 import RaceV2RewardTracker, RaceV2RewardWeights

RewardTrackerFactory = Callable[[], RewardTracker]
DEFAULT_REWARD_NAME = "race_v2"
REWARD_TRACKER_REGISTRY: dict[str, RewardTrackerFactory] = {
    DEFAULT_REWARD_NAME: RaceV2RewardTracker,
}


def build_reward_tracker(name: str = DEFAULT_REWARD_NAME) -> RewardTracker:
    """Construct one registered reward tracker by name."""

    factory = REWARD_TRACKER_REGISTRY.get(name)
    if factory is None:
        raise ValueError(f"Unsupported reward profile: {name!r}")
    return factory()


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
