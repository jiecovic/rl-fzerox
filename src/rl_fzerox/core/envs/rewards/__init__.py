# src/rl_fzerox/core/envs/rewards/__init__.py
from dataclasses import fields
from typing import Any

from rl_fzerox.core.config.schema import RewardConfig
from rl_fzerox.core.envs.rewards.common import (
    RewardActionContext,
    RewardStep,
    RewardSummaryConfig,
    RewardTracker,
)
from rl_fzerox.core.envs.rewards.race_v3 import RaceV3RewardTracker, RaceV3RewardWeights

DEFAULT_REWARD_NAME = "race_v3"


def build_reward_tracker(
    config: RewardConfig | None = None,
    *,
    max_episode_steps: int = 12_000,
) -> RewardTracker:
    """Construct one registered reward tracker by name."""

    resolved_config = config or RewardConfig()
    if resolved_config.name != DEFAULT_REWARD_NAME:
        raise ValueError(f"Unsupported reward profile: {resolved_config.name!r}")
    return RaceV3RewardTracker(
        weights=_race_v3_weights(resolved_config),
        course_weights=_race_v3_course_weights(resolved_config),
        max_episode_steps=max_episode_steps,
    )


def _race_v3_weights(config: RewardConfig) -> RaceV3RewardWeights:
    """Map schema fields to the race_v3 weights dataclass."""

    values: dict[str, Any] = config.model_dump(exclude={"name", "course_overrides"})
    return RaceV3RewardWeights(**values)


def _race_v3_course_weights(config: RewardConfig) -> dict[str, RaceV3RewardWeights]:
    base_values = _weight_values(_race_v3_weights(config))
    course_weights: dict[str, RaceV3RewardWeights] = {}
    for raw_course_id, override in config.course_overrides.items():
        course_id = raw_course_id.strip()
        if not course_id:
            raise ValueError("reward course override keys must be non-empty course ids")
        values = dict(base_values)
        values.update(override.model_dump(include=override.model_fields_set, exclude_none=True))
        course_weights[course_id] = RaceV3RewardWeights(**values)
    return course_weights


def _weight_values(weights: RaceV3RewardWeights) -> dict[str, Any]:
    return {field.name: getattr(weights, field.name) for field in fields(RaceV3RewardWeights)}


def reward_tracker_names() -> tuple[str, ...]:
    """Return the registered reward tracker names in insertion order."""

    return (DEFAULT_REWARD_NAME,)


__all__ = [
    "DEFAULT_REWARD_NAME",
    "RaceV3RewardTracker",
    "RaceV3RewardWeights",
    "RewardActionContext",
    "RewardStep",
    "RewardSummaryConfig",
    "RewardTracker",
    "build_reward_tracker",
    "reward_tracker_names",
]
