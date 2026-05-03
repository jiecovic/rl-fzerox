# src/rl_fzerox/core/envs/rewards/__init__.py
from dataclasses import fields
from typing import Any, TypeVar

from rl_fzerox.core.config.schema import RewardConfig
from rl_fzerox.core.envs.rewards.common import (
    RewardActionContext,
    RewardStep,
    RewardSummaryConfig,
    RewardTracker,
)
from rl_fzerox.core.envs.rewards.race_v3 import RaceV3RewardTracker, RaceV3RewardWeights
from rl_fzerox.core.envs.rewards.reward_main import RewardMainTracker, RewardMainWeights

DEFAULT_REWARD_NAME = "race_v3"
CANONICAL_REWARD_NAME = "reward_main"
WeightT = TypeVar("WeightT", RaceV3RewardWeights, RewardMainWeights)


def build_reward_tracker(
    config: RewardConfig | None = None,
    *,
    max_episode_steps: int = 12_000,
) -> RewardTracker:
    """Construct one registered reward tracker by name."""

    resolved_config = config or RewardConfig()
    if resolved_config.name == DEFAULT_REWARD_NAME:
        return RaceV3RewardTracker(
            weights=_weights_for(resolved_config, RaceV3RewardWeights),
            course_weights=_course_weights_for(resolved_config, RaceV3RewardWeights),
            max_episode_steps=max_episode_steps,
        )
    if resolved_config.name == CANONICAL_REWARD_NAME:
        return RewardMainTracker(
            weights=_weights_for(resolved_config, RewardMainWeights),
            course_weights=_course_weights_for(resolved_config, RewardMainWeights),
            max_episode_steps=max_episode_steps,
        )
    raise ValueError(f"Unsupported reward profile: {resolved_config.name!r}")


def _weights_for(config: RewardConfig, weight_type: type[WeightT]) -> WeightT:
    """Map schema fields to one reward weight dataclass."""

    raw_values: dict[str, Any] = config.model_dump(exclude={"name", "course_overrides"})
    allowed_fields = {field.name for field in fields(weight_type)}
    values = {key: value for key, value in raw_values.items() if key in allowed_fields}
    return weight_type(**values)


def _course_weights_for(
    config: RewardConfig,
    weight_type: type[WeightT],
) -> dict[str, WeightT]:
    base_values = _weight_values(_weights_for(config, weight_type))
    allowed_fields = set(base_values)
    course_weights: dict[str, WeightT] = {}
    for raw_course_id, override in config.course_overrides.items():
        course_id = raw_course_id.strip()
        if not course_id:
            raise ValueError("reward course override keys must be non-empty course ids")
        values = dict(base_values)
        override_values = override.model_dump(include=override.model_fields_set, exclude_none=True)
        values.update(
            {key: value for key, value in override_values.items() if key in allowed_fields}
        )
        course_weights[course_id] = weight_type(**values)
    return course_weights


def _weight_values(weights: RaceV3RewardWeights | RewardMainWeights) -> dict[str, Any]:
    return {field.name: getattr(weights, field.name) for field in fields(type(weights))}


def reward_tracker_names() -> tuple[str, ...]:
    """Return the registered reward tracker names in insertion order."""

    return (DEFAULT_REWARD_NAME, CANONICAL_REWARD_NAME)


__all__ = [
    "CANONICAL_REWARD_NAME",
    "DEFAULT_REWARD_NAME",
    "RaceV3RewardTracker",
    "RaceV3RewardWeights",
    "RewardMainTracker",
    "RewardMainWeights",
    "RewardActionContext",
    "RewardStep",
    "RewardSummaryConfig",
    "RewardTracker",
    "build_reward_tracker",
    "reward_tracker_names",
]
