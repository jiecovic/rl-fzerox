# src/rl_fzerox/core/envs/rewards/__init__.py
from dataclasses import fields
from typing import Any

from pydantic import BaseModel

from rl_fzerox.core.envs.rewards.common import (
    RewardActionContext,
    RewardStep,
    RewardSummaryConfig,
    RewardTracker,
)
from rl_fzerox.core.envs.rewards.reward_main import RewardMainTracker, RewardMainWeights
from rl_fzerox.core.runtime_spec.schema import RewardConfig

CANONICAL_REWARD_NAME = "reward_main"
WeightT = RewardMainWeights


def build_reward_tracker(
    config: RewardConfig | None = None,
    *,
    max_episode_steps: int = 12_000,
) -> RewardTracker:
    """Construct one registered reward tracker by name."""

    resolved_config = config or RewardConfig()
    if resolved_config.name == CANONICAL_REWARD_NAME:
        return RewardMainTracker(
            weights=_weights_for(resolved_config),
            course_weights=_course_weights_for(resolved_config),
            max_episode_steps=max_episode_steps,
        )
    raise ValueError(f"Unsupported reward profile: {resolved_config.name!r}")


def _weights_for(config: RewardConfig) -> RewardMainWeights:
    """Map canonical reward schema fields to the reward-main weight dataclass."""

    return RewardMainWeights(**_weight_values(config))


def _course_weights_for(config: RewardConfig) -> dict[str, RewardMainWeights]:
    base = _weights_for(config)
    course_weights: dict[str, RewardMainWeights] = {}
    for raw_course_id, override in config.course_overrides.items():
        course_id = raw_course_id.strip()
        if not course_id:
            raise ValueError("reward course override keys must be non-empty course ids")
        course_weights[course_id] = RewardMainWeights(
            **{
                **_dataclass_values(base),
                **_weight_values(override, exclude_none=True),
            }
        )
    return course_weights


def _weight_values(config: BaseModel, *, exclude_none: bool = False) -> dict[str, Any]:
    field_names = {field.name for field in fields(RewardMainWeights)}
    return {
        key: value
        for key, value in config.model_dump(exclude_none=exclude_none).items()
        if key in field_names
    }


def _dataclass_values(weights: RewardMainWeights) -> dict[str, Any]:
    return {field.name: getattr(weights, field.name) for field in fields(RewardMainWeights)}


def reward_tracker_names() -> tuple[str, ...]:
    """Return the registered reward tracker names in insertion order."""

    return (CANONICAL_REWARD_NAME,)


__all__ = [
    "CANONICAL_REWARD_NAME",
    "RewardMainTracker",
    "RewardMainWeights",
    "RewardActionContext",
    "RewardStep",
    "RewardSummaryConfig",
    "RewardTracker",
    "build_reward_tracker",
    "reward_tracker_names",
]
