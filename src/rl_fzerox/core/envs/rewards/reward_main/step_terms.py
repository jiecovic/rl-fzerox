# src/rl_fzerox/core/envs/rewards/reward_main/step_terms.py
"""Per-step reward terms keyed by the current telemetry snapshot.

This module collects small, telemetry-local terms such as KO star changes,
danger-speed shaping, and surface effects. Stateful terms stay in dedicated
trackers or in the main reward tracker.
"""

from __future__ import annotations

from dataclasses import dataclass

from fzerox_emulator import FZeroXTelemetry
from rl_fzerox.core.envs.course_effects import CourseEffect, course_effect_raw
from rl_fzerox.core.envs.rewards.reward_main.weights import RewardMainWeights


@dataclass(frozen=True, slots=True)
class KoStarRewardEvent:
    previous_count: int
    current_count: int
    gained: int
    reward: float


def clip_step_reward(reward: float, *, weights: RewardMainWeights) -> float:
    if weights.step_reward_clip_min is not None:
        reward = max(reward, float(weights.step_reward_clip_min))
    if weights.step_reward_clip_max is not None:
        reward = min(reward, float(weights.step_reward_clip_max))
    return reward


def ground_effect_progress_modifier(
    telemetry: FZeroXTelemetry,
    *,
    weights: RewardMainWeights,
) -> tuple[str, float]:
    raw_effect = course_effect_raw(telemetry)
    if raw_effect == CourseEffect.DIRT:
        return "dirt", weights.dirt_progress_multiplier
    if raw_effect == CourseEffect.ICE:
        return "ice", weights.ice_progress_multiplier
    return "ground_effect", 1.0


def ko_star_reward_event(
    *,
    previous_count: int | None,
    telemetry: FZeroXTelemetry,
    weights: RewardMainWeights,
) -> KoStarRewardEvent | None:
    if weights.ko_star_reward <= 0.0 or telemetry.game_mode_name != "gp_race":
        return None
    current_count = ko_star_count(telemetry)
    if current_count is None or previous_count is None:
        return None
    gained = max(current_count - max(previous_count, 0), 0)
    if gained <= 0:
        return None
    return KoStarRewardEvent(
        previous_count=max(previous_count, 0),
        current_count=current_count,
        gained=gained,
        reward=gained * weights.ko_star_reward,
    )


def ko_star_count(telemetry: FZeroXTelemetry | None) -> int | None:
    if telemetry is None:
        return None
    return max(int(telemetry.player.ko_star_count), 0)
