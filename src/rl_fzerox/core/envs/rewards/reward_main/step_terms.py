# src/rl_fzerox/core/envs/rewards/reward_main/step_terms.py
from __future__ import annotations

from fzerox_emulator import FZeroXTelemetry
from rl_fzerox.core.envs.course_effects import CourseEffect, course_effect_raw
from rl_fzerox.core.envs.rewards.reward_main.weights import RewardMainWeights


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


def zero_progress_bonus(progress_reward: float) -> float:
    del progress_reward
    return 0.0


def ko_star_reward(
    *,
    previous_count: int | None,
    telemetry: FZeroXTelemetry,
    weights: RewardMainWeights,
) -> float:
    if weights.ko_star_reward <= 0.0 or telemetry.game_mode_name != "gp_race":
        return 0.0
    current_count = ko_star_count(telemetry)
    if current_count is None or previous_count is None:
        return 0.0
    gained = max(current_count - max(previous_count, 0), 0)
    return gained * weights.ko_star_reward


def ko_star_count(telemetry: FZeroXTelemetry | None) -> int | None:
    if telemetry is None:
        return None
    return max(int(telemetry.player.ko_star_count), 0)
