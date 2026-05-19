# src/rl_fzerox/core/envs/rewards/reward_main/controls.py
from __future__ import annotations

from fzerox_emulator import FZeroXTelemetry, StepSummary
from rl_fzerox.core.envs.rewards.common import RewardActionContext
from rl_fzerox.core.envs.rewards.reward_main.weights import RewardMainWeights


def outside_track_recovery_reward(
    *,
    weights: RewardMainWeights,
    previous_distance: float | None,
    current_distance: float | None,
    enabled: bool,
) -> float:
    recovery_weight = weights.outside_track_recovery_reward
    if (
        recovery_weight <= 0.0
        or not enabled
        or previous_distance is None
        or current_distance is None
    ):
        return 0.0
    reward = recovery_weight * (previous_distance - current_distance)
    cap = max(0.0, float(weights.outside_track_recovery_reward_cap))
    return max(-cap, min(cap, reward))


def lean_request_penalty(
    summary: StepSummary,
    action_context: RewardActionContext | None,
    *,
    weights: RewardMainWeights,
) -> float:
    penalty = weights.lean_request_penalty
    if penalty >= 0.0 or action_context is None or not action_context.lean_requested:
        return 0.0
    return max(int(summary.frames_run), 0) * penalty


def air_brake_request_penalty(
    summary: StepSummary,
    action_context: RewardActionContext | None,
    *,
    weights: RewardMainWeights,
) -> float:
    penalty = weights.air_brake_request_penalty
    if penalty >= 0.0 or action_context is None or not action_context.air_brake_requested:
        return 0.0
    return max(int(summary.frames_run), 0) * penalty


def grounded_pitch_penalty(
    summary: StepSummary,
    telemetry: FZeroXTelemetry,
    action_context: RewardActionContext | None,
    *,
    weights: RewardMainWeights,
) -> float:
    penalty = weights.grounded_pitch_penalty
    if (
        penalty >= 0.0
        or telemetry.player.airborne
        or action_context is None
        or action_context.pitch_level is None
    ):
        return 0.0

    pitch_magnitude = abs(max(-1.0, min(1.0, float(action_context.pitch_level))))
    deadzone = max(0.0, min(1.0, float(action_context.pitch_deadzone)))
    if pitch_magnitude <= deadzone:
        return 0.0
    scale = (pitch_magnitude - deadzone) / max(1.0 - deadzone, 1e-9)
    return max(int(summary.frames_run), 0) * penalty * scale


def manual_boost_reward(
    action_context: RewardActionContext | None,
    *,
    weights: RewardMainWeights,
) -> float:
    reward = weights.manual_boost_reward
    if reward <= 0.0 or action_context is None or not action_context.boost_requested:
        return 0.0
    return reward
