# src/rl_fzerox/core/envs/rewards/race_v3/controls.py
from __future__ import annotations

from fzerox_emulator import FZeroXTelemetry, StepSummary
from rl_fzerox.core.envs.rewards.common import RewardActionContext
from rl_fzerox.core.envs.rewards.race_v3.state import normalized_steer_level
from rl_fzerox.core.envs.rewards.race_v3.weights import RaceV3RewardWeights


class SteerOscillationRewardTracker:
    """Penalize second-order steering oscillation between policy decisions."""

    def __init__(self) -> None:
        self._previous_steer_level: float | None = None
        self._penultimate_steer_level: float | None = None

    def reset(self) -> None:
        self._previous_steer_level = None
        self._penultimate_steer_level = None

    def penalty(
        self,
        action_context: RewardActionContext | None,
        *,
        weights: RaceV3RewardWeights,
    ) -> float:
        steer_level = normalized_steer_level(action_context)
        if steer_level is None:
            self.reset()
            return 0.0

        penalty = 0.0
        if self._penultimate_steer_level is not None and self._previous_steer_level is not None:
            penalty = _steer_oscillation_penalty_for(
                steer_level,
                previous=self._previous_steer_level,
                penultimate=self._penultimate_steer_level,
                weights=weights,
            )

        self._penultimate_steer_level = self._previous_steer_level
        self._previous_steer_level = steer_level
        return penalty


def gas_underuse_penalty(
    summary: StepSummary,
    action_context: RewardActionContext | None,
    *,
    weights: RaceV3RewardWeights,
) -> float:
    penalty = weights.gas_underuse_penalty
    threshold = weights.gas_underuse_threshold
    if (
        penalty >= 0.0
        or threshold <= 0.0
        or action_context is None
        or action_context.gas_level is None
    ):
        return 0.0

    gas_level = min(max(float(action_context.gas_level), 0.0), 1.0)
    if gas_level >= threshold:
        return 0.0

    deficit_scale = (threshold - gas_level) / threshold
    return max(int(summary.frames_run), 0) * penalty * deficit_scale


def lean_low_speed_penalty(
    summary: StepSummary,
    telemetry: FZeroXTelemetry,
    action_context: RewardActionContext | None,
    *,
    weights: RaceV3RewardWeights,
) -> float:
    penalty = weights.lean_low_speed_penalty
    if (
        penalty >= 0.0
        or action_context is None
        or not action_context.lean_requested
        or telemetry.player.speed_kph >= weights.lean_low_speed_penalty_max_speed_kph
    ):
        return 0.0
    return max(int(summary.frames_run), 0) * penalty


def lean_request_penalty(
    summary: StepSummary,
    action_context: RewardActionContext | None,
    *,
    weights: RaceV3RewardWeights,
) -> float:
    penalty = weights.lean_request_penalty
    if penalty >= 0.0 or action_context is None or not action_context.lean_requested:
        return 0.0
    return max(int(summary.frames_run), 0) * penalty


def airborne_pitch_up_penalty(
    summary: StepSummary,
    telemetry: FZeroXTelemetry,
    action_context: RewardActionContext | None,
    *,
    weights: RaceV3RewardWeights,
) -> float:
    penalty = weights.airborne_pitch_up_penalty
    if (
        penalty >= 0.0
        or not telemetry.player.airborne
        or action_context is None
        or action_context.pitch_level is None
    ):
        return 0.0

    pitch_up_level = max(0.0, min(1.0, float(action_context.pitch_level)))
    if pitch_up_level <= 0.0:
        return 0.0
    return max(int(summary.frames_run), 0) * penalty * pitch_up_level


def manual_boost_reward(
    action_context: RewardActionContext | None,
    *,
    weights: RaceV3RewardWeights,
) -> float:
    reward = weights.manual_boost_reward
    if reward <= 0.0 or action_context is None or not action_context.boost_requested:
        return 0.0
    return reward


def _steer_oscillation_penalty_for(
    steer_level: float,
    *,
    previous: float,
    penultimate: float,
    weights: RaceV3RewardWeights,
) -> float:
    penalty = weights.steer_oscillation_penalty
    cap = weights.steer_oscillation_cap
    if penalty >= 0.0 or cap <= 0.0:
        return 0.0

    acceleration = steer_level - (2.0 * previous) + penultimate
    magnitude = max(abs(acceleration) - weights.steer_oscillation_deadzone, 0.0)
    if magnitude <= 0.0:
        return 0.0

    normalized = min(magnitude / cap, 1.0)
    return penalty * (normalized**weights.steer_oscillation_power)
