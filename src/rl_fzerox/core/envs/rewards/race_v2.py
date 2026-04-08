# src/rl_fzerox/core/envs/rewards/race_v2.py
from __future__ import annotations

from dataclasses import dataclass

from rl_fzerox.core.emulator.base import StepSummary
from rl_fzerox.core.envs.rewards.common import (
    RewardStep,
    RewardSummaryConfig,
    apply_flag_penalty,
    finish_placement_bonus,
)
from rl_fzerox.core.game.flags import (
    FLAG_COLLISION_RECOIL,
    FLAG_CRASHED,
    FLAG_FALLING_OFF_TRACK,
    FLAG_FINISHED,
    FLAG_RETIRED,
    FLAG_SPINNING_OUT,
)
from rl_fzerox.core.game.telemetry import FZeroXTelemetry


@dataclass(frozen=True)
class RaceV2RewardWeights:
    """Weights for the simpler race-first reward profile (`race_v2`)."""

    progress_scale: float = 0.001
    reverse_progress_scale: float = 0.001
    progress_epsilon: float = 0.5
    time_penalty_per_frame: float = -0.01
    energy_loss_epsilon: float = 0.1
    energy_loss_penalty_scale: float = 0.05
    stuck_truncation_penalty: float = -100.0
    wrong_way_truncation_penalty: float = -120.0
    timeout_truncation_penalty: float = -80.0
    collision_recoil_penalty: float = -2.0
    spinning_out_penalty: float = -4.0
    falling_off_track_penalty: float = -150.0
    crashed_penalty: float = -150.0
    retired_penalty: float = -150.0
    finish_bonus: float = 150.0
    finish_position_scale: float = 4.0
    max_race_position: int = 30


class RaceV2RewardTracker:
    """Track episode reward state for the simplified `race_v2` profile.

    The tracker keeps only cross-step frontier state; per-step deltas like
    reverse progress, energy loss, and entered flags are now pre-aggregated by
    the native repeated-step path.
    """

    def __init__(self, weights: RaceV2RewardWeights | None = None) -> None:
        self._weights = weights or RaceV2RewardWeights()
        self._best_race_distance = float("-inf")

    def reset(self, telemetry: FZeroXTelemetry | None) -> None:
        """Initialize reward state for a new episode."""

        if telemetry is None:
            self._best_race_distance = float("-inf")
            return
        self._best_race_distance = telemetry.player.race_distance
    
    def summary_config(self) -> RewardSummaryConfig:
        """Describe the native aggregation thresholds needed by `race_v2`."""

        return RewardSummaryConfig(
            reverse_progress_epsilon=self._weights.progress_epsilon,
            energy_loss_epsilon=self._weights.energy_loss_epsilon,
        )

    def step_summary(
        self,
        summary: StepSummary,
        telemetry: FZeroXTelemetry | None,
    ) -> RewardStep:
        """Compute one reward step from one repeated env-step summary."""

        if telemetry is None or not telemetry.in_race_mode:
            return RewardStep(reward=0.0, terminated=False)

        reward = summary.frames_run * self._weights.time_penalty_per_frame
        breakdown: dict[str, float] = {}
        if reward:
            breakdown["time"] = reward

        progress_gain = summary.max_race_distance - self._best_race_distance
        if progress_gain > self._weights.progress_epsilon:
            progress_reward = progress_gain * self._weights.progress_scale
            reward += progress_reward
            if progress_reward:
                breakdown["progress"] = progress_reward
            self._best_race_distance = summary.max_race_distance

        if summary.reverse_progress_total > 0.0:
            reverse_penalty = (
                -summary.reverse_progress_total * self._weights.reverse_progress_scale
            )
            reward += reverse_penalty
            if reverse_penalty:
                breakdown["reverse_progress"] = reverse_penalty

        if summary.energy_loss_total > 0.0:
            energy_loss_penalty = (
                -summary.energy_loss_total * self._weights.energy_loss_penalty_scale
            )
            reward += energy_loss_penalty
            if energy_loss_penalty:
                breakdown["energy_loss"] = energy_loss_penalty

        entered_flags = summary.entered_state_flags
        reward += apply_flag_penalty(
            entered_flags,
            FLAG_COLLISION_RECOIL,
            self._weights.collision_recoil_penalty,
            "collision_recoil",
            breakdown,
        )
        reward += apply_flag_penalty(
            entered_flags,
            FLAG_SPINNING_OUT,
            self._weights.spinning_out_penalty,
            "spinning_out",
            breakdown,
        )
        reward += apply_flag_penalty(
            entered_flags,
            FLAG_FALLING_OFF_TRACK,
            self._weights.falling_off_track_penalty,
            "falling_off_track",
            breakdown,
        )
        reward += apply_flag_penalty(
            entered_flags,
            FLAG_CRASHED,
            self._weights.crashed_penalty,
            "crashed",
            breakdown,
        )
        reward += apply_flag_penalty(
            entered_flags,
            FLAG_RETIRED,
            self._weights.retired_penalty,
            "retired",
            breakdown,
        )

        if entered_flags & FLAG_FINISHED:
            reward += self._weights.finish_bonus
            breakdown["finished"] = self._weights.finish_bonus
            placement_bonus = finish_placement_bonus(
                position=telemetry.player.position,
                max_race_position=self._weights.max_race_position,
                scale=self._weights.finish_position_scale,
            )
            if placement_bonus:
                reward += placement_bonus
                breakdown["finish_position"] = placement_bonus

        terminated = bool(
            telemetry.player.state_flags
            & (FLAG_FINISHED | FLAG_CRASHED | FLAG_RETIRED | FLAG_FALLING_OFF_TRACK)
        )
        return RewardStep(reward=reward, terminated=terminated, breakdown=breakdown)

    def truncation_penalty(self, truncation_reason: str | None) -> tuple[float, str | None]:
        """Return any extra reward penalty that should apply to a truncation."""

        if truncation_reason == "stuck":
            return self._weights.stuck_truncation_penalty, "stuck_truncation"
        if truncation_reason == "wrong_way":
            return self._weights.wrong_way_truncation_penalty, "wrong_way_truncation"
        if truncation_reason == "timeout":
            return self._weights.timeout_truncation_penalty, "timeout_truncation"
        return 0.0, None
