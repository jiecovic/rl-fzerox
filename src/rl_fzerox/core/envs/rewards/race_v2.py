# src/rl_fzerox/core/envs/rewards/race_v2.py
from __future__ import annotations

from dataclasses import dataclass

from fzerox_emulator import FZeroXTelemetry, StepStatus, StepSummary
from rl_fzerox.core.envs.rewards.common import (
    RewardStep,
    RewardSummaryConfig,
    apply_event_penalty,
    finish_placement_bonus,
)


@dataclass(frozen=True)
class RaceV2RewardWeights:
    """Weights for the race-first reward profile (`race_v2`)."""

    time_penalty_per_frame: float = -0.005
    reverse_time_penalty_scale: float = 2.0
    milestone_distance: float = 3_000.0
    milestone_bonus: float = 2.0
    lap_1_completion_bonus: float = 20.0
    lap_2_completion_bonus: float = 35.0
    final_lap_completion_bonus: float = 60.0
    lap_position_scale: float = 1.0
    remaining_lap_penalty: float = 50.0
    energy_loss_epsilon: float = 0.01
    energy_loss_penalty_scale: float = 0.05
    energy_gain_reward_scale: float = 0.02
    collision_recoil_penalty: float = -2.0
    spinning_out_penalty: float = -4.0
    terminal_failure_base_penalty: float = -120.0
    stuck_truncation_base_penalty: float = -150.0
    wrong_way_truncation_base_penalty: float = -170.0
    timeout_truncation_base_penalty: float = -150.0
    finish_position_scale: float = 4.0


class RaceV2RewardTracker:
    """Track episode reward state for the simplified `race_v2` profile.

    The tracker keeps only cross-step frontier state; per-step deltas like
    reverse-warning frames, energy loss, and entered flags are now
    pre-aggregated by the native repeated-step path.
    """

    def __init__(
        self,
        weights: RaceV2RewardWeights | None = None,
        *,
        max_episode_steps: int = 12_000,
    ) -> None:
        self._weights = weights or RaceV2RewardWeights()
        self._next_milestone_index = 1
        self._awarded_laps_completed = 0
        self._max_episode_steps = max_episode_steps

    def reset(self, telemetry: FZeroXTelemetry | None) -> None:
        """Initialize reward state for a new episode."""

        if telemetry is None:
            self._next_milestone_index = 1
            self._awarded_laps_completed = 0
            return
        self._next_milestone_index = self._milestone_index(telemetry.player.race_distance) + 1
        self._awarded_laps_completed = telemetry.player.laps_completed

    def summary_config(self) -> RewardSummaryConfig:
        """Describe the native aggregation thresholds needed by `race_v2`."""

        return RewardSummaryConfig(
            energy_loss_epsilon=self._weights.energy_loss_epsilon,
        )

    def step_summary(
        self,
        summary: StepSummary,
        status: StepStatus,
        telemetry: FZeroXTelemetry | None,
    ) -> RewardStep:
        """Compute one reward step from one repeated env-step summary."""

        if telemetry is None or not telemetry.in_race_mode:
            return RewardStep(reward=0.0)

        reward = summary.frames_run * self._weights.time_penalty_per_frame
        breakdown: dict[str, float] = {}
        if reward:
            breakdown["time"] = reward
        reverse_time_penalty = self._reverse_time_penalty(summary)
        if reverse_time_penalty:
            reward += reverse_time_penalty
            breakdown["reverse_time"] = reverse_time_penalty

        milestone_bonus = self._milestone_bonus(summary.max_race_distance)
        if milestone_bonus:
            reward += milestone_bonus
            breakdown["milestone"] = milestone_bonus

        laps_completed_gain = telemetry.player.laps_completed - self._awarded_laps_completed
        if laps_completed_gain > 0:
            lap_bonus = 0.0
            lap_position_bonus = 0.0
            for lap_number in range(
                self._awarded_laps_completed + 1,
                telemetry.player.laps_completed + 1,
            ):
                lap_bonus += self._lap_completion_bonus(lap_number, telemetry.total_lap_count)
                lap_position_bonus += finish_placement_bonus(
                    position=telemetry.player.position,
                    total_racers=telemetry.total_racers,
                    scale=self._weights.lap_position_scale,
                )
            reward += lap_bonus
            if lap_bonus:
                breakdown["lap_completion"] = lap_bonus
            reward += lap_position_bonus
            if lap_position_bonus:
                breakdown["lap_position"] = lap_position_bonus
            self._awarded_laps_completed = telemetry.player.laps_completed

        if summary.energy_loss_total > 0.0:
            energy_loss_penalty = (
                -summary.energy_loss_total * self._weights.energy_loss_penalty_scale
            )
            reward += energy_loss_penalty
            if energy_loss_penalty:
                breakdown["energy_loss"] = energy_loss_penalty

        if summary.energy_gain_total > 0.0:
            energy_gain_reward = summary.energy_gain_total * self._weights.energy_gain_reward_scale
            reward += energy_gain_reward
            if energy_gain_reward:
                breakdown["energy_gain"] = energy_gain_reward

        reward += apply_event_penalty(
            summary.entered_collision_recoil,
            self._weights.collision_recoil_penalty,
            "collision_recoil",
            breakdown,
        )
        reward += apply_event_penalty(
            summary.entered_spinning_out,
            self._weights.spinning_out_penalty,
            "spinning_out",
            breakdown,
        )

        if status.termination_reason == "finished":
            placement_bonus = finish_placement_bonus(
                position=telemetry.player.position,
                total_racers=telemetry.total_racers,
                scale=self._weights.finish_position_scale,
            )
            if placement_bonus:
                reward += placement_bonus
                breakdown["finish_position"] = placement_bonus
            return RewardStep(reward=reward, breakdown=breakdown)

        if status.termination_reason in {"crashed", "retired", "falling_off_track"}:
            failure_penalty = self._dynamic_penalty(
                self._weights.terminal_failure_base_penalty,
                status,
                telemetry,
            )
            reward += failure_penalty
            breakdown[status.termination_reason] = failure_penalty
            return RewardStep(reward=reward, breakdown=breakdown)

        if status.truncation_reason is not None:
            truncation_penalty = self._dynamic_penalty(
                self._truncation_base_penalty(status.truncation_reason),
                status,
                telemetry,
            )
            reward += truncation_penalty
            breakdown[f"{status.truncation_reason}_truncation"] = truncation_penalty

        return RewardStep(reward=reward, breakdown=breakdown)

    def _milestone_index(self, race_distance: float) -> int:
        if race_distance <= 0.0:
            return 0
        return int(race_distance // self._weights.milestone_distance)

    def _milestone_bonus(self, max_race_distance: float) -> float:
        last_crossed_index = self._milestone_index(max_race_distance)
        crossed_count = max(0, last_crossed_index - self._next_milestone_index + 1)
        if crossed_count <= 0:
            return 0.0
        self._next_milestone_index += crossed_count
        return crossed_count * self._weights.milestone_bonus

    def _lap_completion_bonus(self, lap_number: int, total_lap_count: int) -> float:
        if lap_number <= 1:
            return self._weights.lap_1_completion_bonus
        if lap_number >= max(total_lap_count, 1):
            return self._weights.final_lap_completion_bonus
        if lap_number == 2:
            return self._weights.lap_2_completion_bonus
        return self._weights.lap_2_completion_bonus

    def _dynamic_penalty(
        self,
        base_penalty: float,
        status: StepStatus,
        telemetry: FZeroXTelemetry,
    ) -> float:
        return (
            base_penalty
            - (self._remaining_step_penalty(status))
            - (self._remaining_lap_count(telemetry) * self._weights.remaining_lap_penalty)
        )

    def _remaining_step_penalty(self, status: StepStatus) -> float:
        remaining_steps = max(self._max_episode_steps - status.step_count, 0)
        return remaining_steps * abs(self._weights.time_penalty_per_frame)

    def _reverse_time_penalty(self, summary: StepSummary) -> float:
        extra_scale = self._weights.reverse_time_penalty_scale - 1.0
        if summary.reverse_warning_frames <= 0 or extra_scale == 0.0:
            return 0.0
        return (
            summary.reverse_warning_frames
            * self._weights.time_penalty_per_frame
            * extra_scale
        )

    def _remaining_lap_count(self, telemetry: FZeroXTelemetry) -> int:
        return max(
            telemetry.total_lap_count - telemetry.player.laps_completed,
            0,
        )

    def _truncation_base_penalty(self, truncation_reason: str) -> float:
        if truncation_reason == "stuck":
            return self._weights.stuck_truncation_base_penalty
        if truncation_reason == "wrong_way":
            return self._weights.wrong_way_truncation_base_penalty
        if truncation_reason == "timeout":
            return self._weights.timeout_truncation_base_penalty
        raise ValueError(f"Unsupported truncation reason: {truncation_reason!r}")
