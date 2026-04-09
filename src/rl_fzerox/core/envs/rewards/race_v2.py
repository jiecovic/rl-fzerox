# src/rl_fzerox/core/envs/rewards/race_v2.py
from __future__ import annotations

from dataclasses import dataclass

from fzerox_emulator import FZeroXTelemetry, StepStatus, StepSummary
from rl_fzerox.core.envs.laps import completed_race_laps
from rl_fzerox.core.envs.rewards.common import (
    RewardActionContext,
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
    low_speed_time_penalty_scale: float = 2.0
    milestone_distance: float = 3_000.0
    milestone_bonus: float = 2.0
    milestone_speed_scale: float = 0.0
    milestone_speed_bonus_cap: float = 0.0
    bootstrap_progress_scale: float = 0.001
    bootstrap_regress_penalty_scale: float = 0.002
    bootstrap_position_multiplier_scale: float = 0.0
    bootstrap_lap_count: int = 1
    lap_1_completion_bonus: float = 20.0
    lap_2_completion_bonus: float = 35.0
    final_lap_completion_bonus: float = 60.0
    lap_position_scale: float = 1.0
    remaining_step_penalty_per_frame: float = 0.01
    remaining_lap_penalty: float = 50.0
    energy_loss_epsilon: float = 0.01
    energy_loss_penalty_scale: float = 0.05
    energy_loss_safe_fraction: float = 0.9
    energy_loss_danger_power: float = 2.0
    energy_gain_reward_scale: float = 0.02
    energy_gain_collision_cooldown_frames: int = 0
    boost_redundant_press_penalty: float = 0.0
    collision_recoil_penalty: float = -2.0
    spinning_out_penalty: float = -4.0
    terminal_failure_base_penalty: float = -120.0
    stuck_truncation_base_penalty: float = -150.0
    wrong_way_truncation_base_penalty: float = -170.0
    timeout_truncation_base_penalty: float = -150.0
    finish_position_scale: float = 4.0


class RaceV2RewardTracker:
    """Track episode reward state for the simplified `race_v2` profile.

    The tracker keeps only compact cross-step reward state; per-step deltas like
    reverse-active frames, energy loss, and entered flags are now
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
        self._last_milestone_step_count = 0
        self._awarded_laps_completed = 0
        self._previous_progress_delta_position = 0.0
        self._progress_origin = 0.0
        self._has_progress_origin = False
        self._max_episode_steps = max_episode_steps
        self._energy_gain_cooldown_frames_remaining = 0
        self._previous_boost_active = False

    def reset(self, telemetry: FZeroXTelemetry | None) -> None:
        """Initialize reward state for a new episode."""

        if telemetry is None or not telemetry.in_race_mode:
            self._next_milestone_index = 1
            self._last_milestone_step_count = 0
            self._awarded_laps_completed = 0
            self._previous_progress_delta_position = 0.0
            self._progress_origin = 0.0
            self._has_progress_origin = False
            self._energy_gain_cooldown_frames_remaining = 0
            self._previous_boost_active = False
            return
        self._next_milestone_index = 1
        self._last_milestone_step_count = 0
        self._awarded_laps_completed = self._race_laps_completed(telemetry)
        self._previous_progress_delta_position = 0.0
        self._set_progress_origin(telemetry.player.race_distance)
        self._energy_gain_cooldown_frames_remaining = 0
        self._previous_boost_active = telemetry.player.boost_timer > 0

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
        action_context: RewardActionContext | None = None,
    ) -> RewardStep:
        """Compute one reward step from one repeated env-step summary."""

        if telemetry is None or not telemetry.in_race_mode:
            self._previous_boost_active = False
            return RewardStep(reward=0.0)

        resolved_action_context = action_context or RewardActionContext()
        boost_active_before_step = self._previous_boost_active
        self._ensure_progress_origin(telemetry)
        max_relative_progress = self._relative_progress(summary.max_race_distance)
        current_delta_position = self._progress_delta_position(telemetry.player.race_distance)
        reward = summary.frames_run * self._weights.time_penalty_per_frame
        breakdown: dict[str, float] = {}
        if reward:
            breakdown["time"] = reward
        reverse_time_penalty = self._reverse_time_penalty(summary)
        if reverse_time_penalty:
            reward += reverse_time_penalty
            breakdown["reverse_time"] = reverse_time_penalty
        low_speed_time_penalty = self._low_speed_time_penalty(summary)
        if low_speed_time_penalty:
            reward += low_speed_time_penalty
            breakdown["low_speed_time"] = low_speed_time_penalty

        bootstrap_progress_reward = self._bootstrap_progress_reward(
            current_delta_position,
            telemetry,
        )
        if bootstrap_progress_reward:
            reward += bootstrap_progress_reward
            breakdown_name = (
                "bootstrap_progress" if bootstrap_progress_reward > 0.0 else "bootstrap_regress"
            )
            breakdown[breakdown_name] = bootstrap_progress_reward

        milestone_bonus, milestone_speed_bonus = self._milestone_reward(
            max_relative_progress,
            status,
        )
        if milestone_bonus:
            reward += milestone_bonus
            breakdown["milestone"] = milestone_bonus
        if milestone_speed_bonus:
            reward += milestone_speed_bonus
            breakdown["milestone_speed"] = milestone_speed_bonus

        race_laps_completed = self._race_laps_completed(telemetry)
        laps_completed_gain = race_laps_completed - self._awarded_laps_completed
        if laps_completed_gain > 0:
            lap_bonus = 0.0
            lap_position_bonus = 0.0
            for lap_number in range(
                self._awarded_laps_completed + 1,
                race_laps_completed + 1,
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
            self._awarded_laps_completed = race_laps_completed

        if summary.energy_loss_total > 0.0:
            energy_loss_penalty = (
                -summary.energy_loss_total
                * self._weights.energy_loss_penalty_scale
                * self._energy_loss_danger_weight(telemetry)
            )
            reward += energy_loss_penalty
            if energy_loss_penalty:
                breakdown["energy_loss"] = energy_loss_penalty

        self._start_energy_gain_cooldown(summary)
        if summary.energy_gain_total > 0.0:
            energy_gain_reward = self._energy_gain_reward(summary)
            reward += energy_gain_reward
            if energy_gain_reward:
                breakdown["energy_gain"] = energy_gain_reward

        if (
            resolved_action_context.boost_requested
            and boost_active_before_step
            and self._weights.boost_redundant_press_penalty != 0.0
        ):
            reward += self._weights.boost_redundant_press_penalty
            breakdown["boost_redundant_press"] = self._weights.boost_redundant_press_penalty

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
        self._advance_energy_gain_cooldown(summary.frames_run)
        self._previous_boost_active = telemetry.player.boost_timer > 0

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

        if status.termination_reason in {
            "crashed",
            "retired",
            "falling_off_track",
            "energy_depleted",
        }:
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

    def info(self, telemetry: FZeroXTelemetry | None) -> dict[str, object]:
        completed_milestones = max(self._next_milestone_index - 1, 0)
        info: dict[str, object] = {
            "milestones_completed": completed_milestones,
            "next_milestone_index": self._next_milestone_index,
            "milestone_distance": self._weights.milestone_distance,
            "bootstrap_progress_active": self._bootstrap_progress_active(),
            "bootstrap_lap_count": self._weights.bootstrap_lap_count,
            "energy_gain_cooldown_frames_remaining": self._energy_gain_cooldown_frames_remaining,
            "rewarded_laps_completed": self._awarded_laps_completed,
        }
        if telemetry is None:
            return info
        if not telemetry.in_race_mode:
            return info
        self._ensure_progress_origin(telemetry)
        current_relative_progress = self._relative_progress(telemetry.player.race_distance)
        next_milestone_distance = self._next_milestone_index * self._weights.milestone_distance
        info["race_laps_completed"] = self._race_laps_completed(telemetry)
        info["bootstrap_position_multiplier"] = self._bootstrap_position_multiplier(telemetry)
        info["next_milestone_distance"] = next_milestone_distance
        info["distance_to_next_milestone"] = max(
            next_milestone_distance - current_relative_progress,
            0.0,
        )
        info["relative_progress"] = current_relative_progress
        if self._bootstrap_progress_active():
            info["bootstrap_laps_remaining"] = max(
                self._weights.bootstrap_lap_count - self._awarded_laps_completed,
                0.0,
            )
        return info

    def _relative_progress(self, race_distance: float) -> float:
        if not self._has_progress_origin:
            return 0.0
        return max(race_distance - self._progress_origin, 0.0)

    def _progress_delta_position(self, race_distance: float) -> float:
        if not self._has_progress_origin:
            return 0.0
        return race_distance - self._progress_origin

    def _set_progress_origin(self, race_distance: float) -> None:
        self._progress_origin = race_distance
        self._previous_progress_delta_position = 0.0
        self._has_progress_origin = True

    def _ensure_progress_origin(self, telemetry: FZeroXTelemetry) -> None:
        if self._has_progress_origin or not telemetry.in_race_mode:
            return
        self._set_progress_origin(telemetry.player.race_distance)

    def _milestone_index(self, relative_progress: float) -> int:
        if relative_progress <= 0.0:
            return 0
        return int(relative_progress // self._weights.milestone_distance)

    def _milestone_reward(
        self,
        max_relative_progress: float,
        status: StepStatus,
    ) -> tuple[float, float]:
        last_crossed_index = self._milestone_index(max_relative_progress)
        crossed_count = max(0, last_crossed_index - self._next_milestone_index + 1)
        if crossed_count <= 0:
            return 0.0, 0.0
        speed_bonus = self._milestone_speed_bonus(crossed_count, status)
        self._next_milestone_index += crossed_count
        self._last_milestone_step_count = int(status.step_count)
        return crossed_count * self._weights.milestone_bonus, speed_bonus

    def _milestone_speed_bonus(
        self,
        crossed_count: int,
        status: StepStatus,
    ) -> float:
        scale = self._weights.milestone_speed_scale
        cap = self._weights.milestone_speed_bonus_cap
        if crossed_count <= 0 or scale <= 0.0 or cap <= 0.0:
            return 0.0
        frames_elapsed = max(int(status.step_count) - self._last_milestone_step_count, 1)
        crossed_distance = crossed_count * self._weights.milestone_distance
        raw_bonus = scale * crossed_distance / frames_elapsed
        return min(raw_bonus, crossed_count * cap)

    def _bootstrap_progress_active(self) -> bool:
        return (
            self._weights.bootstrap_progress_scale > 0.0
            or self._weights.bootstrap_regress_penalty_scale > 0.0
        ) and self._awarded_laps_completed < self._weights.bootstrap_lap_count

    def _bootstrap_progress_reward(
        self,
        current_delta_position: float,
        telemetry: FZeroXTelemetry,
    ) -> float:
        if not self._bootstrap_progress_active():
            return 0.0

        progress_delta = current_delta_position - self._previous_progress_delta_position
        self._previous_progress_delta_position = current_delta_position
        position_multiplier = self._bootstrap_position_multiplier(telemetry)
        if progress_delta > 0.0 and self._weights.bootstrap_progress_scale > 0.0:
            return progress_delta * self._weights.bootstrap_progress_scale * position_multiplier
        if progress_delta < 0.0 and self._weights.bootstrap_regress_penalty_scale > 0.0:
            return (
                progress_delta * self._weights.bootstrap_regress_penalty_scale * position_multiplier
            )
        return 0.0

    def _bootstrap_position_multiplier(self, telemetry: FZeroXTelemetry) -> float:
        scale = self._weights.bootstrap_position_multiplier_scale
        if scale <= 0.0:
            return 1.0
        total_racers = max(int(telemetry.total_racers), 1)
        if total_racers <= 1:
            return 1.0
        position = min(max(int(telemetry.player.position), 1), total_racers)
        position_score = (total_racers - position) / (total_racers - 1)
        return 1.0 + (position_score * scale)

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
        return remaining_steps * self._weights.remaining_step_penalty_per_frame

    def _reverse_time_penalty(self, summary: StepSummary) -> float:
        extra_scale = self._weights.reverse_time_penalty_scale - 1.0
        if summary.reverse_active_frames <= 0 or extra_scale == 0.0:
            return 0.0
        return summary.reverse_active_frames * self._weights.time_penalty_per_frame * extra_scale

    def _low_speed_time_penalty(self, summary: StepSummary) -> float:
        extra_scale = self._weights.low_speed_time_penalty_scale - 1.0
        if summary.low_speed_frames <= 0 or extra_scale == 0.0:
            return 0.0
        return summary.low_speed_frames * self._weights.time_penalty_per_frame * extra_scale

    def _energy_loss_danger_weight(self, telemetry: FZeroXTelemetry) -> float:
        safe_fraction = max(float(self._weights.energy_loss_safe_fraction), 0.0)
        if safe_fraction <= 0.0:
            return 1.0

        energy_fraction = self._energy_fraction(telemetry)
        if energy_fraction >= safe_fraction:
            return 0.0

        danger = (safe_fraction - energy_fraction) / safe_fraction
        return danger**self._weights.energy_loss_danger_power

    def _energy_fraction(self, telemetry: FZeroXTelemetry) -> float:
        max_energy = float(telemetry.player.max_energy)
        if max_energy <= 0.0:
            return 0.0
        return max(0.0, min(1.0, float(telemetry.player.energy) / max_energy))

    def _start_energy_gain_cooldown(self, summary: StepSummary) -> None:
        cooldown_frames = max(int(self._weights.energy_gain_collision_cooldown_frames), 0)
        if not summary.entered_collision_recoil or cooldown_frames <= 0:
            return
        self._energy_gain_cooldown_frames_remaining = max(
            self._energy_gain_cooldown_frames_remaining,
            cooldown_frames,
        )

    def _energy_gain_reward(self, summary: StepSummary) -> float:
        if self._energy_gain_cooldown_frames_remaining > 0:
            return 0.0
        return summary.energy_gain_total * self._weights.energy_gain_reward_scale

    def _advance_energy_gain_cooldown(self, frames_run: int) -> None:
        self._energy_gain_cooldown_frames_remaining = max(
            self._energy_gain_cooldown_frames_remaining - max(int(frames_run), 0),
            0,
        )

    def _remaining_lap_count(self, telemetry: FZeroXTelemetry) -> int:
        return max(
            telemetry.total_lap_count - self._race_laps_completed(telemetry),
            0,
        )

    def _race_laps_completed(self, telemetry: FZeroXTelemetry) -> int:
        return completed_race_laps(telemetry)

    def _truncation_base_penalty(self, truncation_reason: str) -> float:
        if truncation_reason == "stuck":
            return self._weights.stuck_truncation_base_penalty
        if truncation_reason == "wrong_way":
            return self._weights.wrong_way_truncation_base_penalty
        if truncation_reason == "timeout":
            return self._weights.timeout_truncation_base_penalty
        raise ValueError(f"Unsupported truncation reason: {truncation_reason!r}")
