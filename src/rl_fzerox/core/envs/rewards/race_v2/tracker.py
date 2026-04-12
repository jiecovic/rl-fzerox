# src/rl_fzerox/core/envs/rewards/race_v2/tracker.py
from __future__ import annotations

from fzerox_emulator import FZeroXTelemetry, StepStatus, StepSummary
from rl_fzerox.core.envs.laps import completed_race_laps
from rl_fzerox.core.envs.rewards.common import (
    RewardActionContext,
    RewardStep,
    RewardSummaryConfig,
    apply_event_penalty,
    finish_placement_bonus,
)
from rl_fzerox.core.envs.rewards.race_v2.energy import (
    EnergyRewardState,
    energy_loss_danger_weight,
)
from rl_fzerox.core.envs.rewards.race_v2.weights import RaceV2RewardWeights
from rl_fzerox.core.seed import normalize_seed

_U64_UNIT = float(1 << 64)


class RaceV2RewardTracker:
    """Track episode reward state for the simplified `race_v2` profile.

    The tracker keeps compact cross-step reward state; per-step deltas like
    reverse-active frames, energy loss, and entered flags are pre-aggregated by
    the native repeated-step path.
    """

    def __init__(
        self,
        weights: RaceV2RewardWeights | None = None,
        *,
        max_episode_steps: int = 12_000,
    ) -> None:
        self._weights = weights or RaceV2RewardWeights()
        self._next_milestone_index = 1
        self._milestone_phase_offset = 0.0
        self._last_milestone_step_count = 0
        self._awarded_laps_completed = 0
        self._previous_progress_delta_position = 0.0
        self._progress_origin = 0.0
        self._has_progress_origin = False
        self._max_episode_steps = max_episode_steps
        self._energy = EnergyRewardState()
        self._boost_pad_reward_cooldown_frames_remaining = 0
        self._previous_airborne = False

    def reset(
        self,
        telemetry: FZeroXTelemetry | None,
        *,
        episode_seed: int | None = None,
    ) -> None:
        """Initialize reward state for a new episode."""

        self._next_milestone_index = 1
        self._milestone_phase_offset = self._resolve_milestone_phase_offset(episode_seed)
        self._last_milestone_step_count = 0
        if telemetry is None or not telemetry.in_race_mode:
            self._awarded_laps_completed = 0
            self._previous_progress_delta_position = 0.0
            self._progress_origin = 0.0
            self._has_progress_origin = False
            self._energy.reset(None)
            self._boost_pad_reward_cooldown_frames_remaining = 0
            self._previous_airborne = False
            return
        self._awarded_laps_completed = self._race_laps_completed(telemetry)
        self._previous_progress_delta_position = 0.0
        self._set_progress_origin(telemetry.player.race_distance)
        self._energy.reset(telemetry)
        self._boost_pad_reward_cooldown_frames_remaining = 0
        self._previous_airborne = telemetry.player.airborne

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
            self._energy.reset(None)
            self._boost_pad_reward_cooldown_frames_remaining = 0
            self._previous_airborne = False
            return RewardStep(reward=0.0)

        resolved_action_context = action_context or RewardActionContext()
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
                * energy_loss_danger_weight(telemetry, self._weights)
            )
            reward += energy_loss_penalty
            if energy_loss_penalty:
                breakdown["energy_loss"] = energy_loss_penalty

        self._energy.start_gain_cooldown(summary, self._weights)
        if summary.energy_gain_total > 0.0:
            energy_gain_reward = self._energy.energy_gain_reward(summary, self._weights)
            reward += energy_gain_reward
            if energy_gain_reward:
                breakdown["energy_gain"] = energy_gain_reward

        self._energy.advance_cooldowns(summary.frames_run)
        full_refill_bonus = self._energy.full_refill_bonus(telemetry, self._weights)
        if full_refill_bonus:
            reward += full_refill_bonus
            breakdown["energy_full_refill"] = full_refill_bonus

        landing_reward = self._landing_reward(telemetry)
        if landing_reward:
            reward += landing_reward
            breakdown["landing"] = landing_reward

        if (
            resolved_action_context.air_brake_requested
            and not self._previous_airborne
            and self._weights.grounded_air_brake_penalty != 0.0
        ):
            reward += self._weights.grounded_air_brake_penalty
            breakdown["grounded_air_brake"] = self._weights.grounded_air_brake_penalty

        self._advance_boost_pad_reward_cooldown(summary.frames_run)
        boost_pad_reward = self._boost_pad_reward(summary)
        if boost_pad_reward:
            reward += boost_pad_reward
            breakdown["boost_pad"] = boost_pad_reward

        if (
            resolved_action_context.boost_requested
            and self._weights.boost_press_penalty != 0.0
        ):
            reward += self._weights.boost_press_penalty
            breakdown["boost_press"] = self._weights.boost_press_penalty

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
        self._energy.update_previous(telemetry)
        self._previous_airborne = telemetry.player.airborne

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
            "milestone_phase_offset": self._milestone_phase_offset,
            "bootstrap_progress_active": self._bootstrap_progress_active(),
            "bootstrap_lap_count": self._weights.bootstrap_lap_count,
            "energy_gain_cooldown_frames_remaining": (
                self._energy.gain_cooldown_frames_remaining
            ),
            "energy_full_refill_cooldown_frames_remaining": (
                self._energy.full_refill_cooldown_frames_remaining
            ),
            "boost_pad_reward_cooldown_frames_remaining": (
                self._boost_pad_reward_cooldown_frames_remaining
            ),
            "rewarded_laps_completed": self._awarded_laps_completed,
        }
        if telemetry is None:
            return info
        if not telemetry.in_race_mode:
            return info
        self._ensure_progress_origin(telemetry)
        current_relative_progress = self._relative_progress(telemetry.player.race_distance)
        next_milestone_distance = self._next_milestone_distance()
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
        if self._milestone_phase_offset <= 0.0:
            return int(relative_progress // self._weights.milestone_distance)
        if relative_progress < self._milestone_phase_offset:
            return 0
        return int(
            ((relative_progress - self._milestone_phase_offset) // self._weights.milestone_distance)
            + 1
        )

    def _next_milestone_distance(self) -> float:
        if self._milestone_phase_offset <= 0.0:
            return self._next_milestone_index * self._weights.milestone_distance
        return self._milestone_phase_offset + (
            (self._next_milestone_index - 1) * self._weights.milestone_distance
        )

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

    def _advance_boost_pad_reward_cooldown(self, frames_run: int) -> None:
        self._boost_pad_reward_cooldown_frames_remaining = max(
            self._boost_pad_reward_cooldown_frames_remaining - max(int(frames_run), 0),
            0,
        )

    def _boost_pad_reward(self, summary: StepSummary) -> float:
        reward = self._weights.boost_pad_reward
        if (
            reward <= 0.0
            or not summary.entered_dash_pad_boost
            or self._boost_pad_reward_cooldown_frames_remaining > 0
        ):
            return 0.0
        self._boost_pad_reward_cooldown_frames_remaining = max(
            int(self._weights.boost_pad_reward_cooldown_frames),
            0,
        )
        return reward

    def _landing_reward(self, telemetry: FZeroXTelemetry) -> float:
        reward = self._weights.airborne_landing_reward
        if reward == 0.0:
            return 0.0
        if not self._previous_airborne or telemetry.player.airborne:
            return 0.0
        return reward

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
        if truncation_reason == "progress_stalled":
            return self._weights.progress_stalled_truncation_base_penalty
        if truncation_reason == "timeout":
            return self._weights.timeout_truncation_base_penalty
        raise ValueError(f"Unsupported truncation reason: {truncation_reason!r}")

    def _resolve_milestone_phase_offset(self, episode_seed: int | None) -> float:
        if not self._weights.randomize_milestone_phase_on_reset:
            return 0.0
        if episode_seed is None:
            return 0.0
        return (normalize_seed(episode_seed) / _U64_UNIT) * self._weights.milestone_distance
