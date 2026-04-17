# src/rl_fzerox/core/envs/rewards/race_v3/tracker.py
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
from rl_fzerox.core.envs.rewards.progress import (
    DamagePenaltyState,
    EpisodeProgressState,
)
from rl_fzerox.core.envs.rewards.race_v3.weights import RaceV3RewardWeights

_FAILURE_TERMINATION_REASONS = frozenset(
    ("spinning_out", "crashed", "retired", "falling_off_track")
)


class RaceV3RewardTracker:
    """Reward newly covered progress buckets on the game spline."""

    def __init__(
        self,
        weights: RaceV3RewardWeights | None = None,
        *,
        max_episode_steps: int = 12_000,
    ) -> None:
        self._weights = weights or RaceV3RewardWeights()
        self._max_episode_steps = max_episode_steps
        self._progress = EpisodeProgressState()
        self._damage = DamagePenaltyState()
        self._frontier_distance = 0.0
        self._frontier_bucket_index = 0
        self._pending_progress_delta = 0.0
        self._pending_progress_reward = 0.0
        self._pending_energy_refill_progress_bonus = 0.0
        self._pending_progress_frames = 0
        self._energy_gain_cooldown_frames_remaining = 0
        self._rewarded_boost_pad_progress_windows: set[int] = set()
        self._rewarded_full_refill_laps: set[int] = set()
        self._energy_refill_since_full_fraction = 0.0
        self._previous_airborne = False
        self._previous_energy_fraction = 0.0
        self._awarded_laps_completed = 0

    def reset(
        self,
        telemetry: FZeroXTelemetry | None,
        *,
        episode_seed: int | None = None,
    ) -> None:
        """Initialize frontier state for a new episode."""

        del episode_seed
        self._progress.reset(telemetry)
        self._damage.reset()
        self._frontier_distance = 0.0
        self._frontier_bucket_index = 0
        self._clear_pending_progress()
        self._energy_gain_cooldown_frames_remaining = 0
        self._rewarded_boost_pad_progress_windows.clear()
        self._rewarded_full_refill_laps.clear()
        self._energy_refill_since_full_fraction = 0.0
        self._previous_airborne = False if telemetry is None else telemetry.player.airborne
        self._previous_energy_fraction = _normalized_energy(telemetry)
        if telemetry is None or not telemetry.in_race_mode:
            self._awarded_laps_completed = 0
            return
        self._awarded_laps_completed = completed_race_laps(telemetry)

    def summary_config(self) -> RewardSummaryConfig:
        """Return native summary thresholds required by this reward."""

        return RewardSummaryConfig(energy_loss_epsilon=self._weights.energy_loss_epsilon)

    def step_summary(
        self,
        summary: StepSummary,
        status: StepStatus,
        telemetry: FZeroXTelemetry | None,
        action_context: RewardActionContext | None = None,
    ) -> RewardStep:
        """Compute frontier-progress reward from one repeated env step."""

        del action_context
        if telemetry is None or not telemetry.in_race_mode:
            self._progress.reset(None)
            self._damage.reset()
            self._frontier_distance = 0.0
            self._frontier_bucket_index = 0
            self._clear_pending_progress()
            self._energy_gain_cooldown_frames_remaining = 0
            self._rewarded_boost_pad_progress_windows.clear()
            self._rewarded_full_refill_laps.clear()
            self._energy_refill_since_full_fraction = 0.0
            self._previous_airborne = False
            self._previous_energy_fraction = 0.0
            return RewardStep(reward=0.0)

        self._progress.ensure_origin(telemetry)
        self._start_energy_gain_cooldown(summary)
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

        progress_reward, energy_refill_progress_bonus = self._progress_reward(
            summary,
            status,
            telemetry,
        )
        if progress_reward:
            reward += progress_reward
            breakdown["frontier_progress"] = progress_reward
        if energy_refill_progress_bonus:
            reward += energy_refill_progress_bonus
            breakdown["energy_refill_progress"] = energy_refill_progress_bonus

        lap_reward = self._lap_rewards(telemetry, breakdown)
        if lap_reward:
            reward += lap_reward

        boost_pad_reward = self._boost_pad_reward(summary)
        if boost_pad_reward:
            reward += boost_pad_reward
            breakdown["boost_pad"] = boost_pad_reward

        self._accumulate_refill_since_full(summary, telemetry)
        full_refill_reward = self._full_refill_lap_reward(summary, telemetry)
        if full_refill_reward:
            reward += full_refill_reward
            breakdown["energy_full_refill_lap"] = full_refill_reward

        landing_reward = self._landing_reward(telemetry)
        if landing_reward:
            reward += landing_reward
            breakdown["landing"] = landing_reward

        damage_penalty = self._damage.penalty(
            summary,
            frame_penalty=self._weights.damage_taken_frame_penalty,
            streak_ramp_penalty=self._weights.damage_taken_streak_ramp_penalty,
            streak_cap_frames=self._weights.damage_taken_streak_cap_frames,
        )
        if damage_penalty:
            reward += damage_penalty
            breakdown["damage_taken"] = damage_penalty

        reward += apply_event_penalty(
            summary.entered_collision_recoil,
            self._weights.collision_recoil_penalty,
            "collision_recoil",
            breakdown,
        )

        terminal_penalty = self._terminal_or_truncation_penalty(status, breakdown)
        if terminal_penalty:
            reward += terminal_penalty

        self._advance_energy_gain_cooldown(summary.frames_run)
        if _normalized_energy(telemetry) >= 1.0:
            self._energy_refill_since_full_fraction = 0.0
        self._previous_airborne = telemetry.player.airborne
        self._previous_energy_fraction = _normalized_energy(telemetry)
        return RewardStep(reward=reward, breakdown=breakdown)

    def info(self, telemetry: FZeroXTelemetry | None) -> dict[str, object]:
        """Expose lightweight frontier reward state for watch/logging."""

        info: dict[str, object] = {
            "reward_profile": "race_v3",
            "frontier_progress_distance": self._frontier_distance,
            "frontier_progress_bucket_index": self._frontier_bucket_index,
            "progress_bucket_distance": self._weights.progress_bucket_distance,
            "progress_bucket_reward": self._weights.progress_bucket_reward,
            "progress_reward_interval_frames": self._weights.progress_reward_interval_frames,
            "pending_progress_reward_delta": self._pending_progress_delta,
            "pending_progress_reward_frames": self._pending_progress_frames,
            "energy_gain_cooldown_frames_remaining": (self._energy_gain_cooldown_frames_remaining),
            "boost_pad_reward_progress_window": self._weights.boost_pad_reward_progress_window,
            "rewarded_boost_pad_progress_windows": len(self._rewarded_boost_pad_progress_windows),
            "rewarded_full_refill_laps": len(self._rewarded_full_refill_laps),
            "energy_refill_since_full_fraction": self._energy_refill_since_full_fraction,
            "damage_taken_streak_frames": self._damage.streak_frames,
            "rewarded_laps_completed": self._awarded_laps_completed,
        }
        if telemetry is None or not telemetry.in_race_mode:
            return info
        self._progress.ensure_origin(telemetry)
        info["relative_progress"] = self._progress.relative_distance(telemetry.player.race_distance)
        info["race_laps_completed"] = completed_race_laps(telemetry)
        return info

    def _progress_reward(
        self,
        summary: StepSummary,
        status: StepStatus,
        telemetry: FZeroXTelemetry,
    ) -> tuple[float, float]:
        max_relative_progress = self._progress.relative_distance(summary.max_race_distance)
        bucket_distance = self._weights.progress_bucket_distance
        if bucket_distance <= 0.0:
            return 0.0, 0.0
        current_bucket_index = int(max_relative_progress // bucket_distance)
        crossed_bucket_count = current_bucket_index - self._frontier_bucket_index
        if crossed_bucket_count <= 0:
            return 0.0, 0.0
        self._frontier_bucket_index = current_bucket_index
        self._frontier_distance = current_bucket_index * bucket_distance
        progress_reward = crossed_bucket_count * self._weights.progress_bucket_reward
        energy_refill_progress_bonus = self._energy_refill_progress_bonus(
            progress_reward,
            summary,
            telemetry,
        )
        interval_frames = max(int(self._weights.progress_reward_interval_frames), 1)
        if interval_frames <= 1:
            return progress_reward, energy_refill_progress_bonus

        self._pending_progress_delta += crossed_bucket_count * bucket_distance
        self._pending_progress_reward += progress_reward
        self._pending_energy_refill_progress_bonus += energy_refill_progress_bonus
        self._pending_progress_frames += max(int(summary.frames_run), 0)
        if (
            self._pending_progress_frames < interval_frames
            and status.termination_reason is None
            and status.truncation_reason is None
        ):
            return 0.0, 0.0

        pending_reward = self._pending_progress_reward
        pending_refill_bonus = self._pending_energy_refill_progress_bonus
        self._clear_pending_progress()
        return pending_reward, pending_refill_bonus

    def _energy_refill_progress_bonus(
        self,
        progress_reward: float,
        summary: StepSummary,
        telemetry: FZeroXTelemetry,
    ) -> float:
        scale = self._weights.energy_gain_reward_scale
        if (
            progress_reward <= 0.0
            or scale <= 0.0
            or summary.energy_gain_total <= 0.0
            or summary.reverse_active_frames > 0
            or self._energy_gain_cooldown_frames_remaining > 0
        ):
            return 0.0

        max_energy = float(telemetry.player.max_energy)
        if max_energy <= 0.0:
            return 0.0
        energy_gain_fraction = max(float(summary.energy_gain_total), 0.0) / max_energy
        missing_energy_fraction = max(1.0 - self._previous_energy_fraction, 0.0)
        return progress_reward * scale * energy_gain_fraction * missing_energy_fraction

    def _clear_pending_progress(self) -> None:
        self._pending_progress_delta = 0.0
        self._pending_progress_reward = 0.0
        self._pending_energy_refill_progress_bonus = 0.0
        self._pending_progress_frames = 0

    def _start_energy_gain_cooldown(self, summary: StepSummary) -> None:
        cooldown_frames = max(int(self._weights.energy_gain_collision_cooldown_frames), 0)
        if cooldown_frames <= 0:
            return
        if not summary.entered_collision_recoil and summary.damage_taken_frames <= 0:
            return
        self._energy_gain_cooldown_frames_remaining = max(
            self._energy_gain_cooldown_frames_remaining,
            cooldown_frames,
        )

    def _advance_energy_gain_cooldown(self, frames_run: int) -> None:
        self._energy_gain_cooldown_frames_remaining = max(
            self._energy_gain_cooldown_frames_remaining - max(int(frames_run), 0),
            0,
        )

    def _boost_pad_reward(self, summary: StepSummary) -> float:
        reward = self._weights.boost_pad_reward
        if reward <= 0.0 or not summary.entered_dash_pad_boost or summary.reverse_active_frames > 0:
            return 0.0
        progress_window = self._weights.boost_pad_reward_progress_window
        if progress_window <= 0.0:
            return 0.0
        relative_progress = self._progress.relative_distance(summary.max_race_distance)
        window_index = int(relative_progress // progress_window)
        if window_index in self._rewarded_boost_pad_progress_windows:
            return 0.0
        self._rewarded_boost_pad_progress_windows.add(window_index)
        return reward

    def _accumulate_refill_since_full(
        self,
        summary: StepSummary,
        telemetry: FZeroXTelemetry,
    ) -> None:
        current_energy_fraction = _normalized_energy(telemetry)
        if self._previous_energy_fraction >= 1.0 and current_energy_fraction >= 1.0:
            self._energy_refill_since_full_fraction = 0.0
            return
        max_energy = float(telemetry.player.max_energy)
        if max_energy <= 0.0 or summary.energy_gain_total <= 0.0:
            return
        self._energy_refill_since_full_fraction = min(
            self._energy_refill_since_full_fraction
            + (max(float(summary.energy_gain_total), 0.0) / max_energy),
            1.0,
        )

    def _full_refill_lap_reward(
        self,
        summary: StepSummary,
        telemetry: FZeroXTelemetry,
    ) -> float:
        reward = self._weights.energy_full_refill_lap_bonus
        if (
            reward <= 0.0
            or summary.energy_gain_total <= 0.0
            or summary.reverse_active_frames > 0
            or telemetry.player.reverse_timer > 0
            or self._energy_gain_cooldown_frames_remaining > 0
        ):
            return 0.0
        current_energy_fraction = _normalized_energy(telemetry)
        if self._previous_energy_fraction >= 1.0 or current_energy_fraction < 1.0:
            return 0.0
        recovered_fraction = self._energy_refill_since_full_fraction
        if recovered_fraction < self._weights.energy_full_refill_min_gain_fraction:
            return 0.0

        lap_index = completed_race_laps(telemetry)
        if lap_index in self._rewarded_full_refill_laps:
            return 0.0
        self._rewarded_full_refill_laps.add(lap_index)
        return reward * min(recovered_fraction, 1.0)

    def _landing_reward(self, telemetry: FZeroXTelemetry) -> float:
        reward = self._weights.airborne_landing_reward
        if reward == 0.0:
            return 0.0
        if not self._previous_airborne or telemetry.player.airborne:
            return 0.0
        return reward

    def _lap_rewards(
        self,
        telemetry: FZeroXTelemetry,
        breakdown: dict[str, float],
    ) -> float:
        race_laps_completed = completed_race_laps(telemetry)
        laps_completed_gain = race_laps_completed - self._awarded_laps_completed
        if laps_completed_gain <= 0:
            return 0.0

        lap_bonus = laps_completed_gain * self._weights.lap_completion_bonus
        lap_position_bonus = laps_completed_gain * finish_placement_bonus(
            position=telemetry.player.position,
            total_racers=telemetry.total_racers,
            scale=self._weights.lap_position_scale,
        )

        if lap_bonus:
            breakdown["lap_completion"] = lap_bonus
        if lap_position_bonus:
            breakdown["lap_position"] = lap_position_bonus
        self._awarded_laps_completed = race_laps_completed
        return lap_bonus + lap_position_bonus

    def _terminal_or_truncation_penalty(
        self,
        status: StepStatus,
        breakdown: dict[str, float],
    ) -> float:
        if status.termination_reason == "finished":
            return 0.0
        if status.termination_reason is not None:
            if status.termination_reason not in _FAILURE_TERMINATION_REASONS:
                return 0.0
            penalty = self._weights.failure_penalty
            breakdown[status.termination_reason] = penalty
            return penalty
        if status.truncation_reason is None:
            return 0.0
        penalty = self._weights.truncation_penalty
        breakdown[f"{status.truncation_reason}_truncation"] = penalty
        return penalty

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


def _normalized_energy(telemetry: FZeroXTelemetry | None) -> float:
    if telemetry is None or not telemetry.in_race_mode:
        return 0.0
    max_energy = float(telemetry.player.max_energy)
    if max_energy <= 0.0:
        return 0.0
    return max(0.0, min(1.0, float(telemetry.player.energy) / max_energy))
