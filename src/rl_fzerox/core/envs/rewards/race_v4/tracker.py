# src/rl_fzerox/core/envs/rewards/race_v4/tracker.py
from __future__ import annotations

from fzerox_emulator import FZeroXTelemetry, StepStatus, StepSummary
from rl_fzerox.core.envs.laps import completed_race_laps
from rl_fzerox.core.envs.rewards.common import (
    RewardActionContext,
    RewardStep,
    RewardSummaryConfig,
    finish_placement_bonus,
)
from rl_fzerox.core.envs.rewards.progress import (
    DamagePenaltyState,
    EpisodeProgressState,
    lap_completion_bonus,
    remaining_lap_count,
    remaining_step_penalty,
    truncation_base_penalty,
)
from rl_fzerox.core.envs.rewards.race_v4.weights import RaceV4RewardWeights
from rl_fzerox.core.seed import normalize_seed

_U64_UNIT = float(1 << 64)


class RaceV4RewardTracker:
    """Checkpoint-style reward: fixed milestones plus explicit frame cost."""

    def __init__(
        self,
        weights: RaceV4RewardWeights | None = None,
        *,
        max_episode_steps: int = 12_000,
    ) -> None:
        self._weights = weights or RaceV4RewardWeights()
        self._max_episode_steps = max_episode_steps
        self._progress = EpisodeProgressState()
        self._damage = DamagePenaltyState()
        self._next_milestone_index = 1
        self._milestone_phase_offset = 0.0
        self._awarded_laps_completed = 0

    def reset(
        self,
        telemetry: FZeroXTelemetry | None,
        *,
        episode_seed: int | None = None,
    ) -> None:
        """Initialize milestone state for a new episode."""

        self._progress.reset(telemetry)
        self._damage.reset()
        self._next_milestone_index = 1
        self._milestone_phase_offset = self._resolve_milestone_phase_offset(episode_seed)
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
        """Compute milestone/time-pressure reward from one repeated env step."""

        del action_context
        if telemetry is None or not telemetry.in_race_mode:
            self._progress.reset(None)
            self._damage.reset()
            self._next_milestone_index = 1
            return RewardStep(reward=0.0)

        self._progress.ensure_origin(telemetry)
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

        milestone_reward = self._milestone_reward(summary)
        if milestone_reward:
            reward += milestone_reward
            breakdown["milestone"] = milestone_reward

        lap_reward = self._lap_rewards(telemetry, breakdown)
        if lap_reward:
            reward += lap_reward

        damage_penalty = self._damage.penalty(
            summary,
            frame_penalty=self._weights.damage_taken_frame_penalty,
            streak_ramp_penalty=self._weights.damage_taken_streak_ramp_penalty,
            streak_cap_frames=self._weights.damage_taken_streak_cap_frames,
        )
        if damage_penalty:
            reward += damage_penalty
            breakdown["damage_taken"] = damage_penalty

        terminal_penalty = self._terminal_or_truncation_penalty(status, telemetry, breakdown)
        if terminal_penalty:
            reward += terminal_penalty

        if status.termination_reason == "finished" and self._weights.finish_position_scale:
            placement_bonus = finish_placement_bonus(
                position=telemetry.player.position,
                total_racers=telemetry.total_racers,
                scale=self._weights.finish_position_scale,
            )
            if placement_bonus:
                reward += placement_bonus
                breakdown["finish_position"] = placement_bonus

        return RewardStep(reward=reward, breakdown=breakdown)

    def info(self, telemetry: FZeroXTelemetry | None) -> dict[str, object]:
        """Expose lightweight milestone reward state for watch/logging."""

        completed_milestones = max(self._next_milestone_index - 1, 0)
        info: dict[str, object] = {
            "reward_profile": "race_v4",
            "milestones_completed": completed_milestones,
            "next_milestone_index": self._next_milestone_index,
            "milestone_distance": self._weights.milestone_distance,
            "milestone_phase_offset": self._milestone_phase_offset,
            "damage_taken_streak_frames": self._damage.streak_frames,
            "rewarded_laps_completed": self._awarded_laps_completed,
        }
        if telemetry is None or not telemetry.in_race_mode:
            return info
        self._progress.ensure_origin(telemetry)
        current_relative_progress = self._progress.relative_distance(
            telemetry.player.race_distance
        )
        next_milestone_distance = self._next_milestone_distance()
        info["race_laps_completed"] = completed_race_laps(telemetry)
        info["next_milestone_distance"] = next_milestone_distance
        info["distance_to_next_milestone"] = max(
            next_milestone_distance - current_relative_progress,
            0.0,
        )
        info["relative_progress"] = current_relative_progress
        return info

    def _milestone_reward(self, summary: StepSummary) -> float:
        max_relative_progress = self._progress.relative_distance(summary.max_race_distance)
        last_crossed_index = self._milestone_index(max_relative_progress)
        crossed_count = max(0, last_crossed_index - self._next_milestone_index + 1)
        if crossed_count <= 0:
            return 0.0
        self._next_milestone_index += crossed_count
        return crossed_count * self._weights.milestone_bonus

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

    def _lap_rewards(
        self,
        telemetry: FZeroXTelemetry,
        breakdown: dict[str, float],
    ) -> float:
        race_laps_completed = completed_race_laps(telemetry)
        laps_completed_gain = race_laps_completed - self._awarded_laps_completed
        if laps_completed_gain <= 0:
            return 0.0

        lap_bonus = 0.0
        lap_position_bonus = 0.0
        for lap_number in range(
            self._awarded_laps_completed + 1,
            race_laps_completed + 1,
        ):
            lap_bonus += lap_completion_bonus(
                lap_number=lap_number,
                total_lap_count=telemetry.total_lap_count,
                lap_1_completion_bonus=self._weights.lap_1_completion_bonus,
                lap_2_completion_bonus=self._weights.lap_2_completion_bonus,
                final_lap_completion_bonus=self._weights.final_lap_completion_bonus,
            )
            lap_position_bonus += finish_placement_bonus(
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
        telemetry: FZeroXTelemetry,
        breakdown: dict[str, float],
    ) -> float:
        if status.termination_reason == "finished":
            return 0.0
        if status.termination_reason is not None:
            penalty = self._dynamic_penalty(
                self._weights.terminal_failure_base_penalty,
                status,
                telemetry,
            )
            breakdown[status.termination_reason] = penalty
            return penalty
        if status.truncation_reason is None:
            return 0.0
        base_penalty = truncation_base_penalty(
            status.truncation_reason,
            stuck_truncation_base_penalty=self._weights.stuck_truncation_base_penalty,
            wrong_way_truncation_base_penalty=self._weights.wrong_way_truncation_base_penalty,
            progress_stalled_truncation_base_penalty=(
                self._weights.progress_stalled_truncation_base_penalty
            ),
            timeout_truncation_base_penalty=self._weights.timeout_truncation_base_penalty,
        )
        penalty = self._dynamic_penalty(base_penalty, status, telemetry)
        breakdown[f"{status.truncation_reason}_truncation"] = penalty
        return penalty

    def _dynamic_penalty(
        self,
        base_penalty: float,
        status: StepStatus,
        telemetry: FZeroXTelemetry,
    ) -> float:
        return (
            base_penalty
            - remaining_step_penalty(
                status=status,
                max_episode_steps=self._max_episode_steps,
                remaining_step_penalty_per_frame=self._weights.remaining_step_penalty_per_frame,
            )
            - (remaining_lap_count(telemetry) * self._weights.remaining_lap_penalty)
        )

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

    def _resolve_milestone_phase_offset(self, episode_seed: int | None) -> float:
        if not self._weights.randomize_milestone_phase_on_reset:
            return 0.0
        if episode_seed is None:
            return 0.0
        return (normalize_seed(episode_seed) / _U64_UNIT) * self._weights.milestone_distance
