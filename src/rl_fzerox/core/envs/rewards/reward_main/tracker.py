# src/rl_fzerox/core/envs/rewards/reward_main/tracker.py
from __future__ import annotations

from fzerox_emulator import FZeroXTelemetry, StepStatus, StepSummary
from rl_fzerox.core.envs.course_effects import CourseEffect, course_effect_raw
from rl_fzerox.core.envs.laps import completed_race_laps
from rl_fzerox.core.envs.rewards.common import (
    RewardActionContext,
    RewardStep,
    RewardSummaryConfig,
    apply_event_penalty,
)
from rl_fzerox.core.envs.rewards.progress import DamagePenaltyState
from rl_fzerox.core.envs.rewards.race_v3.bounds import (
    cap_outside_bounds_reentry_reward,
)
from rl_fzerox.core.envs.rewards.race_v3.events import (
    BadSurfaceEntryPenaltyTracker,
    BoostPadRewardTracker,
    LapRewardTracker,
    landing_reward,
    terminal_or_truncation_penalty,
)
from rl_fzerox.core.envs.rewards.race_v3.progress import (
    FrontierProgressRewardTracker,
    FrontierReward,
)
from rl_fzerox.core.envs.rewards.reward_main.controls import (
    air_brake_request_penalty,
    airborne_pitch_up_penalty,
    lean_request_penalty,
    manual_boost_reward,
    outside_track_frame_penalty,
)
from rl_fzerox.core.envs.rewards.reward_main.energy import EnergyRefillRewardTracker
from rl_fzerox.core.envs.rewards.reward_main.weights import RewardMainWeights
from rl_fzerox.core.envs.track_bounds import telemetry_outside_track_bounds


class RewardMainTracker:
    """Reward newly covered progress buckets on the game spline."""

    def __init__(
        self,
        weights: RewardMainWeights | None = None,
        *,
        course_weights: dict[str, RewardMainWeights] | None = None,
        max_episode_steps: int = 12_000,
    ) -> None:
        self._base_weights = weights or RewardMainWeights()
        self._course_weights = course_weights or {}
        self._weights = self._base_weights
        self._max_episode_steps = max_episode_steps
        self._progress = FrontierProgressRewardTracker()
        self._energy = EnergyRefillRewardTracker()
        self._boost_pads = BoostPadRewardTracker()
        self._bad_surfaces = BadSurfaceEntryPenaltyTracker()
        self._laps = LapRewardTracker()
        self._damage = DamagePenaltyState()
        self._previous_airborne = False
        self._deferred_outside_bounds_progress = False
        self._course_id: str | None = None

    def reset(
        self,
        telemetry: FZeroXTelemetry | None,
        *,
        episode_seed: int | None = None,
        course_id: str | None = None,
    ) -> None:
        """Initialize frontier state for a new episode."""

        del episode_seed
        self._course_id = course_id
        self._weights = self._course_weights.get(course_id or "", self._base_weights)
        self._progress.reset(telemetry)
        self._energy.reset(telemetry)
        self._boost_pads.reset()
        self._bad_surfaces.reset(telemetry)
        self._laps.reset(telemetry)
        self._damage.reset()
        self._previous_airborne = False if telemetry is None else telemetry.player.airborne
        self._deferred_outside_bounds_progress = telemetry_outside_track_bounds(telemetry)

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

        if telemetry is None or not telemetry.in_race_mode:
            self._progress.reset_inactive()
            self._energy.reset_inactive()
            self._boost_pads.reset()
            self._bad_surfaces.reset(None)
            self._laps.reset(None)
            self._damage.reset()
            self._previous_airborne = False
            self._deferred_outside_bounds_progress = False
            return RewardStep(reward=0.0)

        self._progress.ensure_origin(telemetry)
        outside_track_bounds = telemetry_outside_track_bounds(telemetry)
        defer_outside_bounds_progress = outside_track_bounds or (
            self._deferred_outside_bounds_progress and telemetry.player.airborne
        )
        returning_grounded_inside_track_bounds = (
            self._deferred_outside_bounds_progress
            and not outside_track_bounds
            and not telemetry.player.airborne
        )
        self._energy.start_collision_cooldown(summary, weights=self._weights)
        reward = summary.frames_run * self._weights.time_penalty_per_frame
        breakdown: dict[str, float] = {}
        if reward:
            breakdown["time"] = reward

        reverse_time_penalty = self._reverse_time_penalty(summary)
        if reverse_time_penalty:
            reward += reverse_time_penalty
            breakdown["reverse_time"] = reverse_time_penalty
        slow_speed_penalty = self._slow_speed_time_penalty(summary, telemetry)
        if slow_speed_penalty:
            reward += slow_speed_penalty
            breakdown["slow_speed_time"] = slow_speed_penalty

        frontier_distance_before_step = self._progress.frontier_distance
        ground_effect_key, progress_multiplier = _ground_effect_progress_modifier(
            telemetry,
            weights=self._weights,
        )
        frontier_reward = self._frontier_reward(
            summary,
            status,
            telemetry,
            progress_multiplier,
            outside_track_bounds=outside_track_bounds,
            defer_outside_bounds_progress=defer_outside_bounds_progress,
            returning_grounded_inside_track_bounds=returning_grounded_inside_track_bounds,
        )
        if frontier_reward.progress:
            reward += frontier_reward.progress
            breakdown["frontier_progress"] = frontier_reward.progress
        if frontier_reward.ground_effect_adjustment:
            reward += frontier_reward.ground_effect_adjustment
            breakdown[f"{ground_effect_key}_progress"] = frontier_reward.ground_effect_adjustment
        if frontier_reward.energy_refill_bonus:
            reward += frontier_reward.energy_refill_bonus
            breakdown["energy_refill_progress"] = frontier_reward.energy_refill_bonus
        offtrack_penalty = outside_track_frame_penalty(
            summary,
            weights=self._weights,
            outside_track_bounds=outside_track_bounds,
        )
        if offtrack_penalty:
            reward += offtrack_penalty
            breakdown["outside_track"] = offtrack_penalty

        lap_reward = self._laps.reward(
            telemetry,
            weights=self._weights,
            breakdown=breakdown,
        )
        if lap_reward:
            reward += lap_reward

        boost_pad_reward = self._boost_pads.reward(
            summary,
            relative_progress=self._progress.relative_distance(summary.max_race_distance),
            frontier_distance_before_step=frontier_distance_before_step,
            weights=self._weights,
        )
        if boost_pad_reward:
            reward += boost_pad_reward
            breakdown["boost_pad"] = boost_pad_reward

        bad_surface_penalty = self._bad_surfaces.penalty(
            telemetry,
            weights=self._weights,
            breakdown=breakdown,
        )
        if bad_surface_penalty:
            reward += bad_surface_penalty

        air_brake_penalty = air_brake_request_penalty(
            summary,
            action_context,
            weights=self._weights,
        )
        if air_brake_penalty:
            reward += air_brake_penalty
            breakdown["air_brake"] = air_brake_penalty

        lean_penalty = lean_request_penalty(
            summary,
            action_context,
            weights=self._weights,
        )
        if lean_penalty:
            reward += lean_penalty
            breakdown["lean"] = lean_penalty

        pitch_penalty = airborne_pitch_up_penalty(
            summary,
            telemetry,
            action_context,
            weights=self._weights,
        )
        if pitch_penalty:
            reward += pitch_penalty
            breakdown["airborne_pitch_up"] = pitch_penalty

        boost_reward = manual_boost_reward(action_context, weights=self._weights)
        if boost_reward:
            reward += boost_reward
            breakdown["manual_boost"] = boost_reward

        landing = landing_reward(
            previous_airborne=self._previous_airborne,
            telemetry=telemetry,
            weights=self._weights,
        )
        if landing:
            reward += landing
            breakdown["landing"] = landing

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

        terminal_penalty = terminal_or_truncation_penalty(
            status,
            weights=self._weights,
            breakdown=breakdown,
        )
        if terminal_penalty:
            reward += terminal_penalty
        raw_reward = reward
        reward = _clip_step_reward(raw_reward, weights=self._weights)
        if reward != raw_reward:
            breakdown["step_reward_clip"] = reward - raw_reward

        self._energy.advance_cooldown(summary.frames_run)
        self._energy.finish_step(telemetry)
        self._previous_airborne = telemetry.player.airborne
        self._deferred_outside_bounds_progress = (
            defer_outside_bounds_progress and not returning_grounded_inside_track_bounds
        )
        return RewardStep(reward=reward, breakdown=breakdown, raw_reward=raw_reward)

    def info(self, telemetry: FZeroXTelemetry | None) -> dict[str, object]:
        """Expose lightweight frontier reward state for watch/logging."""

        info: dict[str, object] = {
            "reward_profile": "reward_main",
            "boost_pad_reward_progress_window": self._weights.boost_pad_reward_progress_window,
            "rewarded_boost_pad_progress_windows": self._boost_pads.rewarded_window_count,
            "damage_taken_streak_frames": self._damage.streak_frames,
            "rewarded_laps_completed": self._laps.awarded_laps_completed,
        }
        info.update(self._progress.info(telemetry, weights=self._weights))
        info.update(self._energy.info())
        if telemetry is None or not telemetry.in_race_mode:
            return info
        self._progress.ensure_origin(telemetry)
        info["race_laps_completed"] = completed_race_laps(telemetry)
        info["progress_reward_outside_track_bounds"] = telemetry_outside_track_bounds(telemetry)
        info["progress_reward_deferred_outside_track_bounds"] = (
            self._deferred_outside_bounds_progress
        )
        return info

    def _reverse_time_penalty(self, summary: StepSummary) -> float:
        extra_scale = self._weights.reverse_time_penalty_scale - 1.0
        if summary.reverse_active_frames <= 0 or extra_scale == 0.0:
            return 0.0
        return summary.reverse_active_frames * self._weights.time_penalty_per_frame * extra_scale

    def _slow_speed_time_penalty(self, summary: StepSummary, telemetry: FZeroXTelemetry) -> float:
        scale = self._weights.slow_speed_time_penalty_scale
        start_kph = self._weights.slow_speed_time_penalty_start_kph
        if scale == 0.0 or start_kph <= 0.0 or self._weights.time_penalty_per_frame == 0.0:
            return 0.0
        speed_deficit = max(start_kph - float(telemetry.player.speed_kph), 0.0)
        if speed_deficit <= 0.0:
            return 0.0
        deficit_fraction = speed_deficit / start_kph
        shaped_deficit = deficit_fraction**self._weights.slow_speed_time_penalty_power
        return (
            max(int(summary.frames_run), 1)
            * self._weights.time_penalty_per_frame
            * scale
            * shaped_deficit
        )

    def _frontier_reward(
        self,
        summary: StepSummary,
        status: StepStatus,
        telemetry: FZeroXTelemetry,
        progress_multiplier: float,
        *,
        outside_track_bounds: bool,
        defer_outside_bounds_progress: bool,
        returning_grounded_inside_track_bounds: bool,
    ) -> FrontierReward:
        if defer_outside_bounds_progress:
            return FrontierReward(
                progress=0.0,
                ground_effect_adjustment=0.0,
                energy_refill_bonus=0.0,
            )

        if returning_grounded_inside_track_bounds:
            progress_multiplier = 1.0
            race_distance = telemetry.player.race_distance
            energy_refill_bonus_for_progress = _zero_progress_bonus
        else:
            race_distance = None

            def energy_refill_bonus_for_progress(progress_reward: float) -> float:
                return self._energy.progress_bonus(
                    progress_reward,
                    summary,
                    telemetry,
                    weights=self._weights,
                )

        frontier_reward = self._progress.step(
            summary,
            status,
            weights=self._weights,
            progress_multiplier=progress_multiplier,
            airborne=telemetry.player.airborne,
            outside_track_bounds=outside_track_bounds,
            race_distance=race_distance,
            energy_refill_bonus_for_progress=energy_refill_bonus_for_progress,
        )
        if returning_grounded_inside_track_bounds:
            return cap_outside_bounds_reentry_reward(frontier_reward, weights=self._weights)
        return frontier_reward


def _clip_step_reward(reward: float, *, weights: RewardMainWeights) -> float:
    if weights.step_reward_clip_min is not None:
        reward = max(reward, float(weights.step_reward_clip_min))
    if weights.step_reward_clip_max is not None:
        reward = min(reward, float(weights.step_reward_clip_max))
    return reward


def _ground_effect_progress_modifier(
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


def _zero_progress_bonus(progress_reward: float) -> float:
    del progress_reward
    return 0.0
