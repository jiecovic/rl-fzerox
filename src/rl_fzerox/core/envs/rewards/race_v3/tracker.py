# src/rl_fzerox/core/envs/rewards/race_v3/tracker.py
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
from rl_fzerox.core.envs.rewards.progress import (
    DamagePenaltyState,
)
from rl_fzerox.core.envs.rewards.race_v3.controls import (
    SteerOscillationRewardTracker,
    airborne_pitch_up_penalty,
    gas_underuse_penalty,
    lean_low_speed_penalty,
    lean_request_penalty,
    manual_boost_reward,
)
from rl_fzerox.core.envs.rewards.race_v3.energy import EnergyRefillRewardTracker
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
from rl_fzerox.core.envs.rewards.race_v3.weights import RaceV3RewardWeights
from rl_fzerox.core.envs.track_bounds import telemetry_outside_track_bounds, track_edge_state


class RaceV3RewardTracker:
    """Reward newly covered progress buckets on the game spline."""

    def __init__(
        self,
        weights: RaceV3RewardWeights | None = None,
        *,
        course_weights: dict[str, RaceV3RewardWeights] | None = None,
        max_episode_steps: int = 12_000,
    ) -> None:
        self._base_weights = weights or RaceV3RewardWeights()
        self._course_weights = course_weights or {}
        self._weights = self._base_weights
        self._max_episode_steps = max_episode_steps
        self._progress = FrontierProgressRewardTracker()
        self._energy = EnergyRefillRewardTracker()
        self._boost_pads = BoostPadRewardTracker()
        self._bad_surfaces = BadSurfaceEntryPenaltyTracker()
        self._laps = LapRewardTracker()
        self._damage = DamagePenaltyState()
        self._steering = SteerOscillationRewardTracker()
        self._previous_airborne = False
        self._deferred_outside_bounds_progress = False
        self._previous_airborne_offtrack_excess: float | None = None
        self._previous_airborne_height: float | None = None
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
        self._steering.reset()
        self._previous_airborne = False if telemetry is None else telemetry.player.airborne
        self._deferred_outside_bounds_progress = telemetry_outside_track_bounds(telemetry)
        self._previous_airborne_offtrack_excess = _airborne_offtrack_excess(telemetry)
        self._previous_airborne_height = _airborne_height_above_ground(telemetry)

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
            self._steering.reset()
            self._previous_airborne = False
            self._deferred_outside_bounds_progress = False
            self._previous_airborne_offtrack_excess = None
            self._previous_airborne_height = None
            return RewardStep(reward=0.0)

        self._progress.ensure_origin(telemetry)
        outside_track_bounds = telemetry_outside_track_bounds(telemetry)
        airborne_offtrack_excess = _airborne_offtrack_excess(telemetry)
        airborne_descending = _airborne_descending(
            previous_height=self._previous_airborne_height,
            telemetry=telemetry,
            weights=self._weights,
        )
        defer_outside_bounds_progress = (
            outside_track_bounds
            or (self._deferred_outside_bounds_progress and telemetry.player.airborne)
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
        low_speed_time_penalty = self._low_speed_time_penalty(summary)
        if low_speed_time_penalty:
            reward += low_speed_time_penalty
            breakdown["low_speed_time"] = low_speed_time_penalty
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
        airborne_offtrack_penalty = _airborne_offtrack_penalty(
            summary,
            airborne_offtrack_excess,
            weights=self._weights,
        )
        if airborne_offtrack_penalty:
            reward += airborne_offtrack_penalty
            breakdown["airborne_offtrack"] = airborne_offtrack_penalty
        airborne_offtrack_recovery = _airborne_offtrack_recovery_reward(
            previous_excess=self._previous_airborne_offtrack_excess,
            current_excess=airborne_offtrack_excess,
            descending=airborne_descending,
            weights=self._weights,
        )
        if airborne_offtrack_recovery:
            reward += airborne_offtrack_recovery
            breakdown["airborne_offtrack_recovery"] = airborne_offtrack_recovery

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

        self._energy.accumulate_since_full(summary, telemetry)
        full_refill_reward = self._energy.full_refill_lap_reward(
            summary,
            telemetry,
            weights=self._weights,
        )
        if full_refill_reward:
            reward += full_refill_reward
            breakdown["energy_full_refill_lap"] = full_refill_reward

        gas_penalty = gas_underuse_penalty(
            summary,
            action_context,
            weights=self._weights,
        )
        if gas_penalty:
            reward += gas_penalty
            breakdown["gas_underuse"] = gas_penalty

        steer_oscillation_penalty = self._steering.penalty(
            action_context,
            weights=self._weights,
        )
        if steer_oscillation_penalty:
            reward += steer_oscillation_penalty
            breakdown["steer_oscillation"] = steer_oscillation_penalty

        lean_penalty = lean_low_speed_penalty(
            summary,
            telemetry,
            action_context,
            weights=self._weights,
        )
        if lean_penalty:
            reward += lean_penalty
            breakdown["lean_low_speed"] = lean_penalty

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
        self._previous_airborne_offtrack_excess = airborne_offtrack_excess
        self._previous_airborne_height = _airborne_height_above_ground(telemetry)
        return RewardStep(reward=reward, breakdown=breakdown, raw_reward=raw_reward)

    def info(self, telemetry: FZeroXTelemetry | None) -> dict[str, object]:
        """Expose lightweight frontier reward state for watch/logging."""

        info: dict[str, object] = {
            "reward_profile": "race_v3",
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

    def _low_speed_time_penalty(self, summary: StepSummary) -> float:
        extra_scale = self._weights.low_speed_time_penalty_scale - 1.0
        if summary.low_speed_frames <= 0 or extra_scale == 0.0:
            return 0.0
        return summary.low_speed_frames * self._weights.time_penalty_per_frame * extra_scale

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
            race_distance=race_distance,
            energy_refill_bonus_for_progress=energy_refill_bonus_for_progress,
        )
        if returning_grounded_inside_track_bounds:
            return _cap_outside_bounds_reentry_reward(frontier_reward, weights=self._weights)
        return frontier_reward


def _cap_outside_bounds_reentry_reward(
    frontier_reward: FrontierReward,
    *,
    weights: RaceV3RewardWeights,
) -> FrontierReward:
    distance_cap = weights.outside_bounds_reentry_progress_distance_cap
    if distance_cap is None:
        return frontier_reward
    reward_cap = (
        max(float(distance_cap), 0.0)
        / weights.progress_bucket_distance
        * weights.progress_bucket_reward
    )
    if frontier_reward.progress <= reward_cap:
        return frontier_reward
    return FrontierReward(
        progress=reward_cap,
        ground_effect_adjustment=0.0,
        energy_refill_bonus=0.0,
    )


def _clip_step_reward(reward: float, *, weights: RaceV3RewardWeights) -> float:
    if weights.step_reward_clip_min is not None:
        reward = max(reward, float(weights.step_reward_clip_min))
    if weights.step_reward_clip_max is not None:
        reward = min(reward, float(weights.step_reward_clip_max))
    return reward


def _ground_effect_progress_modifier(
    telemetry: FZeroXTelemetry,
    *,
    weights: RaceV3RewardWeights,
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


def _airborne_offtrack_excess(telemetry: FZeroXTelemetry | None) -> float | None:
    if telemetry is None or not telemetry.player.airborne:
        return None
    edge_ratio = track_edge_state(telemetry.player).ratio
    if edge_ratio is None:
        return None
    outside_excess = max(abs(edge_ratio) - 1.0, 0.0)
    if outside_excess <= 0.0:
        return None
    return outside_excess


def _airborne_height_above_ground(telemetry: FZeroXTelemetry | None) -> float | None:
    if telemetry is None or not telemetry.player.airborne:
        return None
    return float(telemetry.player.height_above_ground)


def _airborne_descending(
    *,
    previous_height: float | None,
    telemetry: FZeroXTelemetry,
    weights: RaceV3RewardWeights,
) -> bool:
    current_height = _airborne_height_above_ground(telemetry)
    if previous_height is None or current_height is None:
        return False
    epsilon = float(weights.airborne_offtrack_recovery_descend_epsilon)
    return current_height < previous_height - epsilon


def _airborne_offtrack_penalty(
    summary: StepSummary,
    outside_excess: float | None,
    *,
    weights: RaceV3RewardWeights,
) -> float:
    scale = weights.airborne_offtrack_penalty_scale
    if scale <= 0.0 or outside_excess is None:
        return 0.0
    return -scale * (outside_excess**2) * max(int(summary.frames_run), 1)


def _airborne_offtrack_recovery_reward(
    *,
    previous_excess: float | None,
    current_excess: float | None,
    descending: bool,
    weights: RaceV3RewardWeights,
) -> float:
    scale = weights.airborne_offtrack_recovery_reward_scale
    if scale <= 0.0 or current_excess is None:
        return 0.0
    reward = scale * ((previous_excess or 0.0) - current_excess)
    if (
        reward > 0.0
        and weights.airborne_offtrack_recovery_requires_descending
        and not descending
    ):
        return 0.0
    return reward
