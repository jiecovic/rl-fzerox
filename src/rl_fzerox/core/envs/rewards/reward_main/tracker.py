# src/rl_fzerox/core/envs/rewards/reward_main/tracker.py
"""Episode orchestration for the canonical `reward_main` profile.

`RewardMainTracker` owns the mutable reward state across an episode: progress
frontiers, previous telemetry, airborne/refill state, and debug summaries.
Individual term formulas live in sibling modules.
"""

from __future__ import annotations

from dataclasses import dataclass

from fzerox_emulator import FZeroXTelemetry, StepStatus, StepSummary
from rl_fzerox.core.envs.laps import completed_race_laps
from rl_fzerox.core.envs.rewards.common import (
    RewardActionContext,
    RewardStep,
    RewardSummaryConfig,
)
from rl_fzerox.core.envs.rewards.progress import impact_frame_penalty
from rl_fzerox.core.envs.rewards.reward_main.airborne import (
    landing_airborne_frames,
    landing_airborne_peak_height,
)
from rl_fzerox.core.envs.rewards.reward_main.controls import (
    air_brake_request_penalty,
    grounded_pitch_penalty,
    lean_activation_penalty,
    lean_request_penalty,
    manual_boost_reward,
    outside_track_dip_height,
    outside_track_dip_penalty,
    outside_track_recovery_reward,
    spin_request_penalty,
)
from rl_fzerox.core.envs.rewards.reward_main.energy import EnergyRefillRewardTracker
from rl_fzerox.core.envs.rewards.reward_main.events import (
    BadSurfaceEntryPenaltyTracker,
    BoostPadRewardTracker,
    LandingRewardTracker,
    LapRewardTracker,
    terminal_or_truncation_penalty,
)
from rl_fzerox.core.envs.rewards.reward_main.progress import (
    FrontierProgressRewardTracker,
    FrontierReward,
)
from rl_fzerox.core.envs.rewards.reward_main.recovery import (
    outside_track_recovery_airborne_frames,
    outside_track_recovery_armed,
    outside_track_recovery_baseline,
    outside_track_recovery_distance,
    previous_outside_track_recovery_distance,
)
from rl_fzerox.core.envs.rewards.reward_main.step_terms import (
    clip_step_reward,
    ground_effect_progress_modifier,
    ko_star_count,
    ko_star_reward_event,
)
from rl_fzerox.core.envs.rewards.reward_main.weights import RewardMainWeights
from rl_fzerox.core.envs.telemetry import telemetry_can_boost
from rl_fzerox.core.envs.track_bounds import (
    telemetry_outside_track_bounds,
    track_recovery_segment_distance,
)


@dataclass(frozen=True, slots=True)
class _ActiveStepState:
    outside_track_bounds: bool
    landing_airborne_frames: int
    landing_airborne_peak_height: float
    recovery_airborne_frames: int
    recovery_armed: bool
    current_outside_recovery_distance: float | None
    progress_suspended: bool
    frontier_distance_before_step: float
    ground_effect_key: str
    progress_multiplier: float


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
        self._landings = LandingRewardTracker()
        self._bad_surfaces = BadSurfaceEntryPenaltyTracker()
        self._laps = LapRewardTracker()
        self._previous_airborne = False
        self._landing_airborne_frames = 0
        self._landing_airborne_peak_height = 0.0
        self._outside_track_recovery_airborne_frames = 0
        self._outside_track_recovery_armed = False
        self._outside_track_dip_penalized = False
        self._previous_outside_recovery_distance: float | None = None
        self._previous_ko_star_count: int | None = None
        self._previous_lean_requested = False
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
        self._landings.reset()
        self._bad_surfaces.reset(telemetry)
        self._laps.reset(telemetry)
        self._previous_airborne = False if telemetry is None else telemetry.player.airborne
        self._landing_airborne_frames = 0
        self._landing_airborne_peak_height = landing_airborne_peak_height(
            previous_peak_height=0.0,
            telemetry=telemetry,
        )
        self._outside_track_recovery_airborne_frames = 0
        self._outside_track_recovery_armed = False
        self._outside_track_dip_penalized = False
        self._previous_outside_recovery_distance = previous_outside_track_recovery_distance(
            telemetry
        )
        self._previous_ko_star_count = ko_star_count(telemetry)
        self._previous_lean_requested = False

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
            return self._reset_inactive_step()

        self._progress.ensure_origin(telemetry)
        step_state = self._active_step_state(summary, telemetry)
        self._energy.start_collision_cooldown(summary, telemetry, weights=self._weights)
        reward = summary.frames_run * self._weights.time_penalty_per_frame
        breakdown: dict[str, float] = {}
        debug_info: dict[str, object] = {}
        if reward:
            breakdown["time"] = reward

        reward += self._frontier_reward_terms(
            summary,
            status,
            telemetry,
            step_state=step_state,
            breakdown=breakdown,
        )
        previous_recovery_distance_for_reward = outside_track_recovery_baseline(
            previous_distance=self._previous_outside_recovery_distance,
            current_distance=step_state.current_outside_recovery_distance,
            outside_track_bounds=step_state.outside_track_bounds,
            previous_armed=self._outside_track_recovery_armed,
            current_armed=step_state.recovery_armed,
        )
        recovery_reward = outside_track_recovery_reward(
            weights=self._weights,
            previous_distance=previous_recovery_distance_for_reward,
            current_distance=step_state.current_outside_recovery_distance,
            enabled=step_state.recovery_armed,
        )
        if recovery_reward:
            reward += recovery_reward
            breakdown["outside_track_recovery"] = recovery_reward

        dip_height = outside_track_dip_height(
            summary,
            telemetry,
            outside_track_bounds=step_state.outside_track_bounds,
        )
        dip_penalty = outside_track_dip_penalty(
            dip_height=dip_height,
            already_penalized=self._outside_track_dip_penalized,
            weights=self._weights,
        )
        if dip_penalty:
            reward += dip_penalty
            breakdown["outside_track_dip"] = dip_penalty
            debug_info["outside_track_dip_penalty_event"] = True
            debug_info["outside_track_dip_height"] = dip_height

        lap_reward = self._laps.reward(
            telemetry,
            weights=self._weights,
            breakdown=breakdown,
        )
        if lap_reward:
            reward += lap_reward

        ko_event = ko_star_reward_event(
            previous_count=self._previous_ko_star_count,
            telemetry=telemetry,
            weights=self._weights,
        )
        if ko_event is not None:
            reward += ko_event.reward
            breakdown["ko_star"] = ko_event.reward
            debug_info["ko_star_reward_event"] = True
            debug_info["ko_star_reward_previous_count"] = ko_event.previous_count
            debug_info["ko_star_reward_current_count"] = ko_event.current_count
            debug_info["ko_star_reward_gain"] = ko_event.gained
            debug_info["ko_star_reward_value"] = ko_event.reward

        boost_pad_reward = self._boost_pads.reward(
            summary.entered_dash_surface,
            summary.reverse_active_frames,
            can_boost=telemetry_can_boost(telemetry),
            relative_progress=self._progress.relative_distance(summary.max_race_distance),
            frontier_distance_before_step=step_state.frontier_distance_before_step,
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

        spin_penalty = spin_request_penalty(
            action_context,
            weights=self._weights,
        )
        if spin_penalty:
            reward += spin_penalty
            breakdown["spin"] = spin_penalty

        lean_penalty = lean_request_penalty(
            summary,
            action_context,
            previous_lean_requested=self._previous_lean_requested,
            weights=self._weights,
        )
        if lean_penalty:
            reward += lean_penalty
            breakdown["lean"] = lean_penalty

        lean_activation = lean_activation_penalty(
            summary,
            action_context,
            previous_lean_requested=self._previous_lean_requested,
            weights=self._weights,
        )
        if lean_activation:
            reward += lean_activation
            breakdown["lean_activation"] = lean_activation

        grounded_pitch = grounded_pitch_penalty(
            summary,
            telemetry,
            action_context,
            weights=self._weights,
        )
        if grounded_pitch:
            reward += grounded_pitch
            breakdown["grounded_pitch"] = grounded_pitch

        boost_reward = manual_boost_reward(action_context, telemetry, weights=self._weights)
        if boost_reward:
            reward += boost_reward
            breakdown["manual_boost"] = boost_reward

        landing = self._landings.reward(
            previous_airborne=self._previous_airborne,
            airborne_frames=step_state.landing_airborne_frames,
            airborne_peak_height=step_state.landing_airborne_peak_height,
            telemetry=telemetry,
            frontier_bucket_index=self._progress.frontier_bucket_index,
            weights=self._weights,
        )
        if landing:
            reward += landing
            breakdown["landing"] = landing

        loss_penalty = (
            max(float(summary.energy_loss_total), 0.0) * self._weights.energy_loss_penalty
        )
        if loss_penalty:
            reward += loss_penalty
            breakdown["energy_loss"] = loss_penalty

        impact_penalty = impact_frame_penalty(
            summary,
            frame_penalty=self._weights.impact_frame_penalty,
        )
        if impact_penalty:
            reward += impact_penalty
            breakdown["impact"] = impact_penalty

        terminal_penalty = terminal_or_truncation_penalty(
            status,
            weights=self._weights,
            breakdown=breakdown,
        )
        if terminal_penalty:
            reward += terminal_penalty
        return self._finish_active_step(
            reward=reward,
            breakdown=breakdown,
            debug_info=debug_info,
            summary=summary,
            telemetry=telemetry,
            action_context=action_context,
            step_state=step_state,
            dip_penalty=dip_penalty,
            dip_height=dip_height,
        )

    def info(self, telemetry: FZeroXTelemetry | None) -> dict[str, object]:
        """Expose lightweight frontier reward state for watch/logging."""

        info: dict[str, object] = {
            "reward_profile": "reward_main",
            "boost_pad_reward_cannot_boost": self._weights.boost_pad_reward_cannot_boost,
            "boost_pad_reward_can_boost": self._weights.boost_pad_reward_can_boost,
            "boost_pad_reward_progress_window": self._weights.boost_pad_reward_progress_window,
            "rewarded_boost_pad_progress_windows": self._boost_pads.rewarded_window_count,
            "landing_reward_frontier_bucket_index": (
                self._landings.last_rewarded_frontier_bucket_index
            ),
            "rewarded_laps_completed": self._laps.awarded_laps_completed,
        }
        info.update(self._progress.info(telemetry, weights=self._weights))
        info.update(self._energy.info())
        if telemetry is None or not telemetry.in_race_mode:
            return info
        self._progress.ensure_origin(telemetry)
        info["race_laps_completed"] = completed_race_laps(telemetry)
        info["progress_reward_outside_track_bounds"] = telemetry_outside_track_bounds(telemetry)
        info["progress_reward_track_distance"] = track_recovery_segment_distance(telemetry.player)
        info["progress_reward_suspended"] = self._progress_suspended(telemetry)
        return info

    def _reset_inactive_step(self) -> RewardStep:
        self._progress.reset_inactive()
        self._energy.reset_inactive()
        self._boost_pads.reset()
        self._bad_surfaces.reset(None)
        self._laps.reset(None)
        self._previous_airborne = False
        self._landing_airborne_frames = 0
        self._landing_airborne_peak_height = 0.0
        self._outside_track_recovery_airborne_frames = 0
        self._outside_track_recovery_armed = False
        self._outside_track_dip_penalized = False
        self._previous_outside_recovery_distance = None
        self._previous_ko_star_count = None
        self._previous_lean_requested = False
        return RewardStep(reward=0.0)

    def _active_step_state(
        self,
        summary: StepSummary,
        telemetry: FZeroXTelemetry,
    ) -> _ActiveStepState:
        outside_track_bounds = telemetry_outside_track_bounds(telemetry)
        recovery_excursion_active = (
            self._previous_outside_recovery_distance is not None or outside_track_bounds
        )
        recovery_airborne_frames = outside_track_recovery_airborne_frames(
            previous_frames=self._outside_track_recovery_airborne_frames,
            summary=summary,
            excursion_active=recovery_excursion_active,
        )
        ground_effect_key, progress_multiplier = ground_effect_progress_modifier(
            telemetry,
            weights=self._weights,
        )
        return _ActiveStepState(
            outside_track_bounds=outside_track_bounds,
            landing_airborne_frames=landing_airborne_frames(
                previous_frames=self._landing_airborne_frames,
                previous_airborne=self._previous_airborne,
                current_airborne=telemetry.player.airborne,
                summary=summary,
            ),
            landing_airborne_peak_height=landing_airborne_peak_height(
                previous_peak_height=self._landing_airborne_peak_height,
                telemetry=telemetry,
            ),
            recovery_airborne_frames=recovery_airborne_frames,
            recovery_armed=outside_track_recovery_armed(
                previous_armed=self._outside_track_recovery_armed,
                airborne_frames=recovery_airborne_frames,
                grace_frames=self._weights.outside_track_recovery_airborne_grace_frames,
            ),
            current_outside_recovery_distance=outside_track_recovery_distance(telemetry),
            progress_suspended=self._progress_suspended(telemetry),
            frontier_distance_before_step=self._progress.frontier_distance,
            ground_effect_key=ground_effect_key,
            progress_multiplier=progress_multiplier,
        )

    def _frontier_reward_terms(
        self,
        summary: StepSummary,
        status: StepStatus,
        telemetry: FZeroXTelemetry,
        *,
        step_state: _ActiveStepState,
        breakdown: dict[str, float],
    ) -> float:
        reward = 0.0
        frontier_reward = self._frontier_reward(
            summary,
            status,
            telemetry,
            step_state.progress_multiplier,
            progress_suspended=step_state.progress_suspended,
        )
        if frontier_reward.progress:
            reward += frontier_reward.progress
            breakdown["frontier_progress"] = frontier_reward.progress
        if frontier_reward.ground_effect_adjustment:
            reward += frontier_reward.ground_effect_adjustment
            breakdown[f"{step_state.ground_effect_key}_progress"] = (
                frontier_reward.ground_effect_adjustment
            )
        if frontier_reward.speed_adjustment:
            reward += frontier_reward.speed_adjustment
            breakdown["speed_progress"] = frontier_reward.speed_adjustment
        if frontier_reward.position_adjustment:
            reward += frontier_reward.position_adjustment
            breakdown["position_progress"] = frontier_reward.position_adjustment
        if frontier_reward.energy_refill_bonus:
            reward += frontier_reward.energy_refill_bonus
            breakdown["energy_refill_progress"] = frontier_reward.energy_refill_bonus
        if frontier_reward.energy_gain_reward:
            reward += frontier_reward.energy_gain_reward
            breakdown["energy_gain"] = frontier_reward.energy_gain_reward
        return reward

    def _finish_active_step(
        self,
        *,
        reward: float,
        breakdown: dict[str, float],
        debug_info: dict[str, object],
        summary: StepSummary,
        telemetry: FZeroXTelemetry,
        action_context: RewardActionContext | None,
        step_state: _ActiveStepState,
        dip_penalty: float,
        dip_height: float | None,
    ) -> RewardStep:
        raw_reward = reward
        reward = clip_step_reward(raw_reward, weights=self._weights)
        if reward != raw_reward:
            breakdown["step_reward_clip"] = reward - raw_reward

        self._energy.advance_cooldown(summary.frames_run)
        self._energy.finish_step()
        self._previous_airborne = telemetry.player.airborne
        self._landing_airborne_frames = (
            step_state.landing_airborne_frames if telemetry.player.airborne else 0
        )
        self._landing_airborne_peak_height = (
            step_state.landing_airborne_peak_height if telemetry.player.airborne else 0.0
        )
        self._outside_track_recovery_airborne_frames = (
            step_state.recovery_airborne_frames if step_state.outside_track_bounds else 0
        )
        self._outside_track_recovery_armed = (
            step_state.recovery_armed if step_state.outside_track_bounds else False
        )
        self._outside_track_dip_penalized = (
            (self._outside_track_dip_penalized or bool(dip_penalty))
            if step_state.outside_track_bounds or dip_height is not None
            else False
        )
        self._previous_outside_recovery_distance = previous_outside_track_recovery_distance(
            telemetry
        )
        self._previous_ko_star_count = ko_star_count(telemetry)
        self._previous_lean_requested = (
            False if action_context is None else action_context.lean_requested
        )
        return RewardStep(
            reward=reward,
            breakdown=breakdown,
            raw_reward=raw_reward,
            debug_info=debug_info,
        )

    def _frontier_reward(
        self,
        summary: StepSummary,
        status: StepStatus,
        telemetry: FZeroXTelemetry,
        progress_multiplier: float,
        *,
        progress_suspended: bool,
    ) -> FrontierReward:
        def energy_refill_bonus_for_progress(progress_reward: float) -> float:
            return self._energy.progress_bonus(
                progress_reward,
                summary,
                telemetry,
                weights=self._weights,
            )

        def energy_gain_reward_for_progress(progress_reward: float) -> float:
            return self._energy.gain_reward(
                progress_reward,
                summary,
                telemetry,
                weights=self._weights,
            )

        return self._progress.step(
            summary,
            status,
            telemetry=telemetry,
            weights=self._weights,
            progress_multiplier=progress_multiplier,
            progress_suspended=progress_suspended,
            energy_refill_bonus_for_progress=energy_refill_bonus_for_progress,
            energy_gain_reward_for_progress=energy_gain_reward_for_progress,
        )

    def _progress_suspended(self, telemetry: FZeroXTelemetry) -> bool:
        if not self._weights.suspend_progress_while_outside_track_bounds:
            return False
        if not telemetry_outside_track_bounds(telemetry):
            return False
        track_distance = track_recovery_segment_distance(telemetry.player)
        if track_distance is None:
            return True
        return track_distance > max(float(self._weights.progress_track_distance_tolerance), 0.0)
