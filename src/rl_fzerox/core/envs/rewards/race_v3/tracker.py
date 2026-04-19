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
    gas_underuse_penalty,
    lean_low_speed_penalty,
)
from rl_fzerox.core.envs.rewards.race_v3.energy import EnergyRefillRewardTracker
from rl_fzerox.core.envs.rewards.race_v3.events import (
    BoostPadRewardTracker,
    LapRewardTracker,
    landing_reward,
    terminal_or_truncation_penalty,
)
from rl_fzerox.core.envs.rewards.race_v3.progress import (
    FrontierProgressRewardTracker,
)
from rl_fzerox.core.envs.rewards.race_v3.weights import RaceV3RewardWeights


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
        self._progress = FrontierProgressRewardTracker()
        self._energy = EnergyRefillRewardTracker()
        self._boost_pads = BoostPadRewardTracker()
        self._laps = LapRewardTracker()
        self._damage = DamagePenaltyState()
        self._steering = SteerOscillationRewardTracker()
        self._previous_airborne = False

    def reset(
        self,
        telemetry: FZeroXTelemetry | None,
        *,
        episode_seed: int | None = None,
    ) -> None:
        """Initialize frontier state for a new episode."""

        del episode_seed
        self._progress.reset(telemetry)
        self._energy.reset(telemetry)
        self._boost_pads.reset()
        self._laps.reset(telemetry)
        self._damage.reset()
        self._steering.reset()
        self._previous_airborne = False if telemetry is None else telemetry.player.airborne

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
            self._laps.reset(None)
            self._damage.reset()
            self._steering.reset()
            self._previous_airborne = False
            return RewardStep(reward=0.0)

        self._progress.ensure_origin(telemetry)
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

        ground_effect_key, progress_multiplier = _ground_effect_progress_modifier(
            telemetry,
            weights=self._weights,
        )
        frontier_reward = self._progress.step(
            summary,
            status,
            weights=self._weights,
            progress_multiplier=progress_multiplier,
            energy_refill_bonus_for_progress=lambda progress_reward: (
                self._energy.progress_bonus(
                    progress_reward,
                    summary,
                    telemetry,
                    weights=self._weights,
                )
            ),
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
            weights=self._weights,
        )
        if boost_pad_reward:
            reward += boost_pad_reward
            breakdown["boost_pad"] = boost_pad_reward

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

        self._energy.advance_cooldown(summary.frames_run)
        self._energy.finish_step(telemetry)
        self._previous_airborne = telemetry.player.airborne
        return RewardStep(reward=reward, breakdown=breakdown)

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
