# src/rl_fzerox/core/game/reward.py
from __future__ import annotations

import math
from dataclasses import dataclass, field

from rl_fzerox.core.game.flags import (
    FLAG_AIRBORNE,
    FLAG_COLLISION_RECOIL,
    FLAG_CRASHED,
    FLAG_DASH_PAD_BOOST,
    FLAG_FALLING_OFF_TRACK,
    FLAG_FINISHED,
    FLAG_RETIRED,
    FLAG_SPINNING_OUT,
    is_on_pit_strip,
)
from rl_fzerox.core.game.telemetry import FZeroXTelemetry


@dataclass(frozen=True)
class RewardWeights:
    """Weights for the first telemetry-based reward function."""

    progress_scale: float = 0.001
    reverse_progress_scale: float = 0.001
    progress_epsilon: float = 0.5
    checkpoint_spacing: float = 3_000.0
    checkpoint_fast_time_ms: int = 3_000
    checkpoint_slow_time_ms: int = 8_000
    checkpoint_fast_bonus: float = 1.25
    checkpoint_slow_bonus: float = 0.4
    low_speed_threshold_kph: float = 100.0
    low_speed_penalty: float = -0.05
    energy_loss_epsilon: float = 0.1
    energy_loss_penalty_scale: float = 0.1
    low_energy_boost_threshold_ratio: float = 0.1
    low_energy_boost_penalty: float = -4.0
    refill_reward_energy_cap: float = 20.0
    refill_reward_scale: float = 0.05
    stall_grace_steps: int = 60
    stall_penalty: float = -0.05
    stuck_truncation_penalty: float = -5.0
    wrong_way_truncation_penalty: float = -10.0
    dash_pad_boost_reward: float = 2.0
    dash_pad_min_progress: float = 500.0
    collision_recoil_penalty: float = -4.0
    spinning_out_penalty: float = -2.0
    falling_off_track_penalty: float = -10.0
    crashed_penalty: float = -20.0
    retired_penalty: float = -20.0
    finish_bonus: float = 100.0
    finish_position_scale: float = 4.0
    max_race_position: int = 30


@dataclass(frozen=True)
class RewardStep:
    """Reward, terminal state, and debug breakdown for one telemetry sample."""

    reward: float
    terminated: bool
    breakdown: dict[str, float] = field(default_factory=dict)


class RewardTracker:
    """Track episode reward state from live F-Zero X telemetry."""

    def __init__(self, weights: RewardWeights | None = None) -> None:
        self._weights = weights or RewardWeights()
        self._best_race_distance = float("-inf")
        self._previous_race_distance = float("-inf")
        self._previous_energy = 0.0
        self._previous_boost_timer = 0
        self._previous_state_flags = 0
        self._stall_steps = 0
        self._last_dash_pad_reward_distance = float("-inf")
        self._last_rewarded_checkpoint = -1
        self._last_checkpoint_race_time_ms = 0
        self._pit_visit_rewarded_energy = 0.0
        self._pit_visit_missing_energy_ratio = 0.0
        self._previous_on_pit_strip = False

    def reset(self, telemetry: FZeroXTelemetry | None) -> None:
        """Initialize reward state for a new episode."""

        self._previous_state_flags = 0
        self._stall_steps = 0
        self._last_dash_pad_reward_distance = float("-inf")
        self._pit_visit_rewarded_energy = 0.0
        self._pit_visit_missing_energy_ratio = 0.0
        self._previous_on_pit_strip = False
        if telemetry is None:
            self._best_race_distance = float("-inf")
            self._previous_race_distance = float("-inf")
            self._previous_energy = 0.0
            self._previous_boost_timer = 0
            self._last_rewarded_checkpoint = -1
            self._last_checkpoint_race_time_ms = 0
            return
        self._best_race_distance = telemetry.player.race_distance
        self._previous_race_distance = telemetry.player.race_distance
        self._previous_energy = telemetry.player.energy
        self._previous_boost_timer = telemetry.player.boost_timer
        self._previous_state_flags = telemetry.player.state_flags
        self._last_rewarded_checkpoint = _checkpoint_index(
            telemetry.player.race_distance,
            self._weights.checkpoint_spacing,
        )
        self._last_checkpoint_race_time_ms = telemetry.player.race_time_ms
        self._previous_on_pit_strip = is_on_pit_strip(telemetry.player.state_flags)

    def step(self, telemetry: FZeroXTelemetry | None) -> RewardStep:
        """Compute one reward step from the current telemetry sample."""

        if telemetry is None or not telemetry.in_race_mode:
            return RewardStep(reward=0.0, terminated=False)

        reward = 0.0
        breakdown: dict[str, float] = {}
        progress_delta = telemetry.player.race_distance - self._previous_race_distance

        progress_gain = telemetry.player.race_distance - self._best_race_distance
        if progress_gain > self._weights.progress_epsilon:
            progress_reward = progress_gain * self._weights.progress_scale
            reward += progress_reward
            if progress_reward:
                breakdown["progress"] = progress_reward
            self._best_race_distance = telemetry.player.race_distance
            self._stall_steps = 0

        if progress_delta < -self._weights.progress_epsilon:
            reverse_penalty = progress_delta * self._weights.reverse_progress_scale
            reward += reverse_penalty
            if reverse_penalty:
                breakdown["reverse_progress"] = reverse_penalty
        if progress_gain <= self._weights.progress_epsilon:
            self._stall_steps += 1
            if self._stall_steps > self._weights.stall_grace_steps:
                reward += self._weights.stall_penalty
                breakdown["stall"] = self._weights.stall_penalty

        energy_delta = telemetry.player.energy - self._previous_energy
        if energy_delta < -self._weights.energy_loss_epsilon:
            energy_loss_penalty = energy_delta * self._weights.energy_loss_penalty_scale
            reward += energy_loss_penalty
            if energy_loss_penalty:
                breakdown["energy_loss"] = energy_loss_penalty

        checkpoint_bonus = self._checkpoint_bonus(telemetry)
        if checkpoint_bonus:
            reward += checkpoint_bonus
            breakdown["checkpoint"] = checkpoint_bonus

        current_flags = telemetry.player.state_flags
        entered_flags = current_flags & ~self._previous_state_flags
        low_energy_boost_penalty = self._low_energy_boost_penalty(
            telemetry=telemetry,
            entered_flags=entered_flags,
        )
        if low_energy_boost_penalty:
            reward += low_energy_boost_penalty
            breakdown["low_energy_boost"] = low_energy_boost_penalty

        refill_bonus = self._refill_bonus(telemetry)
        if refill_bonus:
            reward += refill_bonus
            breakdown["refill"] = refill_bonus

        low_speed_penalty = self._low_speed_penalty(
            telemetry=telemetry,
            state_flags=current_flags,
        )
        if low_speed_penalty:
            reward += low_speed_penalty
            breakdown["low_speed"] = low_speed_penalty

        reward += _apply_flag_penalty(
            entered_flags,
            FLAG_COLLISION_RECOIL,
            self._weights.collision_recoil_penalty,
            "collision_recoil",
            breakdown,
        )
        reward += _apply_flag_penalty(
            entered_flags,
            FLAG_SPINNING_OUT,
            self._weights.spinning_out_penalty,
            "spinning_out",
            breakdown,
        )
        reward += _apply_flag_penalty(
            entered_flags,
            FLAG_FALLING_OFF_TRACK,
            self._weights.falling_off_track_penalty,
            "falling_off_track",
            breakdown,
        )
        reward += _apply_flag_penalty(
            entered_flags,
            FLAG_CRASHED,
            self._weights.crashed_penalty,
            "crashed",
            breakdown,
        )
        reward += _apply_flag_penalty(
            entered_flags,
            FLAG_RETIRED,
            self._weights.retired_penalty,
            "retired",
            breakdown,
        )
        reward += self._apply_dash_pad_reward(
            entered_flags=entered_flags,
            race_distance=telemetry.player.race_distance,
            breakdown=breakdown,
        )

        if entered_flags & FLAG_FINISHED:
            reward += self._weights.finish_bonus
            breakdown["finished"] = self._weights.finish_bonus
            placement_bonus = _finish_placement_bonus(
                position=telemetry.player.position,
                weights=self._weights,
            )
            if placement_bonus:
                reward += placement_bonus
                breakdown["finish_position"] = placement_bonus

        self._previous_race_distance = telemetry.player.race_distance
        self._previous_energy = telemetry.player.energy
        self._previous_boost_timer = telemetry.player.boost_timer
        self._previous_state_flags = current_flags
        self._previous_on_pit_strip = is_on_pit_strip(current_flags)

        terminated = bool(
            current_flags
            & (FLAG_FINISHED | FLAG_CRASHED | FLAG_RETIRED | FLAG_FALLING_OFF_TRACK)
        )
        return RewardStep(reward=reward, terminated=terminated, breakdown=breakdown)

    def truncation_penalty(self, truncation_reason: str | None) -> tuple[float, str | None]:
        """Return any extra reward penalty that should apply to a truncation."""

        if truncation_reason == "stuck":
            return self._weights.stuck_truncation_penalty, "stuck_truncation"
        if truncation_reason == "wrong_way":
            return self._weights.wrong_way_truncation_penalty, "wrong_way_truncation"
        return 0.0, None

    def _low_speed_penalty(
        self,
        *,
        telemetry: FZeroXTelemetry,
        state_flags: int,
    ) -> float:
        threshold_kph = self._weights.low_speed_threshold_kph
        if threshold_kph <= 0.0:
            return 0.0
        if state_flags & (
            FLAG_FINISHED
            | FLAG_CRASHED
            | FLAG_RETIRED
            | FLAG_FALLING_OFF_TRACK
            | FLAG_SPINNING_OUT
            | FLAG_AIRBORNE
        ):
            return 0.0

        speed_kph = max(0.0, telemetry.player.speed_kph)
        if speed_kph >= threshold_kph:
            return 0.0

        deficit_ratio = 1.0 - (speed_kph / threshold_kph)
        return self._weights.low_speed_penalty * deficit_ratio

    def _low_energy_boost_penalty(
        self,
        *,
        telemetry: FZeroXTelemetry,
        entered_flags: int,
    ) -> float:
        if telemetry.player.max_energy <= 0.0:
            return 0.0
        if telemetry.player.boost_timer <= self._previous_boost_timer:
            return 0.0
        if entered_flags & FLAG_DASH_PAD_BOOST:
            return 0.0

        energy_ratio = self._previous_energy / telemetry.player.max_energy
        if energy_ratio >= 1.0:
            return 0.0
        if energy_ratio <= self._weights.low_energy_boost_threshold_ratio:
            return self._weights.low_energy_boost_penalty

        threshold_ratio = self._weights.low_energy_boost_threshold_ratio
        if threshold_ratio >= 1.0:
            return self._weights.low_energy_boost_penalty

        penalty_scale = (1.0 - energy_ratio) / (1.0 - threshold_ratio)
        return self._weights.low_energy_boost_penalty * penalty_scale

    def _apply_dash_pad_reward(
        self,
        *,
        entered_flags: int,
        race_distance: float,
        breakdown: dict[str, float],
    ) -> float:
        if not (entered_flags & FLAG_DASH_PAD_BOOST):
            return 0.0
        if (
            race_distance - self._last_dash_pad_reward_distance
            < self._weights.dash_pad_min_progress
        ):
            return 0.0
        self._last_dash_pad_reward_distance = race_distance
        breakdown["dash_pad_boost"] = self._weights.dash_pad_boost_reward
        return self._weights.dash_pad_boost_reward

    def _refill_bonus(self, telemetry: FZeroXTelemetry) -> float:
        current_on_pit_strip = is_on_pit_strip(telemetry.player.state_flags)
        if current_on_pit_strip and not self._previous_on_pit_strip:
            self._pit_visit_rewarded_energy = 0.0
            self._pit_visit_missing_energy_ratio = max(
                0.0,
                1.0 - (telemetry.player.energy / telemetry.player.max_energy),
            ) if telemetry.player.max_energy > 0.0 else 0.0

        if not current_on_pit_strip:
            return 0.0

        progress_delta = telemetry.player.race_distance - self._previous_race_distance
        if progress_delta <= self._weights.progress_epsilon:
            return 0.0

        energy_gain = telemetry.player.energy - self._previous_energy
        if energy_gain <= self._weights.energy_loss_epsilon:
            return 0.0

        remaining_cap = self._weights.refill_reward_energy_cap - self._pit_visit_rewarded_energy
        if remaining_cap <= self._weights.energy_loss_epsilon:
            return 0.0

        rewardable_energy_gain = min(max(0.0, energy_gain), remaining_cap)
        self._pit_visit_rewarded_energy += rewardable_energy_gain
        return (
            rewardable_energy_gain
            * self._weights.refill_reward_scale
            * self._pit_visit_missing_energy_ratio
        )

    def _checkpoint_bonus(self, telemetry: FZeroXTelemetry) -> float:
        current_checkpoint = _checkpoint_index(
            telemetry.player.race_distance,
            self._weights.checkpoint_spacing,
        )
        if current_checkpoint <= self._last_rewarded_checkpoint:
            return 0.0

        elapsed_ms = max(
            0,
            telemetry.player.race_time_ms - self._last_checkpoint_race_time_ms,
        )
        per_checkpoint_bonus = _checkpoint_bonus_from_elapsed(
            elapsed_ms=elapsed_ms,
            weights=self._weights,
        )
        self._last_rewarded_checkpoint = current_checkpoint
        self._last_checkpoint_race_time_ms = telemetry.player.race_time_ms
        return per_checkpoint_bonus


def _apply_flag_penalty(
    entered_flags: int,
    flag: int,
    penalty: float,
    label: str,
    breakdown: dict[str, float],
) -> float:
    if not (entered_flags & flag):
        return 0.0
    breakdown[label] = penalty
    return penalty


def _finish_placement_bonus(*, position: int, weights: RewardWeights) -> float:
    clamped_position = min(max(position, 1), weights.max_race_position)
    better_than_last = weights.max_race_position - clamped_position
    return better_than_last * weights.finish_position_scale


def _checkpoint_index(race_distance: float, spacing: float) -> int:
    if spacing <= 0.0:
        return -1
    return math.floor(max(race_distance, 0.0) / spacing)


def _checkpoint_bonus_from_elapsed(
    *,
    elapsed_ms: int,
    weights: RewardWeights,
) -> float:
    fast_ms = weights.checkpoint_fast_time_ms
    slow_ms = weights.checkpoint_slow_time_ms
    fast_bonus = weights.checkpoint_fast_bonus
    slow_bonus = weights.checkpoint_slow_bonus

    if elapsed_ms <= fast_ms:
        return fast_bonus
    if elapsed_ms >= slow_ms:
        return slow_bonus
    if slow_ms <= fast_ms:
        return slow_bonus

    ratio = (elapsed_ms - fast_ms) / float(slow_ms - fast_ms)
    return fast_bonus + ((slow_bonus - fast_bonus) * ratio)
