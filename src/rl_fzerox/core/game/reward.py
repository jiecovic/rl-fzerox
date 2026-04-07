# src/rl_fzerox/core/game/reward.py
from __future__ import annotations

from dataclasses import dataclass, field

from rl_fzerox.core.game.flags import (
    FLAG_COLLISION_RECOIL,
    FLAG_CRASHED,
    FLAG_DASH_PAD_BOOST,
    FLAG_FALLING_OFF_TRACK,
    FLAG_FINISHED,
    FLAG_RETIRED,
    FLAG_SPINNING_OUT,
)
from rl_fzerox.core.game.telemetry import FZeroXTelemetry


@dataclass(frozen=True)
class RewardWeights:
    """Weights for the first telemetry-based reward function."""

    progress_scale: float = 0.001
    reverse_progress_scale: float = 0.001
    progress_epsilon: float = 0.5
    energy_loss_epsilon: float = 0.1
    energy_loss_penalty_scale: float = 0.05
    stall_grace_steps: int = 60
    stall_penalty: float = -0.05
    stuck_truncation_penalty: float = -5.0
    wrong_way_truncation_penalty: float = -10.0
    dash_pad_boost_reward: float = 0.5
    dash_pad_min_progress: float = 500.0
    collision_recoil_penalty: float = -4.0
    spinning_out_penalty: float = -2.0
    falling_off_track_penalty: float = -10.0
    crashed_penalty: float = -20.0
    retired_penalty: float = -20.0
    finish_bonus: float = 100.0
    finish_position_scale: float = 2.0
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
        self._previous_state_flags = 0
        self._stall_steps = 0
        self._last_dash_pad_reward_distance = float("-inf")

    def reset(self, telemetry: FZeroXTelemetry | None) -> None:
        """Initialize reward state for a new episode."""

        self._previous_state_flags = 0
        self._stall_steps = 0
        self._last_dash_pad_reward_distance = float("-inf")
        if telemetry is None:
            self._best_race_distance = float("-inf")
            self._previous_race_distance = float("-inf")
            self._previous_energy = 0.0
            return
        self._best_race_distance = telemetry.player.race_distance
        self._previous_race_distance = telemetry.player.race_distance
        self._previous_energy = telemetry.player.energy
        self._previous_state_flags = telemetry.player.state_flags

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
            breakdown["progress"] = progress_reward
            self._best_race_distance = telemetry.player.race_distance
            self._stall_steps = 0

        if progress_delta < -self._weights.progress_epsilon:
            reverse_penalty = progress_delta * self._weights.reverse_progress_scale
            reward += reverse_penalty
            breakdown["reverse_progress"] = reverse_penalty
            self._stall_steps = 0
        elif progress_gain <= self._weights.progress_epsilon:
            self._stall_steps += 1
            if self._stall_steps > self._weights.stall_grace_steps:
                reward += self._weights.stall_penalty
                breakdown["stall"] = self._weights.stall_penalty

        energy_delta = telemetry.player.energy - self._previous_energy
        if energy_delta < -self._weights.energy_loss_epsilon:
            energy_loss_penalty = energy_delta * self._weights.energy_loss_penalty_scale
            reward += energy_loss_penalty
            breakdown["energy_loss"] = energy_loss_penalty

        current_flags = telemetry.player.state_flags
        entered_flags = current_flags & ~self._previous_state_flags

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
        self._previous_state_flags = current_flags

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
