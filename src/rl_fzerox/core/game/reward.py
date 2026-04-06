# src/rl_fzerox/core/game/reward.py
from __future__ import annotations

from dataclasses import dataclass, field

from rl_fzerox.core.game.telemetry import FZeroXTelemetry

FLAG_COLLISION_RECOIL = 1 << 13
FLAG_SPINNING_OUT = 1 << 14
FLAG_RETIRED = 1 << 18
FLAG_FALLING_OFF_TRACK = 1 << 19
FLAG_FINISHED = 1 << 25
FLAG_CRASHED = 1 << 27


@dataclass(frozen=True)
class RewardWeights:
    """Weights for the first telemetry-based reward function."""

    progress_scale: float = 0.001
    collision_recoil_penalty: float = -0.05
    spinning_out_penalty: float = -0.1
    falling_off_track_penalty: float = -10.0
    crashed_penalty: float = -20.0
    retired_penalty: float = -20.0
    finish_bonus: float = 50.0


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
        self._previous_state_flags = 0

    def reset(self, telemetry: FZeroXTelemetry | None) -> None:
        """Initialize reward state for a new episode."""

        self._previous_state_flags = 0
        if telemetry is None:
            self._best_race_distance = float("-inf")
            return
        self._best_race_distance = telemetry.player.race_distance
        self._previous_state_flags = telemetry.player.state_flags

    def step(self, telemetry: FZeroXTelemetry | None) -> RewardStep:
        """Compute one reward step from the current telemetry sample."""

        if telemetry is None or not telemetry.in_race_mode:
            return RewardStep(reward=0.0, terminated=False)

        reward = 0.0
        breakdown: dict[str, float] = {}

        progress_gain = telemetry.player.race_distance - self._best_race_distance
        if progress_gain > 0.0:
            progress_reward = progress_gain * self._weights.progress_scale
            reward += progress_reward
            breakdown["progress"] = progress_reward
            self._best_race_distance = telemetry.player.race_distance

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

        if entered_flags & FLAG_FINISHED:
            reward += self._weights.finish_bonus
            breakdown["finished"] = self._weights.finish_bonus

        self._previous_state_flags = current_flags

        terminated = bool(
            current_flags
            & (FLAG_FINISHED | FLAG_CRASHED | FLAG_RETIRED | FLAG_FALLING_OFF_TRACK)
        )
        return RewardStep(reward=reward, terminated=terminated, breakdown=breakdown)


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
