# src/rl_fzerox/core/envs/rewards/reward_main/progress.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from fzerox_emulator import FZeroXTelemetry, StepStatus, StepSummary
from rl_fzerox.core.envs.rewards.progress import EpisodeProgressState
from rl_fzerox.core.envs.rewards.shared_weights import SharedRewardWeights


@dataclass(frozen=True, slots=True)
class FrontierReward:
    """Progress and refill bonus paid from newly crossed spline buckets."""

    progress: float
    ground_effect_adjustment: float
    speed_adjustment: float
    energy_refill_bonus: float
    energy_gain_reward: float


class FrontierProgressRewardTracker:
    """Track one-way coverage of the game spline and pay crossed buckets once."""

    def __init__(self) -> None:
        self._progress = EpisodeProgressState()
        self._frontier_distance = 0.0
        self._frontier_bucket_index = 0
        self._pending_delta = 0.0
        self._pending_reward = 0.0
        self._pending_ground_effect_adjustment = 0.0
        self._pending_speed_adjustment = 0.0
        self._pending_energy_refill_bonus = 0.0
        self._pending_energy_gain_reward = 0.0
        self._pending_frames = 0

    @property
    def frontier_distance(self) -> float:
        return self._frontier_distance

    @property
    def frontier_bucket_index(self) -> int:
        return self._frontier_bucket_index

    @property
    def pending_delta(self) -> float:
        return self._pending_delta

    @property
    def pending_frames(self) -> int:
        return self._pending_frames

    def reset(self, telemetry: FZeroXTelemetry | None) -> None:
        self._progress.reset(telemetry)
        self._frontier_distance = 0.0
        self._frontier_bucket_index = 0
        self._clear_pending()

    def reset_inactive(self) -> None:
        self.reset(None)

    def ensure_origin(self, telemetry: FZeroXTelemetry) -> None:
        self._progress.ensure_origin(telemetry)

    def relative_distance(self, race_distance: float) -> float:
        return self._progress.relative_distance(race_distance)

    def step(
        self,
        summary: StepSummary,
        status: StepStatus,
        *,
        weights: SharedRewardWeights,
        progress_multiplier: float,
        progress_suspended: bool = False,
        race_distance: float | None = None,
        energy_refill_bonus_for_progress: Callable[[float], float],
        energy_gain_reward_for_progress: Callable[[float], float],
    ) -> FrontierReward:
        progress_distance = summary.max_race_distance if race_distance is None else race_distance
        relative_progress = self._progress.relative_distance(progress_distance)
        bucket_distance = weights.progress_bucket_distance
        if bucket_distance <= 0.0:
            return zero_frontier_reward()
        crossed_bucket_count = int((relative_progress - self._frontier_distance) // bucket_distance)
        if crossed_bucket_count <= 0:
            return zero_frontier_reward()

        self._advance_frontier(crossed_bucket_count, weights=weights)
        if progress_suspended:
            return zero_frontier_reward()

        progress_reward = crossed_bucket_count * weights.progress_bucket_reward
        ground_effect_adjustment = progress_reward * (max(float(progress_multiplier), 0.0) - 1.0)
        surface_adjusted_progress = progress_reward + ground_effect_adjustment
        speed_multiplier = progress_speed_multiplier(
            summary.max_race_distance_speed_kph,
            weights=weights,
        )
        speed_adjustment = surface_adjusted_progress * (speed_multiplier - 1.0)
        energy_refill_bonus = energy_refill_bonus_for_progress(progress_reward)
        energy_gain_reward = energy_gain_reward_for_progress(progress_reward)
        interval_frames = max(int(weights.progress_reward_interval_frames), 1)
        if interval_frames <= 1:
            return FrontierReward(
                progress=progress_reward,
                ground_effect_adjustment=ground_effect_adjustment,
                speed_adjustment=speed_adjustment,
                energy_refill_bonus=energy_refill_bonus,
                energy_gain_reward=energy_gain_reward,
            )

        self._pending_delta += crossed_bucket_count * bucket_distance
        self._pending_reward += progress_reward
        self._pending_ground_effect_adjustment += ground_effect_adjustment
        self._pending_speed_adjustment += speed_adjustment
        self._pending_energy_refill_bonus += energy_refill_bonus
        self._pending_energy_gain_reward += energy_gain_reward
        self._pending_frames += max(int(summary.frames_run), 0)
        if (
            self._pending_frames < interval_frames
            and status.termination_reason is None
            and status.truncation_reason is None
        ):
            return FrontierReward(
                progress=0.0,
                ground_effect_adjustment=0.0,
                speed_adjustment=0.0,
                energy_refill_bonus=0.0,
                energy_gain_reward=0.0,
            )

        pending_reward = self._pending_reward
        pending_ground_effect_adjustment = self._pending_ground_effect_adjustment
        pending_speed_adjustment = self._pending_speed_adjustment
        pending_refill_bonus = self._pending_energy_refill_bonus
        pending_gain_reward = self._pending_energy_gain_reward
        self._clear_pending()
        return FrontierReward(
            progress=pending_reward,
            ground_effect_adjustment=pending_ground_effect_adjustment,
            speed_adjustment=pending_speed_adjustment,
            energy_refill_bonus=pending_refill_bonus,
            energy_gain_reward=pending_gain_reward,
        )

    def info(
        self,
        telemetry: FZeroXTelemetry | None,
        *,
        weights: SharedRewardWeights,
    ) -> dict[str, object]:
        info: dict[str, object] = {
            "frontier_progress_distance": self._frontier_distance,
            "frontier_progress_bucket_index": self._frontier_bucket_index,
            "progress_bucket_distance": weights.progress_bucket_distance,
            "suspend_progress_while_outside_track_bounds": (
                _suspend_progress_while_outside_track_bounds(weights=weights)
            ),
            "progress_track_distance_tolerance": weights.progress_track_distance_tolerance,
            "progress_bucket_reward": weights.progress_bucket_reward,
            "progress_reward_interval_frames": weights.progress_reward_interval_frames,
            "progress_speed_min_kph": weights.progress_speed_min_kph,
            "progress_speed_min_multiplier": weights.progress_speed_min_multiplier,
            "progress_speed_reference_kph": weights.progress_speed_reference_kph,
            "progress_speed_max_kph": weights.progress_speed_max_kph,
            "progress_speed_max_multiplier": weights.progress_speed_max_multiplier,
            "progress_speed_curve_power": weights.progress_speed_curve_power,
            "pending_progress_reward_delta": self._pending_delta,
            "pending_progress_reward_frames": self._pending_frames,
        }
        if telemetry is None or not telemetry.in_race_mode:
            return info
        self._progress.ensure_origin(telemetry)
        info["relative_progress"] = self._progress.relative_distance(telemetry.player.race_distance)
        return info

    def _clear_pending(self) -> None:
        self._pending_delta = 0.0
        self._pending_reward = 0.0
        self._pending_ground_effect_adjustment = 0.0
        self._pending_speed_adjustment = 0.0
        self._pending_energy_refill_bonus = 0.0
        self._pending_energy_gain_reward = 0.0
        self._pending_frames = 0

    def _advance_frontier(
        self,
        crossed_bucket_count: int,
        *,
        weights: SharedRewardWeights,
    ) -> None:
        self._frontier_distance += crossed_bucket_count * weights.progress_bucket_distance
        self._frontier_bucket_index = int(
            self._frontier_distance // weights.progress_bucket_distance
        )


def zero_frontier_reward() -> FrontierReward:
    return FrontierReward(
        progress=0.0,
        ground_effect_adjustment=0.0,
        speed_adjustment=0.0,
        energy_refill_bonus=0.0,
        energy_gain_reward=0.0,
    )


def progress_speed_multiplier(speed_kph: float, *, weights: SharedRewardWeights) -> float:
    min_kph = max(float(weights.progress_speed_min_kph), 0.0)
    reference_kph = max(float(weights.progress_speed_reference_kph), min_kph + 1e-9)
    max_kph = max(float(weights.progress_speed_max_kph), reference_kph + 1e-9)
    curve_power = max(float(weights.progress_speed_curve_power), 1e-9)
    speed = max(float(speed_kph), 0.0)
    min_multiplier = max(float(weights.progress_speed_min_multiplier), 0.0)

    if speed <= min_kph:
        return min_multiplier
    if speed <= reference_kph:
        ratio = ((speed - min_kph) / (reference_kph - min_kph)) ** curve_power
        return min_multiplier + ((1.0 - min_multiplier) * ratio)

    ratio = min((speed - reference_kph) / (max_kph - reference_kph), 1.0)
    shaped_ratio = 1.0 - ((1.0 - ratio) ** curve_power)
    max_multiplier = max(float(weights.progress_speed_max_multiplier), 0.0)
    return 1.0 + ((max_multiplier - 1.0) * shaped_ratio)


def _suspend_progress_while_outside_track_bounds(*, weights: SharedRewardWeights) -> bool:
    return bool(getattr(weights, "suspend_progress_while_outside_track_bounds", False))
