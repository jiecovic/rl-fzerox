# src/rl_fzerox/core/envs/rewards/race_v3/progress.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from fzerox_emulator import FZeroXTelemetry, StepStatus, StepSummary
from rl_fzerox.core.envs.rewards.progress import EpisodeProgressState
from rl_fzerox.core.envs.rewards.race_v3.weights import RaceV3RewardWeights


@dataclass(frozen=True, slots=True)
class FrontierReward:
    """Progress and refill bonus paid from newly crossed spline buckets."""

    progress: float
    energy_refill_bonus: float


class FrontierProgressRewardTracker:
    """Track one-way coverage of the game spline and pay crossed buckets once."""

    def __init__(self) -> None:
        self._progress = EpisodeProgressState()
        self._frontier_distance = 0.0
        self._frontier_bucket_index = 0
        self._pending_delta = 0.0
        self._pending_reward = 0.0
        self._pending_energy_refill_bonus = 0.0
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
        weights: RaceV3RewardWeights,
        energy_refill_bonus_for_progress: Callable[[float], float],
    ) -> FrontierReward:
        max_relative_progress = self._progress.relative_distance(summary.max_race_distance)
        bucket_distance = weights.progress_bucket_distance
        if bucket_distance <= 0.0:
            return FrontierReward(progress=0.0, energy_refill_bonus=0.0)
        current_bucket_index = int(max_relative_progress // bucket_distance)
        crossed_bucket_count = current_bucket_index - self._frontier_bucket_index
        if crossed_bucket_count <= 0:
            return FrontierReward(progress=0.0, energy_refill_bonus=0.0)

        self._frontier_bucket_index = current_bucket_index
        self._frontier_distance = current_bucket_index * bucket_distance
        progress_reward = crossed_bucket_count * weights.progress_bucket_reward
        energy_refill_bonus = energy_refill_bonus_for_progress(progress_reward)
        interval_frames = max(int(weights.progress_reward_interval_frames), 1)
        if interval_frames <= 1:
            return FrontierReward(
                progress=progress_reward,
                energy_refill_bonus=energy_refill_bonus,
            )

        self._pending_delta += crossed_bucket_count * bucket_distance
        self._pending_reward += progress_reward
        self._pending_energy_refill_bonus += energy_refill_bonus
        self._pending_frames += max(int(summary.frames_run), 0)
        if (
            self._pending_frames < interval_frames
            and status.termination_reason is None
            and status.truncation_reason is None
        ):
            return FrontierReward(progress=0.0, energy_refill_bonus=0.0)

        pending_reward = self._pending_reward
        pending_refill_bonus = self._pending_energy_refill_bonus
        self._clear_pending()
        return FrontierReward(
            progress=pending_reward,
            energy_refill_bonus=pending_refill_bonus,
        )

    def info(
        self,
        telemetry: FZeroXTelemetry | None,
        *,
        weights: RaceV3RewardWeights,
    ) -> dict[str, object]:
        info: dict[str, object] = {
            "frontier_progress_distance": self._frontier_distance,
            "frontier_progress_bucket_index": self._frontier_bucket_index,
            "progress_bucket_distance": weights.progress_bucket_distance,
            "progress_bucket_reward": weights.progress_bucket_reward,
            "progress_reward_interval_frames": weights.progress_reward_interval_frames,
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
        self._pending_energy_refill_bonus = 0.0
        self._pending_frames = 0
