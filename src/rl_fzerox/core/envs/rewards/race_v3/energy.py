# src/rl_fzerox/core/envs/rewards/race_v3/energy.py
from __future__ import annotations

from fzerox_emulator import FZeroXTelemetry, StepSummary
from rl_fzerox.core.envs.laps import completed_race_laps
from rl_fzerox.core.envs.rewards.race_v3.state import normalized_energy
from rl_fzerox.core.envs.rewards.race_v3.weights import RaceV3RewardWeights


class EnergyRefillRewardTracker:
    """Track refill-specific reward state without rewarding stationary recharge."""

    def __init__(self) -> None:
        self._cooldown_frames_remaining = 0
        self._rewarded_full_refill_laps: set[int] = set()
        self._refill_since_full_fraction = 0.0
        self._previous_energy_fraction = 0.0

    @property
    def cooldown_frames_remaining(self) -> int:
        return self._cooldown_frames_remaining

    @property
    def refill_since_full_fraction(self) -> float:
        return self._refill_since_full_fraction

    @property
    def rewarded_full_refill_lap_count(self) -> int:
        return len(self._rewarded_full_refill_laps)

    def reset(self, telemetry: FZeroXTelemetry | None) -> None:
        self._cooldown_frames_remaining = 0
        self._rewarded_full_refill_laps.clear()
        self._refill_since_full_fraction = 0.0
        self._previous_energy_fraction = normalized_energy(telemetry)

    def reset_inactive(self) -> None:
        self.reset(None)

    def start_collision_cooldown(
        self,
        summary: StepSummary,
        *,
        weights: RaceV3RewardWeights,
    ) -> None:
        cooldown_frames = max(int(weights.energy_refill_collision_cooldown_frames), 0)
        if cooldown_frames <= 0:
            return
        if not summary.entered_collision_recoil and summary.damage_taken_frames <= 0:
            return
        self._cooldown_frames_remaining = max(
            self._cooldown_frames_remaining,
            cooldown_frames,
        )

    def progress_bonus(
        self,
        progress_reward: float,
        summary: StepSummary,
        telemetry: FZeroXTelemetry,
        *,
        weights: RaceV3RewardWeights,
    ) -> float:
        multiplier = weights.energy_refill_progress_multiplier
        if (
            progress_reward <= 0.0
            or multiplier <= 1.0
            or not telemetry.player.on_energy_refill
            or summary.reverse_active_frames > 0
            or telemetry.player.reverse_timer > 0
            or self._cooldown_frames_remaining > 0
        ):
            return 0.0

        return progress_reward * (multiplier - 1.0)

    def accumulate_since_full(
        self,
        summary: StepSummary,
        telemetry: FZeroXTelemetry,
    ) -> None:
        current_energy_fraction = normalized_energy(telemetry)
        if self._previous_energy_fraction >= 1.0 and current_energy_fraction >= 1.0:
            self._refill_since_full_fraction = 0.0
            return
        max_energy = float(telemetry.player.max_energy)
        if max_energy <= 0.0 or summary.energy_gain_total <= 0.0:
            return
        self._refill_since_full_fraction = min(
            self._refill_since_full_fraction
            + (max(float(summary.energy_gain_total), 0.0) / max_energy),
            1.0,
        )

    def full_refill_lap_reward(
        self,
        summary: StepSummary,
        telemetry: FZeroXTelemetry,
        *,
        weights: RaceV3RewardWeights,
    ) -> float:
        reward = weights.energy_full_refill_lap_bonus
        if (
            reward <= 0.0
            or summary.energy_gain_total <= 0.0
            or summary.reverse_active_frames > 0
            or telemetry.player.reverse_timer > 0
            or self._cooldown_frames_remaining > 0
        ):
            return 0.0
        current_energy_fraction = normalized_energy(telemetry)
        if self._previous_energy_fraction >= 1.0 or current_energy_fraction < 1.0:
            return 0.0
        recovered_fraction = self._refill_since_full_fraction
        if recovered_fraction < weights.energy_full_refill_min_gain_fraction:
            return 0.0

        lap_index = completed_race_laps(telemetry)
        if lap_index in self._rewarded_full_refill_laps:
            return 0.0
        self._rewarded_full_refill_laps.add(lap_index)
        return reward * min(recovered_fraction, 1.0)

    def advance_cooldown(self, frames_run: int) -> None:
        self._cooldown_frames_remaining = max(
            self._cooldown_frames_remaining - max(int(frames_run), 0),
            0,
        )

    def finish_step(self, telemetry: FZeroXTelemetry) -> None:
        current_energy_fraction = normalized_energy(telemetry)
        if current_energy_fraction >= 1.0:
            self._refill_since_full_fraction = 0.0
        self._previous_energy_fraction = current_energy_fraction

    def info(self) -> dict[str, object]:
        return {
            "energy_refill_cooldown_frames_remaining": self._cooldown_frames_remaining,
            "rewarded_full_refill_laps": len(self._rewarded_full_refill_laps),
            "energy_refill_since_full_fraction": self._refill_since_full_fraction,
        }
