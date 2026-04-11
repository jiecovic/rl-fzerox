# src/rl_fzerox/core/envs/rewards/race_v2/energy.py
from __future__ import annotations

from fzerox_emulator import FZeroXTelemetry, StepSummary
from rl_fzerox.core.envs.rewards.race_v2.weights import RaceV2RewardWeights


class EnergyRewardState:
    """Track energy reward cooldowns and one-shot full-refill edges."""

    def __init__(self) -> None:
        self.gain_cooldown_frames_remaining = 0
        self.full_refill_cooldown_frames_remaining = 0
        self.previous_energy_full = False

    def reset(self, telemetry: FZeroXTelemetry | None) -> None:
        self.gain_cooldown_frames_remaining = 0
        self.full_refill_cooldown_frames_remaining = 0
        self.previous_energy_full = (
            energy_is_full(telemetry) if telemetry is not None and telemetry.in_race_mode else False
        )

    def update_previous(self, telemetry: FZeroXTelemetry) -> None:
        self.previous_energy_full = energy_is_full(telemetry)

    def start_gain_cooldown(
        self,
        summary: StepSummary,
        weights: RaceV2RewardWeights,
    ) -> None:
        cooldown_frames = max(int(weights.energy_gain_collision_cooldown_frames), 0)
        if not summary.entered_collision_recoil or cooldown_frames <= 0:
            return
        self.gain_cooldown_frames_remaining = max(
            self.gain_cooldown_frames_remaining,
            cooldown_frames,
        )

    def energy_gain_reward(
        self,
        summary: StepSummary,
        weights: RaceV2RewardWeights,
    ) -> float:
        if self.gain_cooldown_frames_remaining > 0:
            return 0.0
        return summary.energy_gain_total * weights.energy_gain_reward_scale

    def full_refill_bonus(
        self,
        telemetry: FZeroXTelemetry,
        weights: RaceV2RewardWeights,
    ) -> float:
        bonus = weights.energy_full_refill_bonus
        if bonus == 0.0 or self.full_refill_cooldown_frames_remaining > 0:
            return 0.0
        if not energy_is_full(telemetry) or self.previous_energy_full:
            return 0.0
        self.full_refill_cooldown_frames_remaining = max(
            int(weights.energy_full_refill_cooldown_frames),
            0,
        )
        return bonus

    def advance_cooldowns(self, frames_run: int) -> None:
        elapsed = max(int(frames_run), 0)
        self.gain_cooldown_frames_remaining = max(
            self.gain_cooldown_frames_remaining - elapsed,
            0,
        )
        self.full_refill_cooldown_frames_remaining = max(
            self.full_refill_cooldown_frames_remaining - elapsed,
            0,
        )


def energy_loss_danger_weight(
    telemetry: FZeroXTelemetry,
    weights: RaceV2RewardWeights,
) -> float:
    safe_fraction = max(float(weights.energy_loss_safe_fraction), 0.0)
    if safe_fraction <= 0.0:
        return 1.0

    energy_fraction = normalized_energy(telemetry)
    if energy_fraction >= safe_fraction:
        return 0.0

    danger = (safe_fraction - energy_fraction) / safe_fraction
    return danger**weights.energy_loss_danger_power


def normalized_energy(telemetry: FZeroXTelemetry) -> float:
    max_energy = float(telemetry.player.max_energy)
    if max_energy <= 0.0:
        return 0.0
    return max(0.0, min(1.0, float(telemetry.player.energy) / max_energy))


def energy_is_full(telemetry: FZeroXTelemetry) -> bool:
    max_energy = float(telemetry.player.max_energy)
    return max_energy > 0.0 and float(telemetry.player.energy) >= max_energy
