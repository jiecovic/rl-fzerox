# src/rl_fzerox/core/envs/rewards/reward_main/energy.py
from __future__ import annotations

from fzerox_emulator import FZeroXTelemetry, StepSummary
from rl_fzerox.core.envs.course_effects import on_refill_surface
from rl_fzerox.core.envs.rewards.reward_main.weights import RewardMainWeights


class EnergyRefillRewardTracker:
    """Track refill reward cooldown without rewarding stationary recharge."""

    def __init__(self) -> None:
        self._cooldown_frames_remaining = 0

    @property
    def cooldown_frames_remaining(self) -> int:
        return self._cooldown_frames_remaining

    def reset(self, _telemetry: FZeroXTelemetry | None) -> None:
        self._cooldown_frames_remaining = 0

    def reset_inactive(self) -> None:
        self.reset(None)

    def start_collision_cooldown(
        self,
        summary: StepSummary,
        telemetry: FZeroXTelemetry,
        *,
        weights: RewardMainWeights,
    ) -> None:
        cooldown_frames = max(int(weights.energy_refill_collision_cooldown_frames), 0)
        if cooldown_frames <= 0:
            return
        if summary.collision_recoil_active_frames <= 0 and summary.damage_taken_frames <= 0:
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
        weights: RewardMainWeights,
    ) -> float:
        multiplier = weights.energy_refill_progress_multiplier
        max_energy = float(telemetry.player.max_energy)
        current_energy = float(telemetry.player.energy)
        if (
            progress_reward <= 0.0
            or multiplier <= 1.0
            or not telemetry.player.on_energy_refill
            or max_energy <= 0.0
            or current_energy >= max_energy
            or summary.reverse_active_frames > 0
            or telemetry.player.reverse_timer > 0
            or self._cooldown_frames_remaining > 0
        ):
            return 0.0

        return progress_reward * (multiplier - 1.0)

    def gain_reward(
        self,
        progress_reward: float,
        summary: StepSummary,
        telemetry: FZeroXTelemetry,
        *,
        weights: RewardMainWeights,
    ) -> float:
        gain_reward = max(float(weights.energy_gain_reward), 0.0)
        energy_gain = max(float(summary.energy_gain_total), 0.0)
        if (
            progress_reward <= 0.0
            or gain_reward <= 0.0
            or energy_gain <= 0.0
            or not on_refill_surface(telemetry)
            or summary.reverse_active_frames > 0
            or telemetry.player.reverse_timer > 0
            or self._cooldown_frames_remaining > 0
        ):
            return 0.0

        return energy_gain * gain_reward

    def advance_cooldown(self, frames_run: int) -> None:
        self._cooldown_frames_remaining = max(
            self._cooldown_frames_remaining - max(int(frames_run), 0),
            0,
        )

    def finish_step(self) -> None:
        """Complete one reward step.

        The hook exists beside the stateful reward trackers that do have
        per-step finalization; energy refill currently only advances cooldown.
        """

    def info(self) -> dict[str, object]:
        return {
            "energy_refill_cooldown_frames_remaining": self._cooldown_frames_remaining,
        }
