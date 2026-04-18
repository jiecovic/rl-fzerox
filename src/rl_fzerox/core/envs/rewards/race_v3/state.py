# src/rl_fzerox/core/envs/rewards/race_v3/state.py
from __future__ import annotations

from fzerox_emulator import FZeroXTelemetry
from rl_fzerox.core.envs.rewards.common import RewardActionContext


def normalized_energy(telemetry: FZeroXTelemetry | None) -> float:
    if telemetry is None or not telemetry.in_race_mode:
        return 0.0
    max_energy = float(telemetry.player.max_energy)
    if max_energy <= 0.0:
        return 0.0
    return max(0.0, min(1.0, float(telemetry.player.energy) / max_energy))


def normalized_steer_level(action_context: RewardActionContext | None) -> float | None:
    if action_context is None or action_context.steer_level is None:
        return None
    return max(-1.0, min(1.0, float(action_context.steer_level)))
