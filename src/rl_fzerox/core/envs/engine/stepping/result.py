# src/rl_fzerox/core/envs/engine/stepping/result.py
from __future__ import annotations

from dataclasses import dataclass

from fzerox_emulator.arrays import RgbFrame
from rl_fzerox.core.envs.observations import ObservationValue


@dataclass(frozen=True)
class WatchEnvStep:
    """Gym step result plus watch-only display frames from repeated inner frames."""

    observation: ObservationValue
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object]
    display_frames: tuple[RgbFrame, ...]

    def gym_result(self) -> tuple[ObservationValue, float, bool, bool, dict[str, object]]:
        return self.observation, self.reward, self.terminated, self.truncated, self.info
