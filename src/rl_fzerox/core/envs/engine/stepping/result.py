# src/rl_fzerox/core/envs/engine/stepping/result.py
"""Step result containers shared by Gym and watch runtimes.

These dataclasses keep observation, reward, terminal flags, info dictionaries,
and optional captured media together after one env step.
"""

from __future__ import annotations

from dataclasses import dataclass

from fzerox_emulator.arrays import (
    AudioFrameCounts,
    ControllerMaskBatch,
    DisplayFrames,
    Pcm16Samples,
)
from rl_fzerox.core.envs.observations import ObservationValue


@dataclass(frozen=True)
class WatchEnvStep:
    """Gym step result plus watch-only display frames from repeated inner frames."""

    observation: ObservationValue
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object]
    display_frames: DisplayFrames
    display_controller_masks: ControllerMaskBatch = ()
    audio_samples: Pcm16Samples = ()
    audio_frame_counts: AudioFrameCounts = ()

    def gym_result(self) -> tuple[ObservationValue, float, bool, bool, dict[str, object]]:
        return self.observation, self.reward, self.terminated, self.truncated, self.info
