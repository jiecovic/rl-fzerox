# src/rl_fzerox/core/envs/engine/policy_drive/frame.py
"""Policy-drive frame data."""

from __future__ import annotations

from dataclasses import dataclass

from fzerox_emulator.arrays import ControllerMaskBatch, DisplayFrames
from rl_fzerox.core.envs.observations import ObservationValue


@dataclass(frozen=True, slots=True)
class PolicyDriveFrame:
    """One policy-owned race frame without Gym episode lifecycle semantics."""

    observation: ObservationValue
    reward: float
    info: dict[str, object]
    display_frames: DisplayFrames
    display_controller_masks: ControllerMaskBatch
