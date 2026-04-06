# src/rl_fzerox/core/actions/steer_drive.py
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

from rl_fzerox._native import JOYPAD_B, joypad_mask
from rl_fzerox.core.actions.base import ActionValue
from rl_fzerox.core.config.models import ActionConfig
from rl_fzerox.core.emulator.control import ControllerState

# Mupen64Plus-Next's standard RetroPad mapping exposes the in-game N64 A button
# on RetroPad B, which is the acceleration input F-Zero X uses during races.
THROTTLE_MASK = joypad_mask(JOYPAD_B)


@dataclass(frozen=True, slots=True)
class DriveMode:
    """One held drive-mode mapping for the current controller profile."""

    label: str
    joypad_mask: int


DRIVE_MODES: tuple[DriveMode, ...] = (
    DriveMode(label="coast", joypad_mask=0),
    DriveMode(label="throttle", joypad_mask=THROTTLE_MASK),
)


class SteerDriveActionAdapter:
    """Map MultiDiscrete steering and drive-mode actions to controls."""

    def __init__(self, config: ActionConfig) -> None:
        self._steer_values = np.linspace(
            -1.0,
            1.0,
            num=config.steer_buckets,
            dtype=np.float32,
        )
        self._action_space = spaces.MultiDiscrete(
            np.array([config.steer_buckets, len(DRIVE_MODES)], dtype=np.int64)
        )
        self._idle_action = np.array([config.steer_buckets // 2, 0], dtype=np.int64)

    @property
    def action_space(self) -> spaces.MultiDiscrete:
        """Return the current policy action space."""

        return self._action_space

    @property
    def idle_action(self) -> np.ndarray:
        """Return the neutral action value for this action space."""

        return np.array(self._idle_action, copy=True)

    def decode(self, action: ActionValue) -> ControllerState:
        """Translate one policy action into a held steering/throttle state."""

        steer_index, drive_mode_index = _parse_action_pair(
            action,
            steer_bucket_count=len(self._steer_values),
        )
        steer_value = float(self._steer_values[steer_index])
        drive_mode = DRIVE_MODES[drive_mode_index]
        return ControllerState(
            joypad_mask=drive_mode.joypad_mask,
            left_stick_x=steer_value,
        )


def _parse_action_pair(
    action: ActionValue,
    *,
    steer_bucket_count: int,
) -> tuple[int, int]:
    if isinstance(action, np.ndarray):
        values = action.astype(np.int64, copy=False).reshape(-1).tolist()
    elif isinstance(action, np.integer):
        values = [int(action)]
    elif isinstance(action, Sequence) and not isinstance(action, str | bytes):
        values = [int(value) for value in action]
    else:
        values = [int(action)]

    if len(values) != 2:
        raise ValueError(
            "Steer-drive actions must contain exactly 2 values: "
            "[steer_bucket, drive_mode]"
        )

    steer_index, drive_mode_index = values
    if not 0 <= steer_index < steer_bucket_count:
        raise ValueError(f"Invalid steer bucket index {steer_index}")
    if not 0 <= drive_mode_index < len(DRIVE_MODES):
        raise ValueError(f"Invalid drive mode index {drive_mode_index}")
    return steer_index, drive_mode_index
