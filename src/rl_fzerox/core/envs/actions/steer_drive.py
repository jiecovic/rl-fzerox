# src/rl_fzerox/core/envs/actions/steer_drive.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

from fzerox_emulator import JOYPAD_B, ControllerState, joypad_mask
from rl_fzerox.core.config.schema import ActionConfig
from rl_fzerox.core.envs.actions.base import (
    ActionValue,
    DiscreteActionDimension,
    idle_action,
    multidiscrete_space,
    parse_discrete_action,
    steer_values,
)

# Mupen64Plus-Next's standard RetroPad mapping exposes the in-game N64 A button
# on RetroPad B. In F-Zero X, N64 A is the acceleration input.
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
        self._steer_values = steer_values(config.steer_buckets)
        self._action_space = multidiscrete_space(config.steer_buckets, len(DRIVE_MODES))
        self._idle_action = idle_action(config.steer_buckets // 2, 0)

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
    steer_index, drive_mode_index = parse_discrete_action(
        action,
        action_label="Steer-drive",
        dimensions=(
            DiscreteActionDimension("steer_bucket", steer_bucket_count),
            DiscreteActionDimension("drive_mode", len(DRIVE_MODES)),
        ),
    )
    return steer_index, drive_mode_index
