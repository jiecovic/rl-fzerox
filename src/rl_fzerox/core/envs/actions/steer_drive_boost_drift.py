# src/rl_fzerox/core/envs/actions/steer_drive_boost_drift.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

from rl_fzerox._native import JOYPAD_L2, JOYPAD_R, JOYPAD_Y, joypad_mask
from rl_fzerox.core.config.schema import ActionConfig
from rl_fzerox.core.emulator.control import ControllerState
from rl_fzerox.core.envs.actions.base import ActionValue, coerce_action_values
from rl_fzerox.core.envs.actions.steer_drive import THROTTLE_MASK

# Mupen64Plus-Next's standard RetroPad mapping exposes the in-game N64 B/Z/R
# controls as RetroPad Y/L2/R respectively. In F-Zero X that means:
# - N64 B -> boost
# - N64 Z -> left drift attack / left DT input
# - N64 R -> right drift attack / right DT input
BOOST_MASK = joypad_mask(JOYPAD_Y)
DRIFT_LEFT_MASK = joypad_mask(JOYPAD_L2)
DRIFT_RIGHT_MASK = joypad_mask(JOYPAD_R)


@dataclass(frozen=True, slots=True)
class DriveMode:
    """One held drive-mode mapping for the current controller profile."""

    label: str
    joypad_mask: int


@dataclass(frozen=True, slots=True)
class DriftMode:
    """One held drift-side mapping for the current controller profile."""

    label: str
    joypad_mask: int


DRIVE_MODES: tuple[DriveMode, ...] = (
    DriveMode(label="coast", joypad_mask=0),
    DriveMode(label="throttle", joypad_mask=THROTTLE_MASK),
)

DRIFT_MODES: tuple[DriftMode, ...] = (
    DriftMode(label="none", joypad_mask=0),
    DriftMode(label="left", joypad_mask=DRIFT_LEFT_MASK),
    DriftMode(label="right", joypad_mask=DRIFT_RIGHT_MASK),
)


class SteerDriveBoostDriftActionAdapter:
    """Map steering, throttle, boost, and drift-side actions to controls."""

    def __init__(self, config: ActionConfig) -> None:
        self._steer_values = np.linspace(
            -1.0,
            1.0,
            num=config.steer_buckets,
            dtype=np.float32,
        )
        self._action_space = spaces.MultiDiscrete(
            np.array(
                [
                    config.steer_buckets,
                    len(DRIVE_MODES),
                    2,
                    len(DRIFT_MODES),
                ],
                dtype=np.int64,
            )
        )
        self._idle_action = np.array([config.steer_buckets // 2, 0, 0, 0], dtype=np.int64)

    @property
    def action_space(self) -> spaces.MultiDiscrete:
        """Return the current policy action space."""

        return self._action_space

    @property
    def idle_action(self) -> np.ndarray:
        """Return the neutral action value for this action space."""

        return np.array(self._idle_action, copy=True)

    def decode(self, action: ActionValue) -> ControllerState:
        """Translate one policy action into a held steering/auxiliary state."""

        steer_index, drive_mode_index, boost_index, drift_mode_index = _parse_action_quad(
            action,
            steer_bucket_count=len(self._steer_values),
        )
        steer_value = float(self._steer_values[steer_index])
        drive_mode = DRIVE_MODES[drive_mode_index]
        drift_mode = DRIFT_MODES[drift_mode_index]
        joypad_mask = drive_mode.joypad_mask | drift_mode.joypad_mask
        if boost_index == 1:
            joypad_mask |= BOOST_MASK
        return ControllerState(
            joypad_mask=joypad_mask,
            left_stick_x=steer_value,
        )


def _parse_action_quad(
    action: ActionValue,
    *,
    steer_bucket_count: int,
) -> tuple[int, int, int, int]:
    values = coerce_action_values(action)
    if len(values) != 4:
        raise ValueError(
            "Steer-drive-boost-drift actions must contain exactly 4 values: "
            "[steer_bucket, drive_mode, boost, drift_mode]"
        )

    steer_index, drive_mode_index, boost_index, drift_mode_index = values
    if not 0 <= steer_index < steer_bucket_count:
        raise ValueError(f"Invalid steer bucket index {steer_index}")
    if not 0 <= drive_mode_index < len(DRIVE_MODES):
        raise ValueError(f"Invalid drive mode index {drive_mode_index}")
    if boost_index not in (0, 1):
        raise ValueError(f"Invalid boost index {boost_index}")
    if not 0 <= drift_mode_index < len(DRIFT_MODES):
        raise ValueError(f"Invalid drift mode index {drift_mode_index}")
    return steer_index, drive_mode_index, boost_index, drift_mode_index
