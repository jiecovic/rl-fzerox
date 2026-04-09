# src/rl_fzerox/core/envs/actions/steer_drive_boost_drift.py
from __future__ import annotations

import numpy as np
from gymnasium import spaces

from fzerox_emulator import JOYPAD_L2, JOYPAD_R, JOYPAD_Y, ControllerState, joypad_mask
from rl_fzerox.core.config.schema import ActionConfig
from rl_fzerox.core.envs.actions.base import (
    ActionValue,
    DiscreteActionDimension,
    idle_action,
    multidiscrete_space,
    parse_discrete_action,
    steer_values,
)
from rl_fzerox.core.envs.actions.steer_drive import DRIVE_MODES

# Mupen64Plus-Next's standard RetroPad mapping exposes the in-game N64 B/Z/R
# controls as RetroPad Y/L2/R respectively. In F-Zero X that means:
# - N64 B -> boost
# - N64 Z -> left slide / left DT input
# - N64 R -> right slide / right DT input
BOOST_MASK = joypad_mask(JOYPAD_Y)
DRIFT_LEFT_MASK = joypad_mask(JOYPAD_L2)
DRIFT_RIGHT_MASK = joypad_mask(JOYPAD_R)
SHOULDER_MASKS = (
    0,
    DRIFT_LEFT_MASK,
    DRIFT_RIGHT_MASK,
)


class SteerDriveBoostDriftActionAdapter:
    """Map steering, drive-mode, boost, and explicit shoulder inputs."""

    def __init__(self, config: ActionConfig) -> None:
        self._steer_values = steer_values(config.steer_buckets)
        self._action_space = multidiscrete_space(
            config.steer_buckets,
            len(DRIVE_MODES),
            2,
            len(SHOULDER_MASKS),
        )
        self._idle_action = idle_action(config.steer_buckets // 2, 0, 0, 0)

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

        steer_index, drive_mode_index, boost_index, shoulder_index = _parse_action_quad(
            action,
            steer_bucket_count=len(self._steer_values),
        )
        steer_value = float(self._steer_values[steer_index])
        drive_mode = DRIVE_MODES[drive_mode_index]
        joypad_mask = drive_mode.joypad_mask | SHOULDER_MASKS[shoulder_index]
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
    steer_index, drive_mode_index, boost_index, shoulder_index = parse_discrete_action(
        action,
        action_label="Steer-drive-boost-drift",
        dimensions=(
            DiscreteActionDimension("steer_bucket", steer_bucket_count),
            DiscreteActionDimension("drive_mode", len(DRIVE_MODES)),
            DiscreteActionDimension("boost", 2),
            DiscreteActionDimension("shoulder", len(SHOULDER_MASKS)),
        ),
    )
    return steer_index, drive_mode_index, boost_index, shoulder_index
