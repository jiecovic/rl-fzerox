# src/rl_fzerox/core/envs/actions/steer_drive.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

from fzerox_emulator import JOYPAD_A, JOYPAD_B, ControllerState, joypad_mask
from fzerox_emulator.arrays import ActionMask, DiscreteAction
from rl_fzerox.core.config.schema import ActionConfig, ActionRuntimeConfig
from rl_fzerox.core.envs.actions.base import (
    ActionValue,
    DiscreteActionDimension,
    build_flat_action_mask,
    idle_action,
    multidiscrete_space,
    parse_discrete_action,
    steer_values,
)

# Mupen64Plus-Next's standard RetroPad mapping exposes the in-game N64 A/C-Down
# buttons on RetroPad B/A. F-Zero X names those controls Accelerate/Air Brake.
ACCELERATE_MASK = joypad_mask(JOYPAD_B)
AIR_BRAKE_MASK = joypad_mask(JOYPAD_A)

# Legacy aliases kept for existing imports and older tests/config-adjacent code.
THROTTLE_MASK = ACCELERATE_MASK
BRAKE_MASK = AIR_BRAKE_MASK


@dataclass(frozen=True, slots=True)
class DriveMode:
    """One held drive-mode mapping for the current controller profile."""

    label: str
    joypad_mask: int


DRIVE_MODES: tuple[DriveMode, ...] = (
    DriveMode(label="coast", joypad_mask=0),
    DriveMode(label="accelerate", joypad_mask=ACCELERATE_MASK),
    DriveMode(label="air_brake", joypad_mask=AIR_BRAKE_MASK),
)


class SteerDriveActionAdapter:
    """Map MultiDiscrete steering and drive-mode actions to controls."""

    def __init__(self, config: ActionConfig | ActionRuntimeConfig) -> None:
        self._steer_values = steer_values(
            config.steer_buckets,
            response_power=float(config.steer_response_power),
        )
        self._action_dimensions = (
            DiscreteActionDimension("steer", config.steer_buckets),
            DiscreteActionDimension("drive", len(DRIVE_MODES)),
        )
        self._action_space = multidiscrete_space(
            *(dimension.size for dimension in self._action_dimensions)
        )
        self._idle_action = idle_action(config.steer_buckets // 2, 0)

    @property
    def action_space(self) -> spaces.MultiDiscrete:
        """Return the current policy action space."""

        return self._action_space

    @property
    def idle_action(self) -> DiscreteAction:
        """Return the neutral action value for this action space."""

        return np.array(self._idle_action, copy=True)

    @property
    def action_dimensions(self) -> tuple[DiscreteActionDimension, ...]:
        """Return the ordered MultiDiscrete heads for this adapter."""

        return self._action_dimensions

    def decode(self, action: ActionValue) -> ControllerState:
        """Translate one policy action into a held steering/drive state."""

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

    def action_mask(
        self,
        *,
        base_overrides: dict[str, tuple[int, ...]] | None = None,
        stage_overrides: dict[str, tuple[int, ...]] | None = None,
        dynamic_overrides: dict[str, tuple[int, ...]] | None = None,
    ) -> ActionMask:
        """Return one flattened boolean action mask for the current adapter."""

        return build_flat_action_mask(
            self._action_dimensions,
            base_overrides=base_overrides,
            stage_overrides=stage_overrides,
            dynamic_overrides=dynamic_overrides,
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
            DiscreteActionDimension("steer", steer_bucket_count),
            DiscreteActionDimension("drive", len(DRIVE_MODES)),
        ),
    )
    return steer_index, drive_mode_index
