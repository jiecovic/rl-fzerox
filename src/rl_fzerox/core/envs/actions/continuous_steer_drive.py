# src/rl_fzerox/core/envs/actions/continuous_steer_drive.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

from fzerox_emulator import ControllerState
from fzerox_emulator.arrays import ActionMask, ContinuousAction
from rl_fzerox.core.config.schema import ActionConfig, ActionRuntimeConfig
from rl_fzerox.core.envs.actions.base import (
    ActionValue,
    DiscreteActionDimension,
    shape_steer_value,
)
from rl_fzerox.core.envs.actions.continuous_controls import (
    ContinuousDriveDecoder,
    continuous_action_array,
)
from rl_fzerox.core.envs.actions.steer_drive_boost_lean import (
    LEAN_LEFT_MASK,
    LEAN_RIGHT_MASK,
)


@dataclass(frozen=True, slots=True)
class _ContinuousActionSizes:
    steer_drive: int = 2
    steer_drive_lean: int = 3


_CONTINUOUS_ACTION_SIZES = _ContinuousActionSizes()


class ContinuousSteerDriveActionAdapter:
    """Map continuous steering and accelerate intent to simple held controls."""

    def __init__(self, config: ActionConfig | ActionRuntimeConfig) -> None:
        self._steer_response_power = float(config.steer_response_power)
        self._drive_decoder = ContinuousDriveDecoder(
            mode=config.continuous_drive_mode,
            deadzone=float(config.continuous_drive_deadzone),
        )
        self._action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._idle_action = np.zeros(_CONTINUOUS_ACTION_SIZES.steer_drive, dtype=np.float32)

    @property
    def action_space(self) -> spaces.Box:
        """Return the current policy action space."""

        return self._action_space

    @property
    def idle_action(self) -> ContinuousAction:
        """Return the neutral action value for this action space."""

        return np.array(self._idle_action, copy=True)

    @property
    def action_dimensions(self) -> tuple[DiscreteActionDimension, ...]:
        """Return no discrete heads; this adapter is not maskable."""

        return ()

    def decode(self, action: ActionValue) -> ControllerState:
        """Translate one continuous action into a held steering/accelerate state."""

        steer, drive = _parse_continuous_pair(action)
        joypad_mask = self._drive_decoder.decode(drive)

        return ControllerState(
            joypad_mask=joypad_mask,
            left_stick_x=shape_steer_value(
                steer,
                response_power=self._steer_response_power,
            ),
        )

    def reset(self) -> None:
        """Reset the drive PWM accumulator for a new episode."""

        self._drive_decoder.reset()

    def action_mask(
        self,
        *,
        base_overrides: dict[str, tuple[int, ...]] | None = None,
        stage_overrides: dict[str, tuple[int, ...]] | None = None,
        dynamic_overrides: dict[str, tuple[int, ...]] | None = None,
    ) -> ActionMask:
        """Return an empty mask so watch can call action_masks() safely."""

        _ = (base_overrides, stage_overrides, dynamic_overrides)
        return np.ones(0, dtype=bool)


class ContinuousSteerDriveLeanActionAdapter:
    """Map continuous steer/drive/lean intent to held controller inputs."""

    def __init__(self, config: ActionConfig | ActionRuntimeConfig) -> None:
        self._steer_response_power = float(config.steer_response_power)
        self._drive_decoder = ContinuousDriveDecoder(
            mode=config.continuous_drive_mode,
            deadzone=float(config.continuous_drive_deadzone),
        )
        self._lean_deadzone = float(config.continuous_lean_deadzone)
        self._action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._idle_action = np.zeros(_CONTINUOUS_ACTION_SIZES.steer_drive_lean, dtype=np.float32)

    @property
    def action_space(self) -> spaces.Box:
        """Return the current policy action space."""

        return self._action_space

    @property
    def idle_action(self) -> ContinuousAction:
        """Return the neutral action value for this action space."""

        return np.array(self._idle_action, copy=True)

    @property
    def action_dimensions(self) -> tuple[DiscreteActionDimension, ...]:
        """Return no discrete heads; this adapter is not maskable."""

        return ()

    def decode(self, action: ActionValue) -> ControllerState:
        """Translate one continuous action into steering, drive, and lean."""

        steer, drive, lean = _parse_continuous_triplet(action)
        joypad_mask = self._drive_decoder.decode(drive)
        if lean > self._lean_deadzone:
            joypad_mask |= LEAN_RIGHT_MASK
        elif lean < -self._lean_deadzone:
            joypad_mask |= LEAN_LEFT_MASK

        return ControllerState(
            joypad_mask=joypad_mask,
            left_stick_x=shape_steer_value(
                steer,
                response_power=self._steer_response_power,
            ),
        )

    def reset(self) -> None:
        """Reset the drive PWM accumulator for a new episode."""

        self._drive_decoder.reset()

    def action_mask(
        self,
        *,
        base_overrides: dict[str, tuple[int, ...]] | None = None,
        stage_overrides: dict[str, tuple[int, ...]] | None = None,
        dynamic_overrides: dict[str, tuple[int, ...]] | None = None,
    ) -> ActionMask:
        """Return an empty mask so watch can call action_masks() safely."""

        _ = (base_overrides, stage_overrides, dynamic_overrides)
        return np.ones(0, dtype=bool)


def _parse_continuous_pair(action: ActionValue) -> tuple[float, float]:
    values = continuous_action_array(
        action,
        expected_size=_CONTINUOUS_ACTION_SIZES.steer_drive,
        action_label="Continuous steer-drive",
        field_labels=("steer", "drive"),
    )
    steer = float(np.clip(values[0], -1.0, 1.0))
    drive = float(np.clip(values[1], -1.0, 1.0))
    return steer, drive


def _parse_continuous_triplet(action: ActionValue) -> tuple[float, float, float]:
    values = continuous_action_array(
        action,
        expected_size=_CONTINUOUS_ACTION_SIZES.steer_drive_lean,
        action_label="Continuous steer-drive-lean",
        field_labels=("steer", "drive", "lean"),
    )
    steer = float(np.clip(values[0], -1.0, 1.0))
    drive = float(np.clip(values[1], -1.0, 1.0))
    lean = float(np.clip(values[2], -1.0, 1.0))
    return steer, drive, lean
