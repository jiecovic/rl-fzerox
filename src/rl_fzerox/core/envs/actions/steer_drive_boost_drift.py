# src/rl_fzerox/core/envs/actions/steer_drive_boost_drift.py
from __future__ import annotations

import numpy as np
from gymnasium import spaces

from fzerox_emulator import JOYPAD_L2, JOYPAD_R, JOYPAD_Y, ControllerState, joypad_mask
from rl_fzerox.core.config.schema import ActionConfig
from rl_fzerox.core.envs.actions.base import (
    ActionValue,
    DiscreteActionDimension,
    build_flat_action_mask,
    idle_action,
    multidiscrete_space,
    parse_discrete_action,
    steer_values,
)
from rl_fzerox.core.envs.actions.steer_drive import ACCELERATE_MASK, AIR_BRAKE_MASK, DRIVE_MODES

# Mupen64Plus-Next's standard RetroPad mapping exposes the in-game N64 B/Z/R
# controls as RetroPad Y/L2/R respectively. In F-Zero X that means:
# - N64 B -> boost
# - N64 Z -> left slide / left DT input
# - N64 R -> right slide / right DT input
BOOST_MASK = joypad_mask(JOYPAD_Y)
SHOULDER_LEFT_MASK = joypad_mask(JOYPAD_L2)
SHOULDER_RIGHT_MASK = joypad_mask(JOYPAD_R)
# COMPAT SHIM: legacy "drift" mask names.
# The controls are the Z/R shoulder lean/slide inputs; keep old exports for
# historical tests, configs, and checkpoint metadata until those are retired.
DRIFT_LEFT_MASK = SHOULDER_LEFT_MASK
DRIFT_RIGHT_MASK = SHOULDER_RIGHT_MASK
SHOULDER_MASKS = (
    0,
    SHOULDER_LEFT_MASK,
    SHOULDER_RIGHT_MASK,
)


class SteerDriveBoostShoulderActionAdapter:
    """Map steering, drive-mode, boost, and explicit shoulder inputs."""

    def __init__(self, config: ActionConfig) -> None:
        self._steer_values = steer_values(config.steer_buckets)
        self._action_dimensions = (
            DiscreteActionDimension("steer", config.steer_buckets),
            DiscreteActionDimension("drive", len(DRIVE_MODES)),
            DiscreteActionDimension("boost", 2),
            DiscreteActionDimension("shoulder", len(SHOULDER_MASKS)),
        )
        self._action_space = multidiscrete_space(
            *(dimension.size for dimension in self._action_dimensions)
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

    @property
    def action_dimensions(self) -> tuple[DiscreteActionDimension, ...]:
        """Return the ordered MultiDiscrete heads for this adapter."""

        return self._action_dimensions

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

    def action_mask(
        self,
        *,
        base_overrides: dict[str, tuple[int, ...]] | None = None,
        stage_overrides: dict[str, tuple[int, ...]] | None = None,
        dynamic_overrides: dict[str, tuple[int, ...]] | None = None,
    ) -> np.ndarray:
        """Return one flattened boolean action mask for the current adapter."""

        return build_flat_action_mask(
            self._action_dimensions,
            base_overrides=base_overrides,
            stage_overrides=stage_overrides,
            dynamic_overrides=dynamic_overrides,
        )


def _parse_action_quad(
    action: ActionValue,
    *,
    steer_bucket_count: int,
) -> tuple[int, int, int, int]:
    steer_index, drive_mode_index, boost_index, shoulder_index = parse_discrete_action(
        action,
        action_label="Steer-drive-boost-shoulder",
        dimensions=(
            DiscreteActionDimension("steer", steer_bucket_count),
            DiscreteActionDimension("drive", len(DRIVE_MODES)),
            DiscreteActionDimension("boost", 2),
            DiscreteActionDimension("shoulder", len(SHOULDER_MASKS)),
        ),
    )
    return steer_index, drive_mode_index, boost_index, shoulder_index


class SteerGasAirBrakeBoostShoulderActionAdapter:
    """Map digital steering plus independent gas, air brake, boost, and shoulder heads."""

    def __init__(self, config: ActionConfig) -> None:
        self._steer_values = steer_values(config.steer_buckets)
        self._action_dimensions = (
            DiscreteActionDimension("steer", config.steer_buckets),
            DiscreteActionDimension("gas", 2),
            DiscreteActionDimension("air_brake", 2),
            DiscreteActionDimension("boost", 2),
            DiscreteActionDimension("shoulder", len(SHOULDER_MASKS)),
        )
        self._action_space = multidiscrete_space(
            *(dimension.size for dimension in self._action_dimensions)
        )
        self._idle_action = idle_action(config.steer_buckets // 2, 0, 0, 0, 0)

    @property
    def action_space(self) -> spaces.MultiDiscrete:
        """Return the current policy action space."""

        return self._action_space

    @property
    def idle_action(self) -> np.ndarray:
        """Return the neutral action value for this action space."""

        return np.array(self._idle_action, copy=True)

    @property
    def action_dimensions(self) -> tuple[DiscreteActionDimension, ...]:
        """Return the ordered MultiDiscrete heads for this adapter."""

        return self._action_dimensions

    def decode(self, action: ActionValue) -> ControllerState:
        """Translate one policy action into a held controller state."""

        steer_index, gas_index, air_brake_index, boost_index, shoulder_index = (
            _parse_gas_air_brake_boost_shoulder_action(
                action,
                steer_bucket_count=len(self._steer_values),
            )
        )
        joypad_mask = SHOULDER_MASKS[shoulder_index]
        if gas_index == 1:
            joypad_mask |= ACCELERATE_MASK
        if air_brake_index == 1:
            joypad_mask |= AIR_BRAKE_MASK
        if boost_index == 1:
            joypad_mask |= BOOST_MASK

        return ControllerState(
            joypad_mask=joypad_mask,
            left_stick_x=float(self._steer_values[steer_index]),
        )

    def action_mask(
        self,
        *,
        base_overrides: dict[str, tuple[int, ...]] | None = None,
        stage_overrides: dict[str, tuple[int, ...]] | None = None,
        dynamic_overrides: dict[str, tuple[int, ...]] | None = None,
    ) -> np.ndarray:
        """Return one flattened boolean action mask for the current adapter."""

        return build_flat_action_mask(
            self._action_dimensions,
            base_overrides=base_overrides,
            stage_overrides=stage_overrides,
            dynamic_overrides=dynamic_overrides,
        )


def _parse_gas_air_brake_boost_shoulder_action(
    action: ActionValue,
    *,
    steer_bucket_count: int,
) -> tuple[int, int, int, int, int]:
    steer_index, gas_index, air_brake_index, boost_index, shoulder_index = parse_discrete_action(
        action,
        action_label="Steer-gas-air-brake-boost-shoulder",
        dimensions=(
            DiscreteActionDimension("steer", steer_bucket_count),
            DiscreteActionDimension("gas", 2),
            DiscreteActionDimension("air_brake", 2),
            DiscreteActionDimension("boost", 2),
            DiscreteActionDimension("shoulder", len(SHOULDER_MASKS)),
        ),
    )
    return steer_index, gas_index, air_brake_index, boost_index, shoulder_index


# COMPAT SHIM: legacy action adapter class names.
# Earlier public adapter names used "drift" for these Z/R shoulder inputs.
SteerDriveBoostDriftActionAdapter = SteerDriveBoostShoulderActionAdapter
SteerGasAirBrakeBoostDriftActionAdapter = SteerGasAirBrakeBoostShoulderActionAdapter
