# src/rl_fzerox/core/envs/actions/hybrid_steer_drive.py
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from gymnasium import spaces

from fzerox_emulator import ControllerState
from fzerox_emulator.arrays import ActionMask, ContinuousAction, DiscreteAction, NumpyArray
from rl_fzerox.core.config.schema import ActionConfig, ActionRuntimeConfig
from rl_fzerox.core.domain.hybrid_action import (
    HYBRID_CONTINUOUS_ACTION_KEY,
    HYBRID_DISCRETE_ACTION_KEY,
)
from rl_fzerox.core.envs.actions.base import (
    ActionValue,
    DiscreteActionDimension,
    build_flat_action_mask,
    shape_steer_value,
)
from rl_fzerox.core.envs.actions.continuous_controls import (
    ContinuousButtonPwmDecoder,
    ContinuousDriveDecoder,
    continuous_action_array,
    discrete_action_array,
    hybrid_branch,
)
from rl_fzerox.core.envs.actions.steer_drive import ACCELERATE_MASK, AIR_BRAKE_MASK
from rl_fzerox.core.envs.actions.steer_drive_boost_lean import (
    BOOST_MASK,
    LEAN_LEFT_MASK,
    LEAN_RIGHT_MASK,
)

_HYBRID_STEER_DRIVE_CONTINUOUS_SIZE = 2
_HYBRID_STEER_DRIVE_AIR_BRAKE_CONTINUOUS_SIZE = 3
_HYBRID_STEER_CONTINUOUS_SIZE = 1
_HYBRID_LEAN_DISCRETE_SIZE = 1
_HYBRID_BOOST_LEAN_DISCRETE_SIZE = 2
_HYBRID_GAS_BOOST_LEAN_DISCRETE_SIZE = 3
_HYBRID_GAS_AIR_BRAKE_BOOST_LEAN_DISCRETE_SIZE = 4
_HYBRID_LEAN_PRIMITIVE_SIZE = 7
_HYBRID_ACTION_DIMENSIONS = (DiscreteActionDimension("lean", 3),)
_HYBRID_BOOST_ACTION_DIMENSIONS = (
    DiscreteActionDimension("lean", 3),
    DiscreteActionDimension("boost", 2),
)
_HYBRID_STEER_GAS_BOOST_LEAN_ACTION_DIMENSIONS = (
    DiscreteActionDimension("gas", 2),
    DiscreteActionDimension("boost", 2),
    DiscreteActionDimension("lean", 3),
)
_HYBRID_STEER_GAS_AIR_BRAKE_BOOST_LEAN_ACTION_DIMENSIONS = (
    DiscreteActionDimension("gas", 2),
    DiscreteActionDimension("air_brake", 2),
    DiscreteActionDimension("boost", 2),
    DiscreteActionDimension("lean", 3),
)
_HYBRID_LEAN_PRIMITIVE_ACTION_DIMENSIONS = (
    DiscreteActionDimension("lean", _HYBRID_LEAN_PRIMITIVE_SIZE),
    DiscreteActionDimension("boost", 2),
)
_HYBRID_ENABLED_LEAN_PRIMITIVES = (0, 1, 2)


def _hybrid_action_space(
    *,
    continuous_size: int,
    dimensions: tuple[DiscreteActionDimension, ...],
) -> spaces.Dict:
    return spaces.Dict(
        {
            HYBRID_CONTINUOUS_ACTION_KEY: spaces.Box(
                low=np.full(continuous_size, -1.0, dtype=np.float32),
                high=np.full(continuous_size, 1.0, dtype=np.float32),
                dtype=np.float32,
            ),
            HYBRID_DISCRETE_ACTION_KEY: spaces.MultiDiscrete(
                np.array([dimension.size for dimension in dimensions], dtype=np.int64)
            ),
        }
    )


def _hybrid_idle_action(*, continuous_size: int, discrete_size: int) -> dict[str, NumpyArray]:
    return {
        HYBRID_CONTINUOUS_ACTION_KEY: np.zeros(continuous_size, dtype=np.float32),
        HYBRID_DISCRETE_ACTION_KEY: np.zeros(discrete_size, dtype=np.int64),
    }


def _hybrid_action_mask(
    dimensions: tuple[DiscreteActionDimension, ...],
    *,
    base_overrides: dict[str, tuple[int, ...]] | None = None,
    stage_overrides: dict[str, tuple[int, ...]] | None = None,
    dynamic_overrides: dict[str, tuple[int, ...]] | None = None,
) -> ActionMask:
    return build_flat_action_mask(
        dimensions,
        base_overrides=base_overrides,
        stage_overrides=stage_overrides,
        dynamic_overrides=dynamic_overrides,
    )


class HybridSteerDriveLeanActionAdapter:
    """Map hybrid PPO actions to continuous steer/drive and discrete lean inputs."""

    def __init__(self, config: ActionConfig | ActionRuntimeConfig) -> None:
        self._steer_response_power = float(config.steer_response_power)
        self._drive_decoder = ContinuousDriveDecoder(
            mode=config.continuous_drive_mode,
            deadzone=float(config.continuous_drive_deadzone),
        )
        self._action_space = _hybrid_action_space(
            continuous_size=_HYBRID_STEER_DRIVE_CONTINUOUS_SIZE,
            dimensions=_HYBRID_ACTION_DIMENSIONS,
        )

    @property
    def action_space(self) -> spaces.Dict:
        """Return the public hybrid action space expected by maskable hybrid PPO."""

        return self._action_space

    @property
    def idle_action(self) -> dict[str, NumpyArray]:
        """Return a neutral hybrid action with no lean input held."""

        return _hybrid_idle_action(
            continuous_size=_HYBRID_STEER_DRIVE_CONTINUOUS_SIZE,
            discrete_size=_HYBRID_LEAN_DISCRETE_SIZE,
        )

    @property
    def action_dimensions(self) -> tuple[DiscreteActionDimension, ...]:
        """Return the discrete lean branch maskable by hybrid PPO."""

        return _HYBRID_ACTION_DIMENSIONS

    def decode(self, action: ActionValue) -> ControllerState:
        """Translate one hybrid action into steering, drive, and lean buttons."""

        steer, drive, lean = _parse_hybrid_steer_drive_lean(action)
        joypad_mask = self._drive_decoder.decode(drive)
        if lean == 1:
            joypad_mask |= LEAN_LEFT_MASK
        elif lean == 2:
            joypad_mask |= LEAN_RIGHT_MASK

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
        """Return the discrete-branch mask used by MaskableHybridActionPPO."""

        return _hybrid_action_mask(
            self.action_dimensions,
            base_overrides=base_overrides,
            stage_overrides=stage_overrides,
            dynamic_overrides=dynamic_overrides,
        )


class HybridSteerDriveBoostLeanActionAdapter:
    """Map hybrid PPO actions to continuous steer/drive, lean, and boost."""

    def __init__(self, config: ActionConfig | ActionRuntimeConfig) -> None:
        self._steer_response_power = float(config.steer_response_power)
        self._drive_decoder = ContinuousDriveDecoder(
            mode=config.continuous_drive_mode,
            deadzone=float(config.continuous_drive_deadzone),
        )
        self._action_space = _hybrid_action_space(
            continuous_size=_HYBRID_STEER_DRIVE_CONTINUOUS_SIZE,
            dimensions=_HYBRID_BOOST_ACTION_DIMENSIONS,
        )

    @property
    def action_space(self) -> spaces.Dict:
        """Return the public hybrid action space expected by maskable hybrid PPO."""

        return self._action_space

    @property
    def idle_action(self) -> dict[str, NumpyArray]:
        """Return a neutral hybrid action with no lean or boost held."""

        return _hybrid_idle_action(
            continuous_size=_HYBRID_STEER_DRIVE_CONTINUOUS_SIZE,
            discrete_size=_HYBRID_BOOST_LEAN_DISCRETE_SIZE,
        )

    @property
    def action_dimensions(self) -> tuple[DiscreteActionDimension, ...]:
        """Return the discrete branches maskable by maskable hybrid PPO."""

        return _HYBRID_BOOST_ACTION_DIMENSIONS

    def decode(self, action: ActionValue) -> ControllerState:
        """Translate one hybrid action into steering, drive, lean, and boost."""

        steer, drive, lean, boost = _parse_hybrid_steer_drive_boost_lean_pair(action)
        joypad_mask = self._drive_decoder.decode(drive)
        if lean == 1:
            joypad_mask |= LEAN_LEFT_MASK
        elif lean == 2:
            joypad_mask |= LEAN_RIGHT_MASK
        if boost == 1:
            joypad_mask |= BOOST_MASK

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
        """Return the discrete-branch mask used by MaskableHybridActionPPO."""

        return _hybrid_action_mask(
            self.action_dimensions,
            base_overrides=base_overrides,
            stage_overrides=stage_overrides,
            dynamic_overrides=dynamic_overrides,
        )


class HybridSteerGasAirBrakeBoostLeanActionAdapter:
    """Map continuous steering plus discrete gas, air-brake, boost, and lean heads."""

    def __init__(self, config: ActionConfig | ActionRuntimeConfig) -> None:
        self._steer_response_power = float(config.steer_response_power)
        self._action_space = _hybrid_action_space(
            continuous_size=_HYBRID_STEER_CONTINUOUS_SIZE,
            dimensions=_HYBRID_STEER_GAS_AIR_BRAKE_BOOST_LEAN_ACTION_DIMENSIONS,
        )

    @property
    def action_space(self) -> spaces.Dict:
        """Return the public hybrid action space expected by maskable hybrid PPO."""

        return self._action_space

    @property
    def idle_action(self) -> dict[str, NumpyArray]:
        """Return a neutral hybrid action with no buttons held."""

        return _hybrid_idle_action(
            continuous_size=_HYBRID_STEER_CONTINUOUS_SIZE,
            discrete_size=_HYBRID_GAS_AIR_BRAKE_BOOST_LEAN_DISCRETE_SIZE,
        )

    @property
    def action_dimensions(self) -> tuple[DiscreteActionDimension, ...]:
        """Return the discrete branches maskable by maskable hybrid PPO."""

        return _HYBRID_STEER_GAS_AIR_BRAKE_BOOST_LEAN_ACTION_DIMENSIONS

    def decode(self, action: ActionValue) -> ControllerState:
        """Translate one hybrid action into steering and independent button heads."""

        steer, gas, air_brake, boost, lean = _parse_hybrid_steer_gas_air_brake_boost_lean(action)
        joypad_mask = 0
        if gas == 1:
            joypad_mask |= ACCELERATE_MASK
        if air_brake == 1:
            joypad_mask |= AIR_BRAKE_MASK
        if boost == 1:
            joypad_mask |= BOOST_MASK
        if lean == 1:
            joypad_mask |= LEAN_LEFT_MASK
        elif lean == 2:
            joypad_mask |= LEAN_RIGHT_MASK

        return ControllerState(
            joypad_mask=joypad_mask,
            left_stick_x=shape_steer_value(
                steer,
                response_power=self._steer_response_power,
            ),
        )

    def action_mask(
        self,
        *,
        base_overrides: dict[str, tuple[int, ...]] | None = None,
        stage_overrides: dict[str, tuple[int, ...]] | None = None,
        dynamic_overrides: dict[str, tuple[int, ...]] | None = None,
    ) -> ActionMask:
        """Return the discrete-branch mask used by MaskableHybridActionPPO."""

        return _hybrid_action_mask(
            self.action_dimensions,
            base_overrides=base_overrides,
            stage_overrides=stage_overrides,
            dynamic_overrides=dynamic_overrides,
        )


class HybridSteerGasBoostLeanActionAdapter:
    """Map continuous steering plus discrete gas, boost, and lean heads."""

    def __init__(self, config: ActionConfig | ActionRuntimeConfig) -> None:
        self._steer_response_power = float(config.steer_response_power)
        self._action_space = _hybrid_action_space(
            continuous_size=_HYBRID_STEER_CONTINUOUS_SIZE,
            dimensions=_HYBRID_STEER_GAS_BOOST_LEAN_ACTION_DIMENSIONS,
        )

    @property
    def action_space(self) -> spaces.Dict:
        """Return the public hybrid action space expected by maskable hybrid PPO."""

        return self._action_space

    @property
    def idle_action(self) -> dict[str, NumpyArray]:
        """Return a neutral hybrid action with no buttons held."""

        return _hybrid_idle_action(
            continuous_size=_HYBRID_STEER_CONTINUOUS_SIZE,
            discrete_size=_HYBRID_GAS_BOOST_LEAN_DISCRETE_SIZE,
        )

    @property
    def action_dimensions(self) -> tuple[DiscreteActionDimension, ...]:
        """Return the discrete branches maskable by maskable hybrid PPO."""

        return _HYBRID_STEER_GAS_BOOST_LEAN_ACTION_DIMENSIONS

    def decode(self, action: ActionValue) -> ControllerState:
        """Translate one hybrid action into steering and independent button heads."""

        steer, gas, boost, lean = _parse_hybrid_steer_gas_boost_lean(action)
        joypad_mask = 0
        if gas == 1:
            joypad_mask |= ACCELERATE_MASK
        if boost == 1:
            joypad_mask |= BOOST_MASK
        if lean == 1:
            joypad_mask |= LEAN_LEFT_MASK
        elif lean == 2:
            joypad_mask |= LEAN_RIGHT_MASK

        return ControllerState(
            joypad_mask=joypad_mask,
            left_stick_x=shape_steer_value(
                steer,
                response_power=self._steer_response_power,
            ),
        )

    def action_mask(
        self,
        *,
        base_overrides: dict[str, tuple[int, ...]] | None = None,
        stage_overrides: dict[str, tuple[int, ...]] | None = None,
        dynamic_overrides: dict[str, tuple[int, ...]] | None = None,
    ) -> ActionMask:
        """Return the discrete-branch mask used by MaskableHybridActionPPO."""

        return _hybrid_action_mask(
            self.action_dimensions,
            base_overrides=base_overrides,
            stage_overrides=stage_overrides,
            dynamic_overrides=dynamic_overrides,
        )


class HybridSteerDriveBoostLeanPrimitiveActionAdapter:
    """Map hybrid PPO actions to continuous steer/drive/air-brake and lean primitives.

    The first three lean values match the current lean/slide primitive:
    `off`, `left`, `right`. Values 3..6 reserve architecture
    capacity for future side-attack and spin primitives and currently decode
    as no-ops while masked out by default.
    """

    def __init__(self, config: ActionConfig | ActionRuntimeConfig) -> None:
        self._steer_response_power = float(config.steer_response_power)
        self._drive_decoder = ContinuousDriveDecoder(
            mode=config.continuous_drive_mode,
            deadzone=float(config.continuous_drive_deadzone),
        )
        self._air_brake_decoder = ContinuousButtonPwmDecoder(
            deadzone=float(config.continuous_drive_deadzone),
        )
        self._air_brake_enabled = config.continuous_air_brake_mode != "off"
        self._action_space = _hybrid_action_space(
            continuous_size=_HYBRID_STEER_DRIVE_AIR_BRAKE_CONTINUOUS_SIZE,
            dimensions=_HYBRID_LEAN_PRIMITIVE_ACTION_DIMENSIONS,
        )

    @property
    def action_space(self) -> spaces.Dict:
        """Return the public hybrid action space expected by maskable hybrid PPO."""

        return self._action_space

    @property
    def idle_action(self) -> dict[str, NumpyArray]:
        """Return a neutral hybrid action with no lean primitive or boost."""

        return _hybrid_idle_action(
            continuous_size=_HYBRID_STEER_DRIVE_AIR_BRAKE_CONTINUOUS_SIZE,
            discrete_size=_HYBRID_BOOST_LEAN_DISCRETE_SIZE,
        )

    @property
    def action_dimensions(self) -> tuple[DiscreteActionDimension, ...]:
        """Return the discrete branches maskable by maskable hybrid PPO."""

        return _HYBRID_LEAN_PRIMITIVE_ACTION_DIMENSIONS

    def decode(self, action: ActionValue) -> ControllerState:
        """Translate one hybrid action into steering, drive, lean, and boost."""

        steer, drive, air_brake, lean, boost = _parse_hybrid_steer_drive_boost_lean(action)
        air_brake_mask = (
            self._air_brake_decoder.decode(air_brake, button_mask=AIR_BRAKE_MASK)
            if self._air_brake_enabled
            else 0
        )
        joypad_mask = self._drive_decoder.decode(drive) | air_brake_mask
        if lean == 1:
            joypad_mask |= LEAN_LEFT_MASK
        elif lean == 2:
            joypad_mask |= LEAN_RIGHT_MASK
        if boost == 1:
            joypad_mask |= BOOST_MASK

        return ControllerState(
            joypad_mask=joypad_mask,
            left_stick_x=shape_steer_value(
                steer,
                response_power=self._steer_response_power,
            ),
        )

    def reset(self) -> None:
        """Reset drive and air-brake PWM accumulators for a new episode."""

        self._drive_decoder.reset()
        self._air_brake_decoder.reset()

    def action_mask(
        self,
        *,
        base_overrides: dict[str, tuple[int, ...]] | None = None,
        stage_overrides: dict[str, tuple[int, ...]] | None = None,
        dynamic_overrides: dict[str, tuple[int, ...]] | None = None,
    ) -> ActionMask:
        """Return a mask that reserves future lean primitives by default."""

        return _hybrid_action_mask(
            self.action_dimensions,
            base_overrides=_with_default_lean_primitive_mask(base_overrides),
            stage_overrides=stage_overrides,
            dynamic_overrides=dynamic_overrides,
        )


def _parse_hybrid_steer_drive_lean(action: ActionValue) -> tuple[float, float, int]:
    continuous_values, discrete_values = _parse_hybrid_branches(
        action,
        expected_size=_HYBRID_LEAN_DISCRETE_SIZE,
        action_label="Hybrid steer-drive-lean discrete",
        field_labels=("lean",),
    )
    lean = int(discrete_values[0])
    if not 0 <= lean < 3:
        raise ValueError(f"Invalid hybrid lean index {lean}")

    steer = float(np.clip(continuous_values[0], -1.0, 1.0))
    drive = float(np.clip(continuous_values[1], -1.0, 1.0))
    return steer, drive, lean


def _parse_hybrid_steer_drive_boost_lean_pair(
    action: ActionValue,
) -> tuple[float, float, int, int]:
    continuous_values, discrete_values = _parse_hybrid_branches(
        action,
        expected_size=_HYBRID_BOOST_LEAN_DISCRETE_SIZE,
        action_label="Hybrid steer-drive-boost-lean discrete",
        field_labels=("lean", "boost"),
    )
    lean = int(discrete_values[0])
    boost = int(discrete_values[1])
    if not 0 <= lean < 3:
        raise ValueError(f"Invalid hybrid lean index {lean}")
    if not 0 <= boost < 2:
        raise ValueError(f"Invalid hybrid boost index {boost}")

    steer = float(np.clip(continuous_values[0], -1.0, 1.0))
    drive = float(np.clip(continuous_values[1], -1.0, 1.0))
    return steer, drive, lean, boost


def _parse_hybrid_steer_gas_air_brake_boost_lean(
    action: ActionValue,
) -> tuple[float, int, int, int, int]:
    continuous_values, discrete_values = _parse_hybrid_branches(
        action,
        continuous_size=_HYBRID_STEER_CONTINUOUS_SIZE,
        continuous_field_labels=("steer",),
        expected_size=_HYBRID_GAS_AIR_BRAKE_BOOST_LEAN_DISCRETE_SIZE,
        action_label="Hybrid steer-gas-air-brake-boost-lean discrete",
        field_labels=("gas", "air_brake", "boost", "lean"),
    )
    gas = int(discrete_values[0])
    air_brake = int(discrete_values[1])
    boost = int(discrete_values[2])
    lean = int(discrete_values[3])
    if not 0 <= gas < 2:
        raise ValueError(f"Invalid hybrid gas index {gas}")
    if not 0 <= air_brake < 2:
        raise ValueError(f"Invalid hybrid air-brake index {air_brake}")
    if not 0 <= boost < 2:
        raise ValueError(f"Invalid hybrid boost index {boost}")
    if not 0 <= lean < 3:
        raise ValueError(f"Invalid hybrid lean index {lean}")

    steer = float(np.clip(continuous_values[0], -1.0, 1.0))
    return steer, gas, air_brake, boost, lean


def _parse_hybrid_steer_gas_boost_lean(
    action: ActionValue,
) -> tuple[float, int, int, int]:
    continuous_values, discrete_values = _parse_hybrid_branches(
        action,
        continuous_size=_HYBRID_STEER_CONTINUOUS_SIZE,
        continuous_field_labels=("steer",),
        expected_size=_HYBRID_GAS_BOOST_LEAN_DISCRETE_SIZE,
        action_label="Hybrid steer-gas-boost-lean discrete",
        field_labels=("gas", "boost", "lean"),
    )
    gas = int(discrete_values[0])
    boost = int(discrete_values[1])
    lean = int(discrete_values[2])
    if not 0 <= gas < 2:
        raise ValueError(f"Invalid hybrid gas index {gas}")
    if not 0 <= boost < 2:
        raise ValueError(f"Invalid hybrid boost index {boost}")
    if not 0 <= lean < 3:
        raise ValueError(f"Invalid hybrid lean index {lean}")

    steer = float(np.clip(continuous_values[0], -1.0, 1.0))
    return steer, gas, boost, lean


def _parse_hybrid_steer_drive_boost_lean(
    action: ActionValue,
) -> tuple[float, float, float, int, int]:
    continuous_values, discrete_values = _parse_hybrid_branches(
        action,
        continuous_size=_HYBRID_STEER_DRIVE_AIR_BRAKE_CONTINUOUS_SIZE,
        continuous_field_labels=("steer", "drive", "air_brake"),
        expected_size=_HYBRID_BOOST_LEAN_DISCRETE_SIZE,
        action_label="Hybrid steer-drive-boost-lean discrete",
        field_labels=("lean", "boost"),
    )
    lean = int(discrete_values[0])
    boost = int(discrete_values[1])
    if not 0 <= lean < _HYBRID_LEAN_PRIMITIVE_SIZE:
        raise ValueError(f"Invalid hybrid lean index {lean}")
    if not 0 <= boost < 2:
        raise ValueError(f"Invalid hybrid boost index {boost}")

    steer = float(np.clip(continuous_values[0], -1.0, 1.0))
    drive = float(np.clip(continuous_values[1], -1.0, 1.0))
    air_brake = float(np.clip(continuous_values[2], -1.0, 1.0))
    return steer, drive, air_brake, lean, boost


def _with_default_lean_primitive_mask(
    base_overrides: dict[str, tuple[int, ...]] | None,
) -> dict[str, tuple[int, ...]]:
    if base_overrides is not None and "lean" in base_overrides:
        return base_overrides
    overrides = dict(base_overrides or {})
    overrides["lean"] = _HYBRID_ENABLED_LEAN_PRIMITIVES
    return overrides


def _parse_hybrid_branches(
    action: ActionValue,
    *,
    continuous_size: int = _HYBRID_STEER_DRIVE_CONTINUOUS_SIZE,
    continuous_field_labels: tuple[str, ...] = ("steer", "drive"),
    expected_size: int,
    action_label: str,
    field_labels: tuple[str, ...],
) -> tuple[ContinuousAction, DiscreteAction]:
    if not isinstance(action, Mapping):
        raise ValueError(
            "Hybrid steer-drive actions must be a mapping with 'continuous' and 'discrete' branches"
        )

    continuous_values = continuous_action_array(
        hybrid_branch(action, HYBRID_CONTINUOUS_ACTION_KEY),
        expected_size=continuous_size,
        action_label="Hybrid steer-drive continuous",
        field_labels=continuous_field_labels,
    )
    discrete_values = discrete_action_array(
        hybrid_branch(action, HYBRID_DISCRETE_ACTION_KEY),
        expected_size=expected_size,
        action_label=action_label,
        field_labels=field_labels,
    )
    return continuous_values, discrete_values
