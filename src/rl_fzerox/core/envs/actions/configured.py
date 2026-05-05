from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from gymnasium import spaces

from fzerox_emulator import ControllerState
from fzerox_emulator.arrays import ActionMask, DiscreteAction, NumpyArray
from rl_fzerox.core.config.schema import ActionConfig, ActionRuntimeConfig
from rl_fzerox.core.envs.actions.base import (
    ActionBranchValue,
    ActionValue,
    DiscreteActionDimension,
    HybridActionValue,
    build_flat_action_mask,
    multidiscrete_space,
    parse_discrete_action,
    steer_values,
)
from rl_fzerox.core.envs.actions.continuous_controls import (
    ContinuousButtonPwmDecoder,
    ContinuousDriveDecoder,
    continuous_action_array,
    discrete_action_array,
)
from rl_fzerox.core.envs.actions.discrete.steer_drive import (
    ACCELERATE_MASK,
    AIR_BRAKE_MASK,
    DRIVE_MODES,
)
from rl_fzerox.core.envs.actions.discrete.steer_drive_boost_lean import (
    BOOST_MASK,
    LEAN_LEFT_MASK,
    LEAN_RIGHT_MASK,
)
from rl_fzerox.core.envs.actions.hybrid.layouts import HybridActionLayout
from rl_fzerox.core.envs.actions.hybrid.spaces import hybrid_action_mask, hybrid_action_space


def _configured_dimensions(
    config: ActionRuntimeConfig,
) -> tuple[DiscreteActionDimension, ...]:
    return tuple(
        DiscreteActionDimension(label=axis, size=_discrete_axis_size(axis, config))
        for axis in config.layout_discrete_axes
    )


def _discrete_axis_size(axis: str, config: ActionRuntimeConfig) -> int:
    if axis == "steer":
        return int(config.steer_buckets)
    if axis == "drive":
        return len(DRIVE_MODES)
    if axis in {"gas", "air_brake", "boost"}:
        return 2
    if axis == "lean":
        return 4 if config.independent_lean_buttons else 3
    if axis == "pitch":
        return int(config.pitch_buckets)
    raise ValueError(f"Unsupported configured discrete axis: {axis!r}")


def _idle_discrete_values(
    dimensions: tuple[DiscreteActionDimension, ...],
    config: ActionRuntimeConfig,
) -> DiscreteAction:
    values: list[int] = []
    for dimension in dimensions:
        if dimension.label == "steer":
            values.append(config.steer_buckets // 2)
        elif dimension.label == "pitch":
            values.append(config.pitch_buckets // 2)
        else:
            values.append(0)
    return np.asarray(values, dtype=np.int64)


def _pitch_bucket_value(index: int, *, bucket_count: int) -> float:
    neutral_index = bucket_count // 2
    if neutral_index <= 0:
        return 0.0
    return float(index - neutral_index) / float(neutral_index)


def _lean_mask(index: int, *, independent_buttons: bool) -> int:
    if independent_buttons:
        if index == 1:
            return LEAN_LEFT_MASK
        if index == 2:
            return LEAN_RIGHT_MASK
        if index == 3:
            return LEAN_LEFT_MASK | LEAN_RIGHT_MASK
        return 0
    if index == 1:
        return LEAN_LEFT_MASK
    if index == 2:
        return LEAN_RIGHT_MASK
    return 0


class ConfiguredDiscreteActionAdapter:
    """Map one configured MultiDiscrete layout into emulator controls."""

    def __init__(self, config: ActionConfig | ActionRuntimeConfig) -> None:
        runtime = config.runtime() if isinstance(config, ActionConfig) else config
        self._config = runtime
        self._dimensions = _configured_dimensions(runtime)
        self._action_space = multidiscrete_space(
            *(dimension.size for dimension in self._dimensions)
        )
        self._idle_action = _idle_discrete_values(self._dimensions, runtime)
        self._steer_values = steer_values(
            runtime.steer_buckets,
            response_power=float(runtime.steer_response_power),
        )

    @property
    def action_space(self) -> spaces.MultiDiscrete:
        return self._action_space

    @property
    def idle_action(self) -> DiscreteAction:
        return np.array(self._idle_action, copy=True)

    @property
    def action_dimensions(self) -> tuple[DiscreteActionDimension, ...]:
        return self._dimensions

    def decode(self, action: ActionValue) -> ControllerState:
        parsed = parse_discrete_action(
            action,
            action_label="Configured discrete action",
            dimensions=self._dimensions,
        )
        steer = 0.0
        pitch = 0.0
        joypad_mask = 0
        for dimension, value in zip(self._dimensions, parsed, strict=True):
            if dimension.label == "steer":
                steer = float(self._steer_values[value])
            elif dimension.label == "drive":
                joypad_mask |= DRIVE_MODES[value].joypad_mask
            elif dimension.label == "gas" and value == 1:
                joypad_mask |= ACCELERATE_MASK
            elif dimension.label == "air_brake" and value == 1:
                joypad_mask |= AIR_BRAKE_MASK
            elif dimension.label == "boost" and value == 1:
                joypad_mask |= BOOST_MASK
            elif dimension.label == "lean":
                joypad_mask |= _lean_mask(
                    value,
                    independent_buttons=self._config.independent_lean_buttons,
                )
            elif dimension.label == "pitch":
                pitch = _pitch_bucket_value(value, bucket_count=self._config.pitch_buckets)

        if self._config.force_full_throttle:
            joypad_mask |= ACCELERATE_MASK

        return ControllerState(
            joypad_mask=joypad_mask,
            left_stick_x=steer,
            left_stick_y=pitch,
        )

    def action_mask(
        self,
        *,
        base_overrides: dict[str, tuple[int, ...]] | None = None,
        stage_overrides: dict[str, tuple[int, ...]] | None = None,
        dynamic_overrides: dict[str, tuple[int, ...]] | None = None,
    ) -> ActionMask:
        return build_flat_action_mask(
            self._dimensions,
            base_overrides=base_overrides,
            stage_overrides=stage_overrides,
            dynamic_overrides=dynamic_overrides,
        )


class ConfiguredHybridActionAdapter:
    """Map one configured Dict(Box, MultiDiscrete) layout into emulator controls."""

    def __init__(self, config: ActionConfig | ActionRuntimeConfig) -> None:
        runtime = config.runtime() if isinstance(config, ActionConfig) else config
        self._config = runtime
        self._dimensions = _configured_dimensions(runtime)
        self._action_space = hybrid_action_space(
            HybridActionLayout(
                continuous_size=len(runtime.layout_continuous_axes),
                dimensions=self._dimensions,
            )
        )
        self._idle_action = {
            "continuous": np.zeros(len(runtime.layout_continuous_axes), dtype=np.float32),
            "discrete": _idle_discrete_values(self._dimensions, runtime),
        }
        self._steer_values = steer_values(
            runtime.steer_buckets,
            response_power=float(runtime.steer_response_power),
        )
        self._drive_decoder = ContinuousDriveDecoder(
            deadzone=float(runtime.continuous_drive_deadzone),
            full_threshold=float(runtime.continuous_drive_full_threshold),
            min_thrust=float(runtime.continuous_drive_min_thrust),
        )
        self._air_brake_decoder = ContinuousButtonPwmDecoder(
            deadzone=float(runtime.continuous_air_brake_deadzone),
            full_threshold=float(runtime.continuous_air_brake_full_threshold),
            min_duty=float(runtime.continuous_air_brake_min_duty),
        )

    @property
    def action_space(self) -> spaces.Dict:
        return self._action_space

    @property
    def idle_action(self) -> dict[str, NumpyArray]:
        return {
            "continuous": np.array(self._idle_action["continuous"], copy=True),
            "discrete": np.array(self._idle_action["discrete"], copy=True),
        }

    @property
    def action_dimensions(self) -> tuple[DiscreteActionDimension, ...]:
        return self._dimensions

    def decode(self, action: ActionValue) -> ControllerState:
        mapping = _hybrid_action_mapping(action)
        continuous_values = continuous_action_array(
            _required_branch(mapping, "continuous"),
            expected_size=len(self._config.layout_continuous_axes),
            action_label="Configured hybrid continuous",
            field_labels=tuple(self._config.layout_continuous_axes),
        )
        discrete_values = discrete_action_array(
            _required_branch(mapping, "discrete"),
            expected_size=len(self._dimensions),
            action_label="Configured hybrid discrete",
            field_labels=tuple(dimension.label for dimension in self._dimensions),
        )

        steer = 0.0
        drive: float | None = None
        air_brake: float | None = None
        pitch = 0.0
        for axis, value in zip(self._config.layout_continuous_axes, continuous_values, strict=True):
            clipped = float(np.clip(value, -1.0, 1.0))
            if axis == "steer":
                steer = clipped
            elif axis == "drive":
                drive = clipped
            elif axis == "air_brake":
                air_brake = clipped
            elif axis == "pitch":
                pitch = clipped

        joypad_mask = 0
        if drive is not None:
            joypad_mask |= self._drive_decoder.decode(
                1.0 if self._config.force_full_throttle else drive
            )
        if air_brake is not None and self._config.continuous_air_brake_mode != "off":
            joypad_mask |= self._air_brake_decoder.decode(air_brake, button_mask=AIR_BRAKE_MASK)

        for dimension, raw_value in zip(self._dimensions, discrete_values, strict=True):
            value = int(raw_value)
            if dimension.label == "steer":
                steer = float(self._steer_values[value])
            elif dimension.label == "drive":
                joypad_mask |= DRIVE_MODES[value].joypad_mask
            elif dimension.label == "gas" and value == 1:
                joypad_mask |= ACCELERATE_MASK
            elif dimension.label == "air_brake" and value == 1:
                joypad_mask |= AIR_BRAKE_MASK
            elif dimension.label == "boost" and value == 1:
                joypad_mask |= BOOST_MASK
            elif dimension.label == "lean":
                joypad_mask |= _lean_mask(
                    value,
                    independent_buttons=self._config.independent_lean_buttons,
                )
            elif dimension.label == "pitch":
                pitch = _pitch_bucket_value(value, bucket_count=self._config.pitch_buckets)

        if self._config.force_full_throttle:
            joypad_mask |= ACCELERATE_MASK

        return ControllerState(
            joypad_mask=joypad_mask,
            left_stick_x=steer,
            left_stick_y=pitch,
        )

    def reset(self) -> None:
        self._drive_decoder.reset()
        self._air_brake_decoder.reset()

    def action_mask(
        self,
        *,
        base_overrides: dict[str, tuple[int, ...]] | None = None,
        stage_overrides: dict[str, tuple[int, ...]] | None = None,
        dynamic_overrides: dict[str, tuple[int, ...]] | None = None,
    ) -> ActionMask:
        return hybrid_action_mask(
            self._dimensions,
            base_overrides=base_overrides,
            stage_overrides=stage_overrides,
            dynamic_overrides=dynamic_overrides,
        )


def _required_branch(
    action: HybridActionValue,
    branch_name: str,
) -> ActionBranchValue:
    try:
        return action[branch_name]
    except KeyError as exc:
        raise ValueError(f"Hybrid action is missing {branch_name!r} branch") from exc


def _hybrid_action_mapping(action: ActionValue) -> HybridActionValue:
    if not isinstance(action, Mapping):
        raise ValueError("Configured hybrid action must be a mapping")
    return action
