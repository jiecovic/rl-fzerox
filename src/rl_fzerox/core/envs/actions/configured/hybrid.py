# src/rl_fzerox/core/envs/actions/configured/hybrid.py
"""Configured mixed continuous/discrete action adapter."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from gymnasium import spaces

from fzerox_emulator import ControllerState
from fzerox_emulator.arrays import ActionMask, NumpyArray
from rl_fzerox.core.envs.actions.base import (
    ActionBranchValue,
    ActionValue,
    DiscreteActionDimension,
    HybridActionValue,
    build_flat_action_mask,
    steer_values,
)
from rl_fzerox.core.envs.actions.buttons import RACE_CONTROL_MASKS
from rl_fzerox.core.envs.actions.configured.layout import (
    HybridActionLayout,
    categorical_lean_mask,
    configured_dimensions,
    idle_discrete_values,
    pitch_bucket_value,
)
from rl_fzerox.core.envs.actions.continuous_controls import (
    ContinuousButtonPwmDecoder,
    ContinuousDriveDecoder,
    continuous_action_array,
)
from rl_fzerox.core.runtime_spec.schema import ActionConfig, ActionRuntimeConfig

from .. import continuous_controls


class ConfiguredHybridActionAdapter:
    """Map one configured Dict(Box, MultiDiscrete) layout into emulator controls."""

    def __init__(self, config: ActionConfig | ActionRuntimeConfig) -> None:
        runtime = config.runtime() if isinstance(config, ActionConfig) else config
        self._config = runtime
        self._dimensions = configured_dimensions(runtime)
        self._action_space = _hybrid_action_space(
            HybridActionLayout(
                continuous_size=len(runtime.layout_continuous_axes),
                dimensions=self._dimensions,
            )
        )
        self._idle_action = {
            "continuous": np.zeros(len(runtime.layout_continuous_axes), dtype=np.float32),
            "discrete": idle_discrete_values(self._dimensions, runtime),
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
        discrete_values = continuous_controls.discrete_action_array(
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
            joypad_mask |= self._air_brake_decoder.decode(
                air_brake,
                button_mask=RACE_CONTROL_MASKS.air_brake,
            )

        for dimension, raw_value in zip(self._dimensions, discrete_values, strict=True):
            value = int(raw_value)
            if dimension.label == "steer":
                steer = float(self._steer_values[value])
            elif dimension.label == "gas" and value == 1:
                joypad_mask |= RACE_CONTROL_MASKS.accelerate
            elif dimension.label == "air_brake" and value == 1:
                joypad_mask |= RACE_CONTROL_MASKS.air_brake
            elif dimension.label == "boost" and value == 1:
                joypad_mask |= RACE_CONTROL_MASKS.boost
            elif dimension.label == "lean":
                joypad_mask |= categorical_lean_mask(
                    value,
                    four_way=self._config.lean_output_mode == "four_way_categorical",
                )
            elif dimension.label == "lean_left" and value == 1:
                joypad_mask |= RACE_CONTROL_MASKS.lean_left
            elif dimension.label == "lean_right" and value == 1:
                joypad_mask |= RACE_CONTROL_MASKS.lean_right
            elif dimension.label == "pitch":
                pitch = pitch_bucket_value(value, bucket_count=self._config.pitch_buckets)

        if self._config.force_full_throttle:
            joypad_mask |= RACE_CONTROL_MASKS.accelerate

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
        return _hybrid_action_mask(
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


def _hybrid_action_space(layout: HybridActionLayout) -> spaces.Dict:
    return spaces.Dict(
        {
            "continuous": spaces.Box(
                low=np.full(layout.continuous_size, -1.0, dtype=np.float32),
                high=np.full(layout.continuous_size, 1.0, dtype=np.float32),
                dtype=np.float32,
            ),
            "discrete": spaces.MultiDiscrete(
                np.array([dimension.size for dimension in layout.dimensions], dtype=np.int64)
            ),
        }
    )


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
