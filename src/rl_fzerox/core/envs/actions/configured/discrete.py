# src/rl_fzerox/core/envs/actions/configured/discrete.py
"""Configured fully discrete action adapter."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from fzerox_emulator import ControllerState
from fzerox_emulator.arrays import ActionMask, DiscreteAction
from rl_fzerox.core.envs.actions.base import (
    ActionValue,
    DiscreteActionDimension,
    build_flat_action_mask,
    multidiscrete_space,
    parse_discrete_action,
    steer_values,
)
from rl_fzerox.core.envs.actions.buttons import RACE_CONTROL_MASKS
from rl_fzerox.core.envs.actions.configured.layout import (
    apply_pitch_deadzone,
    configured_dimensions,
    idle_discrete_values,
    lean_mask,
    pitch_bucket_value,
)
from rl_fzerox.core.runtime_spec.schema import ActionConfig, ActionRuntimeConfig


class ConfiguredDiscreteActionAdapter:
    """Map one configured MultiDiscrete layout into emulator controls."""

    def __init__(self, config: ActionConfig | ActionRuntimeConfig) -> None:
        runtime = config.runtime() if isinstance(config, ActionConfig) else config
        self._config = runtime
        self._dimensions = configured_dimensions(runtime)
        self._action_space = multidiscrete_space(
            *(dimension.size for dimension in self._dimensions)
        )
        self._idle_action = idle_discrete_values(self._dimensions, runtime)
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
            elif dimension.label == "gas" and value == 1:
                joypad_mask |= RACE_CONTROL_MASKS.accelerate
            elif dimension.label == "air_brake" and value == 1:
                joypad_mask |= RACE_CONTROL_MASKS.air_brake
            elif dimension.label == "boost" and value == 1:
                joypad_mask |= RACE_CONTROL_MASKS.boost
            elif dimension.label == "lean":
                joypad_mask |= lean_mask(
                    value,
                    independent_buttons=self._config.independent_lean_buttons,
                )
            elif dimension.label == "pitch":
                pitch = pitch_bucket_value(value, bucket_count=self._config.pitch_buckets)

        pitch = apply_pitch_deadzone(pitch, deadzone=float(self._config.pitch_deadzone))

        if self._config.force_full_throttle:
            joypad_mask |= RACE_CONTROL_MASKS.accelerate

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
