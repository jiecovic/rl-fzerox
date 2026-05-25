# src/rl_fzerox/core/envs/actions/configured/discrete.py
"""Configured fully discrete action adapter."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from fzerox_emulator import RaceControlState, SpinRequest
from fzerox_emulator.arrays import ActionMask, DiscreteAction
from rl_fzerox.core.envs.actions.base import (
    ActionValue,
    DecodedAction,
    DiscreteActionDimension,
    build_flat_action_mask,
    multidiscrete_space,
    parse_discrete_action,
    steer_values,
)
from rl_fzerox.core.envs.actions.configured.layout import (
    categorical_lean_state,
    configured_dimensions,
    idle_discrete_values,
    pitch_bucket_value,
    spin_request_value,
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

    def decode(self, action: ActionValue) -> RaceControlState:
        return self.decode_request(action).control_state

    def decode_request(self, action: ActionValue) -> DecodedAction:
        parsed = parse_discrete_action(
            action,
            action_label="Configured discrete action",
            dimensions=self._dimensions,
        )
        steer = 0.0
        pitch = 0.0
        gas = False
        air_brake = False
        boost = False
        lean_left = False
        lean_right = False
        spin_request: SpinRequest = "none"
        for dimension, value in zip(self._dimensions, parsed, strict=True):
            if dimension.label == "steer":
                steer = float(self._steer_values[value])
            elif dimension.label == "gas" and value == 1:
                gas = True
            elif dimension.label == "air_brake" and value == 1:
                air_brake = True
            elif dimension.label == "boost" and value == 1:
                boost = True
            elif dimension.label == "lean":
                lean_left, lean_right = categorical_lean_state(
                    value,
                    four_way=self._config.lean_output_mode == "four_way_categorical",
                )
            elif dimension.label == "lean_left" and value == 1:
                lean_left = True
            elif dimension.label == "lean_right" and value == 1:
                lean_right = True
            elif dimension.label == "spin":
                spin_request = spin_request_value(value)
            elif dimension.label == "pitch":
                pitch = pitch_bucket_value(value, bucket_count=self._config.pitch_buckets)

        if self._config.force_full_throttle:
            gas = True

        return DecodedAction(
            control_state=RaceControlState(
                gas=gas,
                air_brake=air_brake,
                boost=boost,
                lean_left=lean_left,
                lean_right=lean_right,
                stick_x=steer,
                pitch=pitch,
            ),
            spin_request=spin_request,
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
