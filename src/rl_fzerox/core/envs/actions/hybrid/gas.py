# src/rl_fzerox/core/envs/actions/hybrid/gas.py
from __future__ import annotations

from gymnasium import spaces

from fzerox_emulator import ControllerState
from fzerox_emulator.arrays import ActionMask, NumpyArray
from rl_fzerox.core.config.schema import ActionConfig, ActionRuntimeConfig
from rl_fzerox.core.envs.actions.base import ActionValue, DiscreteActionDimension
from rl_fzerox.core.envs.actions.hybrid.common import (
    action_mask,
    button_mask,
    controller_state,
    pitch_value,
)
from rl_fzerox.core.envs.actions.hybrid.layouts import (
    PITCH_BUCKETS,
    STEER_GAS_AIR_BRAKE_BOOST_LEAN_LAYOUT,
    STEER_GAS_AIR_BRAKE_BOOST_LEAN_PITCH_LAYOUT,
    STEER_GAS_BOOST_LEAN_LAYOUT,
)
from rl_fzerox.core.envs.actions.hybrid.parsing import (
    parse_hybrid_steer_gas_air_brake_boost_lean,
    parse_hybrid_steer_gas_air_brake_boost_lean_pitch,
    parse_hybrid_steer_gas_boost_lean,
)
from rl_fzerox.core.envs.actions.hybrid.spaces import (
    hybrid_action_space,
    hybrid_idle_action,
)


class HybridSteerGasAirBrakeBoostLeanActionAdapter:
    """Map continuous steering plus discrete gas, air-brake, boost, and lean heads."""

    def __init__(self, config: ActionConfig | ActionRuntimeConfig) -> None:
        self._steer_response_power = float(config.steer_response_power)
        self._action_space = hybrid_action_space(STEER_GAS_AIR_BRAKE_BOOST_LEAN_LAYOUT)

    @property
    def action_space(self) -> spaces.Dict:
        """Return the public hybrid action space expected by maskable hybrid PPO."""

        return self._action_space

    @property
    def idle_action(self) -> dict[str, NumpyArray]:
        """Return a neutral hybrid action with no buttons held."""

        return hybrid_idle_action(STEER_GAS_AIR_BRAKE_BOOST_LEAN_LAYOUT)

    @property
    def action_dimensions(self) -> tuple[DiscreteActionDimension, ...]:
        """Return the discrete branches maskable by maskable hybrid PPO."""

        return STEER_GAS_AIR_BRAKE_BOOST_LEAN_LAYOUT.dimensions

    def decode(self, action: ActionValue) -> ControllerState:
        """Translate one hybrid action into steering and independent button heads."""

        steer, gas, air_brake, boost, lean = parse_hybrid_steer_gas_air_brake_boost_lean(action)
        joypad_mask = button_mask(gas=gas, air_brake=air_brake, boost=boost, lean=lean)
        return controller_state(
            joypad_mask=joypad_mask,
            steer=steer,
            response_power=self._steer_response_power,
        )

    def action_mask(
        self,
        *,
        base_overrides: dict[str, tuple[int, ...]] | None = None,
        stage_overrides: dict[str, tuple[int, ...]] | None = None,
        dynamic_overrides: dict[str, tuple[int, ...]] | None = None,
    ) -> ActionMask:
        """Return the discrete-branch mask used by MaskableHybridActionPPO."""

        return action_mask(
            STEER_GAS_AIR_BRAKE_BOOST_LEAN_LAYOUT,
            base_overrides=base_overrides,
            stage_overrides=stage_overrides,
            dynamic_overrides=dynamic_overrides,
        )


class HybridSteerGasAirBrakeBoostLeanPitchActionAdapter:
    """Map continuous steering plus discrete gas, air-brake, boost, lean, and pitch."""

    def __init__(self, config: ActionConfig | ActionRuntimeConfig) -> None:
        self._steer_response_power = float(config.steer_response_power)
        self._action_space = hybrid_action_space(STEER_GAS_AIR_BRAKE_BOOST_LEAN_PITCH_LAYOUT)

    @property
    def action_space(self) -> spaces.Dict:
        """Return the public hybrid action space expected by maskable hybrid PPO."""

        return self._action_space

    @property
    def idle_action(self) -> dict[str, NumpyArray]:
        """Return a neutral hybrid action with no buttons and neutral pitch."""

        action = hybrid_idle_action(STEER_GAS_AIR_BRAKE_BOOST_LEAN_PITCH_LAYOUT)
        action["discrete"][4] = PITCH_BUCKETS.neutral_index
        return action

    @property
    def action_dimensions(self) -> tuple[DiscreteActionDimension, ...]:
        """Return the discrete branches maskable by maskable hybrid PPO."""

        return STEER_GAS_AIR_BRAKE_BOOST_LEAN_PITCH_LAYOUT.dimensions

    def decode(self, action: ActionValue) -> ControllerState:
        """Translate one hybrid action into steering and independent button heads."""

        steer, gas, air_brake, boost, lean, pitch = (
            parse_hybrid_steer_gas_air_brake_boost_lean_pitch(action)
        )
        joypad_mask = button_mask(gas=gas, air_brake=air_brake, boost=boost, lean=lean)
        return controller_state(
            joypad_mask=joypad_mask,
            steer=steer,
            pitch=pitch_value(pitch),
            response_power=self._steer_response_power,
        )

    def action_mask(
        self,
        *,
        base_overrides: dict[str, tuple[int, ...]] | None = None,
        stage_overrides: dict[str, tuple[int, ...]] | None = None,
        dynamic_overrides: dict[str, tuple[int, ...]] | None = None,
    ) -> ActionMask:
        """Return the discrete-branch mask used by MaskableHybridActionPPO."""

        return action_mask(
            STEER_GAS_AIR_BRAKE_BOOST_LEAN_PITCH_LAYOUT,
            base_overrides=base_overrides,
            stage_overrides=stage_overrides,
            dynamic_overrides=dynamic_overrides,
        )


class HybridSteerGasBoostLeanActionAdapter:
    """Map continuous steering plus discrete gas, boost, and lean heads."""

    def __init__(self, config: ActionConfig | ActionRuntimeConfig) -> None:
        self._steer_response_power = float(config.steer_response_power)
        self._action_space = hybrid_action_space(STEER_GAS_BOOST_LEAN_LAYOUT)

    @property
    def action_space(self) -> spaces.Dict:
        """Return the public hybrid action space expected by maskable hybrid PPO."""

        return self._action_space

    @property
    def idle_action(self) -> dict[str, NumpyArray]:
        """Return a neutral hybrid action with no buttons held."""

        return hybrid_idle_action(STEER_GAS_BOOST_LEAN_LAYOUT)

    @property
    def action_dimensions(self) -> tuple[DiscreteActionDimension, ...]:
        """Return the discrete branches maskable by maskable hybrid PPO."""

        return STEER_GAS_BOOST_LEAN_LAYOUT.dimensions

    def decode(self, action: ActionValue) -> ControllerState:
        """Translate one hybrid action into steering and independent button heads."""

        steer, gas, boost, lean = parse_hybrid_steer_gas_boost_lean(action)
        joypad_mask = button_mask(gas=gas, boost=boost, lean=lean)
        return controller_state(
            joypad_mask=joypad_mask,
            steer=steer,
            response_power=self._steer_response_power,
        )

    def action_mask(
        self,
        *,
        base_overrides: dict[str, tuple[int, ...]] | None = None,
        stage_overrides: dict[str, tuple[int, ...]] | None = None,
        dynamic_overrides: dict[str, tuple[int, ...]] | None = None,
    ) -> ActionMask:
        """Return the discrete-branch mask used by MaskableHybridActionPPO."""

        return action_mask(
            STEER_GAS_BOOST_LEAN_LAYOUT,
            base_overrides=base_overrides,
            stage_overrides=stage_overrides,
            dynamic_overrides=dynamic_overrides,
        )
