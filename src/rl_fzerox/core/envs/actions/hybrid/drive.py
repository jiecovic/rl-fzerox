# src/rl_fzerox/core/envs/actions/hybrid/drive.py
from __future__ import annotations

from gymnasium import spaces

from fzerox_emulator import ControllerState
from fzerox_emulator.arrays import ActionMask, NumpyArray
from rl_fzerox.core.config.schema import ActionConfig, ActionRuntimeConfig
from rl_fzerox.core.envs.actions.base import ActionValue, DiscreteActionDimension
from rl_fzerox.core.envs.actions.continuous_controls import (
    ContinuousButtonPwmDecoder,
    ContinuousDriveDecoder,
)
from rl_fzerox.core.envs.actions.discrete.steer_drive import AIR_BRAKE_MASK
from rl_fzerox.core.envs.actions.discrete.steer_drive_boost_lean import BOOST_MASK
from rl_fzerox.core.envs.actions.hybrid.common import (
    action_mask,
    apply_lean_mask,
    button_mask,
    controller_state,
    pitch_value,
)
from rl_fzerox.core.envs.actions.hybrid.layouts import (
    PITCH_BUCKETS,
    STEER_DRIVE_AIR_BRAKE_BOOST_LEAN_PITCH_LAYOUT,
    STEER_DRIVE_AIR_BRAKE_PRIMITIVE_LAYOUT,
    STEER_DRIVE_BOOST_LEAN_LAYOUT,
    STEER_DRIVE_LEAN_LAYOUT,
)
from rl_fzerox.core.envs.actions.hybrid.parsing import (
    parse_hybrid_steer_drive_air_brake_boost_lean_pitch,
    parse_hybrid_steer_drive_boost_lean,
    parse_hybrid_steer_drive_boost_lean_pair,
    parse_hybrid_steer_drive_lean,
    with_default_lean_primitive_mask,
)
from rl_fzerox.core.envs.actions.hybrid.spaces import (
    hybrid_action_space,
    hybrid_idle_action,
)


class HybridSteerDriveLeanActionAdapter:
    """Map hybrid PPO actions to continuous steer/drive and discrete lean inputs."""

    def __init__(self, config: ActionConfig | ActionRuntimeConfig) -> None:
        self._steer_response_power = float(config.steer_response_power)
        self._drive_decoder = ContinuousDriveDecoder(
            deadzone=float(config.continuous_drive_deadzone),
            full_threshold=float(config.continuous_drive_full_threshold),
            min_thrust=float(config.continuous_drive_min_thrust),
        )
        self._action_space = hybrid_action_space(STEER_DRIVE_LEAN_LAYOUT)

    @property
    def action_space(self) -> spaces.Dict:
        """Return the public hybrid action space expected by maskable hybrid PPO."""

        return self._action_space

    @property
    def idle_action(self) -> dict[str, NumpyArray]:
        """Return a neutral hybrid action with no lean input held."""

        return hybrid_idle_action(STEER_DRIVE_LEAN_LAYOUT)

    @property
    def action_dimensions(self) -> tuple[DiscreteActionDimension, ...]:
        """Return the discrete lean branch maskable by hybrid PPO."""

        return STEER_DRIVE_LEAN_LAYOUT.dimensions

    def decode(self, action: ActionValue) -> ControllerState:
        """Translate one hybrid action into steering, drive, and lean buttons."""

        steer, drive, lean = parse_hybrid_steer_drive_lean(action)
        joypad_mask = self._drive_decoder.decode(drive)
        joypad_mask = apply_lean_mask(joypad_mask, lean)

        return controller_state(
            joypad_mask=joypad_mask,
            steer=steer,
            response_power=self._steer_response_power,
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

        return action_mask(
            STEER_DRIVE_LEAN_LAYOUT,
            base_overrides=base_overrides,
            stage_overrides=stage_overrides,
            dynamic_overrides=dynamic_overrides,
        )


class HybridSteerDriveBoostLeanActionAdapter:
    """Map hybrid PPO actions to continuous steer/drive, lean, and boost."""

    def __init__(self, config: ActionConfig | ActionRuntimeConfig) -> None:
        self._steer_response_power = float(config.steer_response_power)
        self._drive_decoder = ContinuousDriveDecoder(
            deadzone=float(config.continuous_drive_deadzone),
            full_threshold=float(config.continuous_drive_full_threshold),
            min_thrust=float(config.continuous_drive_min_thrust),
        )
        self._action_space = hybrid_action_space(STEER_DRIVE_BOOST_LEAN_LAYOUT)

    @property
    def action_space(self) -> spaces.Dict:
        """Return the public hybrid action space expected by maskable hybrid PPO."""

        return self._action_space

    @property
    def idle_action(self) -> dict[str, NumpyArray]:
        """Return a neutral hybrid action with no lean or boost held."""

        return hybrid_idle_action(STEER_DRIVE_BOOST_LEAN_LAYOUT)

    @property
    def action_dimensions(self) -> tuple[DiscreteActionDimension, ...]:
        """Return the discrete branches maskable by maskable hybrid PPO."""

        return STEER_DRIVE_BOOST_LEAN_LAYOUT.dimensions

    def decode(self, action: ActionValue) -> ControllerState:
        """Translate one hybrid action into steering, drive, lean, and boost."""

        steer, drive, lean, boost = parse_hybrid_steer_drive_boost_lean_pair(action)
        joypad_mask = self._drive_decoder.decode(drive)
        joypad_mask = apply_lean_mask(joypad_mask, lean)
        if boost == 1:
            joypad_mask |= BOOST_MASK

        return controller_state(
            joypad_mask=joypad_mask,
            steer=steer,
            response_power=self._steer_response_power,
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

        return action_mask(
            STEER_DRIVE_BOOST_LEAN_LAYOUT,
            base_overrides=base_overrides,
            stage_overrides=stage_overrides,
            dynamic_overrides=dynamic_overrides,
        )


class HybridSteerDriveAirBrakeBoostLeanPitchActionAdapter:
    """Map continuous steer/thrust plus discrete airborne controls."""

    def __init__(self, config: ActionConfig | ActionRuntimeConfig) -> None:
        self._steer_response_power = float(config.steer_response_power)
        self._drive_decoder = ContinuousDriveDecoder(
            deadzone=float(config.continuous_drive_deadzone),
            full_threshold=float(config.continuous_drive_full_threshold),
            min_thrust=float(config.continuous_drive_min_thrust),
        )
        self._action_space = hybrid_action_space(STEER_DRIVE_AIR_BRAKE_BOOST_LEAN_PITCH_LAYOUT)

    @property
    def action_space(self) -> spaces.Dict:
        """Return the public hybrid action space expected by maskable hybrid PPO."""

        return self._action_space

    @property
    def idle_action(self) -> dict[str, NumpyArray]:
        """Return a neutral hybrid action with no buttons and neutral pitch."""

        action = hybrid_idle_action(STEER_DRIVE_AIR_BRAKE_BOOST_LEAN_PITCH_LAYOUT)
        action["discrete"][3] = PITCH_BUCKETS.neutral_index
        return action

    @property
    def action_dimensions(self) -> tuple[DiscreteActionDimension, ...]:
        """Return the discrete branches maskable by maskable hybrid PPO."""

        return STEER_DRIVE_AIR_BRAKE_BOOST_LEAN_PITCH_LAYOUT.dimensions

    def decode(self, action: ActionValue) -> ControllerState:
        """Translate one hybrid action into steer/thrust, buttons, and pitch."""

        steer, drive, air_brake, boost, lean, pitch = (
            parse_hybrid_steer_drive_air_brake_boost_lean_pitch(action)
        )
        joypad_mask = self._drive_decoder.decode(drive)
        joypad_mask = button_mask(
            gas=0,
            air_brake=air_brake,
            boost=boost,
            lean=lean,
        ) | joypad_mask
        return controller_state(
            joypad_mask=joypad_mask,
            steer=steer,
            pitch=pitch_value(pitch),
            response_power=self._steer_response_power,
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

        return action_mask(
            STEER_DRIVE_AIR_BRAKE_BOOST_LEAN_PITCH_LAYOUT,
            base_overrides=base_overrides,
            stage_overrides=stage_overrides,
            dynamic_overrides=dynamic_overrides,
        )


class HybridSteerDriveBoostLeanPrimitiveActionAdapter:
    """Map hybrid PPO actions to continuous steer/drive/air-brake and lean primitives.

    The first three lean values match the current lean/slide primitive:
    `off`, `left`, `right`. Values 3..6 reserve architecture capacity for
    future side-attack and spin primitives and currently decode as no-ops while
    masked out by default.
    """

    def __init__(self, config: ActionConfig | ActionRuntimeConfig) -> None:
        self._steer_response_power = float(config.steer_response_power)
        self._drive_decoder = ContinuousDriveDecoder(
            deadzone=float(config.continuous_drive_deadzone),
            full_threshold=float(config.continuous_drive_full_threshold),
            min_thrust=float(config.continuous_drive_min_thrust),
        )
        self._air_brake_decoder = ContinuousButtonPwmDecoder(
            deadzone=float(config.continuous_drive_deadzone),
        )
        self._air_brake_enabled = config.continuous_air_brake_mode != "off"
        self._action_space = hybrid_action_space(STEER_DRIVE_AIR_BRAKE_PRIMITIVE_LAYOUT)

    @property
    def action_space(self) -> spaces.Dict:
        """Return the public hybrid action space expected by maskable hybrid PPO."""

        return self._action_space

    @property
    def idle_action(self) -> dict[str, NumpyArray]:
        """Return a neutral hybrid action with no lean primitive or boost."""

        return hybrid_idle_action(STEER_DRIVE_AIR_BRAKE_PRIMITIVE_LAYOUT)

    @property
    def action_dimensions(self) -> tuple[DiscreteActionDimension, ...]:
        """Return the discrete branches maskable by maskable hybrid PPO."""

        return STEER_DRIVE_AIR_BRAKE_PRIMITIVE_LAYOUT.dimensions

    def decode(self, action: ActionValue) -> ControllerState:
        """Translate one hybrid action into steering, drive, lean, and boost."""

        steer, drive, air_brake, lean, boost = parse_hybrid_steer_drive_boost_lean(action)
        air_brake_mask = (
            self._air_brake_decoder.decode(air_brake, button_mask=AIR_BRAKE_MASK)
            if self._air_brake_enabled
            else 0
        )
        joypad_mask = self._drive_decoder.decode(drive) | air_brake_mask
        joypad_mask = apply_lean_mask(joypad_mask, lean)
        if boost == 1:
            joypad_mask |= BOOST_MASK

        return controller_state(
            joypad_mask=joypad_mask,
            steer=steer,
            response_power=self._steer_response_power,
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

        return action_mask(
            STEER_DRIVE_AIR_BRAKE_PRIMITIVE_LAYOUT,
            base_overrides=with_default_lean_primitive_mask(base_overrides),
            stage_overrides=stage_overrides,
            dynamic_overrides=dynamic_overrides,
        )
