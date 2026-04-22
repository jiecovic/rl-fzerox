# src/rl_fzerox/core/envs/engine/controls/availability.py
from __future__ import annotations

from fzerox_emulator import ControllerState, FZeroXTelemetry
from rl_fzerox.core.config.schema_models.common import ContinuousAirBrakeMode
from rl_fzerox.core.envs.actions import (
    AIR_BRAKE_MASK,
    BOOST_MASK,
    LEAN_LEFT_MASK,
    LEAN_RIGHT_MASK,
)
from rl_fzerox.core.envs.telemetry import telemetry_boost_active

from ..info import telemetry_can_boost, telemetry_energy_fraction
from .gates import with_left_stick_y, without_joypad_mask
from .history import ControlStateTracker
from .masks import (
    ActionMaskController,
    action_branch_non_neutral_allowed,
    action_branch_value_allowed,
)


def sync_dynamic_action_masks(
    *,
    mask_controller: ActionMaskController,
    control_state: ControlStateTracker,
    telemetry: FZeroXTelemetry | None,
    boost_min_energy_fraction: float,
) -> None:
    """Sync live telemetry-derived mask state into the action-mask controller."""

    if telemetry is None:
        mask_controller.set_boost_unlocked(None)
        mask_controller.set_speed_kph(None)
        mask_controller.set_airborne(None)
        return

    speed_kph = float(telemetry.player.speed_kph)
    mask_controller.set_speed_kph(speed_kph)
    mask_controller.set_airborne(bool(telemetry.player.airborne))

    can_boost = telemetry_can_boost(telemetry)
    can_boost = can_boost and not telemetry_boost_active(telemetry)
    can_boost = can_boost and not telemetry.player.airborne
    can_boost = can_boost and telemetry.player.reverse_timer <= 0
    can_boost = can_boost and control_state.boost_action_allowed_by_timing()

    max_boost_speed = mask_controller.current_boost_unmask_max_speed_kph
    if max_boost_speed is not None:
        can_boost = can_boost and speed_kph < float(max_boost_speed)

    energy_fraction = telemetry_energy_fraction(telemetry)
    min_energy_fraction = mask_controller.current_boost_min_energy_fraction(
        boost_min_energy_fraction
    )
    if energy_fraction is not None:
        can_boost = can_boost and energy_fraction > 0.0
        if min_energy_fraction > 0.0:
            can_boost = can_boost and energy_fraction >= min_energy_fraction

    mask_controller.set_boost_unlocked(can_boost)


def apply_dynamic_control_gates(
    control_state: ControllerState,
    *,
    mask_controller: ActionMaskController,
    continuous_air_brake_mode: ContinuousAirBrakeMode,
    last_telemetry: FZeroXTelemetry | None,
) -> ControllerState:
    """Suppress requested controls that are not available for the current frame."""

    branches = mask_controller.action_mask_branches()
    if not action_branch_value_allowed(
        branches,
        "boost",
        1,
        missing_allowed=True,
    ):
        control_state = without_joypad_mask(control_state, BOOST_MASK)
    if not action_branch_value_allowed(
        branches,
        "lean",
        1,
        missing_allowed=True,
    ):
        control_state = without_joypad_mask(control_state, LEAN_LEFT_MASK)
    if not action_branch_value_allowed(
        branches,
        "lean",
        2,
        missing_allowed=True,
    ):
        control_state = without_joypad_mask(control_state, LEAN_RIGHT_MASK)
    if not action_branch_value_allowed(
        branches,
        "air_brake",
        1,
        missing_allowed=True,
    ):
        control_state = without_joypad_mask(control_state, AIR_BRAKE_MASK)

    if continuous_air_brake_mode == "off":
        control_state = without_joypad_mask(control_state, AIR_BRAKE_MASK)

    pitch_non_neutral_allowed = action_branch_non_neutral_allowed(
        branches,
        "pitch",
        neutral_index=mask_controller.pitch_neutral_index,
        missing_allowed=True,
    )
    if not pitch_non_neutral_allowed:
        control_state = with_left_stick_y(control_state, 0.0)

    if last_telemetry is None or last_telemetry.player.airborne:
        return control_state
    return with_left_stick_y(without_joypad_mask(control_state, AIR_BRAKE_MASK), 0.0)
