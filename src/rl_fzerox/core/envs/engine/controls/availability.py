# src/rl_fzerox/core/envs/engine/controls/availability.py
"""Dynamic control availability based on telemetry and runtime gates.

The functions here update masks for boost, lean, air brake, and spin as the live
race state changes during an episode.
"""
from __future__ import annotations

from fzerox_emulator import FZeroXTelemetry, RaceControlState
from rl_fzerox.core.envs.engine.controls.gates import with_pitch, without_controls
from rl_fzerox.core.envs.engine.controls.history import ControlStateTracker
from rl_fzerox.core.envs.engine.controls.mask_queries import (
    action_branch_non_neutral_allowed,
    action_branch_value_allowed,
)
from rl_fzerox.core.envs.engine.controls.masks import ActionMaskController
from rl_fzerox.core.envs.engine.info import telemetry_boost_unlocked, telemetry_energy_fraction
from rl_fzerox.core.envs.telemetry import telemetry_boost_active
from rl_fzerox.core.runtime_spec.schema.common import ContinuousAirBrakeMode


def sync_dynamic_action_masks(
    *,
    mask_controller: ActionMaskController,
    control_state: ControlStateTracker,
    telemetry: FZeroXTelemetry | None,
    boost_min_energy_fraction: float,
    mask_boost_when_active: bool,
    mask_boost_when_airborne: bool,
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

    boost_unlocked = telemetry_boost_unlocked(telemetry)
    if mask_boost_when_active:
        boost_unlocked = boost_unlocked and not telemetry_boost_active(telemetry)
    if mask_boost_when_airborne:
        boost_unlocked = boost_unlocked and not telemetry.player.airborne
    boost_unlocked = boost_unlocked and telemetry.player.reverse_timer <= 0
    boost_unlocked = boost_unlocked and control_state.boost_action_allowed_by_timing()

    max_boost_speed = mask_controller.current_boost_unmask_max_speed_kph
    if max_boost_speed is not None:
        boost_unlocked = boost_unlocked and speed_kph < float(max_boost_speed)

    energy_fraction = telemetry_energy_fraction(telemetry)
    min_energy_fraction = mask_controller.current_boost_min_energy_fraction(
        boost_min_energy_fraction
    )
    if energy_fraction is not None:
        boost_unlocked = boost_unlocked and energy_fraction > 0.0
        if min_energy_fraction > 0.0:
            boost_unlocked = boost_unlocked and energy_fraction >= min_energy_fraction

    mask_controller.set_boost_unlocked(boost_unlocked)


def apply_dynamic_control_gates(
    control_state: RaceControlState,
    *,
    mask_controller: ActionMaskController,
    mask_air_brake_on_ground: bool,
    continuous_air_brake_mode: ContinuousAirBrakeMode,
    last_telemetry: FZeroXTelemetry | None,
) -> RaceControlState:
    """Suppress requested controls that are not available for the current frame."""

    branches = mask_controller.control_gate_action_mask_branches()
    if not action_branch_value_allowed(
        branches,
        "boost",
        1,
        missing_allowed=True,
    ):
        control_state = without_controls(control_state, boost=True)
    if control_state.lean_left and control_state.lean_right:
        if not _lean_button_pair_allowed(branches):
            control_state = without_controls(control_state, lean_left=True, lean_right=True)
    elif control_state.lean_left and not _lean_button_allowed(
        branches, split_label="lean_left", categorical_index=1
    ):
        control_state = without_controls(control_state, lean_left=True)
    elif control_state.lean_right and not _lean_button_allowed(
        branches, split_label="lean_right", categorical_index=2
    ):
        control_state = without_controls(control_state, lean_right=True)
    if not action_branch_value_allowed(
        branches,
        "air_brake",
        1,
        missing_allowed=True,
    ):
        control_state = without_controls(control_state, air_brake=True)

    if continuous_air_brake_mode == "off":
        control_state = without_controls(control_state, air_brake=True)

    pitch_non_neutral_allowed = action_branch_non_neutral_allowed(
        branches,
        "pitch",
        neutral_index=mask_controller.pitch_neutral_index,
        missing_allowed=True,
    )
    if not pitch_non_neutral_allowed:
        control_state = with_pitch(control_state, 0.0)

    if last_telemetry is not None and not last_telemetry.player.airborne:
        if mask_air_brake_on_ground or continuous_air_brake_mode == "disable_on_ground":
            control_state = without_controls(control_state, air_brake=True)
    return control_state


def _lean_button_allowed(
    branches: dict[str, tuple[bool, ...]],
    *,
    split_label: str,
    categorical_index: int,
) -> bool:
    if split_label in branches:
        return action_branch_value_allowed(
            branches,
            split_label,
            1,
            missing_allowed=True,
        )
    return action_branch_value_allowed(
        branches,
        "lean",
        categorical_index,
        missing_allowed=True,
    )


def _lean_button_pair_allowed(branches: dict[str, tuple[bool, ...]]) -> bool:
    if "lean_left" in branches or "lean_right" in branches:
        return _lean_button_allowed(
            branches, split_label="lean_left", categorical_index=1
        ) and _lean_button_allowed(branches, split_label="lean_right", categorical_index=2)
    lean_branch = branches.get("lean")
    if lean_branch is not None and len(lean_branch) >= 4:
        return action_branch_value_allowed(branches, "lean", 3, missing_allowed=True)
    return False
