# src/rl_fzerox/ui/watch/view/panels/visuals/viz.py
"""Convert controls, policy actions, and masks into Watch cockpit visuals.

The functions here derive UI-ready `ControlViz` and flag data. Actual cockpit
widget drawing lives under `view/components/cockpit`.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from fzerox_emulator import RaceControlState
from fzerox_emulator.arrays import Float32Array
from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.core.envs.engine.controls import (
    ActionMaskBranches,
    action_branch_non_neutral_allowed,
    action_branch_value_allowed,
    selected_action_branches,
)
from rl_fzerox.ui.watch.view.components.cockpit.style import COCKPIT_PANEL_STYLE
from rl_fzerox.ui.watch.view.screen.layout import LAYOUT
from rl_fzerox.ui.watch.view.screen.theme import PALETTE
from rl_fzerox.ui.watch.view.screen.types import (
    ControlViz,
    FlagToken,
    FlagViz,
    RenderFont,
    ViewerFonts,
)


def _control_viz(
    control_state: RaceControlState,
    *,
    gas_level: float,
    thrust_warning_threshold: float | None = None,
    thrust_deadzone_threshold: float | None = None,
    thrust_full_threshold: float | None = None,
    engine_setting_level: float | None = None,
    speed_kph: float | None = None,
    energy_fraction: float | None = None,
    boost_active: bool = False,
    boost_lamp_level: float = 0.0,
    policy_deterministic: bool | None = None,
    policy_action: ActionValue | None = None,
    action_mask_branches: ActionMaskBranches | None = None,
    continuous_drive_deadzone: float = 0.2,
    continuous_drive_enabled: bool = False,
    force_full_throttle: bool = False,
    continuous_pitch_enabled: bool = False,
    continuous_air_brake_axis_index: int | None = 2,
    continuous_air_brake_deadzone: float = 0.05,
    continuous_air_brake_full_threshold: float = 0.85,
    continuous_air_brake_min_duty: float = 0.0,
    continuous_air_brake_mode: str = "always",
    continuous_air_brake_disabled: bool = False,
    spin_requested: bool = False,
    spin_request: str = "none",
    spin_macro_active: bool = False,
    spin_macro_cooldown_frames: int = 0,
) -> ControlViz:
    continuous_air_brake_enabled = continuous_air_brake_mode != "off"
    continuous_air_brake_exposed = _continuous_air_brake_axis_available_from_action(
        policy_action,
        axis_index=continuous_air_brake_axis_index,
        continuous_air_brake_enabled=continuous_air_brake_enabled,
    )
    air_brake_axis = _continuous_air_brake_axis_from_action(
        policy_action,
        axis_index=continuous_air_brake_axis_index,
        continuous_air_brake_deadzone=continuous_air_brake_deadzone,
        continuous_air_brake_full_threshold=continuous_air_brake_full_threshold,
        continuous_air_brake_min_duty=continuous_air_brake_min_duty,
        continuous_air_brake_enabled=continuous_air_brake_enabled,
    )
    if air_brake_axis is None and continuous_air_brake_mode != "off" and control_state.air_brake:
        air_brake_axis = 1.0
    selected_branches = _selected_policy_branches(
        action_mask_branches=action_mask_branches,
        policy_action=policy_action,
    )
    spin_direction = _spin_direction(selected_branches)
    if spin_direction == 0:
        spin_direction = _spin_request_direction(spin_request)
    normalized_spin_cooldown_frames = max(0, int(spin_macro_cooldown_frames))
    spin_runtime_masked = spin_macro_active or normalized_spin_cooldown_frames > 0
    boost_pressed = _branch_pressed(
        selected_branches,
        "boost",
        fallback=control_state.boost,
    )
    lean_left_pressed, lean_right_pressed = _lean_buttons(
        {} if spin_direction != 0 or spin_macro_active else selected_branches,
        fallback_left=control_state.lean_left,
        fallback_right=control_state.lean_right,
    )
    normalized_boost_lamp_level = max(
        0.0,
        min(
            1.0,
            max(
                boost_lamp_level,
                1.0 if boost_pressed else 0.0,
                0.55 if boost_active else 0.0,
            ),
        ),
    )
    displayed_gas_level = _displayed_gas_level(
        gas_pressed=control_state.gas,
        gas_level=gas_level,
        continuous_drive_enabled=continuous_drive_enabled,
        force_full_throttle=force_full_throttle,
    )
    return ControlViz(
        steer_x=max(-1.0, min(1.0, control_state.stick_x)),
        pitch_y=max(-1.0, min(1.0, control_state.pitch)),
        gas_level=displayed_gas_level,
        thrust_warning_threshold=_normalize_optional_level(thrust_warning_threshold),
        thrust_deadzone_threshold=_normalize_optional_level(thrust_deadzone_threshold),
        thrust_full_threshold=_normalize_optional_level(thrust_full_threshold),
        engine_setting_level=_normalize_optional_level(engine_setting_level),
        speed_kph=_normalize_optional_speed(speed_kph),
        energy_fraction=_normalize_optional_level(energy_fraction),
        air_brake_axis=air_brake_axis,
        air_brake_disabled=continuous_air_brake_disabled and air_brake_axis is not None,
        boost_pressed=boost_pressed,
        boost_active=boost_active,
        boost_lamp_level=normalized_boost_lamp_level,
        lean_direction=_lean_direction(
            left_pressed=lean_left_pressed,
            right_pressed=lean_right_pressed,
        ),
        spin_direction=spin_direction,
        spin_requested=spin_requested or spin_direction != 0,
        spin_macro_active=spin_macro_active,
        spin_macro_cooldown_frames=normalized_spin_cooldown_frames,
        spin_left_masked=spin_runtime_masked,
        spin_right_masked=spin_runtime_masked,
        lean_left_pressed=lean_left_pressed,
        lean_right_pressed=lean_right_pressed,
        deterministic_policy=policy_deterministic,
        thrust_masked=(
            False
            if force_full_throttle or continuous_drive_enabled
            else not action_branch_value_allowed(
                action_mask_branches,
                "gas",
                1,
                missing_allowed=False,
            )
        ),
        air_brake_masked=(
            False
            if continuous_air_brake_exposed
            else not action_branch_value_allowed(
                action_mask_branches,
                "air_brake",
                1,
                missing_allowed=False,
            )
        ),
        boost_masked=not action_branch_value_allowed(
            action_mask_branches,
            "boost",
            1,
            missing_allowed=False,
        ),
        lean_left_masked=not _lean_button_allowed(action_mask_branches, "lean_left", 1),
        lean_right_masked=not _lean_button_allowed(action_mask_branches, "lean_right", 2),
        pitch_masked=(
            False
            if continuous_pitch_enabled
            else not action_branch_non_neutral_allowed(
                action_mask_branches,
                "pitch",
                neutral_index=2,
                missing_allowed=False,
            )
        ),
    )


def _displayed_gas_level(
    *,
    gas_pressed: bool,
    gas_level: float,
    continuous_drive_enabled: bool,
    force_full_throttle: bool,
) -> float:
    if force_full_throttle:
        return 1.0
    if continuous_drive_enabled:
        return max(0.0, min(1.0, gas_level))
    return 1.0 if gas_pressed else 0.0


def _selected_policy_branches(
    *,
    action_mask_branches: ActionMaskBranches | None,
    policy_action: ActionValue | None,
) -> dict[str, int]:
    if action_mask_branches is None or policy_action is None:
        return {}
    return selected_action_branches(action_mask_branches, policy_action)


def _branch_pressed(
    selected_branches: dict[str, int],
    label: str,
    *,
    fallback: bool,
) -> bool:
    if label not in selected_branches:
        return fallback
    return selected_branches[label] == 1


def _lean_buttons(
    selected_branches: dict[str, int],
    *,
    fallback_left: bool,
    fallback_right: bool,
) -> tuple[bool, bool]:
    if "lean_left" in selected_branches or "lean_right" in selected_branches:
        return (
            selected_branches.get("lean_left") == 1,
            selected_branches.get("lean_right") == 1,
        )
    lean = selected_branches.get("lean")
    if lean is None:
        return fallback_left, fallback_right
    if lean == 1:
        return True, False
    if lean == 2:
        return False, True
    if lean == 3:
        return True, True
    return False, False


def _lean_direction(*, left_pressed: bool, right_pressed: bool) -> int:
    if left_pressed == right_pressed:
        return 0
    return -1 if left_pressed else 1


def _spin_direction(selected_branches: dict[str, int]) -> int:
    spin = selected_branches.get("spin")
    if spin == 1:
        return -1
    if spin == 2:
        return 1
    return 0


def _spin_request_direction(spin_request: str) -> int:
    if spin_request == "left":
        return -1
    if spin_request == "right":
        return 1
    return 0


def _lean_button_allowed(
    branches: ActionMaskBranches | None,
    split_label: str,
    categorical_index: int,
) -> bool:
    if branches is not None and split_label in branches:
        return action_branch_value_allowed(
            branches,
            split_label,
            1,
            missing_allowed=False,
        )
    pair_allowed = action_branch_value_allowed(
        branches,
        "lean",
        3,
        missing_allowed=False,
    )
    return (
        action_branch_value_allowed(
            branches,
            "lean",
            categorical_index,
            missing_allowed=False,
        )
        or pair_allowed
    )


def _continuous_air_brake_axis_from_action(
    policy_action: ActionValue | None,
    *,
    axis_index: int | None,
    continuous_air_brake_deadzone: float,
    continuous_air_brake_full_threshold: float,
    continuous_air_brake_min_duty: float,
    continuous_air_brake_enabled: bool,
) -> float | None:
    """Return HUD air-brake value when the policy action exposes that axis."""

    values = _continuous_action_values(
        policy_action,
        axis_index=axis_index,
        continuous_air_brake_enabled=continuous_air_brake_enabled,
    )
    if values is None or axis_index is None:
        return None
    return _continuous_air_brake_axis(
        float(values[axis_index]),
        continuous_air_brake_deadzone=continuous_air_brake_deadzone,
        continuous_air_brake_full_threshold=continuous_air_brake_full_threshold,
        continuous_air_brake_min_duty=continuous_air_brake_min_duty,
    )


def _continuous_air_brake_axis_available_from_action(
    policy_action: ActionValue | None,
    *,
    axis_index: int | None,
    continuous_air_brake_enabled: bool,
) -> bool:
    return (
        _continuous_action_values(
            policy_action,
            axis_index=axis_index,
            continuous_air_brake_enabled=continuous_air_brake_enabled,
        )
        is not None
    )


def _continuous_action_values(
    policy_action: ActionValue | None,
    *,
    axis_index: int | None,
    continuous_air_brake_enabled: bool,
) -> Float32Array | None:
    if policy_action is None or axis_index is None or not continuous_air_brake_enabled:
        return None
    source = (
        policy_action.get("continuous") if isinstance(policy_action, Mapping) else policy_action
    )
    action = np.asarray(source)
    if not np.issubdtype(action.dtype, np.floating):
        return None
    values = action.reshape(-1)
    return values if values.size > axis_index else None


def _continuous_air_brake_axis(
    air_brake: float,
    *,
    continuous_air_brake_deadzone: float,
    continuous_air_brake_full_threshold: float,
    continuous_air_brake_min_duty: float,
) -> float | None:
    if not np.isfinite(air_brake):
        return None
    air_brake = max(-1.0, min(1.0, air_brake))
    return _continuous_positive_button_level(
        air_brake,
        deadzone=continuous_air_brake_deadzone,
        full_threshold=continuous_air_brake_full_threshold,
        min_duty=continuous_air_brake_min_duty,
    )


def _continuous_positive_button_level(
    value: float,
    *,
    deadzone: float,
    full_threshold: float,
    min_duty: float,
) -> float:
    if value <= deadzone:
        return 0.0
    if value >= full_threshold:
        return 1.0
    scaled = (value - deadzone) / (full_threshold - deadzone)
    return min(max(min_duty + ((1.0 - min_duty) * scaled), 0.0), 1.0)


def _normalize_optional_level(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


def _normalize_optional_speed(value: float | None) -> float | None:
    if value is None:
        return None
    scalar = float(value)
    if not np.isfinite(scalar):
        return None
    return max(0.0, scalar)


def _control_viz_height(fonts: ViewerFonts) -> int:
    _ = fonts
    return COCKPIT_PANEL_STYLE.wide_panel_height


_FLAG_DISPLAY_LABELS = {
    "reverse_detected": "reverse",
    "low_speed_detected": "slow",
    "can_boost": "boost unlocked",
    "boost_active": "boost",
    "refill_surface": "refill",
    "ground_dirt": "dirt",
    "ground_ice": "ice",
    "energy_loss": "energy loss",
    "damage_taken": "damage taken",
    "track_edge": "edge",
    "outside_track_bounds": "outside",
    "airborne": "airborne",
    "collision_recoil": "recoil",
    "spinning_out": "spin",
    "crashed": "crash",
    "falling_off_track": "off-track",
    "energy_depleted": "depleted",
    "retired": "retired",
    "finished": "finish",
}

_FLAG_ROWS = (
    ("reverse_detected", "low_speed_detected", "track_edge", "outside_track_bounds"),
    ("can_boost", "boost_active", "airborne"),
    ("refill_surface", "ground_dirt", "ground_ice"),
    ("energy_loss", "damage_taken"),
    ("collision_recoil", "spinning_out", "crashed", "falling_off_track"),
    ("energy_depleted", "retired", "finished"),
)


def _flag_viz(
    state_labels: tuple[str, ...],
    *,
    boost_active: bool,
    reverse_detected: bool,
    low_speed_detected: bool,
    energy_depleted: bool,
    refill_surface_detected: bool = False,
    dirt_detected: bool = False,
    ice_detected: bool = False,
    track_edge_detected: bool = False,
    outside_track_bounds: bool = False,
    energy_loss_detected: bool = False,
    damage_taken_detected: bool = False,
) -> FlagViz:
    active_flags = set(state_labels)
    if boost_active:
        active_flags.add("boost_active")
    if reverse_detected:
        active_flags.add("reverse_detected")
    if low_speed_detected:
        active_flags.add("low_speed_detected")
    if energy_depleted:
        active_flags.add("energy_depleted")
    if refill_surface_detected:
        active_flags.add("refill_surface")
    if dirt_detected:
        active_flags.add("ground_dirt")
    if ice_detected:
        active_flags.add("ground_ice")
    if track_edge_detected:
        active_flags.add("track_edge")
    if outside_track_bounds:
        active_flags.add("outside_track_bounds")
    if energy_loss_detected:
        active_flags.add("energy_loss")
    if damage_taken_detected:
        active_flags.add("damage_taken")
    known_flags = {flag_label for row in _FLAG_ROWS for flag_label in row}
    active_flags.intersection_update(known_flags)
    return FlagViz(
        rows=tuple(
            tuple(
                FlagToken(
                    label=_FLAG_DISPLAY_LABELS.get(flag_label, flag_label.replace("_", " ")),
                    active=flag_label in active_flags,
                )
                for flag_label in row
            )
            for row in _FLAG_ROWS
        )
    )


def _flag_viz_height(fonts: ViewerFonts, flag_viz: FlagViz) -> int:
    row_height = fonts.small.render("flags", True, PALETTE.text_primary).get_height()
    pill_height = row_height + (2 * LAYOUT.flag_token_pad_y)
    return row_height + LAYOUT.line_gap + (len(flag_viz.rows) * (pill_height + LAYOUT.line_gap))


def _wrap_text(font: RenderFont, text: str, max_width: int) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return [""]
    if font.render(normalized, True, PALETTE.text_primary).get_width() <= max_width:
        return [normalized]

    wrapped: list[str] = []
    current = ""
    for token in normalized.split(" "):
        candidate = token if not current else f"{current} {token}"
        if current and font.render(candidate, True, PALETTE.text_primary).get_width() > max_width:
            wrapped.append(current)
            current = token
        else:
            current = candidate
    if current:
        wrapped.append(current)
    return wrapped
