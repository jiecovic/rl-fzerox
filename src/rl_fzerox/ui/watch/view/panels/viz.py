# src/rl_fzerox/ui/watch/view/panels/viz.py
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from fzerox_emulator import ControllerState
from fzerox_emulator.arrays import ContinuousAction
from rl_fzerox.core.domain.hybrid_action import HYBRID_CONTINUOUS_ACTION_KEY
from rl_fzerox.core.envs.actions import (
    AIR_BRAKE_MASK,
    BOOST_MASK,
    LEAN_LEFT_MASK,
    LEAN_RIGHT_MASK,
    ActionValue,
)
from rl_fzerox.core.envs.engine.masks import (
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
    control_state: ControllerState,
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
    continuous_air_brake_mode: str = "always",
    continuous_air_brake_disabled: bool = False,
) -> ControlViz:
    joypad_mask = control_state.joypad_mask
    air_brake_axis = _continuous_air_brake_axis_from_action(
        policy_action,
        continuous_drive_deadzone=continuous_drive_deadzone,
        continuous_air_brake_enabled=continuous_air_brake_mode != "off",
    )
    if (
        air_brake_axis is None
        and continuous_air_brake_mode != "off"
        and joypad_mask & AIR_BRAKE_MASK
    ):
        air_brake_axis = 1.0
    selected_branches = _selected_policy_branches(
        action_mask_branches=action_mask_branches,
        policy_action=policy_action,
    )
    boost_pressed = _branch_pressed(
        selected_branches,
        "boost",
        fallback=bool(joypad_mask & BOOST_MASK),
    )
    lean_direction = _lean_direction(
        selected_branches,
        fallback=-1 if joypad_mask & LEAN_LEFT_MASK else 1 if joypad_mask & LEAN_RIGHT_MASK else 0,
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
    return ControlViz(
        steer_x=max(-1.0, min(1.0, control_state.left_stick_x)),
        pitch_y=max(-1.0, min(1.0, control_state.left_stick_y)),
        gas_level=max(0.0, min(1.0, gas_level)),
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
        lean_direction=lean_direction,
        deterministic_policy=policy_deterministic,
        thrust_masked=not action_branch_value_allowed(
            action_mask_branches,
            "gas",
            1,
            missing_allowed=True,
        ),
        air_brake_masked=not action_branch_value_allowed(
            action_mask_branches,
            "air_brake",
            1,
            missing_allowed=False,
        ),
        boost_masked=not action_branch_value_allowed(
            action_mask_branches,
            "boost",
            1,
            missing_allowed=False,
        ),
        lean_left_masked=not action_branch_value_allowed(
            action_mask_branches,
            "lean",
            1,
            missing_allowed=False,
        ),
        lean_right_masked=not action_branch_value_allowed(
            action_mask_branches,
            "lean",
            2,
            missing_allowed=False,
        ),
        pitch_masked=not action_branch_non_neutral_allowed(
            action_mask_branches,
            "pitch",
            neutral_index=2,
            missing_allowed=False,
        ),
    )


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


def _lean_direction(selected_branches: dict[str, int], *, fallback: int) -> int:
    lean = selected_branches.get("lean")
    if lean is None:
        return fallback
    if lean == 1:
        return -1
    if lean == 2:
        return 1
    return 0


def _continuous_air_brake_axis_from_action(
    policy_action: ActionValue | None,
    *,
    continuous_drive_deadzone: float,
    continuous_air_brake_enabled: bool,
) -> float | None:
    """Return HUD air-brake value when the policy action exposes that axis."""

    if policy_action is None:
        return None
    source = (
        policy_action.get(HYBRID_CONTINUOUS_ACTION_KEY)
        if isinstance(policy_action, Mapping)
        else policy_action
    )
    action = np.asarray(source)
    if not np.issubdtype(action.dtype, np.floating):
        return None
    values = action.reshape(-1)
    if values.size < 3 or not continuous_air_brake_enabled:
        return None
    return _continuous_air_brake_axis(
        values,
        continuous_drive_deadzone=continuous_drive_deadzone,
    )


def _continuous_air_brake_axis(
    values: ContinuousAction,
    *,
    continuous_drive_deadzone: float,
) -> float | None:
    if values.size < 3:
        return None
    air_brake = float(values[2])
    if not np.isfinite(air_brake):
        return None
    air_brake = max(-1.0, min(1.0, air_brake))
    return _continuous_positive_button_level(
        air_brake,
        deadzone=continuous_drive_deadzone,
    )


def _continuous_positive_button_level(value: float, *, deadzone: float) -> float:
    if value <= deadzone:
        return 0.0
    return (value - deadzone) / (1.0 - deadzone)


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
    "can_boost": "can boost",
    "boost_active": "boost",
    "energy_refill": "refill",
    "ground_dirt": "dirt",
    "ground_ice": "ice",
    "energy_loss": "energy loss",
    "damage_taken": "damage taken",
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
    ("reverse_detected", "low_speed_detected"),
    ("can_boost", "boost_active", "airborne"),
    ("energy_refill", "ground_dirt", "ground_ice"),
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
    energy_refill_detected: bool = False,
    dirt_detected: bool = False,
    ice_detected: bool = False,
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
    if energy_refill_detected:
        active_flags.add("energy_refill")
    if dirt_detected:
        active_flags.add("ground_dirt")
    if ice_detected:
        active_flags.add("ground_ice")
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
