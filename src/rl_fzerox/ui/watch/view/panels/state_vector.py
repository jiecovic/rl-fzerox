# src/rl_fzerox/ui/watch/view/panels/state_vector.py
from __future__ import annotations

import numpy as np

from fzerox_emulator.arrays import StateVector
from rl_fzerox.ui.watch.view.panels.lines import panel_divider, panel_heading, panel_line
from rl_fzerox.ui.watch.view.screen.theme import PALETTE
from rl_fzerox.ui.watch.view.screen.types import PanelLine, PanelSection


def policy_state_sections(
    *,
    observation_state: StateVector | None,
    feature_names: tuple[str, ...],
    zeroed_components: frozenset[str] = frozenset(),
    zeroed_features: frozenset[str] = frozenset(),
) -> list[PanelSection]:
    if observation_state is None:
        return []

    values: StateVector = np.asarray(observation_state, dtype=np.float32).reshape(-1)
    names = (
        feature_names
        if len(feature_names) == values.size
        else tuple(f"state_{index}" for index in range(values.size))
    )
    section_lines: list[PanelLine] = []
    for group_title, group_prefix, component_name in _state_vector_groups(names):
        group_zeroed = component_name in zeroed_components
        group_lines = _state_vector_group_lines(
            names=names,
            values=values,
            group_prefix=group_prefix,
            zeroed=group_zeroed,
            zeroed_features=zeroed_features,
        )
        if group_lines:
            if section_lines:
                section_lines.append(panel_divider())
            section_lines.append(
                panel_heading(_state_vector_group_title(group_title, group_zeroed))
            )
            section_lines.extend(group_lines)
    if not section_lines:
        return []
    return [PanelSection(title="State Vector", lines=section_lines)]


def _state_vector_groups(names: tuple[str, ...]) -> tuple[tuple[str, str | None, str | None], ...]:
    component_groups = (
        ("Vehicle", "vehicle_state.", "vehicle_state"),
        ("Machine", "machine_context.", "machine_context"),
        ("Track Position", "track_position.", "track_position"),
        ("Surface", "surface_state.", "surface_state"),
        ("Course", "course_context.", "course_context"),
    )
    used_component_names = {
        name
        for _, prefix, _ in component_groups
        for name in names
        if prefix is not None and name.startswith(prefix)
    }
    groups: tuple[tuple[str, str | None, str | None], ...] = tuple(
        (title, prefix, component_name)
        for title, prefix, component_name in component_groups
        if any(name.startswith(prefix) for name in names)
    )
    if any(name.startswith("control_history.") or name.startswith("prev_") for name in names):
        groups = (*groups, ("Control History", "control_history.", "control_history"))
    legacy_names = tuple(
        name
        for name in names
        if name not in used_component_names
        and not name.startswith("control_history.")
        and not name.startswith("prev_")
    )
    if legacy_names:
        return (*groups, ("State", None, None))
    return groups


def _state_vector_group_lines(
    *,
    names: tuple[str, ...],
    values: StateVector,
    group_prefix: str | None,
    zeroed: bool,
    zeroed_features: frozenset[str],
) -> list[PanelLine]:
    if group_prefix == "course_context.":
        return _course_context_state_lines(
            names=names,
            values=values,
            zeroed=zeroed,
            zeroed_features=zeroed_features,
        )
    if group_prefix == "control_history.":
        return _control_history_state_lines(
            names=names,
            values=values,
            zeroed=zeroed,
            zeroed_features=zeroed_features,
        )
    return [
        panel_line(
            _state_vector_line_label(
                name,
                group_prefix=group_prefix,
                zeroed=_state_vector_entry_zeroed(
                    name,
                    zeroed=zeroed,
                    zeroed_features=zeroed_features,
                ),
            ),
            f"{float(value):.3f}",
            _state_vector_line_color(
                _state_vector_entry_zeroed(
                    name,
                    zeroed=zeroed,
                    zeroed_features=zeroed_features,
                )
            ),
        )
        for name, value in zip(names, values, strict=True)
        if _state_vector_name_matches_group(name, group_prefix)
    ]


def _control_history_state_lines(
    *,
    names: tuple[str, ...],
    values: StateVector,
    zeroed: bool,
    zeroed_features: frozenset[str],
) -> list[PanelLine]:
    return [
        panel_line(
            _state_vector_line_label(
                name,
                group_prefix="control_history.",
                zeroed=_state_vector_entry_zeroed(
                    name,
                    zeroed=zeroed,
                    zeroed_features=zeroed_features,
                ),
            ),
            f"{float(value):.3f}",
            _state_vector_line_color(
                _state_vector_entry_zeroed(
                    name,
                    zeroed=zeroed,
                    zeroed_features=zeroed_features,
                )
            ),
        )
        for name, value in zip(names, values, strict=True)
        if name.startswith("control_history.") or name.startswith("prev_")
    ]


def _course_context_state_lines(
    *,
    names: tuple[str, ...],
    values: StateVector,
    zeroed: bool,
    zeroed_features: frozenset[str],
) -> list[PanelLine]:
    course_bits = [
        float(value)
        for name, value in zip(names, values, strict=True)
        if name.startswith("course_context.course_builtin_")
    ]
    if not course_bits:
        return []

    active_index = _one_hot_active_index(course_bits)
    encoded_bits = "".join("1" if value >= 0.5 else "0" for value in course_bits)
    value = f"-- | {encoded_bits}" if active_index is None else f"{active_index} | {encoded_bits}"
    feature_zeroed = _state_vector_entry_zeroed(
        "course_context",
        zeroed=zeroed,
        zeroed_features=zeroed_features,
    )
    return [
        panel_line(
            _zeroed_label("course", zeroed=feature_zeroed),
            value,
            _state_vector_line_color(feature_zeroed),
        ),
    ]


def _one_hot_active_index(values: list[float]) -> int | None:
    active_indices = [index for index, value in enumerate(values) if value >= 0.5]
    if len(active_indices) == 1:
        return active_indices[0]
    return None


def _state_vector_name_matches_group(name: str, group_prefix: str | None) -> bool:
    if group_prefix is None:
        return "." not in name and not name.startswith("prev_")
    return name.startswith(group_prefix)


def _state_vector_label(name: str, *, group_prefix: str | None) -> str:
    if group_prefix is None:
        return name
    if group_prefix == "control_history." and name.startswith("prev_"):
        return name
    return name.removeprefix(group_prefix)


def _state_vector_group_title(title: str, zeroed: bool) -> str:
    return f"// {title}" if zeroed else title


def _state_vector_entry_zeroed(
    name: str,
    *,
    zeroed: bool,
    zeroed_features: frozenset[str],
) -> bool:
    return zeroed or name in zeroed_features


def _state_vector_line_label(
    name: str,
    *,
    group_prefix: str | None,
    zeroed: bool,
) -> str:
    return _zeroed_label(_state_vector_label(name, group_prefix=group_prefix), zeroed=zeroed)


def _zeroed_label(label: str, *, zeroed: bool) -> str:
    return f"// {label}" if zeroed else label


def _state_vector_line_color(zeroed: bool):
    return PALETTE.text_muted if zeroed else PALETTE.text_primary
