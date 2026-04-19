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
    for group_title, group_prefix in _state_vector_groups(names):
        group_lines = _state_vector_group_lines(
            names=names,
            values=values,
            group_prefix=group_prefix,
        )
        if group_lines:
            if section_lines:
                section_lines.append(panel_divider())
            section_lines.append(panel_heading(group_title))
            section_lines.extend(group_lines)
    if not section_lines:
        return []
    return [PanelSection(title="State Vector", lines=section_lines)]


def _state_vector_groups(names: tuple[str, ...]) -> tuple[tuple[str, str | None], ...]:
    component_groups = (
        ("Vehicle", "vehicle_state."),
        ("Track Position", "track_position."),
        ("Surface", "surface_state."),
        ("Course", "course_context."),
    )
    used_component_names = {
        name
        for _, prefix in component_groups
        for name in names
        if prefix is not None and name.startswith(prefix)
    }
    groups: tuple[tuple[str, str | None], ...] = tuple(
        (title, prefix)
        for title, prefix in component_groups
        if any(name.startswith(prefix) for name in names)
    )
    if any(name.startswith("control_history.") or name.startswith("prev_") for name in names):
        groups = (*groups, ("Control History", "control_history."))
    legacy_names = tuple(
        name
        for name in names
        if name not in used_component_names
        and not name.startswith("control_history.")
        and not name.startswith("prev_")
    )
    if legacy_names:
        return (*groups, ("Legacy", None))
    return groups


def _state_vector_group_lines(
    *,
    names: tuple[str, ...],
    values: StateVector,
    group_prefix: str | None,
) -> list[PanelLine]:
    if group_prefix == "course_context.":
        return _course_context_state_lines(names=names, values=values)
    if group_prefix == "control_history.":
        return _control_history_state_lines(names=names, values=values)
    return [
        panel_line(
            _state_vector_label(name, group_prefix=group_prefix),
            f"{float(value):.3f}",
            PALETTE.text_primary,
        )
        for name, value in zip(names, values, strict=True)
        if _state_vector_name_matches_group(name, group_prefix)
    ]


def _control_history_state_lines(
    *,
    names: tuple[str, ...],
    values: StateVector,
) -> list[PanelLine]:
    return [
        panel_line(
            _state_vector_label(name, group_prefix="control_history."),
            f"{float(value):.3f}",
            PALETTE.text_primary,
        )
        for name, value in zip(names, values, strict=True)
        if name.startswith("control_history.") or name.startswith("prev_")
    ]


def _course_context_state_lines(
    *,
    names: tuple[str, ...],
    values: StateVector,
) -> list[PanelLine]:
    course_bits = [
        float(value)
        for name, value in zip(names, values, strict=True)
        if name.startswith("course_context.course_builtin_")
    ]
    if not course_bits:
        return []

    active_index = _one_hot_active_index(course_bits)
    return [
        panel_line(
            "categorical",
            "--" if active_index is None else str(active_index),
            PALETTE.text_primary if active_index is not None else PALETTE.text_muted,
        ),
        panel_line(
            "one hot",
            "".join("1" if value >= 0.5 else "0" for value in course_bits),
            PALETTE.text_primary,
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
