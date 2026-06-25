# src/rl_fzerox/ui/watch/view/panels/content/career.py
"""Career Mode side-panel sections for the Watch UI.

The Career tab summarizes controller progress, FSM observations, and active
policy-control metadata emitted by the Career Mode runtime.
"""

from __future__ import annotations

from rl_fzerox.ui.watch.view.panels.core.format import (
    _float_info,
    _format_mode_name,
    _int_info,
)
from rl_fzerox.ui.watch.view.panels.core.lines import panel_line as _panel_line
from rl_fzerox.ui.watch.view.screen.theme import PALETTE
from rl_fzerox.ui.watch.view.screen.types import PanelSection


def career_mode_sections(info: dict[str, object]) -> list[PanelSection]:
    target = _non_empty_text(info.get("career_mode_target_label"))
    phase = _non_empty_text(info.get("career_mode_phase"))
    save_attempt_id = _non_empty_text(info.get("career_mode_attempt_id"))
    completed_targets = _int_info(info, "career_mode_completed_targets")
    total_targets = _int_info(info, "career_mode_total_targets")
    inspection_status = _non_empty_text(info.get("career_mode_inspection_status"))
    policy_run_name = _non_empty_text(info.get("career_mode_policy_run_name"))
    policy_run_id = _non_empty_text(info.get("career_mode_policy_run_id"))
    policy_artifact = _non_empty_text(info.get("career_mode_policy_artifact"))
    policy_course_id = _non_empty_text(info.get("career_mode_policy_course_id"))
    policy_active = info.get("career_mode_policy_active") is True
    progress_lines = [
        _panel_line(
            "Progress",
            _format_career_progress(completed_targets, total_targets),
            PALETTE.text_primary if total_targets else PALETTE.text_muted,
        ),
        _panel_line(
            "Save state",
            _format_mode_name(inspection_status) if inspection_status else "-",
            PALETTE.text_primary if inspection_status else PALETTE.text_muted,
        ),
        _panel_line(
            "Current target",
            target or "-",
            PALETTE.text_primary if target else PALETTE.text_muted,
            wrap=True,
            min_value_lines=2,
        ),
    ]
    controller_lines = [
        _panel_line(
            "Phase",
            _format_mode_name(phase) if phase else "-",
            PALETTE.text_primary if phase else PALETTE.text_muted,
        ),
        _panel_line(
            "Last input",
            _career_last_input(info),
            PALETTE.text_primary,
            wrap=True,
            min_value_lines=2,
        ),
        _panel_line(
            "Game facts",
            _career_game_facts(info),
            PALETTE.text_primary,
            wrap=True,
            min_value_lines=2,
        ),
        _panel_line(
            "Race boundary",
            _career_boundary_facts(info),
            PALETTE.text_primary,
            wrap=True,
            min_value_lines=2,
        ),
        _panel_line(
            "Race progress",
            _career_race_progress_facts(info),
            PALETTE.text_primary,
            wrap=True,
            min_value_lines=2,
        ),
        _panel_line(
            "Camera",
            _career_camera_facts(info),
            PALETTE.text_primary,
            wrap=True,
            min_value_lines=2,
        ),
    ]
    policy_lines = [
        _panel_line(
            "Policy control",
            "active" if policy_active else "inactive",
            PALETTE.text_accent if policy_active else PALETTE.text_muted,
        ),
        _panel_line(
            "Policy",
            policy_run_name or "-",
            PALETTE.text_primary if policy_run_name else PALETTE.text_muted,
            wrap=True,
            min_value_lines=2,
        ),
        _panel_line(
            "Artifact",
            policy_artifact or "-",
            PALETTE.text_primary if policy_artifact else PALETTE.text_muted,
        ),
        _panel_line(
            "Course",
            policy_course_id or "-",
            PALETTE.text_primary if policy_course_id else PALETTE.text_muted,
        ),
        _panel_line(
            "Attempt",
            save_attempt_id or "-",
            PALETTE.text_primary if save_attempt_id else PALETTE.text_muted,
            wrap=True,
            min_value_lines=2,
        ),
        _panel_line(
            "Policy run id",
            policy_run_id or "-",
            PALETTE.text_primary if policy_run_id else PALETTE.text_muted,
            wrap=True,
            min_value_lines=2,
        ),
    ]
    return [
        PanelSection(
            title="Career Progress",
            lines=progress_lines,
        ),
        PanelSection(
            title="Career Controller",
            lines=controller_lines,
        ),
        PanelSection(
            title="Career Policy",
            lines=policy_lines,
        ),
    ]


def _format_career_progress(completed: int | None, total: int | None) -> str:
    if completed is None or total is None:
        return "-"
    return f"{completed} / {total} targets"


def _career_game_facts(info: dict[str, object]) -> str:
    fields = [
        ("screen", _non_empty_text(info.get("career_mode_fsm_observed_screen"))),
        ("mode", _non_empty_text(info.get("career_mode_fsm_game_mode"))),
        ("course", _format_optional_int(info, "career_mode_fsm_course_index")),
        ("selected", _format_optional_int(info, "career_mode_fsm_selected_mode_raw")),
        ("diff_state", _format_optional_int(info, "career_mode_fsm_difficulty_state_raw")),
        ("diff_cursor", _format_optional_int(info, "career_mode_fsm_difficulty_cursor_raw")),
        ("transition", _format_optional_int(info, "career_mode_fsm_transition_raw")),
        ("popup", _non_empty_text(info.get("career_mode_fsm_popup_state"))),
    ]
    return _join_named_values(fields)


def _career_last_input(info: dict[str, object]) -> str:
    fields = [
        ("input", _non_empty_text(info.get("career_mode_last_input"))),
        ("step", _non_empty_text(info.get("career_mode_last_step"))),
        ("frames", _format_optional_int(info, "career_mode_last_step_frames")),
    ]
    return _join_named_values(fields)


def _career_boundary_facts(info: dict[str, object]) -> str:
    fields = [
        ("terminal", _format_optional_bool(info, "career_mode_fsm_terminal_result")),
        ("result", _format_optional_bool(info, "career_mode_fsm_completed_result_screen")),
        ("fresh", _format_optional_bool(info, "career_mode_fsm_fresh_race_ready")),
        ("reason", _non_empty_text(info.get("career_mode_fsm_terminal_reason"))),
    ]
    return _join_named_values(fields)


def _career_race_progress_facts(info: dict[str, object]) -> str:
    completed_laps = _format_optional_int(info, "career_mode_fsm_completed_laps")
    total_laps = _format_optional_int(info, "career_mode_fsm_total_laps")
    lap_text = None
    if completed_laps is not None or total_laps is not None:
        lap_text = f"{completed_laps or '-'} / {total_laps or '-'}"
    fields = [
        ("laps", lap_text),
        ("intro", _format_optional_int(info, "career_mode_fsm_intro_timer")),
        ("time", _format_optional_float(info, "career_mode_fsm_race_time_ms", suffix="ms")),
        ("comp", _format_optional_percent(info, "career_mode_fsm_completion_fraction")),
    ]
    return _join_named_values(fields)


def _career_camera_facts(info: dict[str, object]) -> str:
    fields = [
        ("target", _non_empty_text(info.get("career_mode_fsm_camera_target"))),
        ("synced", _format_optional_bool(info, "career_mode_fsm_camera_synced")),
    ]
    return _join_named_values(fields)


def _join_named_values(fields: list[tuple[str, str | None]]) -> str:
    parts = [f"{name}={value}" for name, value in fields if value is not None]
    return " · ".join(parts) if parts else "-"


def _format_optional_bool(info: dict[str, object], key: str) -> str | None:
    value = info.get(key)
    if isinstance(value, bool):
        return "yes" if value else "no"
    return None


def _format_optional_int(info: dict[str, object], key: str) -> str | None:
    if key not in info:
        return None
    value = _int_info(info, key)
    return str(value) if value is not None else None


def _format_optional_float(
    info: dict[str, object],
    key: str,
    *,
    suffix: str = "",
) -> str | None:
    if key not in info:
        return None
    value = _float_info(info, key)
    return f"{value:.0f}{suffix}"


def _format_optional_percent(info: dict[str, object], key: str) -> str | None:
    if key not in info:
        return None
    value = _float_info(info, key)
    return f"{value * 100.0:.1f}%"


def _non_empty_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None
