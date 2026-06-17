# src/rl_fzerox/core/career_mode/controller/terminal.py
from __future__ import annotations

from typing import Protocol

from fzerox_emulator import FZeroXTelemetry
from rl_fzerox.core.career_mode.navigation import (
    BUILT_IN_COURSES_BY_INDEX,
    MenuFacts,
    course_id_from_info,
)


class TerminalTelemetryEmulator(Protocol):
    def try_read_telemetry(self) -> FZeroXTelemetry | None: ...


class TerminalRaceSession(Protocol):
    @property
    def emulator(self) -> TerminalTelemetryEmulator: ...


def post_terminal_progress_screen(facts: MenuFacts) -> bool:
    """Return true for visible screens where a whole GP attempt can end.

    Result and next-course screens are still inside the current GP attempt.
    Stale `gp_race` terminal frames are also intentionally excluded; the
    controller handles those once through `observe_step()` and then advances
    the game until a post-GP, menu, title, course-select, or game-over state is
    actually visible.
    """

    return (
        facts.is_post_gp_screen
        or facts.is_title
        or facts.is_mode_select
        or facts.is_course_select
        or facts.game_mode == "game_over"
    )


def race_terminal_reason(
    *,
    session: TerminalRaceSession,
    info: dict[str, object],
) -> str | None:
    info_reason = info_terminal_reason(info=info)
    if info_reason is not None:
        return info_reason

    telemetry = session.emulator.try_read_telemetry()
    if telemetry is None:
        return None
    return telemetry_terminal_reason(telemetry)


def info_terminal_reason(
    *,
    info: dict[str, object],
) -> str | None:
    facts = MenuFacts.from_info(info)
    if facts.is_post_gp_screen:
        return "finished"
    reason = game_terminal_reason(_non_empty_str(info.get("termination_reason")))
    if reason is not None:
        return reason
    for flag_key, flag_reason in (
        ("entered_finished", "finished"),
        ("entered_retired", "retired"),
        ("entered_crashed", "crashed"),
    ):
        if info.get(flag_key) is True:
            return flag_reason
    return None


def terminal_info(
    *,
    session: TerminalRaceSession,
    info: dict[str, object],
    terminal_reason: str,
) -> dict[str, object]:
    resolved_info = dict(info)
    resolved_info["termination_reason"] = terminal_reason
    resolved_info["career_mode_race_terminal"] = True

    telemetry = session.emulator.try_read_telemetry()
    if telemetry is None:
        return resolved_info
    course_index = getattr(telemetry, "course_index", None)
    if isinstance(course_index, int) and not isinstance(course_index, bool):
        resolved_info.setdefault("course_index", course_index)
    resolved_info.setdefault("race_time_ms", telemetry.player.race_time_ms)
    resolved_info.setdefault("position", telemetry.player.position)
    resolved_info.setdefault("ko_star_count", telemetry.player.ko_star_count)
    resolved_info.setdefault("track_id", course_id_from_info(resolved_info))
    _add_course_metadata(resolved_info)
    return resolved_info


def telemetry_terminal_reason(telemetry: FZeroXTelemetry) -> str | None:
    return game_terminal_reason(telemetry.player.terminal_reason)


def game_terminal_reason(reason: str | None) -> str | None:
    if reason in {"finished", "retired", "crashed"}:
        return reason
    return None


def _add_course_metadata(info: dict[str, object]) -> None:
    course_index = info.get("course_index")
    if isinstance(course_index, bool) or not isinstance(course_index, int):
        return
    course = BUILT_IN_COURSES_BY_INDEX.get(course_index)
    if course is None:
        return
    info.setdefault("track_id", course.id)
    info.setdefault("track_course_id", course.id)
    info.setdefault("track_course_name", course.display_name)
    info.setdefault("track_course_index", course.course_index)


def _non_empty_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
