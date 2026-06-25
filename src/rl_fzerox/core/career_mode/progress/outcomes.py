# src/rl_fzerox/core/career_mode/progress/outcomes.py
"""Pure Career Mode outcome helpers for GP-cup save progress.

These functions classify runtime info and unlock targets without touching the
manager store, save RAM, or controller state. `attempt.py` owns persistence and
uses this module for repeatable race/cup outcome decisions.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from rl_fzerox.core.career_mode.navigation import POST_GP_RECORDING_END_MODES
from rl_fzerox.core.domain.courses import BUILT_IN_COURSES, CourseInfo
from rl_fzerox.core.runtime_info import optional_int_info, optional_str_info
from rl_fzerox.core.runtime_spec.schema import CareerModeRaceSetupConfig


def target_succeeded(
    targets: Iterable[object],
    setup: CareerModeRaceSetupConfig,
) -> bool:
    return any(
        getattr(target, "status", None) == "succeeded"
        and getattr(target, "kind", None) == "clear_gp_cup"
        and getattr(target, "difficulty", None) == setup.difficulty
        and getattr(target, "cup_id", None) == setup.cup_id
        for target in targets
    )


@dataclass(frozen=True, slots=True)
class CareerOutcomeInfo:
    """Typed view of the runtime fields used for GP-cup progress decisions."""

    termination_reason: str | None
    game_mode: str | None
    track_course_index: int | None
    course_index: int | None
    track_id: str | None
    course_id: str | None
    career_mode_cup_course_index: int | None
    cup_course_index: int | None
    post_gp_cutscene_complete: bool
    career_gp_final_rank: int | None
    gp_final_rank: int | None
    career_gp_points: int | None
    gp_points: int | None

    @classmethod
    def from_info(cls, info: dict[str, object]) -> CareerOutcomeInfo:
        return cls(
            termination_reason=_str_info(info, "termination_reason"),
            game_mode=_game_mode(info),
            track_course_index=_int_info(info, "track_course_index"),
            course_index=_int_info(info, "course_index"),
            track_id=_str_info(info, "track_id"),
            course_id=_str_info(info, "course_id"),
            career_mode_cup_course_index=_int_info(
                info,
                "career_mode_cup_course_index",
            ),
            cup_course_index=_int_info(info, "cup_course_index"),
            post_gp_cutscene_complete=(info.get("career_mode_post_gp_cutscene_complete") is True),
            career_gp_final_rank=positive_int_info(
                info,
                "career_mode_gp_final_rank",
            ),
            gp_final_rank=positive_int_info(info, "gp_final_rank"),
            career_gp_points=_gp_points_value(info.get("career_mode_gp_points")),
            gp_points=_gp_points_value(info.get("gp_points")),
        )


def target_terminal_succeeded(
    info: dict[str, object],
    setup: CareerModeRaceSetupConfig,
) -> bool:
    outcome = CareerOutcomeInfo.from_info(info)
    if outcome.termination_reason != "finished":
        return False

    final_course = _target_final_course(setup)
    if final_course is None:
        return False
    return _info_matches_course(outcome, final_course) or _info_matches_final_cup_slot(
        outcome,
        setup,
    )


def failure_reason(info: dict[str, object]) -> str:
    reason = CareerOutcomeInfo.from_info(info).termination_reason
    return reason if reason is not None else "race ended before cup clear"


def is_failed_terminal_result(info: dict[str, object]) -> bool:
    return CareerOutcomeInfo.from_info(info).termination_reason in {"crashed", "retired"}


def keeps_current_gp_attempt(info: dict[str, object]) -> bool:
    reason = CareerOutcomeInfo.from_info(info).termination_reason
    return reason in {"finished", "crashed", "retired"}


def is_post_gp_completion(info: dict[str, object]) -> bool:
    outcome = CareerOutcomeInfo.from_info(info)
    mode = outcome.game_mode
    if mode == "gp_end_cutscene":
        return outcome.post_gp_cutscene_complete
    return mode in POST_GP_RECORDING_END_MODES or mode in {
        "title",
        "main_menu",
        "course_select",
    }


def is_failed_gp_exit(info: dict[str, object]) -> bool:
    mode = CareerOutcomeInfo.from_info(info).game_mode
    return mode in {
        "title",
        "main_menu",
        "course_select",
    }


def positive_int_info(info: dict[str, object], key: str) -> int | None:
    return optional_int_info(info, key, float_values=False, minimum=1)


_GP_RESULT_MODES = {
    "gp_race",
    "results",
    "gp_end_cutscene",
}


def gp_final_rank(info: dict[str, object]) -> int | None:
    outcome = CareerOutcomeInfo.from_info(info)
    if outcome.career_gp_final_rank is not None:
        return outcome.career_gp_final_rank
    # The native rank is derived from the live GP points table. Career progress
    # only asks for it while handling post-terminal GP flow, so these modes are
    # the windows where the table still belongs to the current cup attempt.
    if outcome.game_mode not in _GP_RESULT_MODES:
        return None
    return outcome.gp_final_rank


def gp_points(info: dict[str, object]) -> int | None:
    outcome = CareerOutcomeInfo.from_info(info)
    if outcome.career_gp_points is not None:
        return outcome.career_gp_points
    # `Racer.points` remains authoritative from the race result through the GP
    # end cutscene; after menu/reset transitions it can be stale or reset.
    if outcome.game_mode not in _GP_RESULT_MODES:
        return None
    return outcome.gp_points


def _target_final_course(setup: CareerModeRaceSetupConfig) -> CourseInfo | None:
    courses = _target_cup_courses(setup)
    if not courses:
        return None
    return max(courses, key=lambda course: course.course_index)


def _target_cup_courses(setup: CareerModeRaceSetupConfig) -> tuple[CourseInfo, ...]:
    return tuple(
        sorted(
            (course for course in BUILT_IN_COURSES if course.cup == setup.cup_id),
            key=lambda course: course.course_index,
        )
    )


def _info_matches_course(outcome: CareerOutcomeInfo, course: CourseInfo) -> bool:
    course_index = outcome.track_course_index
    if course_index is None:
        course_index = outcome.course_index
    if course_index == course.course_index:
        return True

    course_id = outcome.track_id
    if course_id is None:
        course_id = outcome.course_id
    return course_id in {course.id, course.ref}


def _info_matches_final_cup_slot(
    outcome: CareerOutcomeInfo,
    setup: CareerModeRaceSetupConfig,
) -> bool:
    courses = _target_cup_courses(setup)
    if not courses:
        return False
    final_slot = len(courses) - 1

    # On live GP result handoff some native fields expose the cup-local slot
    # instead of the global built-in course index. Accept that only as terminal
    # final-course evidence; post-GP success/failure still owns the cup result.
    return final_slot in {
        outcome.career_mode_cup_course_index,
        outcome.cup_course_index,
        outcome.course_index,
    }


def _gp_points_value(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if 0 <= value <= 999 else None


def _int_info(info: dict[str, object], key: str) -> int | None:
    return optional_int_info(info, key, float_values=False)


def _game_mode(info: dict[str, object]) -> str | None:
    return _str_info(info, "game_mode") or _str_info(info, "game_mode_name")


def _str_info(info: dict[str, object], key: str) -> str | None:
    return optional_str_info(info, key, non_empty=True)
