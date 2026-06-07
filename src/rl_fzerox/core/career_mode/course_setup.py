# src/rl_fzerox/core/career_mode/course_setup.py
"""Course setup resolution for save-game unlock attempts."""

from __future__ import annotations

from dataclasses import dataclass

from rl_fzerox.core.domain.courses import BUILT_IN_COURSES
from rl_fzerox.core.manager.models import CourseSetupScope, ManagedSaveCourseSetup


@dataclass(frozen=True, slots=True)
class CourseSetupTarget:
    """Difficulty/cup/course identity for one planned unlock race."""

    difficulty: str | None = None
    cup_id: str | None = None
    course_id: str | None = None


COURSE_SETUP_RESOLUTION_ORDER: tuple[CourseSetupScope, ...] = (
    "course",
    "cup",
    "difficulty",
    "global",
)


def resolve_course_setup(
    setups: tuple[ManagedSaveCourseSetup, ...],
    target: CourseSetupTarget,
) -> ManagedSaveCourseSetup | None:
    """Return the most specific course setup that matches a race target."""

    for scope in COURSE_SETUP_RESOLUTION_ORDER:
        matching = [
            setup for setup in setups if setup.scope == scope and _setup_matches(setup, target)
        ]
        if matching:
            return max(matching, key=_setup_preference_key)
    return None


def required_course_setup_targets(
    target: CourseSetupTarget,
) -> tuple[CourseSetupTarget, ...]:
    """Return concrete course targets required to run one unlock target."""

    if target.course_id is not None or target.cup_id is None:
        return (target,)
    course_targets = tuple(
        CourseSetupTarget(
            difficulty=target.difficulty,
            cup_id=target.cup_id,
            course_id=course.id,
        )
        for course in sorted(BUILT_IN_COURSES, key=lambda course: course.course_index)
        if course.cup == target.cup_id
    )
    return course_targets or (target,)


def missing_course_setup_targets(
    setups: tuple[ManagedSaveCourseSetup, ...],
    target: CourseSetupTarget,
) -> tuple[CourseSetupTarget, ...]:
    """Return concrete course targets that cannot resolve a setup."""

    return tuple(
        course_target
        for course_target in required_course_setup_targets(target)
        if resolve_course_setup(setups, course_target) is None
    )


def _setup_matches(
    setup: ManagedSaveCourseSetup,
    target: CourseSetupTarget,
) -> bool:
    match setup.scope:
        case "global":
            return True
        case "difficulty":
            return setup.difficulty == target.difficulty
        case "cup":
            return setup.cup_id == target.cup_id and _optional_match(
                setup.difficulty, target.difficulty
            )
        case "course":
            return (
                setup.course_id == target.course_id
                and _optional_match(setup.cup_id, target.cup_id)
                and _optional_match(setup.difficulty, target.difficulty)
            )


def _setup_preference_key(setup: ManagedSaveCourseSetup) -> tuple[str, str]:
    return setup.updated_at, setup.id


def _optional_match(expected: str | None, actual: str | None) -> bool:
    return expected is None or expected == actual
