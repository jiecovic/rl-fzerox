# src/rl_fzerox/core/career_mode/course_setup.py
"""Course setup resolution for save-game unlock attempts."""

from __future__ import annotations

from dataclasses import dataclass

from rl_fzerox.core.domain.courses import BUILT_IN_COURSES
from rl_fzerox.core.manager.models import ManagedSaveCourseSetup, ManagedSaveCupSetup


@dataclass(frozen=True, slots=True)
class CourseSetupTarget:
    """Difficulty/cup/course identity for one planned unlock race."""

    difficulty: str | None = None
    cup_id: str | None = None
    course_id: str | None = None


def resolve_course_setup(
    setups: tuple[ManagedSaveCourseSetup, ...],
    target: CourseSetupTarget,
) -> ManagedSaveCourseSetup | None:
    """Return the concrete course setup that matches a race target."""

    if target.course_id is None:
        return None
    matching = tuple(setup for setup in setups if _setup_matches(setup, target))
    if not matching:
        return None
    return max(matching, key=lambda setup: _setup_preference_key(setup, target))


def resolve_cup_setup(
    setups: tuple[ManagedSaveCupSetup, ...],
    target: CourseSetupTarget,
) -> ManagedSaveCupSetup | None:
    """Return the cup-level vehicle setup for a race target."""

    if target.cup_id is None:
        return None
    matching = tuple(
        setup
        for setup in setups
        if setup.cup_id == target.cup_id and _optional_match(setup.difficulty, target.difficulty)
    )
    if not matching:
        return None
    return max(matching, key=lambda setup: _cup_setup_preference_key(setup, target))


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


def has_cup_setup(
    setups: tuple[ManagedSaveCupSetup, ...],
    target: CourseSetupTarget,
) -> bool:
    """Return whether a target can resolve its GP cup vehicle setup."""

    return resolve_cup_setup(setups, target) is not None


def _setup_matches(
    setup: ManagedSaveCourseSetup,
    target: CourseSetupTarget,
) -> bool:
    return (
        setup.course_id == target.course_id
        and _optional_match(setup.cup_id, target.cup_id)
        and _optional_match(setup.difficulty, target.difficulty)
    )


def _setup_preference_key(
    setup: ManagedSaveCourseSetup,
    target: CourseSetupTarget,
) -> tuple[int, int, str, str]:
    return (
        _specificity(setup.difficulty, target.difficulty),
        _specificity(setup.cup_id, target.cup_id),
        setup.updated_at,
        setup.id,
    )


def _cup_setup_preference_key(
    setup: ManagedSaveCupSetup,
    target: CourseSetupTarget,
) -> tuple[int, str, str]:
    return (
        _specificity(setup.difficulty, target.difficulty),
        setup.updated_at,
        setup.id,
    )


def _optional_match(expected: str | None, actual: str | None) -> bool:
    return expected is None or expected == actual


def _specificity(expected: str | None, actual: str | None) -> int:
    return 1 if expected is not None and expected == actual else 0
