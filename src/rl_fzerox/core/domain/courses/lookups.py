# src/rl_fzerox/core/domain/courses/lookups.py
from __future__ import annotations

from .catalog import BUILT_IN_COURSES

_COURSES_BY_REF = {course.ref: course for course in BUILT_IN_COURSES}


def built_in_course_configs() -> tuple[tuple[str, dict[str, object]], ...]:
    """Return source-owned built-in course metadata in registry format."""

    return tuple((course.ref, course.as_config()) for course in BUILT_IN_COURSES)


def built_in_course_by_ref(ref: str) -> dict[str, object] | None:
    """Return a built-in course by registry ref, if it exists."""

    course = _COURSES_BY_REF.get(ref)
    return None if course is None else course.as_config()


def built_in_course_ref_by_id(course_id: str, *, cup: str | None = None) -> tuple[str, ...]:
    """Return refs for built-in courses matching id and optional cup."""

    return tuple(
        course.ref
        for course in BUILT_IN_COURSES
        if course.id == course_id and (cup is None or course.cup == cup)
    )


def built_in_course_refs_by_cup(cup: str) -> tuple[str, ...]:
    """Return refs for all built-in courses in one cup, preserving game order."""

    return tuple(course.ref for course in BUILT_IN_COURSES if course.cup == cup)
