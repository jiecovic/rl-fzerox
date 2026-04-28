# src/rl_fzerox/core/domain/courses/__init__.py
from __future__ import annotations

from .catalog import BUILT_IN_COURSES
from .lookups import (
    built_in_course_by_ref,
    built_in_course_configs,
    built_in_course_ref_by_id,
    built_in_course_refs_by_cup,
)
from .model import CourseInfo, CourseRecord, CourseRecords

__all__ = [
    "BUILT_IN_COURSES",
    "CourseInfo",
    "CourseRecord",
    "CourseRecords",
    "built_in_course_by_ref",
    "built_in_course_configs",
    "built_in_course_ref_by_id",
    "built_in_course_refs_by_cup",
]
