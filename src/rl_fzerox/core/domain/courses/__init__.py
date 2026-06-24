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
from .sampling import (
    DeficitBudgetDifficultyMetric,
    ManagedTrackSamplingMode,
    RuntimeTrackSamplingMode,
    TrackSamplingMode,
)
from .x_cup import (
    X_CUP_COURSE,
    GeneratedXCupCourseIdentity,
    XCupCourseSpec,
    XCupGeneratedCourseKind,
    XCupRaceMode,
    XCupRetentionPolicy,
    XCupRotationDefaults,
    generated_x_cup_course_identity,
    generated_x_cup_slot_key,
)

__all__ = [
    "BUILT_IN_COURSES",
    "CourseInfo",
    "CourseRecord",
    "CourseRecords",
    "DeficitBudgetDifficultyMetric",
    "GeneratedXCupCourseIdentity",
    "ManagedTrackSamplingMode",
    "RuntimeTrackSamplingMode",
    "TrackSamplingMode",
    "X_CUP_COURSE",
    "XCupCourseSpec",
    "XCupGeneratedCourseKind",
    "XCupRaceMode",
    "XCupRetentionPolicy",
    "XCupRotationDefaults",
    "built_in_course_by_ref",
    "built_in_course_configs",
    "built_in_course_ref_by_id",
    "built_in_course_refs_by_cup",
    "generated_x_cup_course_identity",
    "generated_x_cup_slot_key",
]
