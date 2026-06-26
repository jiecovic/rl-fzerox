# src/rl_fzerox/core/evaluation/targets.py
"""Expand frozen evaluation target specs into concrete track entries.

Evaluation presets describe filters such as mode, course, cup, difficulty, and
vehicle. This module resolves those filters against the materialized
track-sampling config and returns only normal baseline entries.
"""

from __future__ import annotations

from rl_fzerox.core.domain.courses import BUILT_IN_COURSES
from rl_fzerox.core.evaluation.models import (
    EvaluationCourseTarget,
    EvaluationMode,
    EvaluationTargetSpec,
)
from rl_fzerox.core.runtime_spec.schema import TrainAppConfig
from rl_fzerox.core.runtime_spec.schema.tracks import TrackSamplingEntryConfig

_BUILT_IN_CUP_BY_COURSE_ID = {course.id: course.cup for course in BUILT_IN_COURSES}

_SINGLE_COURSE_ENTRY_MODE: dict[EvaluationMode, str] = {
    "time_attack_course": "time_attack",
    "gp_course": "gp_race",
}


def single_course_targets_from_config(
    config: TrainAppConfig,
    target: EvaluationTargetSpec,
) -> tuple[EvaluationCourseTarget, ...]:
    """Return default-start track entries addressed by a single-course spec."""

    entry_mode = _SINGLE_COURSE_ENTRY_MODE.get(target.mode)
    if entry_mode is None:
        raise ValueError(f"single-course target expansion does not support mode={target.mode!r}")
    course_ids = set(target.course_ids)
    cup_ids = set(target.cup_ids)
    difficulties = set(target.difficulties)
    vehicle_ids = set(target.vehicle_ids)
    return tuple(
        _course_target(entry)
        for entry in config.env.track_sampling.entries
        if _entry_matches(
            entry,
            entry_mode=entry_mode,
            course_ids=course_ids,
            cup_ids=cup_ids,
            difficulties=difficulties,
            vehicle_ids=vehicle_ids,
        )
    )


def _entry_matches(
    entry: TrackSamplingEntryConfig,
    *,
    entry_mode: str,
    course_ids: set[str],
    cup_ids: set[str],
    difficulties: set[str],
    vehicle_ids: set[str],
) -> bool:
    if entry.mode != entry_mode:
        return False
    if entry.alt_baseline_id is not None:
        return False
    if course_ids and entry.course_id not in course_ids and entry.id not in course_ids:
        return False
    course_id = entry.course_id or entry.id
    if cup_ids and _entry_cup_id(entry, course_id=course_id) not in cup_ids:
        return False
    if difficulties and entry.gp_difficulty not in difficulties:
        return False
    if vehicle_ids and entry.vehicle not in vehicle_ids:
        return False
    return True


def _course_target(entry: TrackSamplingEntryConfig) -> EvaluationCourseTarget:
    course_id = entry.course_id or entry.id
    return EvaluationCourseTarget(
        target_id=entry.id,
        course_id=course_id,
        course_name=entry.course_name or entry.display_name,
        cup_id=_entry_cup_id(entry, course_id=course_id),
        difficulty=entry.gp_difficulty,
        vehicle_id=entry.vehicle,
        baseline_state_path=None
        if entry.baseline_state_path is None
        else str(entry.baseline_state_path),
        baseline_group_id=entry.baseline_group_id,
        baseline_variant_index=entry.baseline_variant_index,
        baseline_variant_count=entry.baseline_variant_count,
        baseline_variant_seed=entry.baseline_variant_seed,
        engine_setting_raw_value=entry.engine_setting_raw_value,
    )


def _entry_cup_id(entry: TrackSamplingEntryConfig, *, course_id: str) -> str | None:
    if entry.course_ref is not None:
        cup_id, has_separator, _course_part = entry.course_ref.partition("/")
        if has_separator:
            return cup_id
    return _BUILT_IN_CUP_BY_COURSE_ID.get(course_id)
