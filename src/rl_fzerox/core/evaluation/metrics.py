# src/rl_fzerox/core/evaluation/metrics.py
"""Aggregate evaluation results into primary and diagnostic metrics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, TypeAlias

from rl_fzerox.core.evaluation.models import (
    AttemptStatus,
    CourseResultStatus,
    EvaluationAttemptResult,
    EvaluationCourseResult,
    EvaluationRunResult,
)

EvaluationMetricGroupKind: TypeAlias = Literal["overall", "cup", "course"]


@dataclass(frozen=True, slots=True)
class EvaluationPrimaryMetrics:
    """Comparable policy-quality metrics shown first in reports."""

    attempt_count: int
    success_count: int
    success_rate: float | None
    finish_count: int
    finish_rate: float | None
    completion_rate: float | None
    mean_finish_time_ms: float | None
    best_finish_time_ms: int | None
    mean_total_race_time_ms: float | None
    best_total_race_time_ms: int | None
    mean_position: float | None
    best_position: int | None
    worst_position: int | None
    total_env_steps: int
    mean_episode_length_steps: float | None


@dataclass(frozen=True, slots=True)
class EvaluationDetailMetrics:
    """Diagnostics that are useful but less stable across reward/config changes."""

    mean_episode_return: float | None
    best_episode_return: float | None
    total_episode_return: float | None
    boost_active_count: int
    boost_active_frames: int
    boost_pad_entries: int
    damage_event_count: int
    minimum_height: float | None
    average_speed: float | None


@dataclass(frozen=True, slots=True)
class EvaluationMetricGroup:
    """Metrics for one report group such as overall, one cup, or one course."""

    kind: EvaluationMetricGroupKind
    key: str
    label: str
    primary: EvaluationPrimaryMetrics
    detail: EvaluationDetailMetrics


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """Aggregates over all attempts plus per-cup and per-course breakdowns."""

    overall: EvaluationMetricGroup
    cups: tuple[EvaluationMetricGroup, ...]
    courses: tuple[EvaluationMetricGroup, ...]


@dataclass(frozen=True, slots=True)
class _CourseRecord:
    attempt: EvaluationAttemptResult
    course: EvaluationCourseResult


def aggregate_evaluation_metrics(result: EvaluationRunResult) -> EvaluationMetrics:
    """Return overall, cup, and course aggregates for an evaluation result."""

    course_records = tuple(
        _CourseRecord(attempt=attempt, course=course)
        for attempt in result.attempts
        for course in attempt.course_results
    )
    return EvaluationMetrics(
        overall=_aggregate_group(
            kind="overall",
            key="overall",
            label="Overall",
            attempts=result.attempts,
            course_records=course_records,
        ),
        cups=_aggregate_cups(result.attempts, course_records),
        courses=_aggregate_courses(course_records),
    )


def _aggregate_cups(
    attempts: tuple[EvaluationAttemptResult, ...],
    course_records: tuple[_CourseRecord, ...],
) -> tuple[EvaluationMetricGroup, ...]:
    attempt_groups: dict[str, list[EvaluationAttemptResult]] = defaultdict(list)
    record_groups: dict[str, list[_CourseRecord]] = defaultdict(list)
    labels: dict[str, str] = {}
    for attempt in attempts:
        if attempt.cup_id is None:
            continue
        attempt_groups[attempt.cup_id].append(attempt)
        labels.setdefault(attempt.cup_id, attempt.cup_id)
    for record in course_records:
        cup_id = record.course.cup_id or record.attempt.cup_id
        if cup_id is None:
            continue
        record_groups[cup_id].append(record)
        labels.setdefault(cup_id, cup_id)
    return tuple(
        _aggregate_group(
            kind="cup",
            key=key,
            label=labels[key],
            attempts=tuple(attempt_groups.get(key, ())),
            course_records=tuple(record_groups.get(key, ())),
        )
        for key in sorted(labels)
    )


def _aggregate_courses(
    course_records: tuple[_CourseRecord, ...],
) -> tuple[EvaluationMetricGroup, ...]:
    record_groups: dict[str, list[_CourseRecord]] = defaultdict(list)
    labels: dict[str, str] = {}
    for record in course_records:
        key = record.course.course_id
        record_groups[key].append(record)
        labels.setdefault(key, record.course.course_name or key)
    return tuple(
        _aggregate_group(
            kind="course",
            key=key,
            label=labels[key],
            attempts=(),
            course_records=tuple(record_groups[key]),
        )
        for key in sorted(record_groups, key=lambda group_key: labels[group_key])
    )


def _aggregate_group(
    *,
    kind: EvaluationMetricGroupKind,
    key: str,
    label: str,
    attempts: tuple[EvaluationAttemptResult, ...],
    course_records: tuple[_CourseRecord, ...],
) -> EvaluationMetricGroup:
    course_results = tuple(record.course for record in course_records)
    attempt_count = len(attempts) if attempts else len(course_results)
    success_count = _success_count(attempts, course_results)
    finish_count = sum(1 for course in course_results if _course_finished(course.status))
    finished_courses = tuple(course for course in course_results if _course_finished(course.status))
    completion_values = tuple(
        value for course in course_results if (value := _completion_value(course)) is not None
    )
    course_times = tuple(
        int(course.race_time_ms)
        for course in finished_courses
        if course.race_time_ms is not None and course.race_time_ms > 0
    )
    course_positions = tuple(
        int(course.position)
        for course in course_results
        if course.position is not None and course.position > 0
    )
    total_race_times = tuple(
        int(value)
        for attempt in attempts
        if (value := _attempt_total_race_time_ms(attempt)) is not None
    )
    episode_lengths = _episode_lengths(attempts, course_results)
    returns = _episode_returns(attempts, course_results)
    group = EvaluationMetricGroup(
        kind=kind,
        key=key,
        label=label,
        primary=EvaluationPrimaryMetrics(
            attempt_count=attempt_count,
            success_count=success_count,
            success_rate=_ratio(success_count, attempt_count),
            finish_count=finish_count,
            finish_rate=_ratio(finish_count, len(course_results)),
            completion_rate=_mean(completion_values),
            mean_finish_time_ms=_mean(course_times),
            best_finish_time_ms=None if not course_times else min(course_times),
            mean_total_race_time_ms=_mean(total_race_times),
            best_total_race_time_ms=None if not total_race_times else min(total_race_times),
            mean_position=_mean(course_positions),
            best_position=None if not course_positions else min(course_positions),
            worst_position=None if not course_positions else max(course_positions),
            total_env_steps=_total_env_steps(attempts, course_results),
            mean_episode_length_steps=_mean(episode_lengths),
        ),
        detail=EvaluationDetailMetrics(
            mean_episode_return=_mean(returns),
            best_episode_return=None if not returns else max(returns),
            total_episode_return=None if not returns else sum(returns),
            boost_active_count=_sum_optional(
                course.boost_active_count for course in course_results
            ),
            boost_active_frames=_sum_optional(
                course.boost_active_frames for course in course_results
            ),
            boost_pad_entries=_sum_optional(course.boost_pad_entries for course in course_results),
            damage_event_count=_sum_optional(
                course.damage_event_count for course in course_results
            ),
            minimum_height=_min_optional(course.minimum_height for course in course_results),
            average_speed=_mean(
                tuple(
                    float(course.average_speed)
                    for course in course_results
                    if course.average_speed is not None
                )
            ),
        ),
    )
    return group


def _success_count(
    attempts: tuple[EvaluationAttemptResult, ...],
    course_results: tuple[EvaluationCourseResult, ...],
) -> int:
    if attempts:
        return sum(1 for attempt in attempts if _attempt_succeeded(attempt.status))
    return sum(1 for course in course_results if _course_finished(course.status))


def _attempt_succeeded(status: AttemptStatus) -> bool:
    return status == "succeeded"


def _course_finished(status: CourseResultStatus) -> bool:
    return status == "finished"


def _completion_value(course: EvaluationCourseResult) -> float | None:
    if course.completion_ratio is not None:
        return _clamp_unit(float(course.completion_ratio))
    if _course_finished(course.status):
        return 1.0
    if (
        course.laps_completed is not None
        and course.total_laps is not None
        and course.total_laps > 0
    ):
        return _clamp_unit(float(course.laps_completed) / float(course.total_laps))
    return None


def _attempt_total_race_time_ms(attempt: EvaluationAttemptResult) -> int | None:
    if attempt.total_race_time_ms is not None and attempt.total_race_time_ms > 0:
        return attempt.total_race_time_ms
    course_times = tuple(
        course.race_time_ms
        for course in attempt.course_results
        if _course_finished(course.status)
        and course.race_time_ms is not None
        and course.race_time_ms > 0
    )
    if not course_times:
        return None
    return sum(course_times)


def _episode_lengths(
    attempts: tuple[EvaluationAttemptResult, ...],
    course_results: tuple[EvaluationCourseResult, ...],
) -> tuple[int, ...]:
    values = tuple(
        int(attempt.episode_length_steps)
        for attempt in attempts
        if attempt.episode_length_steps is not None and attempt.episode_length_steps >= 0
    )
    if values:
        return values
    return tuple(
        int(course.episode_length_steps)
        for course in course_results
        if course.episode_length_steps is not None and course.episode_length_steps >= 0
    )


def _episode_returns(
    attempts: tuple[EvaluationAttemptResult, ...],
    course_results: tuple[EvaluationCourseResult, ...],
) -> tuple[float, ...]:
    values = tuple(
        float(attempt.episode_return) for attempt in attempts if attempt.episode_return is not None
    )
    if values:
        return values
    return tuple(
        float(course.episode_return)
        for course in course_results
        if course.episode_return is not None
    )


def _total_env_steps(
    attempts: tuple[EvaluationAttemptResult, ...],
    course_results: tuple[EvaluationCourseResult, ...],
) -> int:
    attempt_steps = _sum_optional(attempt.env_steps for attempt in attempts)
    if attempt_steps:
        return attempt_steps
    return _sum_optional(
        course.env_steps if course.env_steps is not None else course.episode_length_steps
        for course in course_results
    )


def _sum_optional(values: Iterable[int | float | None]) -> int:
    total = 0
    for value in values:
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, int | float):
            total += int(value)
    return total


def _min_optional(values: Iterable[int | float | None]) -> float | None:
    valid_values = tuple(
        float(value)
        for value in values
        if isinstance(value, int | float) and not isinstance(value, bool)
    )
    if not valid_values:
        return None
    return min(valid_values)


def _mean(values: tuple[int | float, ...]) -> float | None:
    if not values:
        return None
    return float(sum(values)) / float(len(values))


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))
