# tests/core/training/track_sampling_support.py
from __future__ import annotations

from collections.abc import Mapping

from rl_fzerox.core.training.session.callbacks.track_sampling import (
    GeneratedCourseMetadata,
    TrackSamplingCourseEntry,
    resolve_track_sampling_courses_from_entries,
)
from rl_fzerox.core.training.session.callbacks.track_sampling.courses import (
    ResolvedTrackSamplingCourses,
)


def resolved_track_sampling_courses(
    base_weights: Mapping[str, float],
    *,
    course_keys: Mapping[str, str] | None = None,
    log_keys: Mapping[str, str] | None = None,
    labels: Mapping[str, str] | None = None,
    log_enabled: Mapping[str, bool] | None = None,
    generated: Mapping[str, GeneratedCourseMetadata] | None = None,
) -> ResolvedTrackSamplingCourses:
    return resolve_track_sampling_courses_from_entries(
        tuple(
            _resolved_entry(
                track_id=track_id,
                base_weight=base_weight,
                course_keys=course_keys,
                log_keys=log_keys,
                labels=labels,
                log_enabled=log_enabled,
                generated=generated,
            )
            for track_id, base_weight in base_weights.items()
        )
    )


def _resolved_entry(
    *,
    track_id: str,
    base_weight: float,
    course_keys: Mapping[str, str] | None,
    log_keys: Mapping[str, str] | None,
    labels: Mapping[str, str] | None,
    log_enabled: Mapping[str, bool] | None,
    generated: Mapping[str, GeneratedCourseMetadata] | None,
) -> TrackSamplingCourseEntry:
    course_key = _mapped_value(course_keys, track_id, track_id) or track_id
    return TrackSamplingCourseEntry(
        entry_id=track_id,
        base_weight=base_weight,
        course_key=course_key,
        log_key=_mapped_value(log_keys, track_id, course_key),
        label=_mapped_value(labels, track_id, course_key) or course_key,
        log_enabled=_mapped_bool(log_enabled, track_id, course_key, default=True),
        generated=(_mapped_generated(generated, track_id, course_key) or GeneratedCourseMetadata()),
    )


def _mapped_value(
    mapping: Mapping[str, str] | None,
    primary_key: str,
    fallback_key: str,
) -> str | None:
    if mapping is None:
        return None
    if primary_key in mapping:
        return mapping[primary_key]
    return mapping.get(fallback_key)


def _mapped_bool(
    mapping: Mapping[str, bool] | None,
    primary_key: str,
    fallback_key: str,
    *,
    default: bool,
) -> bool:
    if mapping is None:
        return default
    if primary_key in mapping:
        return bool(mapping[primary_key])
    if fallback_key in mapping:
        return bool(mapping[fallback_key])
    return default


def _mapped_generated(
    mapping: Mapping[str, GeneratedCourseMetadata] | None,
    primary_key: str,
    fallback_key: str,
) -> GeneratedCourseMetadata | None:
    if mapping is None:
        return None
    if primary_key in mapping:
        return mapping[primary_key]
    return mapping.get(fallback_key)
